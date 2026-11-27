"""HH trigger seeds: HH -> 4b and HH -> bb tau tau.

Event-level seeds on the leading jets of an event. Each goes through the
same stages as the tau seeds (seeds_tau), one section each below:

1. Candidates: events with at least 4 jets (MinBias, or the HH signal of
   config SIGNAL_SAMPLES hh4b, hhbbtt), the tagger's scores summed per
   event (bbbb: the b scores of the 4 leading jets; bbtt: the best tau
   pair of the 4 leading jets and the b scores of the other two), HT,
   and the extra selection.
2. Rate grid: the MinBias rate of every threshold point (bbbb: HT x b
   sum; bbtt: b sum x tau pair score at HT > BBTT_HT_CUT).
3. Working point: the grid points at the target rate, and of those the
   one with the highest signal efficiency.
4. Rate at the working point (bbbb: OR the HT-only trigger).
5. Signal efficiency versus generated HT and m_HH, against the HT-only
   trigger.

``score`` is raw (the class score) or vs_qg / qg (score / (score + light
+ gluon + pileup)). ``sel`` is all (no extra cut) or, for bbbb, sel
(four leading jets with pT > BBBB_SEL_JET_PT), for bbtt, tau (both tau
candidates with pT > BBTT_SEL_TAU_PT) or all (four leading jets with pT
> BBTT_SEL_JET_PT); all with |eta| < JET_ETA_CUT.

The tagger is the model's. For ``-m cmssw`` the bbbb seed uses the CMSSW
L1 baseline b-tag score (jet_bjetscore, raw only) through the same code,
at the fixed working point CMSSW_WPS btag_l1_ht and btag. The bbtt seed
needs the network's tau charge, which the CMSSW L1 baseline does not
have: for cmssw the bbtt variables are the CMSSW reference seeds instead
(the Run-3-like and the di-tau seed, with the NN PUPPI taus). The
network's bbtt working point is derived at the Run-3-like seed rate,
which it computes itself (bbttRun3SeedRate).
"""

import itertools
from typing import Literal

import awkward as ak
import hist
import numpy as np
from scipy.interpolate import interp1d

from tagger.data.tools import group_id_values
from tagger.eval.variable_manager import vm
from tagger.eval.variables.base.efficiency import efficiency
from tagger.eval.variables.base.rates import (
    closest_lower,
    counts_above,
    find_rate,
    hist_2d,
)
from tagger.eval.variables.base.scores import pileup_score, x_vs_y
from tagger.eval.variables.base.seeds import (
    expand_ranges,
    minbias,
    rate_khz,
    signal,
)
from tagger.eval.variables.base.taggers import (
    cmssw_ditau_seed,
    is_cmssw,
    run3_bbtt_seed,
)

BbbbScore, BbbbSel = Literal["raw", "vs_qg"], Literal["all", "sel"]
BbttScore, BbttSel = Literal["raw", "qg"], Literal["tau", "all"]
BbbbTrigger = Literal["wp", "wpPure", "htOnly", "wpOrHtOnly"]
BbttTrigger = Literal["nn", "nnTarget", "nnTargetPure", "htOnly"]


# ====================================================================
# HH -> 4b
# ======================================================================
# -- HH -> 4b: candidates ----------------------------------------------
def _bbbb_default_selection(config, jet_pt, jet_eta, sel):
    if sel == "sel":
        mask = (jet_pt[:, :4] > config["BBBB_SEL_JET_PT"]) & (
            np.abs(jet_eta[:, :4]) < config["JET_ETA_CUT"]
        )
        return np.sum(mask, axis=1) == 4
    return np.ones(len(jet_pt), dtype=bool)


def _bbbb_events(run, sample, score):
    """Events with at least 4 jets: jet pT, eta, the summed b score of
    the 4 leading jets and HT."""
    return run.memo(
        ("bbbb_events", sample, score),
        _compute_bbbb_events,
        run,
        sample,
        score,
    )


def _compute_bbbb_events(run, sample, score):
    """_bbbb_events without the cache."""
    ev = minbias(run) if sample == "minbias" else signal(run, sample)
    if is_cmssw(ev):  # the L1 b-tag score, -1 (no score) counts as 0
        if score != "raw":
            raise ValueError("the cmssw pseudo-model has only the raw b score")
        b = np.asarray(ev.branch("jet_bjetscore"))
        b = np.where(b == -1, 0.0, b)
    else:
        labels = run.class_labels()
        scores, _ = ev.outputs()
        b = x_vs_y(
            scores[:, labels["b"]],
            scores[:, labels["light"]]
            + scores[:, labels["gluon"]]
            + pileup_score(scores, labels),
            score == "vs_qg",
        )
    jet_pt, jet_eta, b = ev.grouped(
        ev.branch("jet_pt"), ev.branch("jet_eta_phys"), b, num_elements=4
    )
    cfg = run.config
    b = ak.where(
        (jet_pt > cfg["JET_PT_CUT"]) & (abs(jet_eta) < cfg["JET_ETA_CUT"]),
        b,
        0,
    )
    bsum = sum(b[:, i] for i in range(4))
    ht = ak.sum(
        jet_pt[
            (jet_pt > cfg["HT_JET_PT_CUT"])
            & (np.abs(jet_eta) < cfg["JET_ETA_CUT"])
        ],
        axis=1,
    )
    return {
        "jet_pt": jet_pt,
        "jet_eta": jet_eta,
        "bsum": bsum,
        "ht": ht,
        "n_events": ev.n_events,
    }


# -- HH -> 4b: rate grid -----------------------------------------------
@vm.variable(inputs={"minbias_events", "model"}, kind="curve")
def bbbbRateGrid(run, score: BbbbScore, sel: BbbbSel):
    """MinBias rate (kHz) of the HH -> 4b seed HT > ht and (sum of the b
    scores of the 4 leading jets) > nn, on a grid of config HH_SCANS
    bbbb_ht and bbbb_nn. Fields ``ht``, ``nn``, ``rate``. The b score is
    the tagger's: the network's class score (raw or vs_qg), or for -m
    cmssw the CMSSW L1 b-tag score (no score counts as 0). Jets with pT
    < JET_PT_CUT or |eta| > JET_ETA_CUT score 0; HT sums jets with pT >
    HT_JET_PT_CUT and |eta| < JET_ETA_CUT. Events need 4 jets."""
    e = _bbbb_events(run, "minbias", score)
    keep = _bbbb_default_selection(run.config, e["jet_pt"], e["jet_eta"], sel)
    scans = run.config["HH_SCANS"]
    ht_edges, nn_edges = expand_ranges(scans, "bbbb_ht"), expand_ranges(
        scans, "bbbb_nn"
    )
    h = hist_2d(ht_edges, nn_edges, "ht", "nn")
    h.fill(ht=e["ht"], nn=e["bsum"][keep] if sel == "sel" else e["bsum"])
    cum = counts_above(
        h.values(flow=True)[1:, 1:]
    )  # events with ht >= edge and nn >= edge
    hh, nn = np.meshgrid(ht_edges[:-1], nn_edges[:-1], indexing="ij")
    return {
        "ht": hh.ravel().astype(float),
        "nn": nn.ravel().astype(float),
        "rate": rate_khz(run, cum.ravel(), e["n_events"]),
    }


# -- HH -> 4b: working point -------------------------------------------
@vm.variable(inputs={"minbias_events", "model"}, kind="curve")
def bbbbWpLine(run, score: BbbbScore, sel: BbbbSel):
    """The bbbbRateGrid(score, sel) points within BBBB_RATE_RANGE of
    BBBB_TARGET_RATE. Fields ``ht``, ``nn``. For -m cmssw for comparison
    only: its working point is the fixed CMSSW_WPS one.
    """
    grid = run.get(f"bbbbRateGrid({score}, {sel})")
    idx = find_rate(
        list(grid["rate"]),
        run.config["BBBB_TARGET_RATE"],
        run.config["BBBB_RATE_RANGE"],
    )
    return {"ht": grid["ht"][idx], "nn": grid["nn"][idx]}


def _bbbb_wp(run, score, sel):
    """(HT, b sum) thresholds of the working point: for -m cmssw the
    fixed CMSSW_WPS btag_l1_ht and btag; for the network, of the HT
    values of config HH_SCANS bbbb_wp_ht with the b sum cut interpolated
    on bbbbWpLine, the one with the highest HH -> 4b efficiency.
    """
    return run.memo(("bbbb_wp", score, sel), _compute_bbbb_wp, run, score, sel)


def _compute_bbbb_wp(run, score, sel):
    """_bbbb_wp without the cache."""
    e = _bbbb_events(run, "hh4b", score)
    if is_cmssw(signal(run, "hh4b")):
        wps = run.config["CMSSW_WPS"]
        return float(wps["btag_l1_ht"]), float(wps["btag"])
    default = _bbbb_default_selection(
        run.config, e["jet_pt"], e["jet_eta"], sel
    )
    line = run.get(f"bbbbWpLine({score}, {sel})")
    f = interp1d(
        list(line["ht"]),
        list(line["nn"]),
        kind="linear",
        fill_value="extrapolate",
    )
    best, wp_ht, wp_nn = -1.0, -1.0, -1.0
    for cut in expand_ranges(run.config["HH_SCANS"], "bbbb_wp_ht"):
        nn = f(cut)
        eff = np.mean((e["ht"] > cut) & (e["bsum"] > nn) & default)
        if eff > best:
            best, wp_ht, wp_nn = eff, cut, float(nn)
    return float(wp_ht), wp_nn


@vm.variable(
    inputs={"signal_events", "minbias_events", "model"}, kind="scalar"
)
def bbbbWpHt(run, score: BbbbScore, sel: BbbbSel):
    """HT threshold (GeV) of the HH -> 4b working point: of the HT
    values of config HH_SCANS bbbb_wp_ht with the b score sum cut
    interpolated on bbbbWpLine(score, sel), the one with the highest HH
    -> 4b efficiency. For -m cmssw the fixed CMSSW_WPS btag_l1_ht."""
    return _bbbb_wp(run, score, sel)[0]


@vm.variable(
    inputs={"signal_events", "minbias_events", "model"}, kind="scalar"
)
def bbbbWpNn(run, score: BbbbScore, sel: BbbbSel):
    """b score sum threshold of the HH -> 4b working point at HT >
    bbbbWpHt(score, sel). For -m cmssw the fixed CMSSW_WPS btag."""
    return _bbbb_wp(run, score, sel)[1]


# -- HH -> 4b: rate at the working point -------------------------------
@vm.variable(
    inputs={"signal_events", "minbias_events", "model"}, kind="scalar"
)
def bbbbOrRate(run, score: BbbbScore, sel: BbbbSel):
    """MinBias rate (kHz) of the HH -> 4b working point OR the HT-only
    trigger."""
    e = _bbbb_events(run, "minbias", score)
    wp_ht, wp_nn = _bbbb_wp(run, score, sel)
    default = _bbbb_default_selection(
        run.config, e["jet_pt"], e["jet_eta"], sel
    )
    model = (e["ht"] > wp_ht) & (e["bsum"] > wp_nn) & default
    passed = model | (e["ht"] > int(run.get("htOnlyCut(bbbb)")))
    return float(np.round(rate_khz(run, np.sum(passed), e["n_events"]), 2))


# -- HH -> 4b: signal efficiency ---------------------------------------
def _bbbb_signal(run, score, sel):
    """Signal selections of every trigger (bbbbEffVsGenHt) and the gen
    HT and m_HH, memoised."""
    return run.memo(
        ("bbbb_signal", score, sel), _compute_bbbb_signal, run, score, sel
    )


def _compute_bbbb_signal(run, score, sel):
    """_bbbb_signal without the cache."""
    sig = signal(run, "hh4b")
    has_mhh = sig.has_branch("genHH_mass")
    event = sig.branch("event")
    all_pt, all_eta = group_id_values(
        event,
        sig.branch("jet_genmatch_pt"),
        sig.branch("jet_genmatch_eta"),
        num_elements=0,
    )[1]
    ht_pt, eta_cut = run.config["HT_JET_PT_CUT"], run.config["JET_ETA_CUT"]
    all_genht = ak.sum(
        all_pt[(all_pt > ht_pt) & (np.abs(all_eta) < eta_cut)], axis=1
    )  # as genht below
    arrays = [sig.branch("jet_genmatch_pt"), sig.branch("jet_genmatch_eta")]
    if has_mhh:
        arrays.append(sig.branch("genHH_mass"))
        all_mhh = ak.firsts(
            group_id_values(event, sig.branch("genHH_mass"), num_elements=0)[
                1
            ][0]
        )
    g = sig.grouped(*arrays, num_elements=4)
    genpt, geneta = g[:2]
    e = _bbbb_events(run, "hh4b", score)
    jet_pt, jet_eta, ht = e["jet_pt"], e["jet_eta"], e["ht"]
    genht = ak.sum(genpt[(genpt > ht_pt) & (np.abs(geneta) < eta_cut)], axis=1)
    ht_only = ht > int(run.get("htOnlyCut(bbbb)"))
    default = _bbbb_default_selection(run.config, jet_pt, jet_eta, sel)
    wp_ht, wp_nn = _bbbb_wp(run, score, sel)
    model = (ht > wp_ht) & (e["bsum"] > wp_nn) & default
    return {
        "n_events": sig.n_events,
        "all_genht": all_genht,
        "genht": genht,
        "all_mhh": all_mhh if has_mhh else None,
        "mhh": ak.firsts(g[2]) if has_mhh else None,
        "sel": {
            "wp": model,
            "wpPure": model & ~ht_only,
            "htOnly": ht_only,
            "wpOrHtOnly": model | ht_only,
        },
    }


@vm.variable(inputs={"signal_events", "minbias_events", "model"}, kind="curve")
def bbbbEffVsGenHt(run, trigger: BbbbTrigger, score: BbbbScore, sel: BbbbSel):
    """HH -> 4b efficiency versus generated HT. ``trigger``: wp (the
    model at its working point, HT > bbbbWpHt and b score sum >
    bbbbWpNn), htOnly (HT > htOnlyCut(bbbb)), wpPure (wp and not
    htOnly), wpOrHtOnly. Gen HT: gen-matched pT of the jets with gen pT
    > HT_JET_PT_CUT and |gen eta| < JET_ETA_CUT. Numerator: passing
    events (they have 4 jets); denominator: all signal events. Fields as
    tauEffVsGenPt.
    """
    s = _bbbb_signal(run, score, sel)
    edges = expand_ranges(run.config, "HH_GEN_HT_EDGES")
    return efficiency(s["genht"][s["sel"][trigger]], s["all_genht"], edges)


@vm.variable(inputs={"signal_events", "minbias_events", "model"}, kind="curve")
def bbbbEffVsGenMhh(run, trigger: BbbbTrigger, score: BbbbScore, sel: BbbbSel):
    """HH -> 4b efficiency versus generated m_HH; triggers as
    bbbbEffVsGenHt. Needs genHH_mass."""
    s = _bbbb_signal(run, score, sel)
    if s["mhh"] is None:
        raise KeyError("the hh4b sample has no genHH_mass branch")
    edges = expand_ranges(run.config, "HH_GEN_MHH_EDGES")
    return efficiency(s["mhh"][s["sel"][trigger]], s["all_mhh"], edges)


@vm.variable(
    inputs={"signal_events", "minbias_events", "model"}, kind="scalar"
)
def bbbbEff(run, trigger: BbbbTrigger, score: BbbbScore, sel: BbbbSel):
    """Integrated HH -> 4b efficiency of a trigger (as bbbbEffVsGenHt):
    passing events over all signal events."""
    s = _bbbb_signal(run, score, sel)
    return float(ak.sum(s["sel"][trigger]) / s["n_events"])


@vm.variable(
    inputs={"signal_events", "minbias_events", "model"}, kind="scalar"
)
def bbbbEffFourJets(run, trigger: BbbbTrigger, score: BbbbScore, sel: BbbbSel):
    """Integrated HH -> 4b efficiency of a trigger over the signal
    events with at least 4 jets (all signal events: bbbbEff)."""
    return float(np.mean(_bbbb_signal(run, score, sel)["sel"][trigger]))


# ====================================================================
# HH -> bb tau tau
# ======================================================================
# -- HH -> bb tau tau: candidates --------------------------------------
def _bbtt_choose(taup, taum, b):
    """Tau pair and b sum: the jet pair of the 4 leading jets with the
    highest (tau+_i + tau+_j)(tau-_i + tau-_j), and the b score sum of
    the other two jets.
    """
    pairs = list(
        itertools.combinations(range(4), 2)
    )  # the order of ak.argcombinations
    topo = np.array(
        [
            (taup[:, i] + taup[:, j]) * (taum[:, i] + taum[:, j])
            for i, j in pairs
        ]
    )
    best = np.argmax(topo, axis=0)
    tsum = np.max(topo, axis=0)
    tau_idx = np.array(pairs)[best]
    idx = np.tile(np.arange(4), (len(b), 1))
    idx[np.arange(len(idx)).reshape(-1, 1), tau_idx] = -1
    b_idx = np.sort(idx, axis=1)[:, -2:]
    return (
        b[np.arange(len(b)).reshape(-1, 1), b_idx].sum(axis=1),
        tsum,
        tau_idx,
    )


def _bbtt_events(run, sample):
    """Events with at least 4 jets: per score type raw and qg
    the b sum and tau pair score, the tau pair, HT of all jets."""
    return run.memo(("bbtt_events", sample), _compute_bbtt_events, run, sample)


def _compute_bbtt_events(run, sample):
    """_bbtt_events without the cache."""
    ev = minbias(run) if sample == "minbias" else signal(run, sample)
    if is_cmssw(ev):
        raise ValueError(
            "the bbtt seed needs the network's tau charge; the cmssw"
            " pseudo-model has none"
        )
    labels = run.class_labels()
    scores, _ = ev.outputs()
    jet_pt, jet_eta, sc = ev.grouped(
        ev.branch("jet_pt"), ev.branch("jet_eta_phys"), scores, num_elements=4
    )
    sc = np.asarray(sc[:, :4])  # (events, 4 jets, classes)
    bkg = sc[:, :, labels["light"]] + sc[:, :, labels["gluon"]]
    if "pileup" in labels:
        bkg = bkg + sc[:, :, labels["pileup"]]
    raw = {
        c: sc[:, :, labels[k]]
        for c, k in (("b", "b"), ("tp", "taup"), ("tm", "taum"))
    }
    out = {
        "jet_pt": jet_pt,
        "jet_eta": jet_eta,
        "ht": ak.sum(jet_pt, axis=1),
        "n_events": ev.n_events,
    }
    for name, d in (
        ("raw", raw),
        ("qg", {c: x_vs_y(v, bkg) for c, v in raw.items()}),
    ):
        out[name] = dict(
            zip(
                ("bsum", "tsum", "tau_idx"),
                _bbtt_choose(d["tp"], d["tm"], d["b"]),
            )
        )
    return out


def _bbtt_default_selection(config, e, score, sel):
    """Extra selection: sel tau needs both jets of the event's tau pair
    (of this score type) with pT > BBTT_SEL_TAU_PT; sel all needs the 4
    leading jets with pT > BBTT_SEL_JET_PT; both |eta| < JET_ETA_CUT.
    """
    jet_pt, jet_eta = e["jet_pt"], e["jet_eta"]
    eta_cut = config["JET_ETA_CUT"]
    if sel == "tau":
        indices = e[score][
            "tau_idx"
        ]  # (events, 2), positions among the 4 leading jets
        rows = np.arange(len(jet_pt)).reshape((-1, 1))
        pt4, eta4 = np.asarray(jet_pt[:, :4]), np.asarray(jet_eta[:, :4])
        mask = (pt4[rows, indices] > config["BBTT_SEL_TAU_PT"]) & (
            np.abs(eta4[rows, indices]) < eta_cut
        )
        return np.asarray(np.sum(mask, axis=1) == 2)
    if sel == "all":
        mask = (jet_pt[:, :4] > config["BBTT_SEL_JET_PT"]) & (
            np.abs(jet_eta[:, :4]) < eta_cut
        )
        return np.asarray(np.sum(mask, axis=1) == 4)
    raise ValueError(f"sel must be tau or all, got {sel!r}")


# -- HH -> bb tau tau: rate grid ---------------------------------------
@vm.variable(inputs={"minbias_events", "signal_events", "model"}, kind="curve")
def bbttRateGrid(run, sel: BbttSel):
    """MinBias rate (kHz) and HH -> bb tau tau efficiency of HT >
    BBTT_HT_CUT and b sum > bb and tau pair score > tt, on a (bb, tt)
    grid, for both score types. Fields ``bb``, ``tt``, ``rate_raw``,
    ``rate_qg``, ``eff_raw``, ``eff_qg``, on the edges of config
    HH_SCANS bbtt_nn. Only points with a rate within
    BBTT_GRID_KEEP_WINDOW of BBTT_TARGET_RATE or of the rounded
    bbttRun3SeedRate (for raw or qg) are kept. Events need 4 jets."""
    seed_rate, target = _bbtt_rate(run, "seed"), run.config["BBTT_TARGET_RATE"]
    window = run.config["BBTT_GRID_KEEP_WINDOW"]
    scans = run.config["HH_SCANS"]
    ht_edges, nn_edges = expand_ranges(scans, "bbtt_ht"), expand_ranges(
        scans, "bbtt_nn"
    )
    i_ht = ht_edges.index(run.config["BBTT_HT_CUT"])
    out = {}
    for sample, key in (("minbias", "rate"), ("hhbbtt", "eff")):
        e = _bbtt_events(run, sample)
        for score in ("raw", "qg"):
            keep = _bbtt_default_selection(run.config, e, score, sel)
            h = hist.Hist(
                *(
                    hist.axis.Variable(x, name=n)
                    for x, n in (
                        (ht_edges, "ht"),
                        (nn_edges, "bb"),
                        (nn_edges, "tt"),
                    )
                )
            )
            h.fill(
                ht=e["ht"][keep],
                bb=e[score]["bsum"][keep],
                tt=e[score]["tsum"][keep],
            )
            cum = counts_above(
                h.values(flow=True)[1 + i_ht :].sum(axis=0)[1:, 1:]
            )  # ht >= cut, overflow included
            out[f"{key}_{score}"] = (
                rate_khz(run, cum, e["n_events"])
                if key == "rate"
                else cum / e["n_events"]
            ).ravel()
    bb, tt = np.meshgrid(nn_edges[:-1], nn_edges[:-1], indexing="ij")
    keep = np.array(
        [
            _in_grid_window(a, target, seed_rate, window)
            or _in_grid_window(b, target, seed_rate, window)
            for a, b in zip(out["rate_raw"], out["rate_qg"])
        ],
        dtype=bool,
    )
    return {
        "bb": bb.ravel()[keep].astype(float),
        "tt": tt.ravel()[keep].astype(float),
        **{k: v[keep] for k, v in out.items()},
    }


def _in_grid_window(rate, target, seed_rate, window):
    """Whether a grid rate is within ``window`` of the target rate or of
    the seed rate."""
    return (
        target - window <= rate <= target + window
        or seed_rate - window <= rate <= seed_rate + window
    )


# -- HH -> bb tau tau: working point -----------------------------------
def _bbtt_rate(run, which):
    if which == "seed":
        return float(np.round(run.get("bbttRun3SeedRate"), 1))
    return run.config["BBTT_TARGET_RATE"]


def _bbtt_wp(run, score, sel, which):
    return run.memo(
        ("bbtt_wp", score, sel, which),
        _compute_bbtt_wp,
        run,
        score,
        sel,
        which,
    )


def _compute_bbtt_wp(run, score, sel, which):
    """_bbtt_wp without the cache."""
    grid = run.get(f"bbttRateGrid({sel})")
    rate = _bbtt_rate(run, which)
    rates = np.asarray(grid[f"rate_{score}"])
    idx = find_rate(list(rates), rate, run.config["BBTT_RATE_RANGE"])
    used = rate
    if not idx:  # none within the range: the closest lower rate
        idx = [closest_lower(rates, rate)]
        used = float(np.round(rates[idx[0]], 1))
    eff = np.asarray(grid[f"eff_{score}"])[idx]
    best = int(np.argmax(eff))
    return {
        "bb": grid["bb"][idx],
        "tt": grid["tt"][idx],
        "wp_bb": float(grid["bb"][idx][best]),
        "wp_tt": float(grid["tt"][idx][best]),
        "rate": float(used),
    }


@vm.variable(inputs={"minbias_events", "signal_events", "model"}, kind="curve")
def bbttWpLine(
    run, score: BbttScore, sel: BbttSel, rate: Literal["seed", "target"]
):
    """The bbttRateGrid(sel) points within BBTT_RATE_RANGE of a rate:
    ``rate`` seed (the rounded bbttRun3SeedRate) or target
    (BBTT_TARGET_RATE). Without any, the point with the closest lower
    rate. Fields ``bb``, ``tt``."""
    w = _bbtt_wp(run, score, sel, rate)
    return {"bb": w["bb"], "tt": w["tt"]}


@vm.variable(
    inputs={"minbias_events", "signal_events", "model"}, kind="scalar"
)
def bbttWpBb(
    run, score: BbttScore, sel: BbttSel, rate: Literal["seed", "target"]
):
    """b sum threshold of the bb tau tau working point: of the
    bbttWpLine(score, sel, rate) points, the one with the highest HH ->
    bb tau tau efficiency."""
    return _bbtt_wp(run, score, sel, rate)["wp_bb"]


@vm.variable(
    inputs={"minbias_events", "signal_events", "model"}, kind="scalar"
)
def bbttWpTt(
    run, score: BbttScore, sel: BbttSel, rate: Literal["seed", "target"]
):
    """Tau pair score threshold of the bb tau tau working point (see
    bbttWpBb)."""
    return _bbtt_wp(run, score, sel, rate)["wp_tt"]


@vm.variable(
    inputs={"minbias_events", "signal_events", "model"}, kind="scalar"
)
def bbttWpRate(
    run, score: BbttScore, sel: BbttSel, rate: Literal["seed", "target"]
):
    """Rate (kHz) the bb tau tau working point was picked at: the
    requested one, or the closest lower grid rate rounded to 0.1 kHz
    when no grid point is within range."""
    return _bbtt_wp(run, score, sel, rate)["rate"]


# -- HH -> bb tau tau: signal efficiency -------------------------------
def _bbtt_signal(run, score, sel):
    return run.memo(
        ("bbtt_signal", score, sel), _compute_bbtt_signal, run, score, sel
    )


def _compute_bbtt_signal(run, score, sel):
    """_bbtt_signal without the cache."""
    sig = signal(run, "hhbbtt")
    event = sig.branch("event")
    has_mhh = sig.has_branch("genHH_mass")
    gen_all = group_id_values(
        event,
        *([sig.branch("genHH_mass")] if has_mhh else []),
        sig.branch("jet_genmatch_pt"),
        num_elements=0,
    )[1]
    g = sig.grouped(
        sig.branch("jet_genmatch_pt"),
        *([sig.branch("genHH_mass")] if has_mhh else []),
        num_elements=4,
    )
    genpt = g[0]
    e = _bbtt_events(run, "hhbbtt")
    keep = _bbtt_default_selection(run.config, e, score, sel)
    target = _bbtt_wp_pass(run, e, score, sel, "target") & keep
    ht_only = e["ht"] > int(run.get("htOnlyCut(bbtt)"))
    return {
        "n_events": sig.n_events,
        "all_genht": ak.sum(gen_all[-1], axis=1),
        "genht": ak.sum(genpt, axis=1),
        "all_mhh": ak.firsts(gen_all[0]) if has_mhh else None,
        "mhh": ak.firsts(g[1]) if has_mhh else None,
        "sel": {
            "nn": _bbtt_wp_pass(run, e, score, sel, "seed") & keep,
            "nnTarget": target,
            "nnTargetPure": target & ~ht_only,
            "htOnly": ht_only,
        },
    }


def _bbtt_wp_pass(run, e, score, sel, which):
    """Events of ``e`` (_bbtt_events) passing the bbtt working point
    ``which`` (seed or target): HT > BBTT_HT_CUT, b sum > bbttWpBb and
    tau pair score > bbttWpTt."""
    bb = run.get(f"bbttWpBb({score}, {sel}, {which})")
    tt = run.get(f"bbttWpTt({score}, {sel}, {which})")
    return (
        (e["ht"] > run.config["BBTT_HT_CUT"])
        & (e[score]["bsum"] > bb)
        & (e[score]["tsum"] > tt)
    )


@vm.variable(inputs={"signal_events", "minbias_events", "model"}, kind="curve")
def bbttEffVsGenHt(run, trigger: BbttTrigger, score: BbttScore, sel: BbttSel):
    """HH -> bb tau tau efficiency versus generated HT (all jets).
    ``trigger``: nn (working point at the rate of the Run-3-like seed),
    nnTarget (at BBTT_TARGET_RATE), htOnly (HT > htOnlyCut(bbtt)),
    nnTargetPure (nnTarget and not htOnly). Numerator: events with 4
    jets; denominator: all signal events. Fields as tauEffVsGenPt."""
    s = _bbtt_signal(run, score, sel)
    edges = expand_ranges(run.config, "HH_GEN_HT_EDGES")
    return efficiency(s["genht"][s["sel"][trigger]], s["all_genht"], edges)


@vm.variable(inputs={"signal_events", "minbias_events", "model"}, kind="curve")
def bbttEffVsGenMhh(run, trigger: BbttTrigger, score: BbttScore, sel: BbttSel):
    """HH -> bb tau tau efficiency versus generated m_HH; triggers as
    bbttEffVsGenHt. Needs genHH_mass."""
    s = _bbtt_signal(run, score, sel)
    if s["mhh"] is None:
        raise KeyError("the hhbbtt sample has no genHH_mass branch")
    edges = expand_ranges(run.config, "HH_GEN_MHH_EDGES")
    return efficiency(s["mhh"][s["sel"][trigger]], s["all_mhh"], edges)


@vm.variable(
    inputs={"signal_events", "minbias_events", "model"}, kind="scalar"
)
def bbttEff(run, trigger: BbttTrigger, score: BbttScore, sel: BbttSel):
    """Integrated HH -> bb tau tau efficiency of a trigger (as
    bbttEffVsGenHt): passing events over all signal events."""
    s = _bbtt_signal(run, score, sel)
    return float(np.sum(s["sel"][trigger]) / s["n_events"])


# -- HH -> bb tau tau: CMSSW reference seeds (-m cmssw) ----------------
@vm.variable(inputs={"minbias_events"}, kind="scalar")
def bbttRun3SeedRate(run):
    """MinBias rate (kHz) of the Run-3-like bb tau tau seed (config
    BBTT_RUN3_SEED), with the CMSSW taus (score >= CMSSW_WPS tau),
    events with at least 2 jets. The bbtt working points are derived at
    this rate rounded to 0.1 kHz, and at BBTT_TARGET_RATE.
    """
    mb = minbias(run)
    jet_pt, jet_eta, tau_pt, tau = mb.grouped(
        mb.branch("jet_pt"),
        mb.branch("jet_eta_phys"),
        mb.branch("jet_taupt"),
        mb.branch("jet_tauscore"),
        num_elements=2,
    )
    passed = np.sum(
        run3_bbtt_seed(
            jet_pt,
            jet_eta,
            tau_pt,
            tau,
            run.config["CMSSW_WPS"]["tau"],
            run.config["BBTT_RUN3_SEED"],
        )
    )
    return float(rate_khz(run, passed, mb.n_events))


def _bbtt_reference(run):
    return run.memo("bbtt_reference", _compute_bbtt_reference, run)


def _compute_bbtt_reference(run):
    """_bbtt_reference without the cache."""
    sig = signal(run, "hhbbtt")
    wps = run.config["CMSSW_WPS"]
    all_genpt = group_id_values(
        sig.branch("event"), sig.branch("jet_genmatch_pt"), num_elements=0
    )[1][0]
    jet_pt, jet_eta, genpt, tau_pt, tau = sig.grouped(
        sig.branch("jet_pt"),
        sig.branch("jet_eta_phys"),
        sig.branch("jet_genmatch_pt"),
        sig.branch("jet_taupt"),
        sig.branch("jet_tauscore"),
        num_elements=4,
    )
    return {
        "n_events": sig.n_events,
        "all_genht": ak.sum(all_genpt, axis=1),
        "genht": ak.sum(genpt, axis=1),
        "sel": {
            "run3": run3_bbtt_seed(
                jet_pt,
                jet_eta,
                tau_pt,
                tau,
                wps["tau"],
                run.config["BBTT_RUN3_SEED"],
            ),
            "ditau": cmssw_ditau_seed(tau_pt, tau, jet_eta, wps),
        },
    }


@vm.variable(inputs={"signal_events"}, kind="curve")
def bbttRefEffVsGenHt(run, trigger: Literal["run3", "ditau"]):
    """HH -> bb tau tau efficiency of a CMSSW reference seed versus
    generated HT (all jets), with the NN PUPPI taus (jet_tauscore,
    jet_taupt). ``trigger``: run3 (the Run-3-like seed, see
    bbttRun3SeedRate) or ditau (the CMSSW di-tau seed: two taus with pT
    > CMSSW_WPS tau_l1_pt, score > CMSSW_WPS tau, |eta| < CMSSW_WPS
    tau_eta). Numerator: events with 4 jets; denominator: all signal
    events. Fields as tauEffVsGenPt."""
    s = _bbtt_reference(run)
    edges = expand_ranges(run.config, "HH_GEN_HT_EDGES")
    return efficiency(s["genht"][s["sel"][trigger]], s["all_genht"], edges)


@vm.variable(inputs={"signal_events"}, kind="scalar")
def bbttRefEff(run, trigger: Literal["run3", "ditau"]):
    """Integrated efficiency of a CMSSW reference seed (as
    bbttRefEffVsGenHt): passing events over all signal events."""
    s = _bbtt_reference(run)
    return float(np.sum(s["sel"][trigger]) / s["n_events"])


# ====================================================================
# Both HH seeds
# ======================================================================
# -- HH: HT-only trigger and gen distributions -------------------------
def _ht_only_cut(run, ht, n_events, ht_edges, target, rate_range):
    h = hist.Hist(hist.axis.Variable(ht_edges, name="ht"))
    h.fill(ht=ht)
    # overflow included, as the old slice(ht*1j, None, sum)
    vals = h.values(flow=True)[1:]
    rates = [
        rate_khz(run, vals[i:].sum(), n_events)
        for i in range(len(ht_edges) - 1)
    ]
    # none within the range: the closest lower rate
    idx = find_rate(rates, target, rate_range) or [
        closest_lower(rates, target)
    ]
    return float(ht_edges[idx[0]])


@vm.variable(inputs={"minbias_events", "model"}, kind="scalar")
def htOnlyCut(run, seed: Literal["bbbb", "bbtt"]):
    """HT threshold (GeV) of the HT-only trigger at the seed's target
    rate: the lowest grid HT whose MinBias rate is within the range of
    the target, else the one with the closest lower rate. ``seed`` bbbb
    (HT of jets with pT > HT_JET_PT_CUT and |eta| < JET_ETA_CUT, the HT
    edges of bbbbRateGrid, range BBBB_HT_ONLY_RATE_RANGE around
    BBBB_TARGET_RATE) or bbtt (HT of all jets, the HT edges of
    bbttRateGrid, range BBTT_RATE_RANGE around BBTT_TARGET_RATE). Events
    need 4 jets.
    """
    if seed == "bbbb":
        e = _bbbb_events(run, "minbias", "raw")
        return _ht_only_cut(
            run,
            e["ht"],
            e["n_events"],
            expand_ranges(run.config["HH_SCANS"], "bbbb_ht"),
            run.config["BBBB_TARGET_RATE"],
            run.config["BBBB_HT_ONLY_RATE_RANGE"],
        )
    e = _bbtt_events(run, "minbias")
    return _ht_only_cut(
        run,
        e["ht"],
        e["n_events"],
        expand_ranges(run.config["HH_SCANS"], "bbtt_ht"),
        run.config["BBTT_TARGET_RATE"],
        run.config["BBTT_RATE_RANGE"],
    )


@vm.variable(inputs={"signal_events"}, kind="curve")
def hhGenHist(
    run, sample: Literal["hh4b", "hhbbtt"], quantity: Literal["ht", "mhh"]
):
    """Normalised distribution of a generated quantity of the signal
    events with at least 4 jets, in config HH_GEN_HIST_EDGES (values
    clipped to HH_GEN_HIST_MAX). ``sample`` hh4b or hhbbtt;
    ``quantity`` ht (hh4b: jets with pT > HT_JET_PT_CUT and |eta| <
    JET_ETA_CUT; hhbbtt: all jets) or mhh. Fields ``edges``,
    ``counts``."""
    sig = signal(run, sample)
    if quantity == "mhh":
        values = ak.firsts(
            sig.grouped(sig.branch("genHH_mass"), num_elements=4)[0]
        )
    else:
        genpt, geneta = sig.grouped(
            sig.branch("jet_genmatch_pt"),
            sig.branch("jet_genmatch_eta"),
            num_elements=4,
        )
        if sample == "hh4b":
            ht_jets = (genpt > run.config["HT_JET_PT_CUT"]) & (
                np.abs(geneta) < run.config["JET_ETA_CUT"]
            )
            values = ak.sum(genpt[ht_jets], axis=1)
        else:
            values = ak.sum(genpt, axis=1)
    return _bkg_hist(run.config, values)


def _bkg_hist(config, values):
    counts, edges = np.histogram(
        np.clip(np.asarray(values), 0, config["HH_GEN_HIST_MAX"]),
        bins=expand_ranges(config, "HH_GEN_HIST_EDGES"),
    )
    return {"edges": edges.astype(float), "counts": counts / np.sum(counts)}
