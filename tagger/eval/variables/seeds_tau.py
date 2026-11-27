"""Tau trigger seeds: single tau, di-tau and di-tau topology.

Every seed goes through the same stages, one section each below:

1. Candidates: the events (MinBias, or the TAU_SIGNAL sample VBF H ->
   tautau), the tagger's tau score and pT per jet, and the candidate
   selection (|eta| < TAU_ETA_CUT; di-tau: deltaR > TAU_PAIR_MIN_DR).
2. Rate grid: the MinBias rate of every (pT threshold, score threshold)
   point.
3. Working point: the grid points within ``rate_range`` of the target
   rate (config TAU_SEEDS), interpolated linearly in pT.
4. Rate at the working point: the MinBias rate versus pT, raw and at the
   working point.
5. Signal efficiency: the efficiency on gen-matched signal taus at the
   working point.

The tagger is the model's: the network (class scores, regressed pT) or,
for ``-m cmssw``, the CMSSW L1 baseline NN PUPPI tau (jet_tauscore,
jet_taupt; not the emulation of our model), through the same code. Its
singleTau working point is derived on its rate grid like the network's;
for diTau and diTauTopo it has the fixed CMSSW_WPS tau score, as in the
old scripts.
"""

from functools import partial
from typing import Literal

import awkward as ak
import numpy as np
from scipy.interpolate import interp1d

from tagger.data.tools import sort_arrays
from tagger.eval.variable_manager import vm
from tagger.eval.variables.base.efficiency import efficiency, ratio_2d
from tagger.eval.variables.base.geometry import (
    delta_r,
    eta_region,
    min_of_leading_two,
)
from tagger.eval.variables.base.rates import (
    find_rate,
    hist_2d,
    rate_points,
    threshold_grid,
    wp_per_pt,
)
from tagger.eval.variables.base.seeds import (
    expand_ranges,
    labels,
    minbias,
    signal,
)
from tagger.eval.variables.base.taggers import is_cmssw, tau_jets, topo_parts

Seed = Literal["singleTau", "diTau", "diTauTopo"]
DiTauSeed = Literal["diTau", "diTauTopo"]
EffSeed = Literal["singleTau", "diTau"]
Region = Literal["barrel", "tau_endcap"]


# -- Candidates: events, tagger score and selection --------------------
def _signal(run):
    return signal(run, run.config["TAU_SIGNAL"])


def _pair_cuts(config, eta, phi):
    """Both leading candidates within |eta| < TAU_ETA_CUT and deltaR >
    TAU_PAIR_MIN_DR between them."""
    eta_cut = config["TAU_ETA_CUT"]
    return (
        (np.abs(eta[:, 0]) < eta_cut)
        & (np.abs(eta[:, 1]) < eta_cut)
        & (
            delta_r(eta[:, 0], phi[:, 0], eta[:, 1], phi[:, 1])
            > config["TAU_PAIR_MIN_DR"]
        )
    )


def _single_tau_candidates(run, ev):
    """(pT, score) of each event's candidate for a pT threshold: of the
    jets above it with |eta| < TAU_ETA_CUT, the one with the highest tau
    score."""
    score, corr_pt = tau_jets(ev, "singleTau", labels(run, ev))
    eta, corr_pt, score = ev.grouped(
        ev.branch("jet_eta_phys"), corr_pt, score, num_elements=1
    )
    score = ak.where(np.abs(eta) < run.config["TAU_ETA_CUT"], score, 0.0)
    return partial(_best_single_tau, corr_pt, score)


def _best_single_tau(corr_pt, score, pt):
    """(pT, score) per event of the jet with the highest score among
    those with pT >= pt."""
    s = ak.where(corr_pt < pt, 0.0, score)
    best = ak.argmax(s, axis=1, keepdims=True)
    return np.asarray(corr_pt[best]).flatten(), np.asarray(s[best]).flatten()


def _ditau_candidates(run, ev, topo):
    """(smaller pT, score) of each event's candidate pair for a pT
    threshold: of the jets above it, the two highest tau scores, both
    |eta| < TAU_ETA_CUT and deltaR > TAU_PAIR_MIN_DR. The score is the
    smaller tau score (diTau) or the pair score (diTauTopo, see
    topo_parts)."""
    cut = run.config["TAU_ETA_CUT"]
    rank, corr_pt = tau_jets(ev, "diTau", labels(run, ev))
    parts, pair = topo_parts(ev, labels(run, ev)) if topo else ((), None)
    g = ev.grouped(
        rank,
        *parts,
        corr_pt,
        ev.branch("jet_eta_phys"),
        ev.branch("jet_phi_phys"),
        num_elements=2,
    )
    rank, g_parts, jet_pt, jet_eta, jet_phi = (
        g[0],
        g[1:-3],
        g[-3],
        g[-2],
        g[-1],
    )
    rank = ak.where(np.abs(jet_eta) > cut, 0.0, rank)
    return partial(
        _best_tau_pair,
        run.config,
        pair,
        rank,
        g_parts,
        jet_pt,
        jet_eta,
        jet_phi,
    )


def _best_tau_pair(config, pair, rank, parts, jet_pt, jet_eta, jet_phi, pt):
    """(smaller pT, score) per event of the two highest ranked jets with
    pT >= pt that pass the pair cuts: the pair score ``pair(*parts)``
    (diTauTopo) or, without ``pair``, the smaller rank (diTau).
    """
    r = ak.where(jet_pt < pt, 0.0, rank)
    s = sort_arrays(
        r, *(parts if pair is not None else (r,)), jet_pt, jet_eta, jet_phi
    )
    cuts = _pair_cuts(config, s[-2], s[-1])
    nn = (
        np.asarray(pair(*s[:-3])[cuts])
        if pair is not None
        else min_of_leading_two(s[0], cuts)
    )
    return min_of_leading_two(s[-3], cuts), nn


def _signal_jets(run, seed):
    """Per-jet signal arrays (flavour, gen pT, gen eta, gen deltaR) and
    the tagger's tau score and pT."""
    return run.memo(("tau_signal_jets", seed), _compute_signal_jets, run, seed)


def _compute_signal_jets(run, seed):
    """_signal_jets without the cache."""
    sig = _signal(run)
    score, pt = tau_jets(sig, seed, labels(run, sig))
    return {
        "flav": sig.branch("jet_tauflav"),
        "gen_pt": sig.branch("jet_genmatch_pt"),
        "gen_eta": sig.branch("jet_genmatch_eta"),
        "gen_dr": sig.branch("jet_genmatch_dR"),
        "score": score,
        "pt": pt,
    }


# -- Rate grid ---------------------------------------------------------
@vm.variable(inputs={"minbias_events", "model"}, kind="curve")
def tauRateGrid(run, seed: Seed):
    """MinBias rate (kHz) of a tau seed on a grid of thresholds (config
    TAU_SCANS <seed>_pt and <seed>_nn). Fields ``pt`` (minimum pT
    threshold, GeV), ``nn`` (score threshold) and ``rate``, one entry
    per grid point.

    Scores and pT are the tagger's: for the network the class scores and
    the regressed pT, for -m cmssw the NN PUPPI tau score and pT. The
    candidates are the jets with the highest score that pass the pT
    threshold and |eta| < TAU_ETA_CUT; di-tau seeds also require deltaR
    > TAU_PAIR_MIN_DR between the two and cut on the smaller pT and
    score. diTauTopo ranks jets by tau score and cuts on the pair score:
    for the network the topology score (tau+_1 tau-_2 + tau-_1 tau+_2) /
    (that + bkg_1 bkg_2), for the CMSSW L1 baseline (no tau charge) the
    smaller NN PUPPI tau score.
    """
    mb = minbias(run)
    if seed == "singleTau":
        candidates = _single_tau_candidates(run, mb)
    else:
        candidates = _ditau_candidates(run, mb, topo=seed == "diTauTopo")
    scans = run.config["TAU_SCANS"]
    pt_edges, nn_edges = expand_ranges(scans, f"{seed}_pt"), expand_ranges(
        scans, f"{seed}_nn"
    )
    return threshold_grid(
        pt_edges, nn_edges, candidates, mb.n_events, run.config["MINBIAS_RATE"]
    )


# -- Working point -----------------------------------------------------
# pT range of the working point line (the y range of the old *_WPs
# plots)
_WP_LINE_PT = {
    "singleTau": (50, 200),
    "diTau": (10, 100),
    "diTauTopo": (10, 100),
}


@vm.variable(inputs={"minbias_events", "model"}, kind="curve")
def tauWpLine(run, seed: Seed):
    """Working point line of a tau seed: the tauRateGrid(seed) points
    within rate_range of the target rate (config TAU_SEEDS), as fields
    ``target_pt`` and ``target_nn``, and their linear interpolation
    (extrapolated) on 100 pT values, fields ``pt`` and ``nn``. Of
    several points at the same pT the interpolation uses the highest
    score. For the CMSSW L1 baseline only the singleTau line is its
    working point; diTau and diTauTopo use the fixed CMSSW_WPS tau score
    (see tauWp).
    """
    grid = run.get(f"tauRateGrid({seed})")
    cfg = run.config["TAU_SEEDS"][seed]
    idx = find_rate(list(grid["rate"]), cfg["target_rate"], cfg["rate_range"])
    target_pt, target_nn = grid["pt"][idx], grid["nn"][idx]
    pt = np.linspace(*_WP_LINE_PT[seed], 100)
    nn = interp1d(
        *wp_per_pt(target_pt, target_nn),
        kind="linear",
        fill_value="extrapolate",
    )(pt)
    return {"pt": pt, "nn": nn, "target_pt": target_pt, "target_nn": target_nn}


@vm.variable(inputs={"minbias_events", "model"}, kind="scalar")
def tauWp(run, seed: DiTauSeed):
    """Score threshold of a di-tau seed at the CMSSW pT threshold
    CMSSW_WPS["tau_l1_pt"]: for the network interpolated on
    tauWpLine(seed) (highest score per pT), for the CMSSW L1 baseline
    (-m cmssw) the fixed CMSSW_WPS["tau"].
    """
    if is_cmssw(minbias(run)):
        return float(run.config["CMSSW_WPS"]["tau"])
    line = run.get(f"tauWpLine({seed})")
    f = interp1d(
        *wp_per_pt(line["target_pt"], line["target_nn"]),
        kind="linear",
        fill_value="extrapolate",
    )
    return float(f(run.config["CMSSW_WPS"]["tau_l1_pt"]))


def _single_tau_wp_interp(run):
    """Score cut versus pT threshold: of the on-target points at the
    same pT the highest (most conservative) score, interpolated linearly
    in pT, clamped outside."""
    line = run.get("tauWpLine(singleTau)")
    pts, nn = wp_per_pt(line["target_pt"], line["target_nn"])
    return interp1d(
        pts,
        nn,
        kind="linear",
        fill_value=(np.max(nn), np.min(nn)),
        bounds_error=False,
    )


# -- Rate at the working point -----------------------------------------
def _curve(pt, rate, err):
    return {"pt": np.asarray(pt, dtype=float), "rate": rate, "err": err}


@vm.variable(inputs={"minbias_events", "model"}, kind="curve")
def tauRateVsPt(run, seed: DiTauSeed, cut: Literal["raw", "wp"]):
    """MinBias rate (kHz) versus a pT threshold (config TAU_SCANS
    <seed>_rate_pt) for a di-tau seed. Fields ``pt``, ``rate`` and
    ``err`` (sqrt(N) of the passing events). The rate is normalised to
    all MinBias events.

    ``cut`` raw: jets without identification, uncorrected jet pT (the
    same for every model); wp: the tagger's pT and score (network or
    CMSSW L1 baseline) at its working point, score > tauWp(seed). diTau:
    jets with |eta| < DITAU_RATE_ETA_CUT, an event passes if any jet
    passes. diTauTopo: the smaller pT of the two leading candidates with
    |eta| < TAU_ETA_CUT and deltaR > TAU_PAIR_MIN_DR; raw takes the two
    leading jets in pT, wp the candidates of tauRateGrid at each
    threshold (the two highest tau scores among the jets above it) and
    cuts on the pair score.
    """
    mb = minbias(run)
    minbias_rate = run.config["MINBIAS_RATE"]
    n_event = mb.n_events
    if seed == "diTau":
        eta_sel = (
            np.abs(mb.branch("jet_eta_phys").to_numpy())
            < run.config["DITAU_RATE_ETA_CUT"]
        )
        event_id = mb.branch("event")[eta_sel]
        cuts = expand_ranges(run.config["TAU_SCANS"], "diTau_rate_pt")
        if cut == "raw":
            return _curve(
                cuts,
                *rate_points(
                    event_id,
                    mb.branch("jet_pt")[eta_sel],
                    cuts,
                    n_event,
                    minbias_rate,
                ),
            )
        score, pt = tau_jets(mb, "diTau", labels(run, mb))
        sel = score[eta_sel] > run.get("tauWp(diTau)")
        return _curve(
            cuts,
            *rate_points(
                event_id[sel], pt[eta_sel][sel], cuts, n_event, minbias_rate
            ),
        )

    # diTauTopo
    cuts = expand_ranges(run.config["TAU_SCANS"], "diTauTopo_rate_pt")
    eta, phi = mb.branch("jet_eta_phys"), mb.branch("jet_phi_phys")

    if cut == "raw":
        event_id, (jet_pt, jet_eta, jet_phi) = mb.grouped(
            mb.branch("jet_pt").to_numpy(),
            eta,
            phi,
            num_elements=2,
            with_event_id=True,
        )
        c = _pair_cuts(run.config, jet_eta, jet_phi)
        return _curve(
            cuts,
            *rate_points(
                event_id[c],
                min_of_leading_two(jet_pt, c),
                cuts,
                n_event,
                minbias_rate,
                flatten=True,
            ),
        )
    wp, candidates = run.get("tauWp(diTauTopo)"), _ditau_candidates(
        run, mb, topo=True
    )
    n_pass = []
    for pt_cut in cuts:
        pt, nn = candidates(pt_cut)
        n_pass.append(np.sum((pt > pt_cut) & (nn > wp)))
    n_pass = np.asarray(n_pass, dtype=float)
    return _curve(
        cuts,
        n_pass / n_event * minbias_rate,
        np.sqrt(n_pass) / n_event * minbias_rate,
    )


# -- Signal efficiency -------------------------------------------------
def _gen_matched_taus(config, j, seed):
    """Jets matched to a gen tau: tau flavour, gen pT > TAU_GEN_PT_MIN
    of the seed, deltaR < TAU_GEN_MATCH_DR."""
    return (
        (j["flav"] == 1)
        & (j["gen_pt"] > config["TAU_GEN_PT_MIN"][seed])
        & (np.abs(j["gen_dr"]) < config["TAU_GEN_MATCH_DR"])
    )


def _selections(run, seed, region):
    """Denominator and numerators per cut, and the bin edges, for the
    gen-pT efficiencies."""
    j = _signal_jets(run, seed)
    wps, cut, barrel = (
        run.config["CMSSW_WPS"],
        run.config["TAU_ETA_CUT"],
        run.config["BARREL_ETA"],
    )
    gen_match = _gen_matched_taus(run.config, j, seed)
    if seed == "diTau":
        deno = gen_match & eta_region(j["gen_eta"], region, barrel, cut)
        pt_wp, wp = wps["tau_l1_pt"], run.get("tauWp(diTau)")
    else:
        deno = (
            gen_match
            & (np.abs(j["gen_eta"]) < cut)
            & eta_region(j["gen_eta"], region, barrel, cut)
        )
        pt_wp = run.config["TAU_SEEDS"]["singleTau"]["wp_pt"]
        wp = _single_tau_wp_interp(run)(pt_wp)
    edges = expand_ranges(run.config["TAU_EFF_PT_EDGES"], seed)
    num = {
        "pt": deno & (j["pt"] > pt_wp),
        "wp": deno & (j["pt"] > pt_wp) & (j["score"] > wp),
    }
    return j, deno, num, edges


@vm.variable(inputs={"signal_events", "minbias_events", "model"}, kind="curve")
def tauEffVsGenPt(
    run, seed: EffSeed, cut: Literal["pt", "wp"], region: Region
):
    """Efficiency of a tau seed on gen-matched signal taus versus gen
    tau pT, per jet. Fields ``edges``, ``num``, ``den``, ``eff`` and
    Clopper-Pearson ``err_low``, ``err_up``.

    ``seed`` is diTau (pT > CMSSW_WPS["tau_l1_pt"], score >
    tauWp(diTau)) or singleTau (pT > TAU_SEEDS singleTau wp_pt, score
    above the working point line there). ``cut`` wp (pT and score) or pt
    (the pT cut alone, the upper bound of the identification). The pT is
    the tagger's: for the network the seeded cone jet with pT
    regression, for the CMSSW L1 baseline the NN PUPPI tau pT.
    ``region`` is barrel (|eta| < BARREL_ETA) or tau_endcap (BARREL_ETA
    < |eta| < TAU_ETA_CUT), on the gen eta.
    """
    j, deno, num, edges = _selections(run, seed, region)
    return efficiency(j["gen_pt"][num[cut]], j["gen_pt"][deno], edges)


@vm.variable(
    inputs={"signal_events", "minbias_events", "model"}, kind="scalar"
)
def tauTotalEff(run, seed: EffSeed, cut: Literal["pt", "wp"], region: Region):
    """Integrated efficiency of tauEffVsGenPt(seed, cut, region):
    passing / denominator jets."""
    _, deno, num, _ = _selections(run, seed, region)
    if seed == "diTau":
        return float(np.mean(num[cut]) / np.mean(deno))
    return float(np.sum(num[cut]) / np.sum(deno))


@vm.variable(inputs={"signal_events", "minbias_events", "model"}, kind="curve")
def singleTauEffVsThreshold(run, cut: Literal["pt", "wp"]):
    """Single tau efficiency versus the pT threshold, with the score cut
    of the working point line at each threshold. Fields ``pt``, ``eff``,
    ``err`` (sqrt(N)/den).

    Gen taus with |eta| < TAU_ETA_CUT, gen pT > TAU_GEN_PT_MIN singleTau
    and deltaR < TAU_GEN_MATCH_DR, and gen pT above the threshold.
    Thresholds: config TAU_SCANS singleTau_eff_pt; thresholds with fewer
    than SINGLE_TAU_MIN_GEN_TAUS such taus get 0. ``cut`` wp (pT and
    score) or pt (pT alone).
    """
    j = _signal_jets(run, "singleTau")
    eta_cut = run.config["TAU_ETA_CUT"]
    gen_cuts = _gen_matched_taus(run.config, j, "singleTau") & (
        np.abs(j["gen_eta"]) < eta_cut
    )
    f = _single_tau_wp_interp(run)
    pt_edges = np.asarray(
        expand_ranges(run.config["TAU_SCANS"], "singleTau_eff_pt")
    )
    counts, denoms = [], []
    for pt_cut in pt_edges:
        deno = gen_cuts & (j["gen_pt"] > pt_cut)
        d = np.sum(deno)
        denoms.append(d)
        if d < run.config["SINGLE_TAU_MIN_GEN_TAUS"]:
            counts.append(0.0)
            continue
        sel = deno & (j["pt"] > pt_cut)
        if cut == "wp":
            sel = sel & (j["score"] > max(f(pt_cut), 0.0))
        counts.append(np.sum(sel))
    denoms = np.maximum(
        np.array(denoms, dtype=float), 1.0
    )  # no gen tau: efficiency 0
    return {
        "pt": pt_edges.astype(float),
        "eff": np.array(counts) / denoms,
        "err": np.sqrt(counts) / denoms,
    }


def _topo_eff_counts(run):
    """Di-tau topology efficiency counts in (gen pT1, gen pT2), memoised."""
    return run.memo("topo_eff_counts", _compute_topo_eff_counts, run)


def _compute_topo_eff_counts(run):
    """_topo_eff_counts without the cache."""
    sig = _signal(run)
    cut, wps = run.config["TAU_ETA_CUT"], run.config["CMSSW_WPS"]
    eta, phi = sig.branch("jet_eta_phys"), sig.branch("jet_phi_phys")
    wp, pt_wp = run.get("tauWp(diTauTopo)"), wps["tau_l1_pt"]
    rank, corr_pt = tau_jets(sig, "diTau", labels(run, sig))
    parts, pair = topo_parts(sig, labels(run, sig))
    order = np.where(
        (corr_pt < pt_wp) | (np.abs(np.asarray(eta)) > cut), 0.0, rank
    )  # as tauRateGrid
    g = sig.grouped(
        *parts, corr_pt, eta, phi, num_elements=2, ordering_var=order
    )
    g_parts, jet_pt, jet_eta, jet_phi = g[:-3], g[-3], g[-2], g[-1]
    flav, genpt = sig.grouped(
        sig.branch("jet_tauflav"),
        sig.branch("jet_genmatch_pt"),
        num_elements=2,
        ordering_var=sig.branch("jet_tauflav"),
    )
    genpt1, genpt2 = np.asarray(genpt[:, 0]), np.asarray(genpt[:, 1])
    gen_pt_min = run.config["TAU_GEN_PT_MIN"]["diTauTopo"]
    gen_mask = np.asarray(
        (flav[:, 0] == 1)
        & (flav[:, 1] == 1)
        & (genpt1 > gen_pt_min)
        & (genpt2 > gen_pt_min)
    )
    cuts = np.asarray(gen_mask & _pair_cuts(run.config, jet_eta, jet_phi))
    passed = (min_of_leading_two(jet_pt, cuts) > pt_wp) & (
        np.asarray(pair(*g_parts))[cuts] > wp
    )

    edges = np.asarray(run.config["TOPO_EFF_PT_EDGES"])
    n1, _ = np.histogram(genpt1, edges)
    n2, _ = np.histogram(genpt2, edges)
    return {
        "edges": edges.astype(float),
        "all": _counts_2d(edges, genpt1[gen_mask], genpt2[gen_mask]),
        "pass": _counts_2d(edges, genpt1[cuts][passed], genpt2[cuts][passed]),
        "n_pass": np.sum(passed),
        "denom": np.sum(gen_mask),
        "genpt1_frac": n1 / np.sum(n1),
        "genpt2_frac": n2 / np.sum(n2),
    }


def _counts_2d(edges, x, y):
    """Counts of (x, y) in bins of ``edges`` on both axes."""
    h = hist_2d(edges, edges, "a", "b")
    h.fill(a=x, b=y)
    return h.values()


@vm.variable(inputs={"signal_events", "minbias_events", "model"}, kind="curve")
def diTauTopoEffVsGenPt2d(run):
    """Di-tau topology seed efficiency in bins of (gen pT of tau 1, gen
    pT of tau 2), on signal events with two gen-matched taus (gen pT > 1
    GeV). The two candidates are the highest tau scores among the jets
    with pT > CMSSW_WPS tau_l1_pt and |eta| < TAU_ETA_CUT, as in
    tauRateGrid; both need |eta| < TAU_ETA_CUT and deltaR >
    TAU_PAIR_MIN_DR, the smaller pT > tau_l1_pt and the pair score >
    tauWp(diTauTopo) (see tauRateGrid). Fields ``edges``, ``num``,
    ``den``, ``eff`` (NaN where den = 0), and the normalised gen pT1 and
    gen pT2 distributions of all signal events with two jets,
    ``genpt1_frac`` and ``genpt2_frac`` (for the ratio map of the
    network against the CMSSW L1 baseline).
    """
    c = _topo_eff_counts(run)
    return {
        "edges": c["edges"],
        "num": c["pass"],
        "den": c["all"],
        "eff": ratio_2d(c["pass"], c["all"]),
        "genpt1_frac": c["genpt1_frac"],
        "genpt2_frac": c["genpt2_frac"],
    }


@vm.variable(
    inputs={"signal_events", "minbias_events", "model"}, kind="scalar"
)
def diTauTopoTotalEff(run):
    """Integrated di-tau topology efficiency of diTauTopoEffVsGenPt2d:
    passing events over all events with two gen-matched taus."""
    c = _topo_eff_counts(run)
    return float(c["n_pass"] / c["denom"])


@vm.variable(
    inputs={"signal_events", "minbias_events", "model"}, kind="scalar"
)
def diTauTopoTotalEffErr(run):
    """sqrt(N) uncertainty of diTauTopoTotalEff."""
    c = _topo_eff_counts(run)
    return float(np.sqrt(c["n_pass"]) / c["denom"])
