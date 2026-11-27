"""FastPUPPI jets: the L1 jet collections of the model's CMSSW emulation
in full events.

The FastPUPPI performance ntuples (input fastpuppi_events, config
FASTPUPPI_DIR) hold two L1 jet collections: the seeded cone jets with
the model's pT regression (``nn``, scPuppiL1TSC4NGJetJets) and the
standard seeded cone jets (``sc``, scPuppiExtendedJets), each as read or
with the standard jet energy corrections (``nnJecs``, ``scJecs``);
``gen`` are the gen jets. The variables follow the stages of a trigger
study:

1. Candidates: the jets of each process, matched to gen jets and, for
the masses, to the decay partons. 2. Response and resolution: L1 pT
against gen pT, in bins of gen pT (...VsGenPt) and of L1 pT (...VsPt).
3. Distributions: jet pT spectra, gen vs reco pT maps, and the top, W
and Higgs mass peaks. 4. Working point: the MinBias threshold of an
observable at a target rate. 5. Signal efficiency: turn-on curves at
that threshold.
"""

from functools import partial
from typing import Literal

import awkward as ak
import numpy as np

from tagger.eval.inputs.fastpuppi import COLLECTIONS, to_coffea
from tagger.eval.variable_manager import vm

Process = Literal[
    "TT_PU200",
    "QCD_Pt15To3000_PU200",
    "MinBias_PU200",
    "VBFHToBB_PU200",
    "VBFHToCC_PU200",
]
Coll = Literal["nn", "sc"]
Jets = Literal["nn", "nnJecs", "sc", "scJecs"]
JetsOrGen = Literal["nn", "nnJecs", "sc", "scJecs", "gen"]
Observable = Literal[
    "jet1",
    "jet2",
    "jet3",
    "ht15",
    "ht30",
    "mjj",
    "max_mjj",
    "dijet",
    "quadjet",
]
Particle = Literal["top", "w", "h"]

NN, SC = COLLECTIONS
_COLL = {"nn": NN, "sc": SC}
_JETS = {
    "nn": (NN, "raw"),
    "nnJecs": (NN, "jecs"),
    "sc": (SC, "raw"),
    "scJecs": (SC, "jecs"),
}
MINBIAS = "MinBias_PU200"

# PDG ids and Pythia status codes of the decay matching
PDG_C, PDG_B, PDG_TOP, PDG_W, PDG_HIGGS = 4, 5, 6, 24, 25
PDG_ELECTRON, PDG_MUON, PDG_TAU = 11, 13, 15
STATUS_INTERMEDIATE, STATUS_OUTGOING = 22, 23  # of the hardest subprocess


# -- Candidates: jets, gen jets and matching ---------------------------
def _events(run, process):
    return run.input("fastpuppi_events", process)


def _jets(run, process, jets):
    """Jets of a process: an L1 collection (as read or corrected) or the
    gen jets."""
    ev = _events(run, process)
    if jets == "gen":
        return ev["genjets"]
    coll, kind = _JETS[jets]
    return ev[coll][kind]


def _flat(run, process, coll):
    """Flat gen pT, L1 pT and correction factor of the jets of a collection."""
    c = _events(run, process)[_COLL[coll]]
    return [
        ak.to_numpy(ak.flatten(a))
        for a in (c["raw"].genpt, c["raw"].pt, c["pt_ratio"])
    ]


def _match_genjets(daughters, genjets, dr_max, rel_pt_max):
    """Per event, partons matched one to one to gen jets: the closest
    pairs first, deltaR < dr_max and |pT jet - pT parton| / pT parton <
    rel_pt_max."""
    matched_parts, matched_jets = [], []
    dr_matrix = daughters.metric_table(genjets)
    for r, (parts, jets) in enumerate(zip(daughters, genjets)):
        if len(parts) == 0 or len(jets) == 0:
            matched_parts.append([])
            matched_jets.append([])
            continue
        dr = dr_matrix[r]
        candidates = []
        for i in range(len(parts)):
            for j in range(len(jets)):
                if dr[i, j] > dr_max:
                    continue
                if abs(jets.pt[j] - parts.pt[i]) / parts.pt[i] > rel_pt_max:
                    continue
                candidates.append((dr[i, j], i, j))
        candidates.sort()
        used_parts, used_jets, evt_parts, evt_jets = set(), set(), [], []
        for _, i, j in candidates:
            if len(used_jets) == len(jets):
                break
            if i in used_parts or j in used_jets:
                continue
            used_parts.add(i)
            used_jets.add(j)
            evt_parts.append(parts[i])
            evt_jets.append(jets[j])
        matched_parts.append(evt_parts)
        matched_jets.append(evt_jets)
    return matched_parts, matched_jets


def _higgs_jets(run, ev, daughter_pdg):
    """H -> q qbar: per L1 jet type the jets matched to the two gen jets
    of the quarks (events with both), and those gen jets."""
    genparts, genjets = to_coffea(ev["genparts"]), to_coffea(ev["genjets"])
    mothers = genparts.pdgId[genparts.genPartIdxMother]
    daughters = genparts[
        (abs(genparts.pdgId) == daughter_pdg)
        & (genparts.status == STATUS_OUTGOING)
        & (mothers == PDG_HIGGS)
    ]
    matched = ak.Array(
        _match_genjets(
            daughters,
            genjets,
            run.config["MATCH_DELTA_R"],
            run.config["MATCH_REL_PT"],
        )[1]
    )
    out = {}
    for jets, (coll, kind) in _JETS.items():
        reco = to_coffea(ev[coll][kind])
        reco = reco[
            ak.any(reco.genpt[:, :, None] == matched.pt[:, None, :], axis=2)
        ]
        out[("h", jets)] = reco[ak.num(reco) == 2]
    out[("h", "gen")] = to_coffea(matched[ak.num(matched) == 2])
    return out


def _decay_mode(particles):
    pdg = abs(particles.pdgId)
    is_quark = (pdg >= 1) & (pdg <= PDG_TOP)
    is_lepton = (pdg == PDG_ELECTRON) | (pdg == PDG_MUON) | (pdg == PDG_TAU)
    mode = ["unknown"] * len(particles)
    mode = np.where(
        ak.sum(is_quark, axis=1) == 2, ["hadronic"] * len(mode), mode
    )
    return np.where(
        ak.sum(is_lepton, axis=1) == 2, ["leptonic"] * len(mode), mode
    )


def _daughters_of(genparts, mothers, pdg_mother):
    """The gen particles (intermediate or outgoing) whose mother has
    ``pdg_mother``, and their mask."""
    status = (genparts.status == STATUS_INTERMEDIATE) | (
        genparts.status == STATUS_OUTGOING
    )
    mask = status & (mothers == pdg_mother)
    return genparts[mask], mask


def _top_jets(run, ev):
    """t -> W b -> q q' b: per L1 jet type the jets matched to the three
    gen jets of the top (``top``) and to the two of the W (``w``),
    hadronic decays only, top and antitop together; and those gen jets.
    """
    genparts, genjets = to_coffea(ev["genparts"]), to_coffea(ev["genjets"])
    mothers = genparts.pdgId[genparts.genPartIdxMother]
    dr_max, rel_pt_max = (
        run.config["MATCH_DELTA_R"],
        run.config["MATCH_REL_PT"],
    )
    top_results, w_results = [], []
    for top in (PDG_TOP, -PDG_TOP):
        w_pdg = PDG_W if top > 0 else -PDG_W
        daughters, mask = _daughters_of(genparts, mothers, top)
        w_indices = ak.local_index(genparts.pt)[
            mask & (genparts.pdgId == w_pdg)
        ]
        bs = daughters[abs(daughters.pdgId) == PDG_B]
        w_daughters, _ = _daughters_of(genparts, mothers, w_pdg)
        w_mask = (ak.num(w_indices) == 1) & (ak.num(w_daughters) == 2)
        top_mask = w_mask & (ak.num(bs) == 1)
        decay = _decay_mode(w_daughters[top_mask])
        all_daughters = ak.concatenate(
            [w_daughters[top_mask], bs[top_mask]], axis=1
        )
        top_jets = _match_genjets(
            all_daughters, genjets[top_mask], dr_max, rel_pt_max
        )[1]
        w_jets = _match_genjets(
            w_daughters[w_mask], genjets[w_mask], dr_max, rel_pt_max
        )[1]
        top_results.append((ak.Array(decay), ak.Array(top_jets), top_mask))
        w_results.append((ak.Array(decay), ak.Array(w_jets), w_mask))

    out = {}
    for particle, results in (("top", top_results), ("w", w_results)):
        n_jets = 3 if particle == "top" else 2
        decay = ak.concatenate([r[0] for r in results], axis=0)
        matched = ak.concatenate([r[1] for r in results], axis=0)
        event_mask = ak.concatenate([r[2] for r in results], axis=0)
        out[(particle, "gen")] = to_coffea(
            matched[(ak.num(matched) == n_jets) & (decay == "hadronic")]
        )
        for jets, (coll, kind) in _JETS.items():
            reco = to_coffea(
                ak.concatenate((ev[coll][kind], ev[coll][kind]), axis=0)[
                    event_mask
                ]
            )
            reco = reco[
                ak.any(
                    reco.genpt[:, :, None] == matched.pt[:, None, :], axis=2
                )
            ]
            out[(particle, jets)] = reco[ak.num(reco) == n_jets]
    return out


def _decay_jets(run, process, particle):
    """The matched jets of a top/W (TT_PU200) or Higgs (VBFHToBB_PU200,
    VBFHToCC_PU200) decay, memoised."""
    return run.memo(
        ("fp_decay_jets", process, "top" if particle == "w" else particle),
        _compute_decay_jets,
        run,
        process,
        particle,
    )


def _compute_decay_jets(run, process, particle):
    """_decay_jets without the cache."""
    ev = _events(run, process)
    if particle in ("top", "w"):
        return _top_jets(run, ev)
    return _higgs_jets(
        run, ev, PDG_B if process == "VBFHToBB_PU200" else PDG_C
    )


# -- Response and resolution -------------------------------------------
def _pt_points(run):
    pt_bins = run.config["PT_BINS"]
    return np.array(
        [
            np.mean((pt_bins[i], pt_bins[i + 1]))
            for i in range(len(pt_bins) - 1)
        ]
    )


def _fp_binned(run, process, coll, binned_by, statistic):
    """``statistic(L1 pT, gen pT)`` -> (value, error) per bin of config
    PT_BINS (open edges), for the jets as read (``raw``) and corrected
    (``jecs``), gen-matched jets only. ``binned_by`` gen: bins in gen
    pT; l1: each L1 pT binned in itself (raw in raw pT, corrected in
    corrected pT).
    """
    gen, reco, ratio = _flat(run, process, coll)
    pt_bins = run.config["PT_BINS"]
    out = {}
    for key, pt in (("raw", reco), ("jecs", np.multiply(reco, ratio))):
        by = gen if binned_by == "gen" else pt
        values, errors = [], []
        for lo, hi in zip(pt_bins[:-1], pt_bins[1:]):
            sel = (by > lo) & (by < hi) & (gen > 0)
            value, error = statistic(pt[sel], gen[sel])
            values.append(value)
            errors.append(error)
        out[key], out[key + "_err"] = np.asarray(
            values, dtype=float
        ), np.asarray(errors, dtype=float)
    return {"x": _pt_points(run), **out}


# Average of the response per bin (fpResponseVsGenPt, fpResponseVsL1Pt
# reduce)
_REDUCERS = {"mean": np.mean, "median": np.median}


def _fp_response(reducer, pt, gen):
    """Response L1 pT / gen pT averaged by ``reducer``, and its error
    std / sqrt(N)."""
    r = pt / gen
    return reducer(r), (np.std(r) / np.sqrt(len(r)) if len(r) > 0 else 0)


def _fp_resolution(pt, gen):
    sigma = np.std(pt - gen) / gen.mean()
    return sigma, (
        sigma if len(pt) <= 1 else sigma / np.sqrt(2 * (len(pt) - 1))
    )


@vm.variable(inputs={"fastpuppi_events"}, kind="curve")
def fpResponseVsGenPt(
    run, process: Process, coll: Coll, reduce: Literal["mean", "median"]
):
    """Mean or median pT response (L1 / gen) of the jets of an L1
    collection in bins of gen pT (config PT_BINS, open edges), as read
    (fields ``raw``, ``raw_err``) and with the standard jet energy
    corrections (``jecs``, ``jecs_err``); ``x`` the bin centres. Errors:
    std / sqrt(n), 0 for empty bins.
    """
    return _fp_binned(
        run, process, coll, "gen", partial(_fp_response, _REDUCERS[reduce])
    )


@vm.variable(inputs={"fastpuppi_events"}, kind="curve")
def fpResponseVsPt(
    run, process: Process, coll: Coll, reduce: Literal["mean", "median"]
):
    """Mean or median pT response (L1 / gen) of the gen-matched jets of
    an L1 collection in bins of their L1 pT (config PT_BINS, open
    edges): as read in bins of the raw pT, corrected in bins of the
    corrected pT. Fields and errors as fpResponseVsGenPt."""
    return _fp_binned(
        run, process, coll, "l1", partial(_fp_response, _REDUCERS[reduce])
    )


@vm.variable(inputs={"fastpuppi_events"}, kind="curve")
def fpResolutionVsGenPt(run, process: Process, coll: Coll):
    """Relative resolution std(L1 pT - gen pT) / mean(gen pT) of the
    jets of an L1 collection in bins of gen pT (config PT_BINS), as read
    (``raw``, ``raw_err``) and corrected (``jecs``, ``jecs_err``); ``x``
    the bin centres. Error: sigma / sqrt(2(n-1)), sigma itself for n <=
    1."""
    return _fp_binned(run, process, coll, "gen", _fp_resolution)


@vm.variable(inputs={"fastpuppi_events"}, kind="curve")
def fpResolutionVsPt(run, process: Process, coll: Coll):
    """Relative resolution std(L1 pT - gen pT) / mean(gen pT) of the
    gen-matched jets of an L1 collection in bins of their L1 pT (config
    PT_BINS): as read in bins of the raw pT, corrected in bins of the
    corrected pT. Fields and errors as fpResolutionVsGenPt."""
    return _fp_binned(run, process, coll, "l1", _fp_resolution)


# -- Distributions -----------------------------------------------------
@vm.variable(inputs={"fastpuppi_events"}, kind="curve")
def fpPtHist(
    run, process: Process, jets: JetsOrGen, which: Literal["leading", "all"]
):
    """pT spectrum of the leading jet per event (events with a jet) or
    of all jets, in config REGULAR_PT_BINS (values clipped into the last
    bin), as fractions of the jets. Fields ``edges``, ``fraction``."""
    j = _jets(run, process, jets)
    pt = (
        ak.to_numpy(ak.max(j.pt, axis=1)[ak.num(j.pt) > 0])
        if which == "leading"
        else ak.to_numpy(ak.flatten(j.pt))
    )
    edges = np.linspace(*run.config["REGULAR_PT_BINS"])
    pt = np.clip(pt, 0, np.max(edges) - 0.5)
    counts, _ = np.histogram(pt, bins=edges)
    return {"edges": edges, "fraction": counts / len(pt)}


@vm.variable(inputs={"fastpuppi_events"}, kind="curve")
def fpGenRecoPtMap(run, process: Process, jets: Jets):
    """Number of gen-matched jets (genpt != 0) in bins of (gen pT, L1
    pT), both in config REGULAR_PT_BINS and clipped to its range. Fields
    ``edges``, ``counts`` (gen pT x L1 pT)."""
    j = _jets(run, process, jets)
    matched = j.genpt != 0
    edges = np.linspace(*run.config["REGULAR_PT_BINS"])
    top = float(edges[-1])
    gen = ak.to_numpy(np.clip(ak.flatten(j.genpt[matched], axis=None), 0, top))
    reco = ak.to_numpy(np.clip(ak.flatten(j.pt[matched], axis=None), 0, top))
    counts, _, _ = np.histogram2d(gen, reco, bins=edges)
    return {"edges": edges, "counts": counts}


def _smallest_interval(data, fraction=0.68):
    """Half width of the smallest interval holding ``fraction`` of the
    values."""
    data = np.sort(data)
    n, k = len(data), int(np.floor(fraction * len(data)))
    min_width = np.inf
    for i in range(n - k):
        min_width = min(min_width, data[i + k] - data[i])
    return min_width / 2


@vm.variable(inputs={"fastpuppi_events"}, kind="curve")
def fpMass(run, process: Process, particle: Particle, jets: JetsOrGen):
    """Invariant mass of the jets of a decay: top (three jets of t -> W
    b -> q q' b) and W (two jets) in TT_PU200, H (two jets of H -> b
    bbar or c cbar) in VBFHToBB_PU200 / VBFHToCC_PU200. The partons are
    matched one to one to gen jets (deltaR < MATCH_DELTA_R, relative pT
    difference < MATCH_REL_PT), the L1 jets to those gen jets by their
    gen pT; only events with all jets matched, hadronic decays. Fields
    ``edges``, ``x`` (bin centres), ``fraction``, ``err`` (sqrt(N) /
    total), ``mean`` and ``sigma_eff`` (half width of the smallest
    interval with 68% of the events). Bins: config MASS_BINS.
    """
    mass = ak.to_numpy(
        _decay_jets(run, process, particle)[(particle, jets)].sum(axis=1).mass
    )
    edges = np.linspace(*run.config["MASS_BINS"][particle])
    counts, _ = np.histogram(mass, bins=edges)
    return {
        "edges": edges,
        "x": 0.5 * (edges[:-1] + edges[1:]),
        "fraction": counts / counts.sum(),
        "err": np.sqrt(counts) / counts.sum(),
        "mean": np.array(np.mean(mass)),
        "sigma_eff": np.array(_smallest_interval(mass)),
    }


# -- Working point -----------------------------------------------------
# Jet pT threshold (GeV) of the HT observables
HT_OBSERVABLE_JET_PT = {"ht15": 15, "ht30": 30}


def _at_least(n_reco, n_gen, n):
    """Events with at least n L1 jets and n gen jets."""
    return (n_reco > n - 1) & (n_gen > n - 1)


def _observable_values(coll, n_reco, n_gen, obj):
    """Per event the observable of a jet collection (jets sorted by pT),
    on the events with enough L1 and gen jets."""
    match obj:
        case "jet1":
            return ak.max(coll.pt[_at_least(n_reco, n_gen, 1)], axis=1)
        case "jet2":
            return ak.sort(
                coll.pt[_at_least(n_reco, n_gen, 2)], ascending=False
            )[:, 1]
        case "jet3":
            return ak.sort(
                coll.pt[_at_least(n_reco, n_gen, 3)], ascending=False
            )[:, 2]
        case "ht15" | "ht30":
            return ak.sum(coll.pt[coll.pt > HT_OBSERVABLE_JET_PT[obj]], axis=1)
        case "mjj":
            cut = _at_least(n_reco, n_gen, 2)
            return (coll[cut][:, 0] + coll[cut][:, 1]).mass
        case "max_mjj":
            cut = _at_least(n_reco, n_gen, 2)
            mjjs = coll[cut].metric_table(
                coll[cut], metric=lambda a, b: (a + b).mass
            )
            return ak.max(ak.max(mjjs, axis=-1), axis=-1)
        case "dijet":
            return ak.min(
                ak.sort(
                    coll[_at_least(n_reco, n_gen, 2)], axis=1, ascending=False
                )[:, :2].pt,
                axis=1,
            )
        case "quadjet":
            return ak.min(
                ak.sort(
                    coll[_at_least(n_reco, n_gen, 4)], axis=1, ascending=False
                )[:, :4].pt,
                axis=1,
            )
    raise ValueError(f"unknown observable {obj!r}")


def _observable(config, coll, n_reco, n_gen, obj):
    """The observable of a jet collection (see _observable_values), its
    threshold scan (GeV) and the x range of the turn-on plot (config
    TURN_ON_SCANS)."""
    first, last, step, x_max = config["TURN_ON_SCANS"][obj]
    return (
        _observable_values(to_coffea(coll), n_reco, n_gen, obj),
        np.arange(first, last, step),
        x_max,
    )


@vm.variable(inputs={"fastpuppi_events"}, kind="scalar")
def fpRateWp(run, obj: Observable, jets: Jets, rate: float):
    """Threshold (GeV) of an observable of an L1 jet type at a MinBias
    rate (kHz): of the threshold scan, the one whose rate is closest; an
    error if that is more than config TURN_ON_RATE_TOLERANCE off.
    Observables: jet1, jet2, jet3 (leading, second, third jet pT), ht15,
    ht30 (scalar sum of the jet pT above 15 / 30 GeV), mjj (mass of the
    two leading jets), max_mjj (largest dijet mass), dijet, quadjet (pT
    of the second / fourth jet).
    """
    reco = _jets(run, MINBIAS, jets)
    total = len(reco)
    sort_idx = ak.argsort(
        reco.pt[ak.num(reco.pt) > 0], axis=1, ascending=False, stable=True
    )
    reco = reco[ak.num(reco.pt) > 0][sort_idx]
    n_reco = ak.num(reco)
    values, cuts, _ = _observable(run.config, reco, n_reco, n_reco, obj)
    rates = [
        (np.sum(values > c) / total) * run.config["MINBIAS_RATE"] for c in cuts
    ]
    i = np.argmin(np.abs(np.array(rates) - rate))
    if abs(rates[i] - rate) > run.config["TURN_ON_RATE_TOLERANCE"]:
        raise ValueError(
            f"no {obj} threshold of {jets} jets close to {rate} kHz: closest"
            f" {rates[i]} kHz at {cuts[i]} GeV"
        )
    return float(cuts[i])


# -- Signal efficiency -------------------------------------------------
@vm.variable(inputs={"fastpuppi_events"}, kind="curve")
def fpTurnOnVsGen(
    run, process: Process, obj: Observable, jets: Jets, rate: float
):
    """Turn-on curve: efficiency of observable > fpRateWp(obj, jets,
    rate) versus the gen jet observable, in the bins of its threshold
    scan, on events with at least one L1 jet. Fields ``x`` (bin
    centres), ``xerr`` (half widths), ``eff``, ``err`` (binomial), and
    ``xmax`` (the x range of the plot).
    """
    coll, kind = _JETS[jets]
    ev = _events(run, process)
    reco, genjets = to_coffea(ev[coll][kind]), to_coffea(ev["genjets"])
    has_jet = ak.num(reco) > 0
    reco, genjets = reco[has_jet], genjets[has_jet]
    n_reco, n_gen = ak.num(reco), ak.num(genjets)
    gen, bins, bin_cut = _observable(run.config, genjets, n_reco, n_gen, obj)
    reco, _, _ = _observable(run.config, reco, n_reco, n_gen, obj)
    wp = run.get(f"fpRateWp({obj}, {jets}, {rate})")
    effs, errs = [], []
    for lower, upper in zip(bins[:-1], bins[1:]):
        sel = (gen > lower) & (gen <= upper)
        n = np.sum(sel)
        eff = np.sum(reco[sel] > wp) / n if n > 0 else 0
        effs.append(eff)
        errs.append(np.sqrt(eff * (1 - eff) / n) if n > 0 else 0)
    xmax = bins[bins >= bin_cut][0] if np.max(bins) >= bin_cut else bins[-1]
    return {
        "x": 0.5 * (bins[:-1] + bins[1:]),
        "xerr": (bins[1:] - bins[:-1]) / 2,
        "eff": np.asarray(effs, dtype=float),
        "err": np.asarray(errs, dtype=float),
        "xmax": np.array(xmax),
    }
