"""Jet pT regression: response and resolution vs generated and vs L1 pT,
pT spectra.

``reco`` is the uncorrected L1 jet pT, ``regressed`` is reco pT x the
model's pT correction, ``gen`` the generator-level (gen-matched) jet pT.
Variables binned in a generated quantity say ``Gen`` in their name
(responseVsGenPt); the others are binned in the L1 quantity
(responseVsPt). Class arguments as in jet_classifier (a class, a
CLASS_GROUPS group or ``all``).
"""

from typing import Literal

import numpy as np

from tagger.eval.variable_manager import vm


def _select(run, cls):
    jets = run.jets()
    sel = sum(jets.y_true[:, i] for i in run.classes(cls)) > 0
    return jets.truth_pt[sel], jets.reco_pt[sel], jets.pt_ratio[sel]


def _bins(run):
    pt_bins = np.asarray(run.config["PT_BINS"], dtype=float)
    return pt_bins, (pt_bins[:-1] + pt_bins[1:]) / 2


def _binned(run, cls, binned_by, statistic):
    """``statistic(L1 pT, gen pT)`` -> (value, error) per bin of config
    PT_BINS (open edges), for the uncorrected (``reco``) and the
    regressed pT (``regressed``). ``binned_by`` gen: the bins are in gen
    pT; l1: each L1 pT is binned in itself (uncorrected in uncorrected
    pT, regressed in regressed pT).
    """
    gen, reco, ratio = _select(run, cls)
    pt_bins, x = _bins(run)
    out = {"x": x}
    for key, pt in (("reco", reco), ("regressed", reco * ratio)):
        by = gen if binned_by == "gen" else pt
        values, errors = [], []
        for lo, hi in zip(pt_bins[:-1], pt_bins[1:]):
            sel = (by > lo) & (by < hi) & (gen > 0)
            value, error = statistic(pt[sel], gen[sel])
            values.append(value)
            errors.append(error)
        out[key], out[key + "_err"] = np.asarray(values), np.asarray(errors)
    return out


def _mean_response(pt, gen):
    """Mean of L1 pT / gen pT and its error std / sqrt(n); NaN and 0 for
    an empty bin."""
    n = len(pt)
    r = pt / gen
    return (np.mean(r) if n else np.nan), (
        np.std(r) / np.sqrt(n) if n else 0.0
    )


def _resolution(pt, gen):
    """Standard deviation sigma of L1 pT - gen pT and its error sigma /
    sqrt(2(n-1)), sigma for n <= 1."""
    n = len(pt)
    sigma = np.std(pt - gen) if n else np.nan
    return sigma, (sigma if n <= 1 else sigma / np.sqrt(2 * (n - 1)))


@vm.variable(inputs={"test_jets"}, kind="curve")
def responseVsGenPt(run, cls: str):
    """Mean pT response (L1 / gen) of ``cls`` jets in bins of generated
    pT (config PT_BINS, open edges).

    Fields ``x`` (bin centres), ``reco``, ``reco_err``, ``regressed``,
    ``regressed_err``. Errors are std / sqrt(n), 0 for empty bins.
    """
    return _binned(run, cls, "gen", _mean_response)


@vm.variable(inputs={"test_jets"}, kind="curve")
def responseVsPt(run, cls: str):
    """Mean pT response (L1 / gen) of gen-matched ``cls`` jets in bins
    of their L1 pT (config PT_BINS, open edges): the uncorrected
    response in bins of the uncorrected pT, the regressed one in bins of
    the regressed pT. Fields and errors as responseVsGenPt.
    """
    return _binned(run, cls, "l1", _mean_response)


@vm.variable(inputs={"test_jets"}, kind="curve")
def resolutionVsGenPt(run, cls: str):
    """Standard deviation of (L1 pT - gen pT) of ``cls`` jets in bins of
    generated pT (config PT_BINS).

    Fields ``x``, ``reco``, ``reco_err``, ``regressed``,
    ``regressed_err``. Error: sigma / sqrt(2(n-1)), sigma itself for n
    <= 1.
    """
    return _binned(run, cls, "gen", _resolution)


@vm.variable(inputs={"test_jets"}, kind="curve")
def resolutionVsPt(run, cls: str):
    """Standard deviation of (L1 pT - gen pT) of gen-matched ``cls``
    jets in bins of their L1 pT (config PT_BINS): uncorrected in bins of
    the uncorrected pT, regressed in bins of the regressed pT. Fields
    and errors as resolutionVsGenPt.
    """
    return _binned(run, cls, "l1", _resolution)


@vm.variable(inputs={"test_jets"}, kind="curve")
def ptHist(run, which: Literal["gen", "reco", "regressed"]):
    """Histogram of jet pT over all test jets, config PT_HIST_BINS bins
    in PT_HIST_RANGE: ``which`` gen (the gen-matched jet pT), reco (the
    uncorrected L1 pT) or regressed. Fields ``edges``, ``counts``.
    """
    jets = run.jets()
    values = {
        "gen": jets.truth_pt,
        "reco": jets.reco_pt,
        "regressed": np.multiply(jets.reco_pt, jets.pt_ratio),
    }
    counts, edges = np.histogram(
        values[which],
        bins=run.config["PT_HIST_BINS"],
        range=run.config["PT_HIST_RANGE"],
    )
    return {"edges": edges, "counts": counts}


@vm.variable(inputs={"test_jets"}, kind="curve")
def ptRatioHist(run, cls: str):
    """Histogram of the model's pT correction for ``cls`` jets, config
    PT_RATIO_HIST_BINS bins in RATIO_HIST_RANGE, normalised to unit sum.

    Fields ``edges``, ``fraction``.
    """
    jets = run.jets()
    (i,) = run.classes(cls)
    ratio = jets.pt_ratio[jets.y_true[:, i] == 1]
    counts, edges = np.histogram(
        ratio,
        bins=run.config["PT_RATIO_HIST_BINS"],
        range=run.config["RATIO_HIST_RANGE"],
    )
    return {"edges": edges, "fraction": counts / len(ratio)}


def _response(run, cls):
    """Regressed pT / gen pT of the ``cls`` jets."""
    truth, reco, ratio = _select(run, cls)
    return reco * ratio / truth


@vm.variable(inputs={"test_jets"}, kind="curve")
def regressionHist(run):
    """Histogram of the regression output (the pT correction) of all
    jets, config RATIO_HIST_BINS bins in RATIO_HIST_RANGE. Fields
    ``edges``, ``counts``."""
    counts, edges = np.histogram(
        run.jets().pt_ratio,
        bins=run.config["RATIO_HIST_BINS"],
        range=run.config["RATIO_HIST_RANGE"],
    )
    return {"edges": edges, "counts": counts}


@vm.variable(inputs={"test_jets"}, kind="curve")
def responseHist(run, cls: str):
    """Histogram of the pT response, regressed pT / gen pT, of ``cls``
    jets, config RATIO_HIST_BINS bins in RATIO_HIST_RANGE. Fields
    ``edges``, ``counts``."""
    counts, edges = np.histogram(
        _response(run, cls),
        bins=run.config["RATIO_HIST_BINS"],
        range=run.config["RATIO_HIST_RANGE"],
    )
    return {"edges": edges, "counts": counts}


@vm.variable(inputs={"test_jets"}, kind="scalar")
def responseMedian(run, cls: str):
    """Median of the pT response (regressed pT / gen pT) of ``cls`` jets."""
    return float(np.median(_response(run, cls)))


@vm.variable(inputs={"test_jets"}, kind="scalar")
def responseRms(run, cls: str):
    """Root mean square of the pT response (regressed pT / gen pT) of
    ``cls`` jets: sqrt(mean(response^2)), summed in double precision."""
    return float(
        np.sqrt(
            np.mean(np.asarray(_response(run, cls), dtype=np.float64) ** 2)
        )
    )


# -- pT weights (models with a pt_weights_output layer) ----------------
_RADIANS = (
    np.pi / 720
)  # eta, deta, dphi of the particle inputs are in units of pi/720


def _compute_pt_weights(run):
    """The pT weight of every real constituent (pT != 0) of the test
    jets, clipped to config PT_WEIGHT_CLIP, and the particle inputs of
    those constituents (column per feature).
    """
    model = run.input("model")
    jets = run.jets()
    layer = model.output_pt_weights_name
    sub = model.get_branch_model(layer)
    weights = sub.predict(
        [jets.model_inputs[name] for name in sub.input_names], verbose=0
    ).flatten()
    basic = jets.model_inputs["basic_input"]
    features = list(model.particle_input_vars)
    if weights.size != basic[:, :, 0].size:
        raise ValueError(
            f"{layer} has {weights.size // len(basic)} outputs per jet,"
            f" basic_input {basic.shape[1]} constituents: no weight per"
            " constituent"
        )
    mask = basic[:, :, features.index("pt")].flatten() != 0
    columns = {
        f: basic[:, :, i].flatten()[mask] for i, f in enumerate(features)
    }
    return (
        np.clip(weights[mask], -np.inf, run.config["PT_WEIGHT_CLIP"]),
        columns,
    )


def _pt_weights(run):
    """:func:`_compute_pt_weights`, memoised for the pT weight variables."""
    return run.memo("pt_weights", _compute_pt_weights, run)


@vm.variable(inputs={"model", "test_jets"}, kind="curve")
def ptWeightHist(run, binned_by: Literal["pt", "eta"]):
    """Distribution of the per-constituent pT weights (layer
    pt_weights_output, clipped to config PT_WEIGHT_CLIP) of the real
    constituents of the test jets, PT_WEIGHT_HIST_BINS bins between the
    smallest and largest weight: of all constituents (field ``all``) and
    in bins of constituent pT (PT_WEIGHT_PT_BINS) or |eta|
    (PT_WEIGHT_ETA_BINS), fields ``bin1``, ``bin2``, ...; ``edges``, and
    ``lo``, ``hi`` the bin limits."""
    weights, columns = _pt_weights(run)
    if binned_by == "pt":
        var, bins = columns["pt"], run.config["PT_WEIGHT_PT_BINS"]
    else:
        # as the old pt_weights.py: |column 3| of the particle inputs
        # (deta in the baseline input set), not eta
        var, bins = (
            abs(list(columns.values())[3]) * _RADIANS,
            run.config["PT_WEIGHT_ETA_BINS"],
        )
    value_range = (weights.min(), weights.max())
    n_bins = run.config["PT_WEIGHT_HIST_BINS"]
    counts, edges = np.histogram(weights, bins=n_bins, range=value_range)
    out = {
        "edges": edges,
        "all": counts,
        "lo": np.asarray(bins[:-1], dtype=float),
        "hi": np.asarray(bins[1:], dtype=float),
    }
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:]), 1):
        out[f"bin{i}"] = np.histogram(
            weights[(var >= lo) & (var < hi)], bins=n_bins, range=value_range
        )[0]
    return out


@vm.variable(inputs={"model", "test_jets"}, kind="curve")
def ptWeightVsInput(run, feature: str):
    """Per-constituent pT weight (clipped to config PT_WEIGHT_CLIP)
    against a particle input feature of the real constituents of the
    test jets: a 2D histogram normalised to its largest bin, with the
    feature clipped to and binned in PT_WEIGHT_BINNING (deta, dphi in
    radians, eta as |eta| in radians) and PT_WEIGHT_MAP_BINS weight
    bins. Fields ``xedges``, ``yedges``, ``counts`` (feature x weight).
    """
    weights, columns = _pt_weights(run)
    if feature not in columns:
        raise ValueError(
            f"ptWeightVsInput: {feature} is not a particle input of this"
            f" model: {sorted(columns)}"
        )
    x = (
        abs(columns[feature]) * _RADIANS
        if feature == "eta"
        else columns[feature].copy()
    )
    if feature in ("deta", "dphi"):
        x *= _RADIANS
    low, high, n_bins = run.config["PT_WEIGHT_BINNING"][feature]
    counts, xedges, yedges = np.histogram2d(
        np.clip(x, low, high),
        weights,
        bins=[n_bins, run.config["PT_WEIGHT_MAP_BINS"]],
    )
    return {
        "xedges": xedges,
        "yedges": yedges,
        "counts": counts / np.max(counts),
    }
