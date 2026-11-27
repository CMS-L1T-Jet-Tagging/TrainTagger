"""Jet classification: ROCs, AUCs, efficiencies at fixed mistag,
confusion matrix, input distributions.

Class arguments are a model class (b, charm, light, gluon, taup, taum,
muon, electron, pileup), a group from config CLASS_GROUPS (taus, jets,
qcd, leptons) or ``all``.
"""

import numpy as np
from sklearn.metrics import auc as _auc
from sklearn.metrics import confusion_matrix, roc_curve

from tagger.eval.variable_manager import Expr, vm

# -- ROC based quantities ----------------------------------------------


def _roc_inputs(jets, sig_idx, bkg_idx):
    """Truth and discriminant sig / (sig + bkg) on jets whose truth is
    in sig or bkg."""
    y_true, y_pred = jets.y_true, jets.y_pred
    signal_mask = sum(y_true[:, i] for i in sig_idx) > 0
    total_mask = signal_mask | (sum(y_true[:, i] for i in bkg_idx) > 0)
    sig_scores = sum(y_pred[:, i] for i in sig_idx)
    bkg_scores = sum(y_pred[:, i] for i in bkg_idx)
    return (
        signal_mask[total_mask],
        (sig_scores / (sig_scores + bkg_scores))[total_mask],
    )


@vm.variable(inputs={"test_jets"}, kind="curve")
def roc(run, sig: str, bkg: str):
    """ROC curve of ``sig`` against ``bkg``.

    Discriminant: summed ``sig`` scores / (summed ``sig`` + summed
    ``bkg`` scores), on jets whose truth is in ``sig`` or ``bkg``.
    Fields ``tpr`` (signal efficiency), ``fpr`` (mistag rate),
    ``thresholds``.
    """
    y_true, y_score = _roc_inputs(
        run.jets(), run.classes(sig), run.classes(bkg)
    )
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    return {"tpr": tpr, "fpr": fpr, "thresholds": thresholds}


@vm.variable(inputs={"test_jets"}, kind="scalar")
def auc(run, sig: str, bkg: str):
    """Area under ``roc(sig, bkg)``."""
    curve = run.get(f"roc({sig}, {bkg})")
    return _auc(curve["fpr"], curve["tpr"])


@vm.variable(inputs={"test_jets"}, kind="scalar")
def effAtMistag(run, sig: str, bkg: str, mistag: float):
    """Signal efficiency at a fixed mistag rate: the fraction of ``sig``
    jets that pass the discriminant cut at which a fraction ``mistag``
    of ``bkg`` jets passes.

    Read off ``roc(sig, bkg)``: the true positive rate at false positive
    rate ``mistag``, linearly interpolated between the ROC points.
    Example: ``effAtMistag(b, light, 0.1)`` is the b-tagging efficiency
    at 10% light-jet mistag.
    """
    curve = run.get(f"roc({sig}, {bkg})")
    return np.interp(mistag, curve["fpr"], curve["tpr"])


@vm.variable(inputs={"test_jets"}, kind="curve")
def rocOneVsRest(run, cls: str):
    """One-vs-rest ROC of class ``cls``: its raw score on all jets,
    truth = jet is ``cls``.

    Fields ``tpr``, ``fpr``, ``thresholds``.
    """
    jets = run.jets()
    (i,) = run.classes(cls)
    fpr, tpr, thresholds = roc_curve(jets.y_true[:, i], jets.y_pred[:, i])
    return {"tpr": tpr, "fpr": fpr, "thresholds": thresholds}


@vm.variable(inputs={"test_jets"}, kind="scalar")
def aucOneVsRest(run, cls: str):
    """Area under ``rocOneVsRest(cls)``: how well the raw ``cls`` score
    separates ``cls`` jets from all other jets. 1 is perfect separation,
    0.5 is random."""
    curve = run.get(f"rocOneVsRest({cls})")
    return _auc(curve["fpr"], curve["tpr"])


@vm.variable(inputs={"test_jets"}, kind="curve")
def effVsPt(run, cls: str, mistag: float):
    """Tagging efficiency of ``cls`` jets vs their L1 (uncorrected) jet
    pT, at a one-vs-rest working point.

    Discriminant: summed ``cls`` scores. The cut is the ROC threshold
    (``cls`` against all other jets) whose mistag rate is closest to
    ``mistag``. Bins: edges are the centres of config PT_BINS. Fields
    ``x`` (bin centres), ``edges``, ``eff``, ``err`` (binomial), ``n``,
    ``threshold``.
    """
    jets = run.jets()
    pt_bins = np.asarray(run.config["PT_BINS"], dtype=float)
    return _eff_vs(
        run, cls, mistag, jets.reco_pt, (pt_bins[:-1] + pt_bins[1:]) / 2
    )


@vm.variable(inputs={"test_jets"}, kind="curve")
def effVsGenPt(run, cls: str, mistag: float):
    """Tagging efficiency of ``cls`` jets vs their generated
    (gen-matched) jet pT, at the working point of
    ``effVsPt(cls, mistag)``. Bins and fields as effVsPt.
    """
    jets = run.jets()
    pt_bins = np.asarray(run.config["PT_BINS"], dtype=float)
    return _eff_vs(
        run, cls, mistag, jets.truth_pt, (pt_bins[:-1] + pt_bins[1:]) / 2
    )


@vm.variable(inputs={"test_jets"}, kind="curve")
def effVsEta(run, cls: str, mistag: float):
    """Tagging efficiency of ``cls`` jets vs L1 jet eta, at the working
    point of ``effVsPt(cls, mistag)``. Bins: config ETA_BINS. Fields as
    effVsPt.
    """
    return _eff_vs(
        run,
        cls,
        mistag,
        _reco_eta(run),
        np.asarray(run.config["ETA_BINS"], dtype=float),
    )


@vm.variable(inputs={"test_jets"}, kind="curve")
def effVsAbsEta(run, cls: str, mistag: float):
    """Tagging efficiency of ``cls`` jets vs L1 jet |eta|, at the
    working point of ``effVsPt(cls, mistag)``. Bins: config
    ABS_ETA_BINS. Fields as effVsPt.
    """
    return _eff_vs(
        run,
        cls,
        mistag,
        np.abs(_reco_eta(run)),
        np.asarray(run.config["ABS_ETA_BINS"], dtype=float),
    )


def _reco_eta(run):
    eta = run.jets().reco_eta
    if eta is None:
        raise ValueError(
            "this model's test split has no jet eta"
            " (testing_data/reco_eta_test.npz): it is saved by train.py since"
            " the eval refactor, so retrain the model"
        )
    return eta


def _eff_vs(run, cls, mistag, values, edges):
    """Efficiency of ``cls`` jets at the one-vs-rest working point with
    mistag rate ``mistag``, in bins ``edges`` of a per-jet quantity
    ``values``."""
    jets = run.jets()
    idx = run.classes(cls)
    truth = sum(jets.y_true[:, i] for i in idx)
    score = sum(jets.y_pred[:, i] for i in idx)
    fpr, _, thresholds = roc_curve(truth, score)
    cut = thresholds[np.argmin(np.abs(np.array(fpr) - mistag))]
    sel = truth > 0
    x, passed = np.asarray(values)[sel], score[sel] > cut
    n = np.array(
        [np.sum((x >= lo) & (x < hi)) for lo, hi in zip(edges[:-1], edges[1:])]
    )
    k = np.array(
        [
            np.sum(passed[(x >= lo) & (x < hi)])
            for lo, hi in zip(edges[:-1], edges[1:])
        ]
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        eff = k / n
        err = np.sqrt(eff * (1 - eff) / n)
    return {
        "x": (edges[:-1] + edges[1:]) / 2,
        "edges": edges,
        "eff": eff,
        "err": err,
        "n": n,
        "threshold": float(cut),
    }


# -- Classification output ---------------------------------------------


@vm.variable(inputs={"test_jets"}, kind="curve")
def confusionMatrix(run):
    """Confusion matrix of argmax(truth) vs argmax(score), normalised
    per true class.

    Fields ``matrix`` (true class x predicted class) and ``classes``
    (row/column names).
    """
    jets = run.jets()
    matrix = confusion_matrix(
        np.argmax(jets.y_true, axis=1),
        np.argmax(jets.y_pred, axis=1),
        normalize="true",
    )
    by_index = {v: k for k, v in run.class_labels().items()}
    return {
        "matrix": matrix,
        "classes": [by_index[i] for i in range(jets.y_true.shape[1])],
    }


@vm.variable(inputs={"test_jets"}, kind="curve")
def scoreHist(run, cls: str):
    """Histogram of the ``cls`` score (summed over the classes of a
    group) of all jets, config SCORE_HIST_BINS bins in 0-1. Fields
    ``edges``, ``counts``."""
    jets = run.jets()
    score = sum(jets.y_pred[:, i] for i in run.classes(cls))
    counts, edges = np.histogram(
        score, bins=run.config["SCORE_HIST_BINS"], range=(0, 1)
    )
    return {"edges": edges, "counts": counts}


@vm.variable(inputs={"test_jets"}, kind="curve")
def jetOutputs(run):
    """The outputs per jet, in sample order: one field per model class
    with its score, and ``regression`` (the pT correction). For per-jet
    comparisons of two model versions.
    """
    jets = run.jets()
    out = {c: jets.y_pred[:, i] for c, i in run.class_labels().items()}
    out["regression"] = jets.pt_ratio
    return out


# -- Input based quantities --------------------------------------------


def _feature_column(run, feature):
    names = run.sources.particle_input_vars()
    if feature not in names:
        raise KeyError(
            f"model has no input feature {feature!r}, features {names}"
        )
    return names.index(feature)


@vm.variable(inputs={"test_jets"}, kind="curve")
def inputHist(run, feature: str, cls: str):
    """Histogram of input feature ``feature`` over real (not
    zero-padded) candidates of ``cls`` jets.

    Binning: config INPUT_HIST_BINS over INPUT_FEATURE_RANGES[feature].
    Fields ``edges``, ``counts``.
    """
    jets = run.jets()
    x = jets.model_inputs["basic_input"]
    filled = x[:, :, _feature_column(run, "isfilled")] == 1
    in_cls = sum(jets.y_true[:, i] for i in run.classes(cls)) > 0
    values = x[:, :, _feature_column(run, feature)][
        in_cls[:, None] & filled
    ].flatten()
    counts, edges = np.histogram(
        values,
        bins=run.config["INPUT_HIST_BINS"],
        range=run.config["INPUT_FEATURE_RANGES"][feature],
    )
    return {"edges": edges, "counts": counts}


@vm.variable(inputs={"test_jets"}, kind="curve")
def multiplicity(run, cls: str):
    """Histogram of the number of real candidates per ``cls`` jet, jets
    with none left out.

    Binning: n + 1 bins over 0 to n, n the number of candidate slots per
    jet. Fields ``edges``, ``counts``.
    """
    jets = run.jets()
    x = jets.model_inputs["basic_input"]
    n = np.sum(x[:, :, _feature_column(run, "isfilled")] != 0, axis=1)
    slots = x.shape[1]
    in_cls = sum(jets.y_true[:, i] for i in run.classes(cls)) > 0
    counts, edges = np.histogram(
        n[in_cls & (n > 0)], bins=slots + 1, range=(0, slots)
    )
    return {"edges": edges, "counts": counts}


# -- Metavariables -----------------------------------------------------


@vm.variable(inputs={"test_jets", "process_jets"}, kind="inherit")
def onTestSubset(run, process: str, expr: Expr):
    """``expr`` evaluated on the test jets that also occur in
    signal_process_data/<process>: the part of the test split that
    belongs to one physics process."""
    return _on_sample(run, f"subset:{process}", expr)


@vm.variable(inputs={"process_jets"}, kind="inherit")
def onProcess(run, process: str, expr: Expr):
    """``expr`` evaluated on all jets of signal_process_data/<process>,
    also those used in training."""
    return _on_sample(run, f"full:{process}", expr)


@vm.variable(inputs={"emulation_jets"}, kind="inherit")
def onEmulation(run, expr: Expr):
    """``expr`` evaluated on the jets of the emulation ntuple (config
    EMULATION_SAMPLE), with the outputs of the model version: the
    network (-m <model>), its hls4ml build (-m hls4ml:<model>, class
    scores quantised to HLS4ML_SCORE_BITS) or the CMSSW emulator (-m
    emulator:<model>, no pileup score).
    """
    return _on_sample(run, "emulation", expr)


def _on_sample(run, sample, expr):
    previous, run.sample = run.sample, sample
    try:
        return run.get(expr)
    finally:
        run.sample = previous
