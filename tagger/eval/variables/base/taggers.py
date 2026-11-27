"""Per-jet tagger view for the trigger seeds, the network (class scores,
regressed pT) or the CMSSW L1 baseline of the -m cmssw pseudo-model (NN
PUPPI tau: jet_tauscore, jet_taupt), and the CMSSW reference seeds.

Plain functions: no evaluation run, everything is passed as arguments.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import awkward as ak
import numpy as np
import numpy.typing as npt

from tagger.eval.variables.base.scores import bkg_score, tau_score, x_vs_y

if TYPE_CHECKING:
    from tagger.eval.inputs.events import EventSample


def is_cmssw(ev: EventSample) -> bool:
    """Whether an event sample belongs to the CMSSW pseudo-model (no
    network outputs).

    Args:
        ev: Event sample.

    Returns:
        bool: True for the samples of -m cmssw.
    """
    return getattr(ev, "pseudo", "") == "cmssw"


def tau_jets(
    ev: EventSample, seed: str, labels: dict[str, int] | None
) -> tuple[npt.NDArray, npt.NDArray]:
    """Tau score and pT per jet, from the tagger of the sample.

    Network: the seed's tau score of the class scores (tau_score) and
    the regressed pT. CMSSW L1 baseline: the NN PUPPI tau score and pT
    (jet_tauscore, jet_taupt).

    Args:
        ev: Event sample.
        seed: Tau seed, for the network's tau score.
        labels: Class name -> output index; unused for the CMSSW L1
            baseline.

    Returns:
        tuple: Tau score and pT, one entry per jet.
    """
    if is_cmssw(ev):
        return np.asarray(ev.branch("jet_tauscore")), np.asarray(
            ev.branch("jet_taupt")
        )
    assert labels is not None, "the network's tau score needs its class labels"
    scores, ratio = ev.outputs()
    return (
        tau_score(seed, scores, labels),
        np.asarray(ev.branch("jet_pt")) * ratio,
    )


def topo_parts(
    ev: EventSample, labels: dict[str, int] | None
) -> tuple[tuple[npt.NDArray, ...], Callable[..., npt.NDArray]]:
    """Inputs of the di-tau pair score, and the pair score function.

    Network: the per-jet (tau+, tau-, bkg) scores and topo_pair_score.
    CMSSW L1 baseline, which has no tau charge: the NN PUPPI tau score
    and the smaller score of the two leading candidates.

    Args:
        ev: Event sample.
        labels: Class name -> output index; unused for the CMSSW L1
            baseline.

    Returns:
        tuple: The per-jet parts, and a function of the grouped, sorted
            parts that gives the pair score
        of the two leading candidates.
    """
    if is_cmssw(ev):
        return (np.asarray(ev.branch("jet_tauscore")),), lambda s: np.minimum(
            np.asarray(s[:, 0]), np.asarray(s[:, 1])
        )
    assert (
        labels is not None
    ), "the network's pair score needs its class labels"
    scores, _ = ev.outputs()
    return (
        scores[:, labels["taup"]],
        scores[:, labels["taum"]],
        bkg_score(scores, labels),
    ), topo_pair_score


def topo_pair_score(
    taup: ak.Array, taum: ak.Array, bkg: ak.Array
) -> npt.NDArray:
    """Di-tau topology score of the two leading candidates of each
    event.

    (tau+_1 tau-_2 + tau-_1 tau+_2) / (that + bkg_1 bkg_2).

    Args:
        taup: tau+ score per candidate (events x candidates, sorted).
        taum: tau- score per candidate.
        bkg: Background score per candidate.

    Returns:
        np.ndarray: The pair score per event.
    """
    p1 = np.array(taup[:, 0] * taum[:, 1])
    p2 = np.array(taum[:, 0] * taup[:, 1])
    return x_vs_y(p1 + p2, np.array(bkg[:, 0] * bkg[:, 1]))


# -- CMSSW reference seeds (NN PUPPI taus: jet_tauscore, jet_taupt) ----
def _leading_two_taus(tau_scores, *arrays):
    order = ak.argsort(tau_scores, axis=1, ascending=False)
    return [a[order] for a in (tau_scores, *arrays)]


def cmssw_ditau_seed(
    tau_pts: ak.Array,
    tau_scores: ak.Array,
    tau_etas: ak.Array,
    wps: dict[str, float],
) -> ak.Array:
    """Events passing the CMSSW di-tau seed.

    The two highest NN PUPPI tau scores need pT > wps tau_l1_pt, score >
    wps tau and |eta| < wps tau_eta.

    Args:
        tau_pts: NN PUPPI tau pT per jet (events x jets).
        tau_scores: NN PUPPI tau score per jet.
        tau_etas: Eta per jet.
        wps: config CMSSW_WPS.

    Returns:
        Boolean mask of the events.
    """
    tau_scores, tau_pts, tau_etas = _leading_two_taus(
        tau_scores, tau_pts, tau_etas
    )
    pt = (tau_pts[:, 0] > wps["tau_l1_pt"]) & (
        tau_pts[:, 1] > wps["tau_l1_pt"]
    )
    eta = (np.abs(tau_etas[:, 0]) < wps["tau_eta"]) & (
        np.abs(tau_etas[:, 1]) < wps["tau_eta"]
    )
    score = (tau_scores[:, 0] > wps["tau"]) & (tau_scores[:, 1] > wps["tau"])
    return pt & score & eta


def run3_bbtt_seed(
    jet_pt: ak.Array,
    tau_eta: ak.Array,
    tau_pt: ak.Array,
    tau_scores: ak.Array,
    cmssw_tau_wp: float,
    seed: dict[str, float],
) -> ak.Array:
    """Events passing the Run-3-like bb tau tau seed.

    HT > ht OR (HT > ht_with_taus and two taus with pT > tau_pt_with_ht)
    OR two taus with pT > tau_pt. The middle term is left out when
    ht_with_taus is None. Taus: the two highest NN PUPPI tau scores, both
    with |eta| <= tau_eta and a score at or above the CMSSW working
    point. HT of all jets.

    Args:
        jet_pt: Jet pT per event (events x jets).
        tau_eta: Eta per jet.
        tau_pt: NN PUPPI tau pT per jet.
        tau_scores: NN PUPPI tau score per jet.
        cmssw_tau_wp: Tau score working point (config CMSSW_WPS tau).
        seed: The thresholds, config BBTT_RUN3_SEED.

    Returns:
        Boolean mask of the events.
    """
    tau_scores, tau_pt, tau_eta = _leading_two_taus(
        tau_scores, tau_pt, tau_eta
    )
    eta = (np.abs(tau_eta[:, 0]) <= seed["tau_eta"]) & (
        np.abs(tau_eta[:, 1]) <= seed["tau_eta"]
    )
    ident = (tau_scores[:, 0] >= cmssw_tau_wp) & (
        tau_scores[:, 1] >= cmssw_tau_wp
    )
    taus = eta & ident
    ht = ak.sum(jet_pt, axis=1)
    passed = (ht > seed["ht"]) | _both_above(taus, tau_pt, seed["tau_pt"])
    if seed["ht_with_taus"] is not None:
        taus_with_ht = _both_above(taus, tau_pt, seed["tau_pt_with_ht"])
        passed = passed | ((ht > seed["ht_with_taus"]) & taus_with_ht)
    return passed


def _both_above(passing, pt, pt_cut):
    """Events in ``passing`` whose two leading candidates have pT >= pt_cut."""
    return passing & (pt[:, 0] >= pt_cut) & (pt[:, 1] >= pt_cut)
