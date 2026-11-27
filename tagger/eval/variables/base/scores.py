"""Combinations of the class scores of the network, and class groups
(config CLASS_GROUPS).

Plain functions: no evaluation run, everything is passed as arguments.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from tagger.eval.variable_manager import ModelInfo


def x_vs_y(
    x: npt.NDArray, y: npt.NDArray, apply_light: bool = True
) -> npt.NDArray:
    """Score x against score y.

    Args:
        x: Signal score.
        y: Background score.
        apply_light: Whether to return x / (x + y); if False, x itself.

    Returns:
        The combined score, elementwise.
    """
    return x / (x + y) if apply_light else x


def pileup_score(scores: npt.NDArray, labels: dict[str, int]) -> npt.NDArray:
    """Pileup class score per jet.

    Args:
        scores: Class scores (jets x classes).
        labels: Class name -> output index.

    Returns:
        np.ndarray: The pileup score, zeros for models without a pileup
            class.
    """
    return (
        scores[:, labels["pileup"]]
        if "pileup" in labels
        else np.zeros(len(scores))
    )


def bkg_score(scores: npt.NDArray, labels: dict[str, int]) -> npt.NDArray:
    """Background score per jet: gluon + light (+ pileup).

    Args:
        scores: Class scores (jets x classes).
        labels: Class name -> output index.

    Returns:
        np.ndarray: The summed background score.
    """
    return (
        scores[:, labels["gluon"]]
        + scores[:, labels["light"]]
        + pileup_score(scores, labels)
    )


def tau_score(
    seed: str, scores: npt.NDArray, labels: dict[str, int]
) -> npt.NDArray:
    """Tau score of a seed per jet: the tau classes against the
    background (bkg_score).

    Args:
        seed: Tau seed (singleTau, diTau or diTauTopo); singleTau also
            counts electrons as signal.
        scores: Class scores (jets x classes).
        labels: Class name -> output index.

    Returns:
        np.ndarray: The score.
    """
    signal = (
        ["taup", "taum", "electron"]
        if seed == "singleTau"
        else ["taup", "taum"]
    )
    tau = sum((scores[:, labels[c]] for c in signal), np.zeros(len(scores)))
    return x_vs_y(tau, bkg_score(scores, labels))


def has_group(group: str, config: dict[str, Any], info: ModelInfo) -> bool:
    """Whether the model has every class of a class or class group.

    Args:
        group: Class name, class group of config CLASS_GROUPS, or all.
        config: Eval config.
        info: ModelInfo of the model.

    Returns:
        bool: True if every class is in info.classes.
    """
    if group == "all":
        return True
    return all(
        c in info.classes for c in config["CLASS_GROUPS"].get(group, [group])
    )


def roc_pairs(
    config: dict[str, Any], info: ModelInfo
) -> list[tuple[str, str]]:
    """The binary ROC pairs of the jetClassifier collection.

    Args:
        config: Eval config; ROC_PAIRS is "all" or a list of [signal,
            background].
        info: ModelInfo of the model.

    Returns:
        list: (signal, background) tuples, every ordered pair of the
            model's classes for "all".
    """
    if config["ROC_PAIRS"] == "all":
        return [(a, b) for a in info.classes for b in info.classes if a != b]
    return [tuple(p) for p in config["ROC_PAIRS"]]
