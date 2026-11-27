"""MinBias rates and working points: rate grids, the points at a target
rate, rate versus threshold.

Plain functions: no evaluation run, everything is passed as arguments.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import awkward as ak
import hist
import numpy as np
import numpy.typing as npt


def rate(
    n_pass: float | npt.NDArray, n_events: int, minbias_rate: float
) -> float | npt.NDArray:
    """Trigger rate of a selection.

    Args:
        n_pass: Number of passing MinBias events.
        n_events: Number of MinBias events.
        minbias_rate: Rate (kHz) of all MinBias events (config
            MINBIAS_RATE).

    Returns:
        The rate in kHz.
    """
    return n_pass / n_events * minbias_rate


def find_rate(
    rate_list: Sequence[float], target: float, rate_range: float
) -> list[int]:
    """Indices of the rates within a window around a target rate.

    Args:
        rate_list: Rates (kHz).
        target: Target rate (kHz).
        rate_range: Half width of the window (kHz).

    Returns:
        list: Indices of the rates in [target - rate_range, target +
            rate_range].
    """
    return [
        i
        for i, r in enumerate(rate_list)
        if target - rate_range <= r <= target + rate_range
    ]


def wp_per_pt(
    pt: npt.ArrayLike, nn: npt.ArrayLike
) -> tuple[npt.NDArray, npt.NDArray]:
    """The highest score per pT of the on-target grid points.

    The on-target points of a rate grid hold several scores per pT (the
    rate range). The highest, the most conservative cut, is kept, so
    that the interpolation in pT is well defined.

    Args:
        pt: pT of the points.
        nn: Score of the points.

    Returns:
        tuple: The distinct pT values (sorted) and the highest score of
            each.
    """
    pts, inv = np.unique(np.asarray(pt, dtype=float), return_inverse=True)
    best = np.full(len(pts), -np.inf)
    np.maximum.at(best, inv, np.asarray(nn, dtype=float))
    return pts, best


def closest_lower(rates: npt.ArrayLike, target: float) -> int:
    """Index of the highest rate that is not above the target.

    The fallback when no rate is within the window of the target.

    Args:
        rates: Rates (kHz).
        target: Target rate (kHz).

    Returns:
        int: The index.
    """
    return int(np.argmax(np.where(np.asarray(rates) > target, -1, rates)))


def hist_2d(
    x_edges: Sequence[float],
    y_edges: Sequence[float],
    x: str = "pt",
    y: str = "nn",
) -> hist.Hist:
    """Empty 2D histogram for a rate grid.

    Args:
        x_edges: Bin edges of the x axis.
        y_edges: Bin edges of the y axis.
        x: Name of the x axis, used to fill.
        y: Name of the y axis.

    Returns:
        hist.Hist: The empty histogram.
    """
    return hist.Hist(
        hist.axis.Variable(x_edges, name=x),
        hist.axis.Variable(y_edges, name=y),
    )


def counts_above(values: npt.ArrayLike) -> npt.NDArray:
    """Counts of a 2D histogram above every pair of lower bin edges.

    A reversed cumulative sum: for each (x edge, y edge) the entries
    with x >= x edge and y >= y edge.

    Args:
        values: Bin contents without the underflow and with the overflow
            as last row and column, h.values(flow=True)[1:, 1:]. Append
            a zero row and column to leave the overflow out.

    Returns:
        np.ndarray: Counts per lower edge pair, the shape of the
            histogram without flow bins.
    """
    v = np.asarray(values)
    return v[::-1, ::-1].cumsum(axis=0).cumsum(axis=1)[::-1, ::-1][:-1, :-1]


def threshold_grid(
    pt_edges: Sequence[float],
    nn_edges: Sequence[float],
    candidates: Callable[[float], tuple[npt.NDArray, npt.NDArray]],
    n_events: int,
    minbias_rate: float,
    overflow: bool = True,
) -> dict[str, npt.NDArray]:
    """Rate grid of a seed whose pT threshold also selects its
    candidates.

    Args:
        pt_edges: pT thresholds (GeV), the last an upper edge.
        nn_edges: Score thresholds, the last an upper edge.
        candidates: Function of a pT threshold that gives the (pT,
            score) of each event's candidate.
        n_events: Number of MinBias events.
        minbias_rate: Rate (kHz) of all MinBias events (config
            MINBIAS_RATE).
        overflow: Whether candidates above the last edges count.

    Returns:
        dict: ``pt``, ``nn`` and ``rate`` (kHz), one entry per grid
            point.
    """
    h = hist_2d(pt_edges, nn_edges)
    rows = []
    for pt in pt_edges[:-1]:
        h.reset()
        cand_pt, cand_nn = candidates(pt)
        h.fill(pt=cand_pt, nn=cand_nn)
        v = (
            h.values(flow=True)[1:, 1:]
            if overflow
            else np.pad(h.values(), ((0, 1), (0, 1)))
        )
        rows.append(counts_above(v)[h.axes[0].index(pt)])
    pts, nns = np.meshgrid(
        np.asarray(pt_edges[:-1], dtype=float),
        np.asarray(nn_edges[:-1], dtype=float),
        indexing="ij",
    )
    return {
        "pt": pts.ravel(),
        "nn": nns.ravel(),
        "rate": np.asarray(rate(np.concatenate(rows), n_events, minbias_rate)),
    }


def rate_points(
    event_ids: ak.Array | npt.NDArray,
    pts: ak.Array | npt.NDArray,
    cuts: Sequence[float],
    n_event: int,
    minbias_rate: float,
    flatten: bool = False,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Rate and its sqrt(N) error versus a pT threshold.

    Args:
        event_ids: Event number of each candidate.
        pts: pT of each candidate.
        cuts: pT thresholds (GeV).
        n_event: Number of MinBias events.
        minbias_rate: Rate (kHz) of all MinBias events (config
            MINBIAS_RATE).
        flatten: Whether event_ids is jagged and is flattened before
            counting.

    Returns:
        tuple: Rates (kHz) and their errors, one per threshold. An event
            counts once if any of its
        candidates is above the threshold.
    """
    rate, err = [], []
    for c in cuts:
        ids = event_ids[pts > c]
        n = len(np.unique(ak.flatten(ids) if flatten else ids))
        rate.append(n / n_event * minbias_rate)
        err.append(np.sqrt(n) / n_event * minbias_rate)
    return np.array(rate), np.array(err)
