"""Binned efficiencies with Clopper-Pearson errors, and ratios of 2D
histograms.

Plain functions: no evaluation run, everything is passed as arguments.
"""

from __future__ import annotations

from collections.abc import Sequence

import hist
import numpy as np
import numpy.typing as npt
from hist.intervals import ratio_uncertainty


def efficiency(
    num_values: npt.ArrayLike,
    den_values: npt.ArrayLike,
    edges: Sequence[float],
) -> dict[str, npt.NDArray]:
    """Binned efficiency with Clopper-Pearson errors.

    The same as hist's plot_ratio(uncertainty_type="efficiency").

    Args:
        num_values: Values of the passing entries (numerator).
        den_values: Values of all entries (denominator).
        edges: Bin edges.

    Returns:
        dict: ``edges``, ``num`` and ``den`` (counts per bin), ``eff``
            (NaN where den = 0), and the
        errors ``err_low`` and ``err_up``.
    """
    axis = hist.axis.Variable(edges)
    num, den = hist.Hist(axis), hist.Hist(axis)
    num.fill(np.asarray(num_values))
    den.fill(np.asarray(den_values))
    n, d = num.values(), den.values()
    with np.errstate(divide="ignore", invalid="ignore"):
        eff = n / d
        err = ratio_uncertainty(num=n, denom=d, uncertainty_type="efficiency")
    return {
        "edges": np.asarray(edges, dtype=float),
        "num": n,
        "den": d,
        "eff": eff,
        "err_low": err[0],
        "err_up": err[1],
    }


def ratio_2d(num: npt.NDArray, den: npt.NDArray) -> npt.NDArray:
    """Ratio of two histograms per bin.

    Args:
        num: Numerator counts.
        den: Denominator counts, same shape.

    Returns:
        np.ndarray: num / den, NaN where den = 0.
    """
    ratio = np.divide(num, den, where=den != 0)
    ratio[den == 0] = np.nan
    return ratio
