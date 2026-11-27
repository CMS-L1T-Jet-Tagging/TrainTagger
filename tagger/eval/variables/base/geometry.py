"""Angular helpers: deltaR, eta regions, and the two leading candidates
of an event.

Plain functions: no evaluation run, everything is passed as arguments.
"""

from __future__ import annotations

import awkward as ak
import numpy as np
import numpy.typing as npt


def delta_r(
    eta1: npt.NDArray, phi1: npt.NDArray, eta2: npt.NDArray, phi2: npt.NDArray
) -> npt.NDArray:
    """deltaR between two directions, with phi wrapped to [-pi, pi).

    Args:
        eta1: Eta of the first direction.
        phi1: Phi of the first direction.
        eta2: Eta of the second direction.
        phi2: Phi of the second direction.

    Returns:
        The deltaR, elementwise for arrays.
    """
    dphi = (phi1 - phi2 + np.pi) % (2 * np.pi) - np.pi
    return np.sqrt((eta1 - eta2) ** 2 + dphi**2)


def eta_region(
    eta: npt.ArrayLike, region: str, barrel_eta: float, tau_eta_cut: float
) -> npt.NDArray:
    """Mask of an eta region.

    Args:
        eta: Values of eta.
        region: barrel (|eta| < barrel_eta) or tau_endcap (barrel_eta <
            |eta| < tau_eta_cut); anything else selects all.
        barrel_eta: |eta| boundary of the barrel (config BARREL_ETA).
        tau_eta_cut: Upper |eta| of the tau endcap (config TAU_ETA_CUT).

    Returns:
        Boolean mask, same shape as eta.
    """
    if region == "barrel":
        return np.abs(eta) < barrel_eta
    if region == "tau_endcap":
        return (np.abs(eta) > barrel_eta) & (np.abs(eta) < tau_eta_cut)
    return np.abs(eta) > 0.0


def min_of_leading_two(
    a: ak.Array | npt.NDArray, cuts: npt.NDArray
) -> npt.NDArray:
    """The smaller value of the first two candidates of each selected
    event.

    Args:
        a: Per-event candidate values, at least two per event (events x
            candidates).
        cuts: Boolean mask of the events to keep.

    Returns:
        np.ndarray: One value per kept event.
    """
    return np.min(
        np.vstack([np.asarray(a[:, 0][cuts]), np.asarray(a[:, 1][cuts])]).T,
        axis=1,
    )
