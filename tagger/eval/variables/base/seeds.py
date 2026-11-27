"""Shared access of the trigger seed variables to the evaluation run:
the MinBias and signal events, the class labels of the tagger, and rates
in kHz.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from tagger.eval.variables.base.rates import rate
from tagger.eval.variables.base.taggers import is_cmssw

if TYPE_CHECKING:
    from tagger.eval.inputs.events import EventSample
    from tagger.eval.variable_manager import EvalRun

# Decimals the expanded ranges are rounded to, against the float noise
# of np.arange
RANGE_DECIMALS = 6


def expand_ranges(specs: Mapping[str, list], name: str) -> list[float]:
    """The values of a config list of ranges (threshold scans, bin
    edges).

    Args:
        specs: Mapping that holds the list, e.g. the config or config
            TAU_SCANS.
        name: Key of the list in specs. Each item is [first, last,
            step], expanded as np.arange (last excluded, values rounded
            to RANGE_DECIMALS), or a single value, kept as it is.

    Returns:
        list: The values, in order.
    """
    values = []
    for part in specs[name]:
        if isinstance(part, (tuple, list)):
            values += [round(x, RANGE_DECIMALS) for x in np.arange(*part)]
        else:
            values.append(part)
    return values


def minbias(run: EvalRun) -> EventSample:
    """The MinBias events of the eval run.

    Args:
        run: Eval run.

    Returns:
        EventSample: The first config SEED_N_ENTRIES minbias jets of
            config MINBIAS_SAMPLE.

    Raises:
        ValueError: The sample has no events (no rate can be computed).
    """
    events = run.input(
        "minbias_events", run.config["SEED_N_ENTRIES"]["minbias"]
    )
    if events.n_events == 0:
        raise ValueError(
            f"no events in the MinBias sample {run.config['MINBIAS_SAMPLE']}"
        )
    return events


def signal(run: EvalRun, sample: str) -> EventSample:
    """The events of a signal sample.

    Args:
        run: Eval run.
        sample: Key of config SIGNAL_SAMPLES.

    Returns:
        EventSample: The first config SEED_N_ENTRIES signal jets of the
            sample.
    """
    return run.input(
        "signal_events", sample, run.config["SEED_N_ENTRIES"]["signal"]
    )


def labels(run: EvalRun, ev: EventSample) -> dict[str, int] | None:
    """Class labels of the tagger of an event sample.

    Args:
        run: Eval run.
        ev: Event sample.

    Returns:
        dict or None: Class name -> output index; None for the CMSSW
            pseudo-model, which has no class
        scores.
    """
    return None if is_cmssw(ev) else run.class_labels()


def rate_khz(
    run: EvalRun, n_pass: float | npt.NDArray, n_events: int
) -> float | npt.NDArray:
    """Rate of a selection of MinBias events, with config MINBIAS_RATE.

    Args:
        run: Eval run.
        n_pass: Number of passing events.
        n_events: Number of MinBias events.

    Returns:
        The rate in kHz.
    """
    return rate(n_pass, n_events, run.config["MINBIAS_RATE"])
