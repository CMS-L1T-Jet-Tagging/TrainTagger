"""Inputs of the variables: INPUTS lists them, each method of
:class:`.sources.Sources` provides one.

This module only names the inputs and resolves ``-m``; :mod:`.sources`
reads them. Loading is lazy: the model and TensorFlow are imported only
when a variable needs them. Jet samples are in :mod:`.jets`, event
samples (ntuples) and the CMSSW pseudo-model in :mod:`.events`.

Besides a model folder, ``-m`` takes the CMSSW pseudo-model ``cmssw``
(the CMSSW L1 baseline, for the seed collections), and two other
versions of a model for the emulation collection: ``hls4ml:<model>``
(its hls4ml build) and ``emulator:<model>`` (the CMSSW emulator scores
in the emulation ntuple). Their results go to ``<model>/hls4ml`` and
``<model>/emulator``.
"""

from __future__ import annotations

import os
from typing import Any

# The fixed set of inputs a variable may read. Anything else goes
# through run.get() on another variable.
INPUTS = {
    "test_jets": (
        "The frozen test split (output/<model>/testing_data) with the model"
        " outputs on it."
    ),
    "process_jets": (
        "Jets of one signal process (signal_process_data/<process>) with the"
        " model outputs on them."
    ),
    "minbias_events": (
        "MinBias events (config MINBIAS_SAMPLE, SEED_N_ENTRIES), with the"
        " model outputs on their jets."
    ),
    "signal_events": (
        "Signal events of config SIGNAL_SAMPLES[sample], with the model"
        " outputs on their jets."
    ),
    "training_history": (
        "Loss per epoch from the training (training_history.json in the model"
        " folder)."
    ),
    "model": (
        "The model object itself, for quantities that need its weights or"
        " graph."
    ),
    "trials": (
        "Several trainings of the same config that differ only in the seed"
        " (repeated -m)."
    ),
    "emulation_jets": (
        "Jets of the emulation ntuple (config EMULATION_SAMPLE) with the"
        " outputs of the model version: the network, its hls4ml build or the"
        " CMSSW emulator."
    ),
    "fastpuppi_events": (
        "Events of one process of the FastPUPPI performance ntuples of the"
        " model's CMSSW emulation (config FASTPUPPI_DIR): gen jets, gen"
        " particles and the L1 jet collections."
    ),
}

# -m names besides model folders: the CMSSW pseudo-model, and versions
# of a model (version:<model>)
PSEUDO_MODELS = ("cmssw",)
MODEL_VERSIONS = ("hls4ml", "emulator")


def resolve_model(
    name: str, config: dict[str, Any]
) -> tuple[str, str, str | None]:
    """What one -m argument names.

    Args:
        name: A model folder, cmssw (the CMSSW pseudo-model), or
            hls4ml:<model> / emulator:<model> (versions of the model).
        config: Eval config, for CMSSW_MODEL_DIR.

    Returns:
        tuple: The results folder (config CMSSW_MODEL_DIR for cmssw,
            <model>/hls4ml or
        <model>/emulator for a version), the pseudo-model or version
            name ("" for a model folder),
        and the model folder (None for cmssw).
    """
    if name in PSEUDO_MODELS:
        return (
            config.get("CMSSW_MODEL_DIR", os.path.join("output", name)),
            name,
            None,
        )
    version, sep, model_dir = name.partition(":")
    if sep and version in MODEL_VERSIONS:
        return os.path.join(model_dir, version), version, model_dir
    return name, "", name
