"""Emulation collection on toy jets: the model versions (network,
hls4ml, CMSSW emulator), evaluate and draw."""

import os
import tempfile

import numpy as np
import toy
from tagger.eval import config as config_mod
from tagger.eval.inputs import resolve_model
from tagger.eval.utils.cli import evaluate
from tagger.plot.cli import draw, ModelResults, specs_for
from tagger.plot.registry import load_all


class ToyEmulationSources(toy.ToySources):
    """The same toy jets for every version; hls4ml and the emulator get
    their scores rounded to 1/128, the emulator also its pT correction,
    so that they agree on part of the jets.
    """

    def __init__(self, pseudo=""):
        super().__init__()
        self.pseudo = pseudo

    def emulation_jets(self):
        j = toy.jets(1000, seed=3)
        if self.pseudo:
            j.y_pred = np.round(j.y_pred * 128) / 128
        if self.pseudo == "emulator":
            j.pt_ratio = np.round(j.pt_ratio, 2)
        return j


def test_resolve_model_versions():
    cfg = config_mod.load()
    assert resolve_model("output/m", cfg) == ("output/m", "", "output/m")
    assert resolve_model("hls4ml:output/m", cfg) == (
        os.path.join("output/m", "hls4ml"),
        "hls4ml",
        "output/m",
    )
    assert resolve_model("emulator:output/m", cfg) == (
        os.path.join("output/m", "emulator"),
        "emulator",
        "output/m",
    )
    assert resolve_model("cmssw", cfg)[1:] == ("cmssw", None)


def test_on_emulation_reads_the_emulation_jets():
    run = toy.run(ToyEmulationSources())
    j = run.sources.emulation_jets()
    counts = run.get("onEmulation(scoreHist(b))")["counts"]
    assert counts.sum() == len(j) and not np.array_equal(
        counts, run.get("scoreHist(b)")["counts"]
    )
    response = j.reco_pt * j.pt_ratio / j.truth_pt
    assert np.isclose(
        run.get("onEmulation(responseMedian(all))"), np.median(response)
    )
    assert np.isclose(
        run.get("onEmulation(responseRms(all))"), np.sqrt(np.mean(response**2))
    )
    assert np.array_equal(
        run.get("onEmulation(jetOutputs)")["taup"],
        j.y_pred[:, toy.CLASSES["taup"]],
    )


def test_emulation_collection_evaluates_and_draws():
    load_all()
    with tempfile.TemporaryDirectory() as d:
        cfg = config_mod.load()
        model = os.path.join(d, "m")
        evaluate(model, ["emulation"], cfg, sources=ToyEmulationSources())
        one = [ModelResults(model, "m", ["emulation"])]
        n_specs = len(specs_for("emulation", one[0]))
        # without the other versions only the network series are drawn;
        # the per-jet comparisons are skipped, and the response plot
        # too: its network curve carries the emulator's numbers (swapped
        # labels)
        assert (
            draw("emulation", one, os.path.join(d, "alone"))
            == n_specs - 1 - len(toy.CLASSES) - 1 - 1
        )
        for version in ("hls4ml", "emulator"):
            evaluate(
                os.path.join(model, version),
                ["emulation"],
                cfg,
                sources=ToyEmulationSources(version),
            )
        one = [ModelResults(model, "m", ["emulation"])]
        assert draw("emulation", one, os.path.join(d, "plots")) == n_specs
        text = (
            open(os.path.join(d, "plots", "emulation", "bit_accuracy.txt"))
            .read()
            .splitlines()
        )
        hls, emu = (
            ToyEmulationSources("hls4ml").emulation_jets(),
            ToyEmulationSources("emulator").emulation_jets(),
        )
        expected = 100 * np.mean(hls.pt_ratio == emu.pt_ratio)
        assert text[0].endswith(f"regression: {expected:.4f} %") and text[
            1
        ].endswith("b: 100.0000 %")
