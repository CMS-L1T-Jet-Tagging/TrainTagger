"""Plot specs draw from stored results; labels resolve result references."""

import os
import tempfile

import toy
from tagger.eval import config as config_mod
from tagger.eval.utils.cli import evaluate
from tagger.plot.cli import ModelResults, draw, specs_for
from tagger.plot.registry import PLOT_COLLECTIONS, format_label, load_all

load_all()


def test_format_label():
    values = {"auc(b, light)": 0.8123, "effAtMistag(b, light, 0.1)": 0.25}
    lookup = values.__getitem__
    assert (
        format_label("b (AUC = [[auc(b, light):.2f]])", lookup)
        == "b (AUC = 0.81)"
    )
    assert (
        format_label(r"$\tau_{h}$ [[effAtMistag(b, light, 0.1):.0%]]", lookup)
        == r"$\tau_{h}$ 25%"
    )


def test_plot_collections_are_documented():
    for coll in PLOT_COLLECTIONS.values():
        assert coll.doc and coll.reads, coll.name


def test_all_specs_draw_and_overlay():
    with tempfile.TemporaryDirectory() as d:
        a, b = os.path.join(d, "a"), os.path.join(d, "b")
        cfg = config_mod.load(overrides=["ROC_PAIRS=[[b, light]]"])
        evaluate(
            a,
            ["train", "jetClassifier", "jetPt"],
            cfg,
            sources=toy.ToySources(),
        )
        evaluate(
            b,
            ["train", "jetClassifier", "jetPt"],
            cfg,
            sources=toy.ToySources(toy.jets(seed=5)),
        )
        for name in ("train", "jetClassifier", "jetPt"):
            models = [
                ModelResults(a, "a", [name]),
                ModelResults(b, "b", [name]),
            ]
            n = draw(name, models, os.path.join(d, "plots"))
            assert n == len(specs_for(name, models[0]))
        assert os.path.exists(
            os.path.join(d, "plots", "training", "basic_ROC.png")
        )
        assert os.path.exists(
            os.path.join(
                d, "plots", "training", "roc_binary", "ROC_b_vs_light.pdf"
            )
        )
