"""VariableManager, parser, config, collections, results and docs."""

import os
import tempfile

import numpy as np
import pytest

import toy
from tagger.eval import config as config_mod
from tagger.eval.utils import docs, output
from tagger.eval.variable_manager import ModelInfo
from tagger.eval.variable_manager import InputNotDeclared
from tagger.eval.variable_manager import INPUTS, KINDS, vm

# -- parser ------------------------------------------------------------


def test_canonical_spelling():
    assert (
        vm.canonical("effAtMistag(b,light,.1)") == "effAtMistag(b, light, 0.1)"
    )
    assert vm.canonical(" auc( taus , jets ) ") == "auc(taus, jets)"
    assert vm.canonical("confusionMatrix") == "confusionMatrix"
    assert (
        vm.canonical("effAtMistag(b, light, 1e-3)")
        == "effAtMistag(b, light, 0.001)"
    )


def test_nested_expression():
    call = vm.parse("onProcess(TT_PU200, roc(b, light))")
    assert call.name == "onProcess(TT_PU200, roc(b, light))"
    assert call.args[1].variable.name == "roc"


def test_parse_errors():
    with pytest.raises(KeyError):
        vm.parse("noSuchVariable(b)")
    with pytest.raises(TypeError):
        vm.parse("auc(b)")
    with pytest.raises(TypeError):
        vm.parse("effAtMistag(b, light, tight)")
    with pytest.raises(SyntaxError):
        vm.parse("auc(b, light")
    with pytest.raises(SyntaxError):
        vm.parse("auc(b, light) extra")


def test_unknown_variable_hint():
    with pytest.raises(KeyError, match="did you mean auc"):
        vm.parse("AUC(b, light)")


# -- VariableManager ---------------------------------------------------


def test_every_variable_is_documented():
    for var in vm.get_variables():
        assert var.doc and len(var.doc) > 20, var.name
        assert set(var.inputs) <= set(INPUTS), var.name
        assert var.kind in KINDS, var.name


def test_registration_checks():
    def undocumented(run):
        return 1.0

    with pytest.raises(ValueError):
        vm.variable(inputs={"test_jets"}, kind="scalar")(undocumented)

    def bad_input(run):
        """Reads something that is not an input."""

    with pytest.raises(ValueError):
        vm.variable(inputs={"my_file"}, kind="scalar")(bad_input)


def test_undeclared_input_is_an_error():
    vm.remove_variable("_toyNoInputs")

    @vm.variable(name="_toyNoInputs", inputs={"model"}, kind="scalar")
    def _toy(run):
        """Test helper that reads test_jets without declaring it."""
        return len(run.jets())

    try:
        with pytest.raises(InputNotDeclared):
            toy.run().get("_toyNoInputs")
    finally:
        vm.remove_variable("_toyNoInputs")


# -- config ------------------------------------------------------------


def test_config_layers():
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write("MISTAG_POINTS: [0.05]\nROC_PAIRS: [[b, light]]\n")
    try:
        cfg = config_mod.load(
            [f.name], ["MISTAG_POINTS=[0.2, 0.3]", "ROC_PAIRS=all"]
        )
    finally:
        os.unlink(f.name)
    assert cfg["MISTAG_POINTS"] == [0.2, 0.3]
    assert cfg["ROC_PAIRS"] == "all"
    with pytest.raises(KeyError):
        config_mod.load((), ["NOT_A_KEY=1"])
    assert config_mod.config_hash(cfg) != config_mod.config_hash(
        config_mod.load()
    )


# -- collections -------------------------------------------------------


def test_collection_expansion():
    cfg = config_mod.load(
        overrides=["ROC_PAIRS=[[b, light]]", "MISTAG_POINTS=[0.1]"]
    )
    info = ModelInfo(list(toy.CLASSES), toy.FEATURES)
    exprs = vm.get_collection("jetClassifier", cfg, info)
    assert "effAtMistag(b, light, 0.1)" in exprs
    assert "auc(taus, jets)" in exprs
    assert len(exprs) == len(set(exprs))
    assert set(vm.get_collection("jetPt", cfg, info)) <= set(
        vm.get_collection("ci", cfg, info)
    )


def test_user_collections_nest_and_detect_cycles():
    info = ModelInfo(list(toy.CLASSES))
    cfg = config_mod.load(
        overrides=[
            'COLLECTIONS={"a": ["auc(b,light)", "b"], "b":'
            ' ["confusionMatrix"]}'
        ]
    )
    assert vm.get_collection("a", cfg, info) == [
        "auc(b, light)",
        "confusionMatrix",
    ]
    cyc = config_mod.load(overrides=['COLLECTIONS={"a": ["b"], "b": ["a"]}'])
    with pytest.raises(ValueError):
        vm.get_collection("a", cyc, info)


def test_add_collection():
    info = ModelInfo(list(toy.CLASSES))
    vm.add_collection("_toyMine", ["auc(b,light)", "train"])
    try:
        # nested: train has no loss history for this model, only the
        # SHAP importances
        assert vm.get_collection("_toyMine", config_mod.load(), info) == [
            "auc(b, light)",
            "shapImportance(classification)",
            "shapImportance(regression)",
        ]
        with pytest.raises(ValueError):
            vm.add_collection("_toyMine", ["confusionMatrix"])
    finally:
        vm.remove_collection("_toyMine")


def test_groups_missing_from_the_model_are_skipped():
    info = ModelInfo(["b", "light", "gluon"])
    exprs = vm.get_collection("jetClassifier", config_mod.load(), info)
    assert "auc(b, qcd)" in exprs
    assert not any("taus" in e for e in exprs)


# -- results -----------------------------------------------------------


def test_results_round_trip():
    values = {
        "auc(b, light)": 0.75,
        "roc(b, light)": {
            "tpr": np.array([0, 0.5, 1.0]),
            "fpr": np.array([0, 0.1, 1.0]),
        },
        "confusionMatrix": {"matrix": np.eye(2), "classes": ["b", "light"]},
    }
    with tempfile.TemporaryDirectory() as d:
        output.write(d, "toy", values, {"git_commit": "x"})
        res = output.read(d, "toy")
        assert res["auc(b, light)"] == 0.75
        assert np.array_equal(res["roc(b, light)"]["tpr"], [0, 0.5, 1.0])
        assert res["confusionMatrix"]["classes"] == ["b", "light"]
        assert res.provenance["git_commit"] == "x"
        with pytest.raises(output.MissingResult):
            output.read(d, "other")


# -- docs --------------------------------------------------------------


def test_docs_are_current():
    path = os.path.normpath(docs.DOCS_PATH)
    with open(path) as f:
        committed = f.read()
    assert committed == docs.markdown(
        config_mod.load()
    ), "docs/variables.md is stale: python -m tagger.eval --docs"
