"""Each variable on toy data, against an independent calculation where
one is simple."""

import numpy as np
import pytest
import tensorflow as tf
from sklearn.metrics import roc_auc_score

import toy
from tagger.eval import config as config_mod
from tagger.eval.variable_manager import ModelInfo
from tagger.eval.variable_manager import vm


def test_auc_matches_sklearn():
    run = toy.run()
    j = run.jets()
    sig, bkg = [4, 5], [0, 1, 2, 3]  # taus vs jets
    mask = j.y_true[:, sig + bkg].sum(axis=1) > 0
    s = j.y_pred[:, sig].sum(axis=1)
    disc = s / (s + j.y_pred[:, bkg].sum(axis=1))
    truth = j.y_true[:, sig].sum(axis=1) > 0
    assert np.isclose(
        run.get("auc(taus, jets)"), roc_auc_score(truth[mask], disc[mask])
    )


def test_auc_one_vs_rest_matches_sklearn():
    run = toy.run()
    j = run.jets()
    assert np.isclose(
        run.get("aucOneVsRest(light)"),
        roc_auc_score(j.y_true[:, 2], j.y_pred[:, 2]),
    )


def test_perfect_and_random_classifier():
    perfect = toy.run(toy.ToySources(toy.jets(skill=50)))
    assert perfect.get("auc(b, light)") > 0.999
    assert perfect.get("effAtMistag(b, light, 0.01)") > 0.99
    random = toy.run(toy.ToySources(toy.jets(n=20000, skill=0)))
    assert abs(random.get("auc(b, light)") - 0.5) < 0.03
    assert abs(random.get("effAtMistag(b, light, 0.1)") - 0.1) < 0.03


def test_eff_at_mistag_interpolates_the_roc():
    run = toy.run()
    roc = run.get("roc(b, qcd)")
    for m in (0.1, 0.01):
        assert np.isclose(
            run.get(f"effAtMistag(b, qcd, {m})"),
            np.interp(m, roc["fpr"], roc["tpr"]),
        )
    assert run.get("effAtMistag(b, qcd, 0.1)") >= run.get(
        "effAtMistag(b, qcd, 0.01)"
    )


def test_confusion_matrix_rows_sum_to_one():
    m = toy.run().get("confusionMatrix")
    assert np.allclose(m["matrix"].sum(axis=1), 1)
    assert m["classes"] == list(toy.CLASSES)
    perfect = toy.run(toy.ToySources(toy.jets(skill=50))).get(
        "confusionMatrix"
    )
    assert np.allclose(perfect["matrix"], np.eye(8))


def test_eff_vs_pt_brute_force():
    run = toy.run()
    j = run.jets()
    c = run.get("effVsPt(b, 0.2)")
    is_b, score = j.y_true[:, 0] == 1, j.y_pred[:, 0]
    passed = score[is_b] > c["threshold"]
    pt = j.reco_pt[is_b]
    for lo, hi, eff, n in zip(
        c["edges"][:-1], c["edges"][1:], c["eff"], c["n"]
    ):
        sel = (pt >= lo) & (pt < hi)
        assert n == sel.sum()
        if n:
            assert np.isclose(eff, passed[sel].mean())
    # the cut gives roughly the asked one-vs-rest mistag rate
    assert abs(np.mean(score[~is_b] > c["threshold"]) - 0.2) < 0.02
    # the same working point in bins of gen pT
    g = run.get("effVsGenPt(b, 0.2)")
    assert g["threshold"] == c["threshold"]
    gen = j.truth_pt[is_b]
    for lo, hi, eff, n in zip(
        g["edges"][:-1], g["edges"][1:], g["eff"], g["n"]
    ):
        sel = (gen >= lo) & (gen < hi)
        assert n == sel.sum() and (
            not n or np.isclose(eff, passed[sel].mean())
        )


def test_eff_vs_eta_brute_force():
    """Same working point as effVsPt; per bin the passing fraction of
    the b jets in that eta range."""
    run = toy.run()
    j = run.jets()
    is_b, score = j.y_true[:, 0] == 1, j.y_pred[:, 0]
    for var, eta in (
        ("effVsEta", j.reco_eta),
        ("effVsAbsEta", np.abs(j.reco_eta)),
    ):
        c = run.get(f"{var}(b, 0.2)")
        assert c["threshold"] == run.get("effVsPt(b, 0.2)")["threshold"]
        passed, x = score[is_b] > c["threshold"], eta[is_b]
        assert (
            c["n"].sum() == ((x >= c["edges"][0]) & (x < c["edges"][-1])).sum()
        )
        for lo, hi, eff, n in zip(
            c["edges"][:-1], c["edges"][1:], c["eff"], c["n"]
        ):
            sel = (x >= lo) & (x < hi)
            assert n == sel.sum() and (
                not n or np.isclose(eff, passed[sel].mean())
            )


def test_eff_vs_eta_needs_eta_in_the_test_split():
    jets = toy.jets()
    jets.reco_eta = None
    run = toy.run(toy.ToySources(sample=jets, process=toy.jets(500, seed=1)))
    with pytest.raises(ValueError, match=r"reco_eta_test\.npz"):
        run.get("effVsEta(b, 0.2)")


def test_input_hist_counts_filled_candidates():
    run = toy.run()
    x = run.jets().model_inputs["basic_input"]
    is_b = run.jets().y_true[:, 0] == 1
    h = run.get("inputHist(pt_rel, b)")
    assert h["counts"].sum() == (x[is_b][:, :, 16] == 1).sum()
    assert len(h["edges"]) == 51
    total = sum(
        run.get(f"inputHist(pt_rel, {c})")["counts"].sum() for c in toy.CLASSES
    )
    assert total == run.get("inputHist(pt_rel, all)")["counts"].sum()


def test_multiplicity_leaves_out_empty_jets():
    j = toy.jets()
    j.model_inputs["basic_input"][:100] = 0
    run = toy.run(toy.ToySources(j))
    n = (j.model_inputs["basic_input"][:, :, 16] != 0).sum(axis=1)
    h = run.get("multiplicity(all)")
    assert h["counts"].sum() == (n > 0).sum() == len(j) - 100


def test_response_and_resolution_of_a_perfect_regression():
    j = toy.jets()
    j.reco_pt = j.truth_pt.copy()
    j.pt_ratio = np.ones(len(j))
    run = toy.run(toy.ToySources(j))
    for binned in ("GenPt", "Pt"):
        r = run.get(f"responseVs{binned}(all)")
        filled = ~np.isnan(r["reco"])
        assert np.allclose(r["reco"][filled], 1) and np.allclose(
            r["regressed"][filled], 1
        )
        s = run.get(f"resolutionVs{binned}(taus)")
        assert np.allclose(np.nan_to_num(s["regressed"]), 0)


def test_response_brute_force_one_bin():
    run = toy.run()
    j = run.jets()
    i = list(run.config["PT_BINS"]).index(50)
    is_b = j.y_true[:, 0] == 1
    regressed = j.reco_pt * j.pt_ratio
    # in bins of gen pT
    r = run.get("responseVsGenPt(b)")
    sel = is_b & (j.truth_pt > 50) & (j.truth_pt < 60)
    assert np.isclose(r["reco"][i], np.mean(j.reco_pt[sel] / j.truth_pt[sel]))
    assert np.isclose(
        r["regressed_err"][i],
        np.std(regressed[sel] / j.truth_pt[sel]) / np.sqrt(sel.sum()),
    )
    # in bins of L1 pT: each pT binned in itself
    r = run.get("responseVsPt(b)")
    sel = is_b & (j.reco_pt > 50) & (j.reco_pt < 60)
    assert np.isclose(r["reco"][i], np.mean(j.reco_pt[sel] / j.truth_pt[sel]))
    sel = is_b & (regressed > 50) & (regressed < 60)
    assert np.isclose(
        r["regressed"][i], np.mean(regressed[sel] / j.truth_pt[sel])
    )
    s = run.get("resolutionVsPt(b)")
    assert np.isclose(
        s["regressed"][i], np.std(regressed[sel] - j.truth_pt[sel])
    )


def test_pt_hist_and_ratio_hist():
    run = toy.run()
    j = run.jets()
    assert run.get("ptHist(gen)")["counts"].sum() == np.sum(j.truth_pt < 300)
    assert np.isclose(
        run.get("ptRatioHist(b)")["fraction"].sum(),
        np.mean((j.pt_ratio[j.y_true[:, 0] == 1] < 2)),
    )


def test_shap_importance():
    run = toy.run()
    s = run.get("shapImportance(classification)")
    assert s["values"].shape == (len(toy.CLASSES), len(toy.FEATURES))
    assert s["outputs"] == list(toy.CLASSES) and s["features"] == toy.FEATURES
    assert np.all(s["values"] >= 0)
    # fixed seed: a fresh run gives the same numbers
    again = toy.run(run.sources).get("shapImportance(classification)")
    assert np.array_equal(s["values"], again["values"])
    assert toy.run().get("shapImportance(regression)")["values"].shape == (
        1,
        len(toy.FEATURES),
    )


def test_loss_history():
    c = toy.run().get("lossHistory(jet_id_output_loss)")
    assert list(c["epoch"]) == [1, 2, 3]
    assert list(c["train"]) == [2.0, 1.5, 1.2] and list(c["val"]) == [
        2.1,
        1.6,
        1.4,
    ]


def test_sample_metavariables():
    run = toy.run()
    sub = run.get("onTestSubset(TT, auc(b, light))")
    full = run.get("onProcess(TT, auc(b, light))")
    test = run.get("auc(b, light)")
    assert len({sub, full, test}) == 3  # three different samples
    # the subset is exactly the 300 shared jets
    ref = toy.run(toy.ToySources(run.jets().subset(np.arange(300)))).get(
        "auc(b, light)"
    )
    assert np.isclose(sub, ref)
    # the switch does not leak: the plain variable is unchanged
    # afterwards
    assert run.get("auc(b, light)") == test


def test_every_collection_member_evaluates():
    """Smoke test: every expression of the built-in collections on toy
    data, with one signal process."""
    cfg = config_mod.load(overrides=["SIGNAL_PROCESSES=['TT']"])
    run = toy.run(overrides=["SIGNAL_PROCESSES=['TT']"])
    info = ModelInfo(list(toy.CLASSES) + [], toy.FEATURES)
    exprs = vm.get_collection("ci", cfg, info)
    for e in exprs:
        value = run.get(e)
        assert isinstance(value, (float, dict)), e
    assert len(exprs) > 100


class ToyPtWeightModel(toy.ToyModel):
    """The toy model plus a pt_weights_output layer: one weight per
    constituent slot (16)."""

    output_pt_weights_name = "pt_weights_output"

    def build_model(self, inputs_shape, outputs_shape):
        """The toy model with a third output, the per-candidate pT weights."""
        super().build_model(inputs_shape, outputs_shape)
        inp = self.jet_model.inputs[0]
        weights = tf.keras.layers.Dense(
            1, activation="relu", name="pt_weights_dense"
        )(inp)
        weights = tf.keras.layers.Flatten(name=self.output_pt_weights_name)(
            weights
        )
        self.jet_model = tf.keras.Model(
            inp, self.jet_model.outputs + [weights]
        )


class ToyPtWeightSources(toy.ToySources):
    def model(self):
        if not hasattr(self, "_model"):
            self._model = ToyPtWeightModel()
        return self._model


def test_pt_weights_per_constituent():
    run = toy.run(ToyPtWeightSources())
    basic = run.jets().model_inputs["basic_input"]
    n_real = int(np.sum(basic[:, :, toy.FEATURES.index("pt")] != 0))
    h = run.get("ptWeightHist(pt)")
    assert h["all"].sum() == n_real
    pt = basic[:, :, toy.FEATURES.index("pt")]
    in_bin = np.sum((pt[pt != 0] >= h["lo"][0]) & (pt[pt != 0] < h["hi"][0]))
    assert h["bin1"].sum() == in_bin
    m = run.get("ptWeightVsInput(pt_rel)")
    assert m["counts"].max() == 1 and m["counts"].shape == (10, 15)
    cfg = run.config
    info = ModelInfo(
        list(toy.CLASSES), list(toy.FEATURES), [], pt_weights=True
    )
    assert "ptWeightVsInput(pt_rel)" in vm.get_collection("jetPt", cfg, info)
    assert not any(
        e.startswith("ptWeight")
        for e in vm.get_collection(
            "jetPt", cfg, ModelInfo(list(toy.CLASSES), list(toy.FEATURES), [])
        )
    )
