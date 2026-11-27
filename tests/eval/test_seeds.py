"""Trigger seed variables on toy events: definitions against brute
force, and the seed collections evaluate and draw."""

import itertools
import os
import tempfile

import awkward as ak
import hist
import numpy as np
import toy
from tagger.eval import config as config_mod
from tagger.eval.utils.cli import evaluate
from tagger.eval.variable_manager import ModelInfo, vm
from tagger.eval.variables.base.efficiency import efficiency
from tagger.eval.variables import seeds_hh
from tagger.plot.cli import draw, ModelResults, specs_for
from tagger.plot.registry import load_all


def seed_run():
    return toy.run(toy.ToySeedSources(), overrides=toy.SEED_OVERRIDES)


def test_efficiency_counts_and_ratio():
    rng = np.random.default_rng(1)
    den = rng.uniform(0, 100, 5000)
    num = den[rng.uniform(size=5000) < 0.4]
    edges = [0, 10, 25, 50, 100]
    e = efficiency(num, den, edges)
    assert np.array_equal(e["den"], np.histogram(den, edges)[0])
    assert np.array_equal(e["num"], np.histogram(num, edges)[0])
    assert np.allclose(e["eff"], e["num"] / e["den"])
    assert np.all(e["err_low"] >= 0) and np.all(e["err_up"] >= 0)


def test_bbtt_tau_pair_brute_force():
    rng = np.random.default_rng(2)
    tp, tm, b = (rng.uniform(size=(200, 4)) for _ in range(3))
    bsum, tsum, idx = seeds_hh._bbtt_choose(tp, tm, b)
    for ev in range(200):
        pairs = list(itertools.combinations(range(4), 2))
        scores = [
            (tp[ev, i] + tp[ev, j]) * (tm[ev, i] + tm[ev, j]) for i, j in pairs
        ]
        best = pairs[int(np.argmax(scores))]
        assert tuple(idx[ev]) == best and np.isclose(tsum[ev], max(scores))
        assert np.isclose(
            bsum[ev], sum(b[ev, k] for k in range(4) if k not in best)
        )


def test_bbbb_rate_grid_matches_hist_slicing():
    """The cumulative sum equals the old per-point hist slicing,
    overflow included."""
    run = seed_run()
    grid = run.get("bbbbRateGrid(raw, all)")
    e = seeds_hh._bbbb_events(run, "minbias", "raw")
    h = hist.Hist(
        hist.axis.Variable(list(np.arange(0, 500, 2)) + [10000], name="ht"),
        hist.axis.Variable(
            [round(i, 2) for i in np.arange(0, 2.5, 0.01)] + [4.0], name="nn"
        ),
    )
    h.fill(ht=e["ht"], nn=e["bsum"])
    for k in np.random.default_rng(3).integers(0, len(grid["ht"]), 50):
        ht, nn = grid["ht"][k], grid["nn"][k]
        count = h[{"ht": slice(ht * 1j, None, sum)}][
            {"nn": slice(nn * 1.0j, None, sum)}
        ]
        assert np.isclose(grid["rate"][k], count / e["n_events"] * 100)


def test_bbtt_tau_selection_uses_each_events_pair():

    jet_pt = ak.Array([[50, 50, 3, 3], [3, 3, 50, 50]])
    e = {
        "jet_pt": jet_pt,
        "jet_eta": ak.zeros_like(jet_pt),
        "raw": {"tau_idx": np.array([[0, 1], [2, 3]])},
        "qg": {"tau_idx": np.array([[2, 3], [2, 3]])},
    }
    config = config_mod.load()
    assert list(seeds_hh._bbtt_default_selection(config, e, "raw", "tau")) == [
        True,
        True,
    ]
    assert list(seeds_hh._bbtt_default_selection(config, e, "qg", "tau")) == [
        False,
        True,
    ]


def test_tau_wp_is_on_the_line():
    run = seed_run()
    line = run.get("tauWpLine(diTau)")
    rates = run.get("tauRateGrid(diTau)")["rate"]
    assert len(line["target_pt"]) >= 2
    on = np.abs(rates - 28) <= 1
    assert on.sum() == len(line["target_pt"])
    wp = run.get("tauWp(diTau)")
    assert np.isfinite(wp)


def test_ht_only_cut_falls_back_to_a_lower_rate():
    run = toy.run(
        toy.ToySeedSources(),
        overrides=["MINBIAS_RATE=100", "BBBB_TARGET_RATE=1000"],
    )
    cut = run.get("htOnlyCut(bbbb)")
    # every grid rate is below 1000: the lowest threshold has the
    # closest one
    assert cut == 0.0


def cmssw_run():
    return toy.run(
        toy.ToySeedSources(pseudo="cmssw"), overrides=toy.SEED_OVERRIDES
    )


def test_cmssw_pseudo_model_reads_the_l1_baseline():
    """-m cmssw: the same selection code on the CMSSW L1 baseline score
    and pT, fixed di-tau working point."""
    run = cmssw_run()
    assert run.get("tauWp(diTau)") == run.config["CMSSW_WPS"]["tau"]
    sig = run.sources.signal_events("vbfHtautau", 0)
    flav, gen_pt, gen_dr = (
        np.asarray(sig.branch(b))
        for b in ("jet_tauflav", "jet_genmatch_pt", "jet_genmatch_dR")
    )
    gen_eta, score, pt = (
        np.asarray(sig.branch(b))
        for b in ("jet_genmatch_eta", "jet_tauscore", "jet_taupt")
    )
    wps = run.config["CMSSW_WPS"]
    deno = (
        (flav == 1)
        & (gen_pt > 1)
        & (np.abs(gen_dr) < 0.4)
        & (np.abs(gen_eta) < 1.5)
    )
    num = deno & (pt > wps["tau_l1_pt"]) & (score > wps["tau"])
    assert np.isclose(
        run.get("tauTotalEff(diTau, wp, barrel)"), num.mean() / deno.mean()
    )
    assert (
        run.get("bbbbWpHt(raw, all)") == wps["btag_l1_ht"]
        and run.get("bbbbWpNn(raw, all)") == wps["btag"]
    )


def test_cmssw_collections_skip_what_needs_the_network():

    cfg = config_mod.load(
        overrides=["BBBB_VARIANTS=[[raw, all], [vs_qg, all]]"]
    )
    info = ModelInfo([], pseudo="cmssw")
    bbtt = vm.get_collection(
        "seedBbtt", cfg, info
    )  # only the CMSSW reference seeds
    assert bbtt and all(
        e == "bbttRun3SeedRate" or e.startswith("bbttRef") for e in bbtt
    )
    assert not any(
        "vs_qg" in e for e in vm.get_collection("seedBbbb", cfg, info)
    )


def test_seed_collections_evaluate_and_draw():

    load_all()
    with tempfile.TemporaryDirectory() as d:
        cfg = config_mod.load(
            overrides=toy.SEED_OVERRIDES + [f"CMSSW_MODEL_DIR='{d}/cmssw'"]
        )
        a, b = os.path.join(d, "a"), os.path.join(d, "b")
        # without cmssw results the CMSSW series are left out, a plot of
        # CMSSW alone is skipped
        evaluate(a, ["seedSingleTau"], cfg, sources=toy.ToySeedSources())
        one = [ModelResults(a, "a", ["seedSingleTau"])]
        assert (
            draw("seedSingleTau", one, os.path.join(d, "noref"))
            == len(specs_for("seedSingleTau", one[0])) - 1
        )
        assert (
            "CMSSW" not in open(
                os.path.join(
                    d, "noref", "physics", "tau", "TotalEff_barrel.txt"
                )
            ).read()
        )
        evaluate(
            os.path.join(d, "cmssw"),
            toy.SEED_COLLECTIONS,
            cfg,
            sources=toy.ToySeedSources(pseudo="cmssw"),
        )
        evaluate(a, toy.SEED_COLLECTIONS, cfg, sources=toy.ToySeedSources())
        evaluate(b, toy.SEED_COLLECTIONS, cfg, sources=toy.ToySeedSources())
        for name in toy.SEED_COLLECTIONS:
            one = [ModelResults(a, "a", [name])]
            assert draw(name, one, os.path.join(d, "plots")) == len(
                specs_for(name, one[0])
            ), name
            two = one + [ModelResults(b, "b", [name])]
            assert draw(name, two, os.path.join(d, "overlay")) == len(
                specs_for(name, one[0])
            ), name
        assert os.path.exists(
            os.path.join(
                d, "plots", "physics", "tautau", "TotalEff_barrel.txt"
            )
        )
        assert os.path.exists(
            os.path.join(
                d, "plots", "physics", "bbbb", "HH_eff_HT_raw_all.png"
            )
        )
