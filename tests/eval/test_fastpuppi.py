"""FastPUPPI regression collection on synthetic perfNano files
(toy_fastpuppi): definitions against brute force, and the reduced
collection evaluates and draws."""

import os
import tempfile

import awkward as ak
import numpy as np
import toy
import toy_fastpuppi
from tagger.eval import config as config_mod
from tagger.eval.inputs.fastpuppi import load_process
from tagger.eval.utils.cli import evaluate
from tagger.plot.cli import draw, ModelResults, specs_for
from tagger.plot.registry import load_all

SMALL = {
    "TT_PU200": 300,
    "QCD_Pt15To3000_PU200": 300,
    "MinBias_PU200": 20000,
    "VBFHToBB_PU200": 200,
    "VBFHToCC_PU200": 200,
}
OVERRIDES = [
    "FASTPUPPI_PROCESSES=['TT_PU200']",
    "TURN_ON_OBSERVABLES=['jet1']",
    "TURN_ON_RATES=[50]",
]
_FOLDER = []


def folder():
    if not _FOLDER:
        _FOLDER.append(
            toy_fastpuppi.write(tempfile.mkdtemp(prefix="fastpuppi_"), SMALL)
        )
    return _FOLDER[0]


class ToyFastPuppiSources(toy.ToySources):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def fastpuppi_events(self, process):
        return load_process(self.config["FASTPUPPI_DIR"], process, self.config)


def fp_run():
    cfg = config_mod.load(
        overrides=OVERRIDES + [f"FASTPUPPI_DIR='{folder()}'"]
    )
    return (
        toy.run(
            ToyFastPuppiSources(cfg),
            overrides=OVERRIDES + [f"FASTPUPPI_DIR='{folder()}'"],
        ),
        cfg,
    )


def test_response_and_leading_pt_against_brute_force():
    run, cfg = fp_run()
    ev = load_process(folder(), "TT_PU200", cfg)
    jets = ev["scPuppiExtendedJets"]
    truth, reco, ratio = (
        ak.to_numpy(ak.flatten(a))
        for a in (jets["raw"].genpt, jets["raw"].pt, jets["pt_ratio"])
    )
    lo, hi = cfg["PT_BINS"][8], cfg["PT_BINS"][9]
    sel = (truth > lo) & (truth < hi)
    r = run.get("fpResponseVsGenPt(TT_PU200, sc, median)")
    assert np.isclose(r["raw"][8], np.median(reco[sel] / truth[sel]))
    assert np.isclose(
        r["jecs"][8], np.median(reco[sel] * ratio[sel] / truth[sel])
    )
    # in bins of L1 pT, gen-matched jets only: corrected response in
    # bins of the corrected pT
    corrected = reco * ratio
    sel = (corrected > lo) & (corrected < hi) & (truth > 0)
    r = run.get("fpResponseVsPt(TT_PU200, sc, median)")
    assert np.isclose(r["jecs"][8], np.median(corrected[sel] / truth[sel]))
    leading = [max(e) for e in ak.to_list(jets["raw"].pt) if len(e)]
    h = run.get("fpPtHist(TT_PU200, sc, leading)")
    assert np.allclose(
        h["fraction"],
        np.histogram(np.clip(leading, 0, 999.5), bins=h["edges"])[0]
        / len(leading),
    )


def test_rate_working_point_is_at_the_target_rate():
    run, cfg = fp_run()
    wp = run.get("fpRateWp(jet1, nn, 50)")
    mb = load_process(folder(), "MinBias_PU200", cfg)[
        "scPuppiL1TSC4NGJetJets"
    ]["raw"]
    leading = np.array([max(e) if len(e) else 0 for e in ak.to_list(mb.pt)])
    assert (
        abs(np.sum(leading > wp) / len(leading) * cfg["MINBIAS_RATE"] - 50)
        <= 2
    )


def test_mass_fractions_and_collection_draws():
    run, cfg = fp_run()
    top = run.get("fpMass(TT_PU200, top, gen)")
    assert np.isclose(top["fraction"].sum(), 1) and top["mean"] > 0
    load_all()
    with tempfile.TemporaryDirectory() as d:
        evaluate(
            os.path.join(d, "m"),
            ["emulationRegression"],
            cfg,
            sources=ToyFastPuppiSources(cfg),
        )
        one = [
            ModelResults(os.path.join(d, "m"), "m", ["emulationRegression"])
        ]
        assert draw(
            "emulationRegression", one, os.path.join(d, "plots")
        ) == len(specs_for("emulationRegression", one[0]))
        assert os.path.exists(
            os.path.join(
                d,
                "plots",
                "emulation_regression",
                "TT_PU200",
                "turn_ons",
                "turn_on_curve_jet1_50kHz.png",
            )
        )
