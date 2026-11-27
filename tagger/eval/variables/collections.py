"""Collections: named lists of variable expressions, built from the
config and the model's classes.

A member is a variable expression or the name of another collection
(nesting). Users add their own through config COLLECTIONS, e.g.
``COLLECTIONS = {"mine": ["jetPt", "auc(b, qcd)"]}``, or with
``vm.add_collection``. The VariableManager
(tagger.eval.variable_manager.vm) holds and expands them; it loads this
module together with the variable modules.
"""

from tagger.eval.variable_manager import vm
from tagger.eval.variables.base.scores import has_group, roc_pairs

# -- helpers -----------------------------------------------------------


def _rocs(config, info):
    """roc + auc for all pairs, the combined plots and the extra pairs;
    effAtMistag for pairs and extras."""
    out = []
    pairs = roc_pairs(config, info)
    extra = [
        tuple(p)
        for p in config["ROC_EXTRA_PAIRS"]
        if has_group(p[0], config, info) and has_group(p[1], config, info)
    ]
    combined = [
        (sig, bkg)
        for sig, bkgs in config["ROC_COMBINED"].values()
        if has_group(sig, config, info)
        for bkg in bkgs
        if has_group(bkg, config, info)
    ]
    for sig, bkg in pairs + combined + extra:
        out += [f"roc({sig}, {bkg})", f"auc({sig}, {bkg})"]
    for sig, bkg in pairs + extra:
        out += [
            f"effAtMistag({sig}, {bkg}, {m})" for m in config["MISTAG_POINTS"]
        ]
    return out


# -- built-in collections ----------------------------------------------


@vm.collection()
def jetClassifier(config, info):
    """Jet classification on the test split: one-vs-rest and binary
    ROCs, AUCs, efficiency at fixed mistag (ROC_PAIRS x MISTAG_POINTS
    and ROC_EXTRA_PAIRS), confusion matrix, SHAP importance, efficiency
    vs L1 and gen pT (and L1 eta), input feature distributions, and the
    binary ROCs per SIGNAL_PROCESSES sample.
    """
    out = []
    for c in info.classes:
        out += [f"rocOneVsRest({c})", f"aucOneVsRest({c})"]
    out += _rocs(config, info)
    out.append("confusionMatrix")
    out.append("shapImportance(classification)")
    for g in config["EFF_VS_PT_GROUPS"]:
        if has_group(g, config, info):
            out += [
                f"{v}({g}, {m})"
                for v in ("effVsPt", "effVsGenPt")
                for m in config["EFF_VS_PT_MISTAGS"]
            ]
            # models trained before the jet eta was saved have no eta
            # variables
            if info.has_eta:
                out += [
                    f"{v}({g}, {m})"
                    for v in ("effVsEta", "effVsAbsEta")
                    for m in config["EFF_VS_PT_MISTAGS"]
                ]
    for f in info.particle_features:
        out += [f"inputHist({f}, {c})" for c in info.classes] + [
            f"inputHist({f}, all)"
        ]
    out += [f"multiplicity({c})" for c in info.classes] + ["multiplicity(all)"]
    for p in config["SIGNAL_PROCESSES"]:
        rocs = [
            m for m in _rocs(config, info) if m.startswith(("roc(", "auc("))
        ]
        out += [f"onTestSubset({p}, {r})" for r in rocs] + [
            f"onProcess({p}, {r})" for r in rocs
        ]
    return out


@vm.collection()
def jetPt(config, info):
    """Jet pT regression on the test split: response and resolution vs
    gen and vs L1 pT (inclusive, per class and per RESPONSE_GROUPS), pT
    spectra, SHAP importance of the regression, the pT correction per
    class for models with pileup, and the per-constituent pT weights of
    models that have them.
    """
    groups = (
        ["all"]
        + list(info.classes)
        + [g for g in config["RESPONSE_GROUPS"] if has_group(g, config, info)]
    )
    out = [
        f"{v}({g})"
        for v in (
            "responseVsGenPt",
            "responseVsPt",
            "resolutionVsGenPt",
            "resolutionVsPt",
        )
        for g in groups
    ]
    out += [
        "ptHist(gen)",
        "ptHist(reco)",
        "ptHist(regressed)",
        "shapImportance(regression)",
    ]
    if "pileup" in info.classes:
        out += [f"ptRatioHist({c})" for c in info.classes]
    if info.pt_weights:
        out += ["ptWeightHist(pt)", "ptWeightHist(eta)"] + [
            f"ptWeightVsInput({f})" for f in info.particle_features
        ]
    return out


@vm.collection()
def emulation(config, info):
    """The model on the jets of the emulation ntuple: score histograms
    and one-vs-rest ROCs per class, regression output, pT response, and
    the outputs per jet. Run it for the network, -m hls4ml:<model> and
    -m emulator:<model>; tagger.plot compares the three. Needs
    EMULATION_SAMPLE.
    """
    if info.pseudo == "cmssw":
        return []
    out = []
    for c in info.classes:
        if c != "pileup":  # the CMSSW emulator has no pileup score
            out += [
                f"onEmulation(scoreHist({c}))",
                f"onEmulation(rocOneVsRest({c}))",
                f"onEmulation(aucOneVsRest({c}))",
            ]
    return out + [
        "onEmulation(regressionHist)",
        "onEmulation(responseHist(all))",
        "onEmulation(responseMedian(all))",
        "onEmulation(responseRms(all))",
        "onEmulation(jetOutputs)",
    ]


@vm.collection()
def emulationRegression(config, info):
    """The model's CMSSW emulation in full events (FastPUPPI ntuples):
    response and resolution, pT spectra and gen vs L1 pT maps per
    FASTPUPPI_PROCESSES, top, W and Higgs mass peaks, and turn-on curves
    at the TURN_ON_RATES working points, of the SC4NN jets (nn) against
    the standard jets (sc, scJecs). Needs FASTPUPPI_DIR."""
    if info.pseudo:
        return []
    out = []
    for p in config["FASTPUPPI_PROCESSES"]:
        for coll in ("nn", "sc"):
            out += [
                f"{v}({p}, {coll}, {r})"
                for v in ("fpResponseVsGenPt", "fpResponseVsPt")
                for r in ("mean", "median")
            ]
            out += [
                f"{v}({p}, {coll})"
                for v in ("fpResolutionVsGenPt", "fpResolutionVsPt")
            ]
        out += [
            f"fpPtHist({p}, {j}, {w})"
            for w in ("leading", "all")
            for j in ("nn", "sc", "scJecs", "gen")
        ]
        out += [f"fpGenRecoPtMap({p}, nn)", f"fpGenRecoPtMap({p}, scJecs)"]
        out += [
            f"fpTurnOnVsGen({p}, {o}, {j}, {r})"
            for o in config["TURN_ON_OBSERVABLES"]
            for r in config["TURN_ON_RATES"]
            for j in ("nn", "sc", "scJecs")
        ]
    out += [
        f"fpRateWp({o}, {j}, {r})"
        for o in config["TURN_ON_OBSERVABLES"]
        for r in config["TURN_ON_RATES"]
        for j in ("nn", "sc", "scJecs")
    ]
    out += [
        f"fpMass(TT_PU200, {particle}, {j})"
        for particle in ("top", "w")
        for j in ("gen", "nn", "sc", "scJecs")
    ]
    out += [
        f"fpMass({p}, h, {j})"
        for p in ("VBFHToBB_PU200", "VBFHToCC_PU200")
        for j in ("gen", "nn", "sc", "scJecs")
    ]
    return out


@vm.collection()
def train(config, info):
    """Training diagnostics: loss per epoch (train and validation) for
    each output, and the SHAP feature importance of the classification
    and regression outputs. The loss is empty for models trained before
    training_history.json existed."""
    if info.pseudo:  # the CMSSW pseudo-model has no network
        return []
    return [f"lossHistory({loss})" for loss in info.losses] + [
        "shapImportance(classification)",
        "shapImportance(regression)",
    ]


@vm.collection()
def seedSingleTau(config, info):
    """Single tau seed: MinBias rate grid and working point line,
    efficiency versus threshold, and efficiency versus gen pT at
    TAU_SEEDS singleTau wp_pt in barrel and tau endcap, each at the
    working point (wp) and with the pT cut alone (pt). Run it for the
    network and for -m cmssw. Needs MINBIAS_SAMPLE and
    SIGNAL_SAMPLES[TAU_SIGNAL]."""
    out = ["tauRateGrid(singleTau)", "tauWpLine(singleTau)"]
    out += [f"singleTauEffVsThreshold({c})" for c in ("pt", "wp")]
    for r in ("barrel", "tau_endcap"):
        for c in ("pt", "wp"):
            out += [
                f"tauEffVsGenPt(singleTau, {c}, {r})",
                f"tauTotalEff(singleTau, {c}, {r})",
            ]
    return out


@vm.collection()
def seedDiTau(config, info):
    """Di-tau seed: MinBias rate grid, working point, rate versus pT
    threshold, and efficiency versus gen pT in barrel and tau endcap.
    Needs MINBIAS_SAMPLE and SIGNAL_SAMPLES[TAU_SIGNAL].
    """
    out = ["tauRateGrid(diTau)", "tauWpLine(diTau)", "tauWp(diTau)"]
    out += [f"tauRateVsPt(diTau, {c})" for c in ("raw", "wp")]
    for r in ("barrel", "tau_endcap"):
        for c in ("pt", "wp"):
            out += [
                f"tauEffVsGenPt(diTau, {c}, {r})",
                f"tauTotalEff(diTau, {c}, {r})",
            ]
    return out


@vm.collection()
def seedDiTauTopo(config, info):
    """Di-tau topology seed: MinBias rate grid, working point, rate
    versus pT threshold, and efficiency in (gen pT1, gen pT2). Needs
    MINBIAS_SAMPLE and SIGNAL_SAMPLES[TAU_SIGNAL].
    """
    out = [
        "tauRateGrid(diTauTopo)",
        "tauWpLine(diTauTopo)",
        "tauWp(diTauTopo)",
    ]
    out += [f"tauRateVsPt(diTauTopo, {c})" for c in ("raw", "wp")]
    return out + [
        "diTauTopoEffVsGenPt2d",
        "diTauTopoTotalEff",
        "diTauTopoTotalEffErr",
    ]


@vm.collection()
def seedBbbb(config, info):
    """HH -> 4b seed per BBBB_VARIANTS (score, sel): MinBias rate grid
    and working point line, HT-only threshold, working point,
    efficiencies versus gen HT and gen m_HH, integrated efficiencies and
    the rate of the working point OR HT-only. For -m cmssw only the raw
    score variants. Needs MINBIAS_SAMPLE and SIGNAL_SAMPLES hh4b."""
    out = ["htOnlyCut(bbbb)", "hhGenHist(hh4b, ht)", "hhGenHist(hh4b, mhh)"]
    for score, sel in config["BBBB_VARIANTS"]:
        if info.pseudo == "cmssw" and score != "raw":
            continue
        v = f"{score}, {sel}"
        out += [
            f"bbbbRateGrid({v})",
            f"bbbbWpLine({v})",
            f"bbbbWpHt({v})",
            f"bbbbWpNn({v})",
            f"bbbbOrRate({v})",
        ]
        for t in ("wp", "wpPure", "htOnly"):
            out += [f"bbbbEffVsGenHt({t}, {v})", f"bbbbEffVsGenMhh({t}, {v})"]
        for t in ("wp", "wpPure", "htOnly", "wpOrHtOnly"):
            out += [f"bbbbEff({t}, {v})", f"bbbbEffFourJets({t}, {v})"]
    return out


@vm.collection()
def seedBbtt(config, info):
    """HH -> bb tau tau seed per BBTT_VARIANTS (score, sel): the
    Run-3-like seed rate, (bb, tt) rate grids, working points at the
    seed rate and at BBTT_TARGET_RATE, HT-only threshold, and
    efficiencies versus gen HT and gen m_HH. For -m cmssw (no tau
    charge, no bbtt working point) the CMSSW reference seeds: the
    Run-3-like seed rate and the run3 and ditau efficiencies. Needs
    MINBIAS_SAMPLE and SIGNAL_SAMPLES hhbbtt."""
    if info.pseudo == "cmssw":
        return ["bbttRun3SeedRate"] + [
            f"bbttRef{v}({t})"
            for t in ("run3", "ditau")
            for v in ("EffVsGenHt", "Eff")
        ]
    out = [
        "bbttRun3SeedRate",
        "htOnlyCut(bbtt)",
        "hhGenHist(hhbbtt, ht)",
        "hhGenHist(hhbbtt, mhh)",
    ]
    for score, sel in config["BBTT_VARIANTS"]:
        v = f"{score}, {sel}"
        out.append(f"bbttRateGrid({sel})")
        for which in ("seed", "target"):
            out += [
                f"bbttWpLine({v}, {which})",
                f"bbttWpBb({v}, {which})",
                f"bbttWpTt({v}, {which})",
                f"bbttWpRate({v}, {which})",
            ]
        for t in ("nn", "nnTarget", "nnTargetPure", "htOnly"):
            out += [f"bbttEffVsGenHt({t}, {v})", f"bbttEff({t}, {v})"]
        out += [
            f"bbttEffVsGenMhh({t}, {v})"
            for t in ("htOnly", "nnTarget", "nnTargetPure")
        ]
    return out


@vm.collection()
def seeds(config, info):
    """All trigger seed collections. They share one MinBias load and one
    model evaluation per sample."""
    return [
        "seedSingleTau",
        "seedDiTau",
        "seedDiTauTopo",
        "seedBbbb",
        "seedBbtt",
    ]


@vm.collection()
def ci(config, info):
    """Everything CI plots: the union of all collections that can run in
    the CI pipeline."""
    return ["train", "jetClassifier", "jetPt"]
