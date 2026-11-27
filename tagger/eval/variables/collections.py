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
def ci(config, info):
    """Everything CI plots: the union of all collections that can run in
    the CI pipeline."""
    return ["train", "jetClassifier", "jetPt"]
