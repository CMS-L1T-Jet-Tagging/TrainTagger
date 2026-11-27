"""Plots of the jetClassifier results, under training/. File names are
those of the old basic.py.

Each plot is a function that returns its PlotSpec; the collection at the
end lists the plots.
"""

from tagger.eval.variables.base.scores import roc_pairs
from tagger.plot import style
from tagger.plot.registry import PlotSpec, Series, plot_collection

CLS = style.CLASS_LABEL_STYLE
ROC_AXES = dict(
    xlabel="Signal Efficiency",
    ylabel="Mistag Rate",
    logy=True,
    ylim=(1e-3, 1.1),
)
ROC_XY = dict(x="tpr", y="fpr")
WP_NAMES = {0: "Loose", 1: "Medium", 2: "Tight"}
WP_COLOURS = ["blue", "orange", "green"]
HIST_OPTS = dict(
    ylabel="a.u",
    logy=True,
    xlabel_kw=dict(ha="right", x=1),
    ylabel_kw=dict(ha="right", y=1),
    legend=dict(loc="upper right"),
)


def _wrap(expr, sample):
    """Expression on the test split (sample None) or via a sample
    metavariable."""
    if sample is None:
        return expr
    kind, process = sample
    return f"{kind}({process}, {expr})"


def _has(config, info, *groups):
    """Whether the model has every class of the classes or class groups."""
    return all(
        c in info.classes
        for g in groups
        for c in config["CLASS_GROUPS"].get(g, [g])
    )


def _hist_series(exprs_labels):
    n = len(exprs_labels)
    colours = style.colours if n <= len(style.colours) else style.color_cycle
    linestyles = (
        [style.LINESTYLES[n - 1 - i] for i in range(n)]
        if n <= len(style.colours)
        else ["-"] * n
    )
    return [
        Series(
            e,
            label=label,
            style=dict(
                color=colours[i],
                linestyle=linestyles[i],
                linewidth=style.THIN_LINEWIDTH,
            ),
        )
        for i, (e, label) in enumerate(exprs_labels)
    ]


# -- ROC curves --------------------------------------------------------
def roc_one_vs_rest(info):
    """training/basic_ROC: one-vs-rest ROC of every class."""
    return PlotSpec(
        name="training/basic_ROC",
        plotter="curve_overlay",
        series=[
            Series(
                f"rocOneVsRest({c})",
                label=f"{CLS[c]} (AUC = [[aucOneVsRest({c}):.2f]])",
                style=dict(
                    color=style.SHAP_COLORS[i], linewidth=style.LINEWIDTH
                ),
                **ROC_XY,
            )
            for i, c in enumerate(info.classes)
        ],
        options=dict(
            ROC_AXES,
            legend=dict(loc="upper left", ncol=2, fontsize=style.SMALLER_SIZE),
            legend_sort=[f"aucOneVsRest({c})" for c in info.classes],
        ),
    )


def roc_binary(a, b, folder, sample=None, legend_title=None):
    """<folder>/roc_binary/ROC_<a>_vs_<b>: ROC of class a against class b."""
    return PlotSpec(
        name=f"{folder}/roc_binary/ROC_{a}_vs_{b}",
        plotter="curve_overlay",
        series=[
            Series(
                _wrap(f"roc({a}, {b})", sample),
                label=(
                    f"{CLS[a]} vs {CLS[b]} (AUC ="
                    f" [[{_wrap(f'auc({a}, {b})', sample)}:.2f]])"
                ),
                style=dict(color="blue", linewidth=style.LINEWIDTH),
                **ROC_XY,
            ),
        ],
        options=dict(
            ROC_AXES,
            legend=dict(
                loc="lower right",
                fontsize=style.SMALL_SIZE + 3,
                title=legend_title,
            ),
        ),
    )


def roc_combined(
    sig,
    bkgs,
    labels,
    name,
    sample=None,
    legend_loc="lower right",
    legend_title=None,
):
    """One plot with the ROCs of sig against each of bkgs."""
    return PlotSpec(
        name=name,
        plotter="curve_overlay",
        series=[
            Series(
                _wrap(f"roc({sig}, {b})", sample),
                label=(
                    f"{label} (AUC ="
                    f" [[{_wrap(f'auc({sig}, {b})', sample)}:.2f]])"
                ),
                style=dict(linewidth=style.LINEWIDTH),
                **ROC_XY,
            )
            for b, label in zip(bkgs, labels)
        ],
        options=dict(
            ROC_AXES,
            legend=dict(
                loc=legend_loc,
                fontsize=style.SMALL_SIZE + 3,
                title=legend_title,
            ),
        ),
    )


def roc_light_vs_jets(info, folder, sample=None, legend_title=None):
    """<folder>/roc_jets/ROC_light_vs_all_jets: light against b, charm
    and gluon."""
    jets = [b for b in ("b", "charm", "gluon") if b in info.classes]
    return roc_combined(
        "light",
        jets,
        [f"light vs {b}" for b in jets],
        f"{folder}/roc_jets/ROC_light_vs_all_jets",
        sample,
        legend_loc="lower right",
        legend_title=legend_title,
    )


def roc_taus(folder, sample=None, legend_title=None):
    """<folder>/roc_taus/ROC_taus_combined: taus against jets, muons and
    electrons."""
    return roc_combined(
        "taus",
        ["jets", "muon", "electron"],
        [
            r"$\tau_h^{\pm}$ vs Jets (b, c, light, gluon)",
            r"$\tau_h^{\pm}$ vs Muons",
            r"$\tau_h^{\pm}$ vs Electrons",
        ],
        f"{folder}/roc_taus/ROC_taus_combined",
        sample,
        legend_loc="upper left",
        legend_title=legend_title,
    )


def roc_set(config, info, folder, sample=None, legend_title=None):
    """Binary ROCs plus the light-vs-jets and taus combined ROCs, into
    folder."""
    plots = [
        roc_binary(a, b, folder, sample, legend_title)
        for a, b in roc_pairs(config, info)
    ]
    if "light" in info.classes and any(
        b in info.classes for b in ("b", "charm", "gluon")
    ):
        plots.append(roc_light_vs_jets(info, folder, sample, legend_title))
    if all(
        c in info.classes
        for c in (
            "taup",
            "taum",
            "b",
            "charm",
            "light",
            "gluon",
            "muon",
            "electron",
        )
    ):
        plots.append(roc_taus(folder, sample, legend_title))
    return plots


# -- Confusion matrix and SHAP -----------------------------------------
def confusion_matrix():
    """training/confusion_matrix"""
    return PlotSpec(
        name="training/confusion_matrix",
        plotter="grid_2d",
        series=[
            Series("confusionMatrix"),
        ],
    )


def shap_classification():
    """training/shap_summary_class: mean |SHAP value| per input feature
    and class."""
    return PlotSpec(
        name="training/shap_summary_class",
        plotter="stacked_bars",
        series=[
            Series("shapImportance(classification)"),
        ],
        options=dict(
            xlabel="mean (|Shapley value|)",
        ),
    )


# -- Efficiency at fixed mistag rates ----------------------------------
EFF_VS = {  # variable -> (file name suffix, x label)
    "effVsPt": ("", r"Jet $p_T^{L1}$ [GeV]"),
    "effVsGenPt": ("_genpt", r"Jet $p_T^{Gen}$ [GeV]"),
    "effVsEta": ("_eta", r"Jet $\eta$"),
    "effVsAbsEta": ("_abseta", r"Jet $|\eta|$"),
}


def eff_vs(config, group, var):
    """training/efficiencies/<group>_efficiency<suffix>: efficiency of a
    class group versus L1 pT, gen pT, eta or |eta| (``var`` of EFF_VS),
    one line per mistag rate of EFF_VS_PT_MISTAGS.
    """
    suffix, xlabel = EFF_VS[var]
    return PlotSpec(
        name=f"training/efficiencies/{group}_efficiency{suffix}",
        plotter="curve_overlay",
        series=[
            Series(
                f"{var}({group}, {m})",
                label=f"{WP_NAMES.get(i, 'WP')} (mistag = {m:.0%})",
                y="eff",
                yerr="err",
                style=dict(color=WP_COLOURS[i % 3], linestyle="-"),
            )
            for i, m in enumerate(config["EFF_VS_PT_MISTAGS"])
        ],
        options=dict(
            xlabel=xlabel,
            ylabel="Tagging efficiency",
            ylim=(0.0, 1.3),
            grid=False,
        ),
    )


def eff_at_mistag(config, sig, bkg):
    """training/eff_at_mistag/<sig>_vs_<bkg>: signal efficiency at the
    MISTAG_POINTS, with the axes of the ROC curves."""
    return PlotSpec(
        name=f"training/eff_at_mistag/{sig}_vs_{bkg}",
        plotter="scalar_vs_x",
        series=[
            Series(
                f"effAtMistag({sig}, {bkg}, {{x}})",
                label=f"{CLS.get(sig, sig)} vs {CLS.get(bkg, bkg)}",
            ),
        ],
        options=dict(
            x_values=config["MISTAG_POINTS"],
            swap_axes=True,
            **ROC_AXES,
            xlim=(0, 1),
            legend=dict(loc="lower right"),
        ),
    )


# -- Particle input features -------------------------------------------
def input_hist_split(info, particle_feature):
    """training/inputs/<particle_feature>_split: a particle input
    feature per class."""
    fs = style.INPUT_FEATURE_STYLE[particle_feature]
    return PlotSpec(
        name=f"training/inputs/{particle_feature}_split",
        plotter="histogram",
        series=_hist_series(
            [
                (f"inputHist({particle_feature}, {c})", f"{fs} {CLS[c]}")
                for c in info.classes
            ]
        ),
        options=dict(HIST_OPTS, xlabel=fs),
    )


def input_hist(particle_feature):
    """training/inputs/<particle_feature>: a particle input feature of
    all jets."""
    fs = style.INPUT_FEATURE_STYLE[particle_feature]
    return PlotSpec(
        name=f"training/inputs/{particle_feature}",
        plotter="histogram",
        series=_hist_series([(f"inputHist({particle_feature}, all)", fs)]),
        options=dict(HIST_OPTS, xlabel=fs),
    )


def multiplicity_split(info):
    """training/inputs/multiplicies_split: number of constituents per class."""
    return PlotSpec(
        name="training/inputs/multiplicies_split",
        plotter="histogram",
        series=_hist_series(
            [
                (f"multiplicity({c})", f"Multiplicities:  {CLS[c]}")
                for c in info.classes
            ]
        ),
        options=dict(HIST_OPTS, xlabel="multiplicities"),
    )


def multiplicity():
    """training/inputs/multiplicity: number of constituents of all jets."""
    return PlotSpec(
        name="training/inputs/multiplicity",
        plotter="histogram",
        series=_hist_series([("multiplicity(all)", "Multiplicity")]),
        options=dict(HIST_OPTS, xlabel="Multiplicity"),
    )


# -- Collection --------------------------------------------------------
@plot_collection(reads=["jetClassifier"])
def jetClassifier(config, info):
    """ROC curves (one-vs-rest, binary, combined; per signal process),
    confusion matrix, SHAP, efficiency vs pT, input feature and
    multiplicity distributions, efficiency at fixed mistag.
    """
    plots = [roc_one_vs_rest(info)]
    plots += roc_set(config, info, "training")
    for p in config["SIGNAL_PROCESSES"]:
        title = style.PROCESS_STYLE.get(p, p)
        plots += roc_set(
            config,
            info,
            f"physics/binary_rocs_{p}/test_set",
            ("onTestSubset", p),
            title,
        )
        plots += roc_set(
            config,
            info,
            f"physics/binary_rocs_{p}/full_sample",
            ("onProcess", p),
            title,
        )

    plots += [
        confusion_matrix(),
        shap_classification(),
    ]

    for g in config["EFF_VS_PT_GROUPS"]:
        if not _has(config, info, g):
            continue
        plots += [
            eff_vs(config, g, "effVsPt"),
            eff_vs(config, g, "effVsGenPt"),
        ]
        if info.has_eta:
            plots += [
                eff_vs(config, g, "effVsEta"),
                eff_vs(config, g, "effVsAbsEta"),
            ]

    for particle_feature in info.particle_features:
        plots += [
            input_hist_split(info, particle_feature),
            input_hist(particle_feature),
        ]
    plots += [
        multiplicity_split(info),
        multiplicity(),
    ]

    for sig, bkg in config["ROC_EXTRA_PAIRS"]:
        if _has(config, info, sig, bkg):
            plots.append(eff_at_mistag(config, sig, bkg))
    return plots
