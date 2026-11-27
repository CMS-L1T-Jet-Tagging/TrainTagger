"""Plots of the jetPt results, under training/, the pT weights under
training/pt_weights. File names are those of the old basic.py and
pt_weights.py.

Each plot is a function that returns its PlotSpec; the collection at the
end lists the plots.
"""

import matplotlib

from tagger.plot import style
from tagger.plot.registry import PlotSpec, Series, plot_collection

CLS = dict(style.CLASS_LABEL_STYLE, all=style.CLASS_LABEL_STYLE["inclusive"])
ERR = dict(
    fmt="o",
    capsize=style.CAPSIZE,
    ms=style.MARKERSIZE,
    elinewidth=style.ELINEWIDTH,
)
# in bins of gen pT (the old file names) and of L1 pT (folders
# ..._vs_l1pt): variable suffix -> (folder suffix, x label)
BINNED_BY = {
    "GenPt": ("", r"Jet $p_T^{Gen}$ [GeV]"),
    "Pt": ("_vs_l1pt", r"Jet $p_T^{L1}$ [GeV]"),
}


def _groups(config, info):
    """(argument, file name) of the inclusive jets, every class and the
    RESPONSE_GROUPS the model has."""
    groups = list(info.classes) + [
        g
        for g in config["RESPONSE_GROUPS"]
        if all(c in info.classes for c in config["CLASS_GROUPS"].get(g, [g]))
    ]
    return [("all", "inclusive")] + [(g, g) for g in groups]


# -- Response and resolution -------------------------------------------
def response(group, name, binned_by):
    """training/response<suffix>/<name>_response: mean response L1/gen,
    uncorrected and regressed."""
    suffix, xlabel = BINNED_BY[binned_by]
    return PlotSpec(
        name=f"training/response{suffix}/{name}_response",
        plotter="curve_overlay",
        series=[
            Series(
                f"responseVs{binned_by}({group})",
                label=f"Uncorrected - {CLS[group]}",
                y="reco",
                yerr="reco_err",
                style=dict(
                    ERR,
                    capsize=style.CAPSIZE,
                    ms=style.RESPONSE_MARKERSIZE,
                    elinewidth=style.ELINEWIDTH,
                ),
            ),
            Series(
                f"responseVs{binned_by}({group})",
                label=f"Regressed - {CLS[group]}",
                y="regressed",
                yerr="regressed_err",
                style=dict(
                    ERR,
                    capsize=style.CAPSIZE,
                    ms=style.RESPONSE_MARKERSIZE,
                    elinewidth=style.ELINEWIDTH,
                ),
            ),
        ],
        options=dict(
            xlabel=xlabel,
            ylabel="Response (L1/Gen)",
        ),
    )


def resolution(group, name, binned_by):
    """training/residual_rms<suffix>/<name>_rms (inclusive: inclusive):
    RMS of gen - L1 pT."""
    suffix, xlabel = BINNED_BY[binned_by]
    rms_name = "inclusive" if group == "all" else f"{name}_rms"
    return PlotSpec(
        name=f"training/residual_rms{suffix}/{rms_name}",
        plotter="curve_overlay",
        series=[
            Series(
                f"resolutionVs{binned_by}({group})",
                label=r"Uncorrected $\sigma$- " + CLS[group],
                y="reco",
                yerr="reco_err",
                style=ERR,
            ),
            Series(
                f"resolutionVs{binned_by}({group})",
                label=r"Regressed $\sigma$ - " + CLS[group],
                y="regressed",
                yerr="regressed_err",
                style=ERR,
            ),
        ],
        options=dict(
            xlabel=xlabel,
            ylabel=r"$\sigma_{(p_T^{Gen} - p_T^{Reco})}$",
        ),
    )


# -- pT spectra, SHAP, pT correction per class -------------------------
def pt_hist():
    """training/pt_hist: gen, reconstructed and regressed pT."""
    pts = [
        ("ptHist(gen)", "Gen"),
        ("ptHist(reco)", "Reconstructed"),
        ("ptHist(regressed)", "NN Predicted"),
    ]
    return PlotSpec(
        name="training/pt_hist",
        plotter="histogram",
        series=[
            Series(
                e,
                label=label,
                style=dict(
                    color=style.colours[i],
                    linestyle=style.LINESTYLES[2 - i],
                    linewidth=style.THIN_LINEWIDTH,
                ),
            )
            for i, (e, label) in enumerate(pts)
        ],
        options=dict(
            xlabel=r"$p_T$ [GeV]",
            ylabel="a.u",
            xlabel_kw=dict(ha="right", x=1),
            ylabel_kw=dict(ha="right", y=1),
            legend=dict(loc="upper right"),
        ),
    )


def shap_regression():
    """training/shap_summary_reg: mean |SHAP value| per input feature of
    the pT regression."""
    return PlotSpec(
        name="training/shap_summary_reg",
        plotter="stacked_bars",
        series=[
            Series("shapImportance(regression)"),
        ],
        options=dict(
            xlabel="mean (|Shapley value|)",
        ),
    )


def pt_ratio_per_class(info):
    """training/hist_response/<last class>_response: the pT correction
    per class (models with pileup)."""
    cmap = matplotlib.colormaps.get_cmap("Set1")
    return PlotSpec(
        name=f"training/hist_response/{info.classes[-1]}_response",
        plotter="histogram",
        series=[
            Series(
                f"ptRatioHist({c})",
                label=f"{c} Jets",
                y="fraction",
                style=dict(linewidth=style.LINEWIDTH, color=cmap(i)),
            )
            for i, c in enumerate(info.classes)
        ],
        options=dict(
            density=False,
            cms=False,
            xlabel=r"$p_T$ Ratio",
            ylabel="Normalized Entries",
            legend=dict(loc="upper right"),
        ),
    )


# -- Per-constituent pT weights ----------------------------------------
# models with a pt_weights_output layer
def pt_weight_hist(config, binned_by):
    """training/pt_weights/<binned_by>_weights_distribution_pT_binned:
    weight distributions in bins of constituent pT or |eta|."""
    bins = (
        config["PT_WEIGHT_PT_BINS"]
        if binned_by == "pt"
        else config["PT_WEIGHT_ETA_BINS"]
    )
    labels = ["full distribution"] + [
        f"{lo} < {style.INPUT_FEATURE_STYLE[binned_by]} < {hi}"
        for lo, hi in zip(bins[:-1], bins[1:])
    ]
    fields = ["all"] + [f"bin{i}" for i in range(1, len(bins))]
    return PlotSpec(
        name=f"training/pt_weights/{binned_by}_weights_distribution_pT_binned",
        plotter="histogram",
        series=[
            Series(
                f"ptWeightHist({binned_by})",
                label=label,
                y=field,
                style=dict(
                    color=style.color_cycle[i],
                    linewidth=style.HISTOGRAM_LINEWIDTH,
                ),
            )
            for i, (field, label) in enumerate(zip(fields, labels))
        ],
        options=dict(
            logy=True,
            xlim="edges",
            xlabel="$p_T$ weights",
            ylabel="Entries",
            grid=False,
            cms_fontsize=style.MEDIUM_SIZE,
            legend=dict(loc="upper right"),
        ),
    )


def pt_weight_vs_input(particle_feature):
    """training/pt_weights/pt_weights_vs_<particle_feature>: the weight
    against a particle input feature."""
    return PlotSpec(
        name=f"training/pt_weights/pt_weights_vs_{particle_feature}",
        plotter="hist_2d",
        series=[
            Series(f"ptWeightVsInput({particle_feature})"),
        ],
        options=dict(
            xlabel=style.INPUT_FEATURE_STYLE[particle_feature],
            ylabel=r"$p_T$ weights",
            zlabel="Entries (normalized)",
            cms_fontsize=style.MEDIUM_SIZE,
        ),
    )


# -- Collection --------------------------------------------------------
@plot_collection(reads=["jetPt"])
def jetPt(config, info):
    """pT response and resolution vs gen pT and vs L1 pT (inclusive, per
    class, per group), pT spectra, SHAP, the pT correction per class for
    models with a pileup class, and the per-constituent pT weights of
    models that have them."""
    plots = []
    for binned_by in BINNED_BY:
        for group, name in _groups(config, info):
            plots += [
                response(group, name, binned_by),
                resolution(group, name, binned_by),
            ]
    plots += [
        pt_hist(),
        shap_regression(),
    ]
    if "pileup" in info.classes:
        plots.append(pt_ratio_per_class(info))
    if info.pt_weights:
        plots += [
            pt_weight_hist(config, "pt"),
            pt_weight_hist(config, "eta"),
        ]
        plots += [
            pt_weight_vs_input(particle_feature)
            for particle_feature in info.particle_features
        ]
    return plots
