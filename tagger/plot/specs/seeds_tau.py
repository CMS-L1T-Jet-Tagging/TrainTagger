"""Plots of the tau seed collections, under physics/tau, physics/tautau
and physics/tautau_topo. File names are those of the old singleTau.py,
diTaus.py and diTaus_topo.py. The CMSSW curves are cmssw/
series, from the results of python -m tagger.eval -m cmssw.

Each plot is a function that returns its PlotSpec; the collections at
the end list the plots.
"""

from tagger.plot import style
from tagger.plot.registry import PlotSpec, Series, plot_collection

C = style.color_cycle
EFF = r"$\int \epsilon$"
REGIONS = ("barrel", "tau_endcap")
PT1, PT2 = r"Gen. $\tau_h$ $p_T^1$ [GeV]", r"Gen. $\tau_h$ $p_T^2$ [GeV]"


def _eta_label(region, eta_cut):
    if region == "barrel":
        return r"Barrel ($|\eta| < 1.5$)"
    return rf"EndCap (1.5 < $|\eta|$ < {eta_cut})"


# -- Shared by the tau seeds -------------------------------------------
def wp_grid(name, seed, prefix, config, xlabel, ylabel, zlabel, xlim, ylim):
    """MinBias rate grid of a seed with its working point line.
    ``prefix`` "" or "cmssw/"."""
    cfg = config["TAU_SEEDS"][seed]
    return PlotSpec(
        name=name,
        plotter="rate_grid",
        series=[
            Series(
                f"{prefix}tauRateGrid({seed})",
                x="nn",
                y="pt",
            ),
            Series(
                f"{prefix}tauWpLine({seed})",
                label=r"${} \pm {}$ kHz".format(
                    cfg["target_rate"], cfg["rate_range"]
                ),
                x="nn",
                y="pt",
            ),
        ],
        options=dict(
            xlabel=xlabel,
            ylabel=ylabel,
            zlabel=zlabel,
            xlim=xlim,
            ylim=ylim,
            legend=dict(loc="upper right", fontsize=style.MEDIUM_SIZE),
            cms_fontsize=style.CMS_LABEL_SMALL_SIZE,
        ),
    )


def total_eff_text(name, title, seed, region):
    """Integrated efficiencies of a seed in a region, as text."""
    return PlotSpec(
        name=name,
        plotter="text_summary",
        series=[],
        options=dict(
            lines=[
                title + " ",
                (
                    "SeededCone + pt regression, no ID cut (Multiclass limit)"
                    f" [[tauTotalEff({seed}, pt, {region}):.4f]] "
                ),
                f"Multiclass NN [[tauTotalEff({seed}, wp, {region}):.4f]] ",
                f"CMSSW  [[cmssw/tauTotalEff({seed}, wp, {region}):.4f]] ",
            ]
        ),
    )


# -- Single tau --------------------------------------------------------
def single_tau_wp_grid(config, prefix=""):
    """physics/tau/tau_WPs (the network) or cmssw_tau_WPs (prefix "cmssw/")."""
    return wp_grid(
        f"physics/tau/{prefix.replace('/', '_')}tau_WPs",
        "singleTau",
        prefix,
        config,
        xlabel=r"$\tau$ Score Threshold",
        ylabel=r"Min L1 $p_T$ [GeV]",
        zlabel=r"Single-tau rate [kHZ]",
        xlim=(0, 1.0),
        ylim=(50, 200),
    )


def single_tau_eff_vs_threshold(config):
    """physics/tau/tau_eff_pt_comparison: efficiency versus the pT
    threshold."""
    rate = config["TAU_SEEDS"]["singleTau"]["target_rate"]
    return PlotSpec(
        name="physics/tau/tau_eff_pt_comparison",
        plotter="band_overlay",
        series=[
            Series(
                "singleTauEffVsThreshold(pt)",
                label=r"SeededCone + $p_T$ regression, no ID cut",
                x="pt",
                y="eff",
                style=dict(color=C[2]),
            ),
            Series(
                "singleTauEffVsThreshold(wp)",
                label=f"SeedCone Tau Tagger Eff., {rate} kHz Rate",
                x="pt",
                y="eff",
                style=dict(color=C[0]),
            ),
            Series(
                "cmssw/singleTauEffVsThreshold(wp)",
                label=f"CMSSW PuppiTau Emulator Eff., {rate} kHz Rate",
                x="pt",
                y="eff",
                style=dict(color=C[1]),
            ),
        ],
        options=dict(
            header=rf"$|\eta| < {config['TAU_ETA_CUT']}$",
            xlabel=r"Single $\tau_h$ $p_T$ Threshold [GeV]",
            ylabel="Trigger Efficiency",
            legend=dict(loc="upper left", fontsize=style.LEGEND_WIDTH),
            cms_fontsize=style.MEDIUM_SIZE,
        ),
    )


def _single_tau_label(text, prefix, cut, region):
    """Legend label of single_tau_eff_vs_gen_pt: ``text`` and the total
    efficiency of the curve."""
    return (
        f"{text}, {EFF}=[[{prefix}tauTotalEff(singleTau, {cut},"
        f" {region}):.2f]]"
    )


def single_tau_eff_vs_gen_pt(config, region):
    """physics/tau/sc_and_tau_eff_<region>: efficiency versus gen pT at
    the wp_pt threshold."""
    rate = config["TAU_SEEDS"]["singleTau"]["target_rate"]
    wp_pt = config["TAU_SEEDS"]["singleTau"]["wp_pt"]
    return PlotSpec(
        name=f"physics/tau/sc_and_tau_eff_{region}",
        plotter="eff_points",
        series=[
            Series(
                f"tauEffVsGenPt(singleTau, pt, {region})",
                label=_single_tau_label(
                    r"SeededCone + $p_T$ regression, no ID cut",
                    "",
                    "pt",
                    region,
                ),
                style=dict(c=C[2]),
            ),
            Series(
                f"cmssw/tauEffVsGenPt(singleTau, wp, {region})",
                label=_single_tau_label(
                    f"Tau CMSSW Emulator @ {rate}kHz", "cmssw/", "wp", region
                ),
                style=dict(c=C[0]),
            ),
            Series(
                f"tauEffVsGenPt(singleTau, wp, {region})",
                label=_single_tau_label(
                    f"SeededCone Tau Tagger @ {rate}kHz", "", "wp", region
                ),
                style=dict(c=C[1]),
            ),
        ],
        options=dict(
            header="$p_T$ > %i GeV Trigger WP " % wp_pt
            + _eta_label(region, config["TAU_ETA_CUT"]),
            zero_nan_err=True,
            hline=1,
            xlim=(0, 150),
            ylim=(0.0, 1.1),
            xlabel=r"$\tau_h$ $p_T^{gen}$ [GeV]",
            ylabel="Efficiency",
            legend=dict(loc="lower right", fontsize=style.LEGEND_WIDTH),
            cms_fontsize=style.MEDIUM_SIZE,
        ),
    )


def single_tau_total_eff(config, region):
    """physics/tau/TotalEff_<region>.txt"""
    return total_eff_text(
        f"physics/tau/TotalEff_{region}", "Total Tau Eff", "singleTau", region
    )


# -- Di-tau ------------------------------------------------------------
def di_tau_wp_grid(config):
    """physics/tautau/tautau_WPs"""
    return wp_grid(
        "physics/tautau/tautau_WPs",
        "diTau",
        "",
        config,
        xlabel=r"Min Tau NN ($\tau^{+} + \tau^{-}$) Score",
        ylabel=r"Min L1 $p_T$ [GeV]",
        zlabel=r"Di-tau rate [kHZ]",
        xlim=(0, 0.4),
        ylim=(10, 100),
    )


def di_tau_rate_vs_pt(config):
    """physics/tautau/tautau_BkgRate: MinBias rate versus the pT threshold."""
    return PlotSpec(
        name="physics/tautau/tautau_BkgRate",
        plotter="band_overlay",
        series=[
            Series(
                "tauRateVsPt(diTau, raw)",
                label=r"No ID/$p_T$ correction",
                x="pt",
                y="rate",
                style=dict(color=C[2]),
            ),
            Series(
                "cmssw/tauRateVsPt(diTau, wp)",
                label="CMSSW PuppiTau Emulator",
                x="pt",
                y="rate",
                style=dict(color=C[0]),
            ),
            Series(
                "tauRateVsPt(diTau, wp)",
                label="SeedCone Tau",
                x="pt",
                y="rate",
                style=dict(color=C[1]),
            ),
        ],
        options=dict(
            header=r"$|\eta| < 2.5$",
            logy=True,
            ylabel=r"$\tau_h$ trigger rate [kHz]",
            xlabel=r"L1 $p_T$ [GeV]",
            legend=dict(loc="upper right", fontsize=style.MEDIUM_SIZE),
            cms_fontsize=style.MEDIUM_SIZE,
        ),
    )


def di_tau_eff_vs_gen_pt(config, region):
    """physics/tautau/sc_and_tau_eff_<region>: efficiency versus gen pT."""
    rate = config["TAU_SEEDS"]["diTau"]["target_rate"]
    return PlotSpec(
        name=f"physics/tautau/sc_and_tau_eff_{region}",
        plotter="eff_points",
        series=[
            Series(
                f"cmssw/tauEffVsGenPt(diTau, wp, {region})",
                label=f"Tau CMSSW Emulator @ {rate}kHz",
                style=dict(c=C[0]),
            ),
            Series(
                f"tauEffVsGenPt(diTau, wp, {region})",
                label=f"SeededCone Tau Tagger @ {rate}kHz",
                style=dict(c=C[1]),
            ),
        ],
        options=dict(
            header=_eta_label(region, config["TAU_ETA_CUT"]),
            hline=1,
            xlim=(0, 150),
            ylim=(0.0, 1.1),
            xlabel=r"$\tau_h$ $p_T^{gen}$ [GeV]",
            ylabel=r"$\epsilon$ (VBF H $\rightarrow$ $\tau\tau$)",
            legend=dict(loc="lower right", fontsize=style.LEGEND_WIDTH),
            cms_fontsize=style.MEDIUM_SIZE,
        ),
    )


def di_tau_total_eff(config, region):
    """physics/tautau/TotalEff_<region>.txt"""
    return total_eff_text(
        f"physics/tautau/TotalEff_{region}", "Total diTau Eff", "diTau", region
    )


# -- Di-tau topology ---------------------------------------------------
def topo_wp_grid(config):
    """physics/tautau_topo/tautau_topo_WPs"""
    return wp_grid(
        "physics/tautau_topo/tautau_topo_WPs",
        "diTauTopo",
        "",
        config,
        xlabel="Tau Topology Score",
        ylabel=r"Min reco $p_T$ [GeV]",
        zlabel=r"Di-tau rate [kHZ]",
        xlim=(0, 1),
        ylim=(10, 100),
    )


def topo_rate_vs_pt(config):
    """physics/tautau_topo/tautau_topo_BkgRate: MinBias rate versus the
    smaller pT threshold."""
    return PlotSpec(
        name="physics/tautau_topo/tautau_topo_BkgRate",
        plotter="band_overlay",
        series=[
            Series(
                "tauRateVsPt(diTauTopo, raw)",
                label=r"Dijet trigger without $\tau_{h}$ idenficitation",
                x="pt",
                y="rate",
                style=dict(color=C[2]),
            ),
            Series(
                "cmssw/tauRateVsPt(diTauTopo, wp)",
                label=r"NN PUPPI Tau $\tau^\pm$ trigger",
                x="pt",
                y="rate",
                style=dict(color=C[0]),
            ),
            Series(
                "tauRateVsPt(diTauTopo, wp)",
                label=r"Multiclass $\tau$ topology trigger",
                x="pt",
                y="rate",
                style=dict(color=C[1]),
            ),
        ],
        options=dict(
            logy=True,
            ylabel=r"VBF H$\to \tau_h \tau_h$ trigger rate [kHz]",
            xlabel=r"Min($p^1_T$,$p^2_T$) [GeV]",
            legend=dict(loc="upper right", fontsize=style.MEDIUM_SIZE),
            cms_fontsize=style.MEDIUM_SIZE,
        ),
    )


def topo_eff_map(config):
    """physics/tautau_topo/topo_vbf_eff: efficiency in (gen pT1, gen
    pT2), CMSSW and the network."""
    rate = config["TAU_SEEDS"]["diTauTopo"]["target_rate"]
    return PlotSpec(
        name="physics/tautau_topo/topo_vbf_eff",
        plotter="mesh_panels",
        series=[
            Series("cmssw/diTauTopoEffVsGenPt2d", y="eff"),
            Series("diTauTopoEffVsGenPt2d", y="eff"),
        ],
        options=dict(
            titles=[
                f"NN PUPPI Tau Efficiency @ {rate}kHz",
                f"Multiclass Tagger Efficiency @ {rate}kHz",
            ],
            clim=(0, 0.5),
            xlabel=PT1,
            ylabel=PT2,
            cms_fontsize=style.CMS_LABEL_SMALL_SIZE,
        ),
    )


def topo_eff_ratio_map(config):
    """physics/tautau_topo/topo_vbf_eff_model_cmssw_ratio: network over
    CMSSW efficiency."""
    return PlotSpec(
        name="physics/tautau_topo/topo_vbf_eff_model_cmssw_ratio",
        plotter="ratio_map",
        series=[
            Series("diTauTopoEffVsGenPt2d"),
            Series("cmssw/diTauTopoEffVsGenPt2d"),
        ],
        options=dict(
            title=r"$\epsilon$ Jet Tagger / $\epsilon$ PUPPI $\tau$",
            xlabel=PT1,
            ylabel=PT2,
            cms_fontsize=style.CMS_LABEL_SMALL_SIZE,
        ),
    )


def topo_total_eff(config):
    """physics/tautau_topo/TotalEff.txt"""
    return PlotSpec(
        name="physics/tautau_topo/TotalEff",
        plotter="text_summary",
        series=[],
        options=dict(
            lines=[
                "Total diTau Eff ",
                (
                    "Multiclass NN [[diTauTopoTotalEff:.4f]] +/-"
                    " [[diTauTopoTotalEffErr:.4f]] "
                ),
                (
                    "CMSSW  [[cmssw/diTauTopoTotalEff:.4f]] +/-"
                    " [[cmssw/diTauTopoTotalEffErr:.4f]] "
                ),
            ]
        ),
    )


# -- Collections -------------------------------------------------------
@plot_collection(reads=["seedSingleTau"])
def seedSingleTau(config, info):
    """Single tau seed: working point grids of the model and the CMSSW
    L1 baseline, efficiency versus threshold, efficiency versus gen pT
    in barrel and tau endcap, and the integrated efficiencies.
    """
    plots = [
        single_tau_wp_grid(config),
        single_tau_wp_grid(config, prefix="cmssw/"),
        single_tau_eff_vs_threshold(config),
    ]
    for region in REGIONS:
        plots += [
            single_tau_eff_vs_gen_pt(config, region),
            single_tau_total_eff(config, region),
        ]
    return plots


@plot_collection(reads=["seedDiTau"])
def seedDiTau(config, info):
    """Di-tau seed: working point grid, MinBias rate versus pT
    threshold, efficiency versus gen pT in barrel and tau endcap, and
    the integrated efficiencies."""
    plots = [
        di_tau_wp_grid(config),
        di_tau_rate_vs_pt(config),
    ]
    for region in REGIONS:
        plots += [
            di_tau_eff_vs_gen_pt(config, region),
            di_tau_total_eff(config, region),
        ]
    return plots


@plot_collection(reads=["seedDiTauTopo"])
def seedDiTauTopo(config, info):
    """Di-tau topology seed: working point grid, MinBias rate versus pT
    threshold, efficiency maps in (gen pT1, gen pT2) of the model and
    the CMSSW di-tau seed, their ratio, and the integrated efficiencies.
    """
    return [
        topo_wp_grid(config),
        topo_rate_vs_pt(config),
        topo_eff_map(config),
        topo_eff_ratio_map(config),
        topo_total_eff(config),
    ]
