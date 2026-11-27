"""Plots of the model's emulation: two plot collections, one section
each below.

- emulation: the network (TensorFlow), its hls4ml build and the CMSSW
  emulator on the same jets of the emulation ntuple, under emulation/.
  File names are those of the old makeEmulationPlot.py.
- emulationRegression: the model's CMSSW emulation in full events, the
  SC4NN jets (seeded cone R = 0.4 jets with the model's pT regression)
  against the standard seeded cone jets (SC4 Raw, SC4 JECs), under
  emulation_regression/<process>/. File names are those of the old
  emulation_regression scripts.

Each plot is a function that returns its PlotSpec; the collections list
the plots.
"""

from tagger.plot import style
from tagger.plot.registry import PlotSpec, Series, plot_collection
from tagger.plot.style import (
    COLORS_DICT,
    LABELS_DICT,
    LINESTYLES_DICT,
    PROCESS_STYLE,
)

# =====================================================================
# emulation: network, hls4ml build and CMSSW emulator on the same jets
# ======================================================================
# (series prefix, legend label): the network, the CMSSW emulator, the
# hls4ml build
VERSIONS = [
    ("", "Tensorflow"),
    ("emulator/", "CMSSW Emulation"),
    ("hls4ml/", "hls4ml"),
]
JETS = "onEmulation(jetOutputs)"


def _hist_series(expr, label_suffix=lambda prefix: ""):
    n = len(VERSIONS)
    return [
        Series(
            f"{prefix}{expr}",
            label=label + label_suffix(prefix),
            style=dict(
                color=style.colours[i],
                linestyle=style.LINESTYLES[n - 1 - i],
                linewidth=style.THIN_LINEWIDTH,
            ),
        )
        for i, (prefix, label) in enumerate(VERSIONS)
    ]


def _hist_opts(xlabel, xlim):
    return dict(
        xlabel=xlabel,
        ylabel="a.u.",
        xlim=xlim,
        xlabel_kw=dict(ha="right", x=1),
        ylabel_kw=dict(ha="right", y=1),
        legend=dict(loc="upper right"),
    )


def _pairwise(name, field, lim, title):
    """Per-jet 2D histogram of the network against the CMSSW emulator."""
    return PlotSpec(
        name=name,
        plotter="pairwise_2d",
        series=[
            Series(JETS, y=field),
            Series("emulator/" + JETS, y=field),
        ],
        options=dict(
            xrange=lim,
            yrange=lim,
            xlabel="Tensorflow",
            ylabel="CMSSW Emulation",
            title=title,
        ),
    )


def regression_2d():
    """emulation/jetRegression_2D: regression output, network against
    emulator."""
    return _pairwise(
        "emulation/jetRegression_2D", "regression", (0, 2), "Jet Regression"
    )


def regression_1d():
    """emulation/jetRegression_1D: regression output of the three versions."""
    return PlotSpec(
        name="emulation/jetRegression_1D",
        plotter="histogram",
        series=_hist_series("onEmulation(regressionHist)"),
        options=_hist_opts("Regression Output", (0, 2)),
    )


def score_1d(cls):
    """emulation/<cls>_score_1D: class score of the three versions."""
    return PlotSpec(
        name=f"emulation/{cls}_score_1D",
        plotter="histogram",
        series=_hist_series(f"onEmulation(scoreHist({cls}))"),
        options=_hist_opts(style.CLASS_LABEL_STYLE[cls] + " score", (0, 1)),
    )


def score_2d(cls):
    """emulation/<cls>_score_2D: class score, network against emulator."""
    return _pairwise(
        f"emulation/{cls}_score_2D",
        cls,
        (0, 1),
        style.CLASS_LABEL_STYLE[cls] + " score",
    )


def roc_comparison(cls):
    """emulation/ROC_Emulation_comparison_<cls>: one-vs-rest ROC of the
    three versions."""
    return PlotSpec(
        name=f"emulation/ROC_Emulation_comparison_{cls}",
        plotter="curve_overlay",
        series=[
            Series(
                f"{prefix}onEmulation(rocOneVsRest({cls}))",
                label=(
                    f"{name} Tagger, AUC ="
                    f" [[{prefix}onEmulation(aucOneVsRest({cls})):.2%]]"
                ),
                x="tpr",
                y="fpr",
                style=dict(
                    color=style.colours[i], linestyle=style.LINESTYLES[i]
                ),
            )
            for i, (prefix, name) in enumerate(VERSIONS)
        ],
        options=dict(
            xlabel="Signal Efficiency",
            ylabel="Mistag Rate",
            logy=True,
            xlim=(0.0, 1.0),
            ylim=(0.001, 1),
            legend=dict(loc="best"),
        ),
    )


def _response_stats(prefix):
    """Legend suffix of one version in response_hist: its response
    median and RMS."""
    return (
        f" median: [[{prefix}onEmulation(responseMedian(all)):.3f]]"
        f" rms: [[{prefix}onEmulation(responseRms(all)):.3f]]"
    )


def response_hist():
    """emulation/response_emulation: jet response L1/gen of the three
    versions, with median and RMS. As in the old script the first two
    legend entries are swapped: the network's curve has the label and
    numbers of the emulator, the emulator's those of the network."""
    labels = [
        ("Emulation", "emulator/"),
        ("Tensorflow", ""),
        ("hls4ml", "hls4ml/"),
    ]
    n = len(VERSIONS)
    return PlotSpec(
        name="emulation/response_emulation",
        plotter="histogram",
        series=[
            Series(
                f"{prefix}onEmulation(responseHist(all))",
                label=label + _response_stats(stats_prefix),
                style=dict(
                    color=style.colours[i],
                    linestyle=style.LINESTYLES[n - 1 - i],
                    linewidth=style.THIN_LINEWIDTH,
                ),
            )
            for i, ((prefix, _), (label, stats_prefix)) in enumerate(
                zip(VERSIONS, labels)
            )
        ],
        options=_hist_opts("Jet Response (L1/Gen)", (0, 2)),
    )


def bit_accuracy(classes):
    """emulation/bit_accuracy.txt: fraction of jets where hls4ml and the
    emulator agree exactly."""
    return PlotSpec(
        name="emulation/bit_accuracy",
        plotter="agreement",
        series=[
            Series("emulator/" + JETS),
            Series("hls4ml/" + JETS),
        ],
        options=dict(
            fields=["regression"] + classes,
            text=(
                "Percent bit accuracy between CMSSW emulator and hls4ml for"
                " {field}: {percent:.4f} %"
            ),
        ),
    )


@plot_collection(reads=["emulation"])
def emulation(config, info):
    """The network, its hls4ml build and the CMSSW emulator on the same
    jets: class scores and regression output (1D, and 2D network vs
    emulator), one-vs-rest ROCs, the pT response, and the bit accuracy
    of hls4ml against the emulator. The hls4ml and emulator series need
    -m hls4ml:<model> and -m emulator:<model> evaluated."""
    classes = [c for c in info.classes if c != "pileup"]
    plots = [
        regression_2d(),
        regression_1d(),
    ]
    for c in classes:
        plots += [
            score_1d(c),
            score_2d(c),
            roc_comparison(c),
        ]
    plots += [
        response_hist(),
        bit_accuracy(classes),
    ]
    return plots


# ====================================================================
# emulationRegression: the SC4NN jets against the standard jets in full
# events (FastPUPPI)
# ======================================================================
REGRESSION_DIR = "emulation_regression"
# jets argument of the variables -> key of the style dicts
KEY = {
    "nn": "scPuppiL1TSC4NGJetJets_raw",
    "sc": "scPuppiExtendedJets_raw",
    "scJecs": "scPuppiExtendedJets_jecs",
    "gen": "genjets",
}
# in bins of gen pT (the old file names) and of L1 pT (folders
# ..._vs_l1pt): variable suffix -> (folder suffix, x label)
BINNED_BY = {
    "GenPt": ("", r"Jet $p_T^{Gen}$ [GeV]"),
    "Pt": ("_vs_l1pt", r"Jet $p_T^{L1}$ [GeV]"),
}
# Nominal top, W and Higgs masses (GeV, PDG values rounded), drawn as a
# line on the mass peaks
TOP_MASS, W_MASS, HIGGS_MASS = 173, 80, 125
MASSES = {
    "top": dict(
        mass=TOP_MASS,
        mass_label=rf"$m_{{t}}$ = {TOP_MASS} GeV",
        xlabel="Trijet invariant mass [GeV]",
    ),
    "w": dict(
        mass=W_MASS,
        mass_label=rf"$m_{{W}}$ = {W_MASS} GeV",
        xlabel="Dijet invariant mass [GeV]",
    ),
    "h": dict(
        mass=HIGGS_MASS,
        mass_label=rf"$m_{{H}}$ = {HIGGS_MASS} GeV",
        xlabel="Dijet invariant mass [GeV]",
    ),
}
# (process, particle) of the mass peaks
MASS_PEAKS = [
    ("TT_PU200", "top"),
    ("TT_PU200", "w"),
    ("VBFHToBB_PU200", "h"),
    ("VBFHToCC_PU200", "h"),
]


def _vs_pt(name, expr_of, ylabel, proc, legend, xlabel, xlim=None, ylim=None):
    """<name>_x<log|linear>_y<log|linear>: one plot per x and y scale,
    the SC4NN jets as read, the standard jets as read and corrected.
    ``expr_of(jets)`` gives the expression of the jets nn or sc.
    """
    err = dict(
        fmt="o",
        capsize=style.CAPSIZE,
        ms=style.MARKERSIZE,
        elinewidth=style.ELINEWIDTH,
    )
    series = [
        Series(
            expr_of(jets),
            label=LABELS_DICT[KEY[key]],
            y=field,
            yerr=f"{field}_err",
            style=dict(err, color=COLORS_DICT[KEY[key]]),
        )
        for jets, key, field in (
            ("nn", "nn", "raw"),
            ("sc", "sc", "raw"),
            ("sc", "scJecs", "jecs"),
        )
    ]
    plots = []
    for lx in ("log", "linear"):
        for ly in ("log", "linear"):
            options = dict(
                xlabel=xlabel,
                ylabel=ylabel,
                logx=lx == "log",
                logy=ly == "log",
                grid=dict(alpha=1, linestyle="-", lw=style.GRID_LINEWIDTH),
                legend=dict(title=PROCESS_STYLE[proc], **legend),
            )
            if xlim:
                options["xlim"] = xlim
            if ylim:
                options["ylim"] = ylim
            plots.append(
                PlotSpec(
                    name=f"{name}_x{lx}_y{ly}",
                    plotter="curve_overlay",
                    series=series,
                    options=options,
                )
            )
    return plots


def fp_response(proc, binned_by, reduce):
    """<proc>/response<suffix>/response_<mean|median>_x.._y..: response
    versus gen or L1 pT (4 plots)."""
    suffix, xlabel = BINNED_BY[binned_by]
    return _vs_pt(
        f"{REGRESSION_DIR}/{proc}/response{suffix}/response_{reduce}",
        lambda jets: f"fpResponseVs{binned_by}({proc}, {jets}, {reduce})",
        f"Response {reduce.capitalize()}(L1/Gen)",
        proc,
        dict(fontsize=style.LARGE_SIZE, title_fontsize=style.BIGGER_SIZE),
        xlabel=xlabel,
        xlim=(0, 1000),
        ylim=(0.6, 1.7),
    )


def fp_resolution(proc, binned_by):
    """<proc>/rms<suffix>/residual_rms_x.._y..: resolution versus gen or
    L1 pT (4 plots)."""
    suffix, xlabel = BINNED_BY[binned_by]
    return _vs_pt(
        f"{REGRESSION_DIR}/{proc}/rms{suffix}/residual_rms",
        lambda jets: f"fpResolutionVs{binned_by}({proc}, {jets})",
        r"$\sigma(p_T^{\mathrm{L1}} - p_T^{\mathrm{Gen}})\, / \,"
        r" \mathrm{Mean}(p_T^{\mathrm{Gen}})$",
        proc,
        dict(loc="upper center"),
        xlabel=xlabel,
    )


def fp_pt_hist(proc, which):
    """<proc>/distributions/<Leading|Full>_pt_distribution: pT of the
    leading or all jets."""
    name = {"leading": "Leading", "all": "Full"}[which]
    return PlotSpec(
        name=f"{REGRESSION_DIR}/{proc}/distributions/{name}_pt_distribution",
        plotter="histogram",
        series=[
            Series(
                f"fpPtHist({proc}, {j}, {which})",
                label=LABELS_DICT[KEY[j]],
                y="fraction",
                style=dict(
                    color=COLORS_DICT[KEY[j]],
                    linewidth=style.SPECTRUM_LINEWIDTH,
                ),
            )
            for j in ("nn", "sc", "scJecs", "gen")
        ],
        options=dict(
            density=False,
            logy=True,
            xlabel=f"{name} Jet $p_T$ [GeV]",
            ylabel="Fraction",
            cms=False,
            legend=dict(title=PROCESS_STYLE[proc]),
        ),
    )


def fp_pt_map(proc, jets):
    """<proc>/heatmaps/<jets key>_heatmap: gen against L1 pT."""
    return PlotSpec(
        name=f"{REGRESSION_DIR}/{proc}/heatmaps/{KEY[jets]}_heatmap",
        plotter="pt_map",
        series=[
            Series(f"fpGenRecoPtMap({proc}, {jets})"),
        ],
        options=dict(
            xlabel="GenJet $p_T$",
            ylabel=LABELS_DICT[KEY[jets]],
            zlabel="Entries",
        ),
    )


def fp_pt_map_ratio(proc):
    """<proc>/heatmaps/ratio_heatmap: the gen vs L1 pT map of the SC4NN
    jets over the corrected standard jets."""
    return PlotSpec(
        name=f"{REGRESSION_DIR}/{proc}/heatmaps/ratio_heatmap",
        plotter="pt_map",
        series=[
            Series(f"fpGenRecoPtMap({proc}, nn)"),
            Series(f"fpGenRecoPtMap({proc}, scJecs)"),
        ],
        options=dict(
            xlabel="GenJet $p_T$",
            ylabel=r"Reco Jet $p_{T}$",
            zlabel=(
                f"Ratio {LABELS_DICT['scPuppiL1TSC4NGJetJets']} vs"
                f" {LABELS_DICT['scPuppiExtendedJets_jecs']}"
            ),
        ),
    )


def fp_turn_on(proc, obj, rate):
    """<proc>/turn_ons/turn_on_curve_<obj>_<rate>kHz: turn-on at the
    threshold of a MinBias rate."""
    return PlotSpec(
        name=f"{REGRESSION_DIR}/{proc}/turn_ons/turn_on_curve_{obj}_{rate}kHz",
        plotter="turn_on",
        series=[
            Series(
                f"fpTurnOnVsGen({proc}, {obj}, {j}, {rate})",
                label=(
                    f"{LABELS_DICT[KEY[j]]}, [[fpRateWp({obj}, {j},"
                    f" {rate}):.1f]] GeV"
                ),
                style=dict(
                    color=COLORS_DICT[KEY[j]],
                    linestyle=LINESTYLES_DICT[KEY[j]],
                ),
            )
            for j in ("nn", "sc", "scJecs")
        ],
        options=dict(
            xlabel=f"{LABELS_DICT[obj]} [GeV]",
            ylabel=f"Eff (L1 rate at {rate} kHz)",
            legend=dict(
                title=PROCESS_STYLE[proc],
                fontsize=style.LARGE_SIZE,
                title_fontsize=style.LARGE_SIZE,
            ),
        ),
    )


def fp_mass(proc, particle):
    """<proc>/masses/<particle>_mass: the top, W or Higgs mass peak."""
    return PlotSpec(
        name=f"{REGRESSION_DIR}/{proc}/masses/{particle}_mass",
        plotter="mass_peak",
        series=[
            Series(
                f"fpMass({proc}, {particle}, {j})",
                label=LABELS_DICT[KEY[j]],
                style=dict(color=COLORS_DICT[KEY[j]]),
            )
            for j in ("gen", "nn", "sc", "scJecs")
        ],
        options=dict(
            MASSES[particle],
            legend=dict(
                title=PROCESS_STYLE[proc],
                fontsize=style.MEDIUM_SIZE,
                title_fontsize=style.MEDIUM_SIZE,
            ),
        ),
    )


@plot_collection(reads=["emulationRegression"])
def emulationRegression(config, info):
    """The model's CMSSW emulation in full events: response and
    resolution vs gen and L1 pT, pT spectra, gen vs L1 pT maps, turn-on
    curves at fixed MinBias rates, and the top, W and Higgs mass peaks.
    """
    plots = []
    for p in config["FASTPUPPI_PROCESSES"]:
        for binned_by in BINNED_BY:
            plots += fp_response(p, binned_by, "mean")
            plots += fp_response(p, binned_by, "median")
            plots += fp_resolution(p, binned_by)
        plots += [
            fp_pt_hist(p, "leading"),
            fp_pt_hist(p, "all"),
            fp_pt_map(p, "nn"),
            fp_pt_map(p, "scJecs"),
            fp_pt_map_ratio(p),
        ]
        for obj in config["TURN_ON_OBSERVABLES"]:
            for rate in config["TURN_ON_RATES"]:
                plots.append(fp_turn_on(p, obj, rate))
    for proc, particle in MASS_PEAKS:
        plots.append(fp_mass(proc, particle))
    return plots
