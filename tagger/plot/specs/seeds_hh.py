"""Plots of the HH seed collections, under physics/bbbb and
physics/bbtt. File names are those of the old bbbb.py and bbtt.py. The
CMSSW curves are cmssw/ series, from the results of
``python -m tagger.eval -m cmssw``.

Each plot is a function that returns its PlotSpec; the collections at
the end list the plots. ``v`` is the variant "score, sel" of the
variables, ``tag`` its "score_sel" in the file names.
"""

from tagger.plot import style
from tagger.plot.registry import PlotSpec, Series, plot_collection
from tagger.plot.specs import seeds_tau

C = style.color_cycle
EFF = r"$\int \epsilon$"
# (gen HT in GeV, efficiency) of the summary text of the HH -> 4b
# efficiency plot
BBBB_TEXT_POSITION = (63, 0.95)
TAU_TOPO, TAU12 = r"$\tau_{topo}$", r"$\tau_{1,2}$"
HH_OPTS = dict(
    errorbar_kw=dict(linewidth=style.ELINEWIDTH),
    grid=True,
    legend=dict(loc="upper left"),
    cms_fontsize=style.CMS_LABEL_SMALL_SIZE,
)


# ====================================================================
# HH -> 4b
# ======================================================================
BBBB_YLABEL = r"$\epsilon$(HH $\to$ 4b)"
BBBB_HT = dict(expr="hhGenHist(hh4b, ht)", label=r"$HT^{gen}$")


def _bbbb_tag(score, sel):
    return f"{'vs_qg' if score == 'vs_qg' else 'raw'}_{sel}"


def _bbbb_labels(config, v):
    """Legend labels of the model at its working point, the HT-only
    trigger and the pure gain."""
    rate = config["BBBB_TARGET_RATE"]
    return dict(
        wp=(
            f"Multiclass @ {rate} kHz, {EFF}=[[bbbbEff(wp, {v}):.2f]] (L1 $HT$"
            f" > [[bbbbWpHt({v}):.0f]] GeV, $\\sum$ 4b >"
            f" [[bbbbWpNn({v}):.2f]])"
        ),
        htOnly=(
            f"HT + QuadJets @ {rate} kHz, {EFF}=[[bbbbEff(htOnly, {v}):.2f]]"
            " (L1 $HT$ > [[htOnlyCut(bbbb):.0f]] GeV)"
        ),
        wpPure=(
            "Multiclass Gain (Pure eff. wrt HT trigger)"
            f" {EFF}=[[bbbbEff(wpPure, {v}):.2f]] "
        ),
    )


def bbbb_rate_grid(config, score, sel):
    """physics/bbbb/bbbb_rate_<tag>: MinBias rate grid in (b score sum,
    HT) with the working point line."""
    v = f"{score}, {sel}"
    return PlotSpec(
        name=f"physics/bbbb/bbbb_rate_{_bbbb_tag(score, sel)}",
        plotter="rate_grid",
        series=[
            Series(
                f"bbbbRateGrid({v})",
                x="nn",
                y="ht",
            ),
            Series(
                f"bbbbWpLine({v})",
                label=r"${} \pm {}$ kHz".format(
                    config["BBBB_TARGET_RATE"], config["BBBB_RATE_RANGE"]
                ),
                x="nn",
                y="ht",
                style=dict(linewidth=style.LINEWIDTH),
            ),
        ],
        options=dict(
            xlabel=r"$\sum_{4~leading~jets}$ b scores",
            ylabel=r"HT [GeV]",
            zlabel=r"4-b rate [kHZ]",
            xlim=(0, 1.3),
            ylim=(10, 500),
            legend=dict(loc="upper right"),
            cms_fontsize=style.CMS_LABEL_SMALL_SIZE,
        ),
    )


def bbbb_eff_vs_ht(config, score, sel):
    """physics/bbbb/HH_eff_HT_<tag>: efficiency versus gen HT, the model
    against the CMSSW b tag seed."""
    v, ref = (
        f"{score}, {sel}",
        f"raw, {sel}",
    )  # the cmssw pseudo-model has the raw b score only
    rate, wps = config["BBBB_TARGET_RATE"], config["CMSSW_WPS"]
    return PlotSpec(
        name=f"physics/bbbb/HH_eff_HT_{_bbbb_tag(score, sel)}",
        plotter="eff_points",
        series=[
            Series(
                f"cmssw/bbbbEffVsGenHt(wp, {ref})",
                label=(
                    f"BTag CMSSW Emulator @ {rate} kHz,"
                    f" {EFF}=[[cmssw/bbbbEff(wp, {ref}):.2f]] (L1 $HT$ >"
                    f" {wps['btag_l1_ht']} GeV, $\\sum$ 4b > {wps['btag']})"
                ),
                style=dict(c=C[0]),
            ),
            Series(
                f"bbbbEffVsGenHt(wp, {v})",
                label=_bbbb_labels(config, v)["wp"],
                style=dict(c=C[1]),
            ),
        ],
        options=dict(
            HH_OPTS,
            background=BBBB_HT,
            hlines=(1, 0, 800),
            ylim=(0.0, 1.17),
            xlim=(0, 800),
            xlabel=r"$HT^{gen}$ [GeV]",
            ylabel=BBBB_YLABEL,
        ),
    )


def bbbb_eff_vs_ht_vs_ht_only(config, score, sel):
    """physics/bbbb/HH_eff_HT_vs_HTonly_<tag>: efficiency versus gen HT,
    the model against the HT-only trigger."""
    v = f"{score}, {sel}"
    labels = _bbbb_labels(config, v)
    return PlotSpec(
        name=f"physics/bbbb/HH_eff_HT_vs_HTonly_{_bbbb_tag(score, sel)}",
        plotter="eff_points",
        series=[
            Series(
                f"bbbbEffVsGenHt(htOnly, {v})",
                label=labels["htOnly"],
                style=dict(c=C[2]),
            ),
            Series(
                f"bbbbEffVsGenHt(wp, {v})",
                label=labels["wp"],
                style=dict(c=C[1]),
            ),
            Series(
                f"bbbbEffVsGenHt(wpPure, {v})",
                label=labels["wpPure"],
                style=dict(c=C[3]),
            ),
        ],
        options=dict(
            HH_OPTS,
            background=BBBB_HT,
            hlines=(1, 0, 800),
            text=(
                *BBBB_TEXT_POSITION,
                (
                    "Multiclass | HT + QuadJets selection:"
                    f" [[bbbbOrRate({v})]] kHz, {EFF}=[[bbbbEff(wpOrHtOnly,"
                    f" {v}):.2f]]"
                ),
            ),
            ylim=(0.0, 1.22),
            xlim=(0, 800),
            xlabel=r"$HT^{gen}$ [GeV]",
            ylabel=BBBB_YLABEL,
        ),
    )


def bbbb_eff_vs_mhh(config, score, sel):
    """physics/bbbb/HH_eff_mHH_<tag>: efficiency versus gen m_HH."""
    v = f"{score}, {sel}"
    labels = _bbbb_labels(config, v)
    return PlotSpec(
        name=f"physics/bbbb/HH_eff_mHH_{_bbbb_tag(score, sel)}",
        plotter="eff_points",
        series=[
            Series(
                f"bbbbEffVsGenMhh(htOnly, {v})",
                label=labels["htOnly"],
                style=dict(c=C[2]),
            ),
            Series(
                f"bbbbEffVsGenMhh(wp, {v})",
                label=labels["wp"],
                style=dict(c=C[1]),
            ),
            Series(
                f"bbbbEffVsGenMhh(wpPure, {v})",
                label=labels["wpPure"],
                style=dict(c=C[3]),
            ),
        ],
        options=dict(
            HH_OPTS,
            background=dict(
                expr="hhGenHist(hh4b, mhh)", label=r"$m_{HH}^{gen}$"
            ),
            hlines=(1, 0, 1000),
            ylim=(0.0, 1.23),
            xlim=(0, 1000),
            xlabel=r"$m_{HH}^{gen}$ [GeV]",
            ylabel=BBBB_YLABEL,
        ),
    )


def bbbb_total_eff(config, score, sel):
    """physics/bbbb/TotalEff_<tag>.txt: integrated efficiencies of the
    events with 4 jets."""
    v, ref = f"{score}, {sel}", f"raw, {sel}"
    return PlotSpec(
        name=f"physics/bbbb/TotalEff_{_bbbb_tag(score, sel)}",
        plotter="text_summary",
        series=[],
        options=dict(
            lines=[
                "Total HH Eff ",
                f"HT Trigger [[bbbbEffFourJets(htOnly, {v}):.4f]] ",
                f"Model Eff [[bbbbEffFourJets(wp, {v}):.4f]] ",
                f"CMSSW Model Eff [[cmssw/bbbbEffFourJets(wp, {ref}):.4f]] ",
                (
                    "Model Pure Eff (exclude HT overlap)"
                    f" [[bbbbEffFourJets(wpPure, {v}):.4f]] "
                ),
                (
                    "CMSSW Model Pure Eff (exclude HT overlap)"
                    f" [[cmssw/bbbbEffFourJets(wpPure, {ref}):.4f]] "
                ),
            ]
        ),
    )


# ====================================================================
# HH -> bb tau tau
# ======================================================================
BBTT_YLABEL = r"$\epsilon$(HH $\to$ bb$\tau \tau$)"
SEED_RATE = "[[bbttRun3SeedRate:.1f]]"


def _bbtt_model_label(config, v, which, rate):
    """Legend label of the model at the working point at the seed rate
    or the target rate."""
    trigger = "nn" if which == "seed" else "nnTarget"
    return (
        f"Multiclass @ {rate} kHz, {EFF}=[[bbttEff({trigger}, {v}):.2f]] (L1"
        f" $HT$ > {config['BBTT_HT_CUT']} GeV, {TAU_TOPO} > [[bbttWpTt({v},"
        f" {which}):.2f]], $\\sum$ bb > [[bbttWpBb({v}, {which}):.2f]])"
    )


def _bbtt_labels(config, v):
    """Legend labels of the model at the target rate, the HT-only
    trigger and the pure gain."""
    return dict(
        nnTarget=_bbtt_model_label(
            config, v, "target", f"[[bbttWpRate({v}, target):g]]"
        ),
        htOnly=(
            f"HT + QuadJets @ {config['BBTT_TARGET_RATE']} kHz,"
            f" {EFF}=[[bbttEff(htOnly, {v}):.2f]] (L1 $HT$ >"
            " [[htOnlyCut(bbtt):.0f]] GeV)"
        ),
        nnTargetPure=(
            "Multiclass Gain (Pure eff. wrt HT trig)"
            f" {EFF}=[[bbttEff(nnTargetPure, {v}):.2f]]"
        ),
    )


def _bbtt_ht_opts(**extra):
    return dict(
        HH_OPTS,
        background=dict(expr="hhGenHist(hhbbtt, ht)", label=r"$HT^{gen}$"),
        hlines=(1, 0, 800),
        xlim=(0, 800),
        xlabel=r"$HT^{gen}$ [GeV]",
        ylabel=BBTT_YLABEL,
        **extra,
    )


def bbtt_rate_grid(config, score, sel, which):
    """physics/bbtt/bbtt_rate_<tag>_<rate>: MinBias rate grid in (bb,
    tau tau score) with the working point line at the seed rate
    (``which`` seed) or the target rate."""
    v = f"{score}, {sel}"
    rate = SEED_RATE if which == "seed" else str(config["BBTT_TARGET_RATE"])
    return PlotSpec(
        name=f"physics/bbtt/bbtt_rate_{score}_{sel}_{rate}",
        plotter="rate_grid",
        series=[
            Series(
                f"bbttRateGrid({sel})",
                x="bb",
                y="tt",
            ),
            Series(
                f"bbttWpLine({v}, {which})",
                label="$"
                + rate
                + r" \pm "
                + str(config["BBTT_RATE_RANGE"])
                + "$ kHz",
                x="bb",
                y="tt",
                style=dict(linewidth=style.LINEWIDTH),
            ),
        ],
        options=dict(
            z=f"rate_{score}",
            xlabel="bb score",
            ylabel=r"$\tau\tau$ score",
            zlabel="Rate [kHZ]",
            xlim=(0, 1.3),
            ylim=(0, 1.3),
            legend=dict(loc="upper right"),
            cms_fontsize=style.CMS_LABEL_SMALL_SIZE,
        ),
    )


def bbtt_eff_vs_run3_seed(config, score, sel):
    """physics/bbtt/HHbbtt_eff_run3_like_seed_<tag>: efficiency versus
    gen HT at the seed rate, against the Run-3-like seed."""
    v = f"{score}, {sel}"
    return PlotSpec(
        name=f"physics/bbtt/HHbbtt_eff_run3_like_seed_{score}_{sel}",
        plotter="eff_points",
        series=[
            Series(
                "cmssw/bbttRefEffVsGenHt(run3)",
                label=(
                    f"Run-3-like Seeds @ {SEED_RATE} kHz (HT OR $\\tau\\tau$)"
                    f" {EFF}=[[cmssw/bbttRefEff(run3):.2f]] (L1 $HT$ > 380"
                    f" GeV, {TAU12} > 34 GeV)"
                ),
                style=dict(c=C[0]),
            ),
            Series(
                f"bbttEffVsGenHt(nn, {v})",
                label=_bbtt_model_label(config, v, "seed", SEED_RATE),
                style=dict(c=C[1]),
            ),
        ],
        options=_bbtt_ht_opts(ylim=(0.0, 1.15)),
    )


def bbtt_eff_vs_ht_only(config, score, sel):
    """physics/bbtt/HHbbtt_eff_HT_only_<tag>: efficiency versus gen HT
    at the target rate, against the HT-only trigger."""
    v = f"{score}, {sel}"
    labels = _bbtt_labels(config, v)
    return PlotSpec(
        name=f"physics/bbtt/HHbbtt_eff_HT_only_{score}_{sel}",
        plotter="eff_points",
        series=[
            Series(
                f"bbttEffVsGenHt(htOnly, {v})",
                label=labels["htOnly"],
                style=dict(c=C[2]),
            ),
            Series(
                f"bbttEffVsGenHt(nnTarget, {v})",
                label=labels["nnTarget"],
                style=dict(c=C[1]),
            ),
            Series(
                f"bbttEffVsGenHt(nnTargetPure, {v})",
                label=labels["nnTargetPure"],
                style=dict(c=C[3]),
            ),
        ],
        options=_bbtt_ht_opts(ylim=(0.0, 1.23)),
    )


def bbtt_eff_vs_di_tau(config, score, sel):
    """physics/bbtt/HHbbtt_eff_diTau_<tag>: efficiency versus gen HT at
    the target rate, against the CMSSW di-tau seed."""
    v = f"{score}, {sel}"
    return PlotSpec(
        name=f"physics/bbtt/HHbbtt_eff_diTau_{score}_{sel}",
        plotter="eff_points",
        series=[
            Series(
                "cmssw/bbttRefEffVsGenHt(ditau)",
                label=(
                    "CMSSW di-$\\tau$ Seed @ 28 kHz,"
                    f" {EFF}=[[cmssw/bbttRefEff(ditau):.2f]] ({TAU12} >"
                    f" {config['CMSSW_WPS']['tau_l1_pt']} GeV)"
                ),
                style=dict(c=C[0]),
            ),
            Series(
                f"bbttEffVsGenHt(nnTarget, {v})",
                label=_bbtt_labels(config, v)["nnTarget"],
                style=dict(c=C[1]),
            ),
        ],
        options=_bbtt_ht_opts(ylim=(0.0, 1.15)),
    )


def bbtt_eff_vs_mhh(config, score, sel):
    """physics/bbtt/HH_eff_mHH_<tag>: efficiency versus gen m_HH at the
    target rate."""
    v = f"{score}, {sel}"
    labels = _bbtt_labels(config, v)
    options = _bbtt_ht_opts(ylim=(0.0, 1.23))
    options.update(
        background=dict(
            expr="hhGenHist(hhbbtt, mhh)", label=r"$m_{HH}^{gen}$"
        ),
        hlines=(1, 0, 1000),
        xlim=(0, 1000),
        xlabel=r"$m_{HH}^{gen}$ [GeV]",
    )
    return PlotSpec(
        name=f"physics/bbtt/HH_eff_mHH_{score}_{sel}",
        plotter="eff_points",
        series=[
            Series(
                f"bbttEffVsGenMhh(htOnly, {v})",
                label=labels["htOnly"],
                style=dict(c=C[2]),
            ),
            Series(
                f"bbttEffVsGenMhh(nnTarget, {v})",
                label=labels["nnTarget"],
                style=dict(c=C[1]),
            ),
            Series(
                f"bbttEffVsGenMhh(nnTargetPure, {v})",
                label=labels["nnTargetPure"],
                style=dict(c=C[3]),
            ),
        ],
        options=options,
    )


# -- Collections -------------------------------------------------------
@plot_collection(reads=["seedBbbb"])
def seedBbbb(config, info):
    """HH -> 4b seed per BBBB_VARIANTS: MinBias rate grid with the
    working point line, efficiency versus gen HT against the CMSSW seed
    and the HT-only trigger, versus gen m_HH, and the integrated
    efficiencies.
    """
    plots = []
    for score, sel in config["BBBB_VARIANTS"]:
        plots += [
            bbbb_rate_grid(config, score, sel),
            bbbb_eff_vs_ht(config, score, sel),
            bbbb_eff_vs_ht_vs_ht_only(config, score, sel),
            bbbb_eff_vs_mhh(config, score, sel),
            bbbb_total_eff(config, score, sel),
        ]
    return plots


@plot_collection(reads=["seedBbtt"])
def seedBbtt(config, info):
    """HH -> bb tau tau seed per BBTT_VARIANTS: (bb, tt) rate grids with
    the working point lines at the Run-3-like seed rate and at
    BBTT_TARGET_RATE, and efficiencies versus gen HT against the
    Run-3-like seed, the HT-only trigger and the CMSSW di-tau seed, and
    versus gen m_HH.
    """
    plots = []
    for score, sel in config["BBTT_VARIANTS"]:
        plots += [
            bbtt_rate_grid(config, score, sel, "seed"),
            bbtt_rate_grid(config, score, sel, "target"),
            bbtt_eff_vs_run3_seed(config, score, sel),
            bbtt_eff_vs_ht_only(config, score, sel),
            bbtt_eff_vs_di_tau(config, score, sel),
            bbtt_eff_vs_mhh(config, score, sel),
        ]
    return plots


@plot_collection(reads=["seeds"])
def seeds(config, info):
    """All trigger seed plots, from the results of the eval collection
    seeds."""
    return (
        seeds_tau.seedSingleTau(config, info)
        + seeds_tau.seedDiTau(config, info)
        + seeds_tau.seedDiTauTopo(config, info)
        + seedBbbb(config, info)
        + seedBbtt(config, info)
    )
