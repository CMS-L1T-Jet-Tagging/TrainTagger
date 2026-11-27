import matplotlib
import matplotlib.pyplot as plt
import mplhep as hep

colours = ["black", "red", "orange", "green", "blue"]
LINESTYLES = [
    "-",
    "--",
    "dotted",
    (0, (3, 5, 1, 5)),
    (
        0,
        (
            3,
            5,
            1,
            1,
            1,
            5,
        ),
    ),
    (0, (3, 10, 1, 10)),
    (0, (3, 10, 1, 10, 1, 10)),
]

color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
# Fixed colours per class, for reproducible colour and class
# combinations (style of the first DP note)
SHAP_COLORS = [
    "#E41A1C",  # red
    "#377EB8",  # blue
    "#4DAF4A",  # green
    "#984EA3",  # purple
    "#FFFF33",  # yellow
    "#A65628",  # brown
    "#F781BF",  # pink
    "#999999",  # gray
    "#FF7F00",  # orange
]

SMALL_SIZE = 25
MEDIUM_SIZE = 28
BIGGER_SIZE = 35

# Font sizes besides SMALL, MEDIUM and BIGGER
SMALLER_SIZE = SMALL_SIZE - 3  # legends with many entries (multiclass ROCs)
CMS_LABEL_SMALL_SIZE = MEDIUM_SIZE - 2  # CMS label of the seed plots
LARGE_SIZE = 40  # colour bar labels, FastPUPPI legends
LARGEST_SIZE = 44  # axis labels of the efficiency points plots

LEGEND_WIDTH = 20
LINEWIDTH = 5
ELINEWIDTH = 3
MARKERSIZE = 10
CAPSIZE = 4

# Line widths besides LINEWIDTH
THIN_LINEWIDTH = (
    LINEWIDTH - 1.5
)  # one curve per model version (emulation, response)
HISTOGRAM_LINEWIDTH = 2.5  # pT weight distributions
SPECTRUM_LINEWIDTH = 2  # FastPUPPI pT spectra
ERRORBAR_LINEWIDTH = 2  # efficiency points without errorbar_kw
MASS_LINEWIDTH = 1.5  # nominal mass line of the mass peaks
GRID_LINEWIDTH = 0.75  # grid of the FastPUPPI turn-on curves
# dashed line at efficiency 1 (option hline)
REFERENCE_LINEWIDTH = 3
# dashed lines of the HH seed plots (option hlines)
HH_REFERENCE_LINEWIDTH = 4

# Marker and error bar cap sizes besides MARKERSIZE and CAPSIZE
RESPONSE_MARKERSIZE = 8  # pT response and resolution points
EFF_POINTS_CAPSIZE = 3  # efficiency points (eff_points)
RATE_GRID_MARKERSIZE = 500  # squares of the rate grid scatter plots (points^2)

# Transparency
BAND_ALPHA = 0.3  # error bands
BAR_ALPHA = 0.7  # gen pT distributions next to the ratio map
TEXT_BOX_ALPHA = 0.8  # text boxes on plots

FIGURE_SIZE = (17, 17)

# Layout of single plotters
MESH_PANELS_WIDTH_FACTOR = (
    2.5  # figure width of mesh_panels in FIGURE_SIZE widths
)
MESH_PANELS_TITLE_PAD = 45  # space between the panel titles and the CMS label
RATIO_MAP_SIZE_FACTORS = (
    1.4,
    1.1,
)  # ratio_map figure size in FIGURE_SIZE units
RATIO_MAP_GRID = dict(
    width_ratios=[0.1, 0.2, 0.5, 4, 1],
    height_ratios=[1.05, 4],
    hspace=0.05,
    wspace=0.11,
)  # title, colour bar, gap, map, gen pT2 / gen pT1, map
COLORBAR_SHRINK = 0.7  # colour bar height relative to the axes (pt_map)
COLORBAR_PAD = 0.5  # gap between a plot and its colour bar (grid_2d)
PROFILE_EXTRA_WIDTH = 2  # extra figure width of plot_2d for its colour bar
PROFILE_HIST_BINS = 50  # bins per axis of plot_2d (firmware profiling)
OVERLAY_ALPHA = 0.6  # series of the second and further models in an overlay
STACKED_BARS_FONTSIZE = (
    30  # default font size of stacked_bars (option fontsize)
)
STACKED_BARS_HEIGHT = (
    0.7  # bar height of stacked_bars, in units of the bar spacing
)
RATIO_MAP_TITLE_Y = (
    0.62  # height of the vertical title of ratio_map (axes fraction)
)
RATIO_MAP_FRACTION_TICKS = [
    0,
    0.15,
    0.3,
]  # ticks of the gen pT fractions next to the ratio map
RATIO_MAP_COLORBAR_ASPECT = 10  # length / width of the ratio_map colour bar
TURN_ON_SPLINE_POINTS = 500  # points of the spline through the turn-on points
TURN_ON_YMAX = 1.05  # top of the y axis of turn_on (efficiencies up to 1)
RATIO_MAP_VMAX = 5.0  # top of the ratio_map colour scale (option vmax)
MATRIX_DECIMALS = 3  # decimals of the grid_2d cell values (option round)
MASS_PEAK_LINEWIDTH = 2  # mass_peak distributions and points
MASS_PEAK_MARKERSIZE = 6.5
MASS_PEAK_CAPSIZE = 3

CMSHEADER_LEFT = "Phase 2 Simulation Preliminary"
CMSHEADER_RIGHT = "PU 200 (14 TeV)"
CMSHEADER_SIZE = BIGGER_SIZE

CLASS_LABEL_STYLE = {
    'b': 'b',
    'charm': 'c',
    'light': 'light',
    'gluon': 'gluon',
    'taum': '$\\tau_{h}^{-}$',
    'taup': '$\\tau_{h}^{+}$',
    'electron': 'Electron',
    'muon': 'Muon',
    'pileup': 'Pile-up',
    'inclusive': 'Inclusive',
    'Regression': 'Regression',
    "taus": "Taus",
    "jets": "Jets (b, c, light, gluon)",
    "leptons": "Leptons (muon, electron)",
}

INPUT_FEATURE_STYLE = {
    'pt': '$p_T$',
    'pt_rel': 'relative $p_T$',
    'pt_log': '$log(p_T)$',
    'eta': '$\\eta$',
    'deta': '$\\Delta\\eta$',
    'dphi': '$\\Delta\\phi$',
    'phi': '$\\phi$',
    'mass': 'mass',
    'isPhoton': 'PID: photon',
    'isElectronPlus': 'PID: electron +',
    'isElectronMinus': 'PID: electron -',
    'isMuonPlus': 'PID: muon +',
    'isMuonMinus': 'PID: muon -',
    'isNeutralHadron': 'PID: hadron neutral',
    'isChargedHadronPlus': 'PID: hadron +',
    'isChargedHadronMinus': 'PID: hadron -',
    'z0': '$z_0$',
    'dxy': '$d_{xy}$',
    'isfilled': 'record filled',
    'puppiweight': 'PUPPI Weight',
    'quality': 'Track Quality',
    'emid': 'ElectroMagnetic ID',
    'jet_eta': 'Jet $\\eta$',
    'jet_pt': 'Jet $p_T$',
    'jet_pt_log': 'Jet $log(p_T)$',
    'charge': 'Charge',
    'id': 'Particle ID',
    'eta_phys': '$\\eta_{phys}$',
    'phi_phys': '$\\phi_{phys}$',
}

PROCESS_STYLE = {
    'TT_PU200': r't$\bar{t}$',
    'ggHHbbbb_PU200': (
        r'gg $\rightarrow$ HH $\rightarrow$ b$\bar{b}$b$\bar{b}$'
    ),
    'VBFHtt_PU200': r'VBF $\rightarrow$ H $\rightarrow$ t$\bar{t}$',
    'ggHHbbtt_PU200': (
        r'gg $\rightarrow$ HH $\rightarrow$ b$\bar{b}$t$\bar{t}$'
    ),
    'ggHtt_PU200': r'gg $\rightarrow$ HH $\rightarrow$ t$\bar{t}$',
    'MinBias_PU200': 'MinBias',
    'VBFHToBB_PU200': r"$VBF H \to b\bar{b}$",
    'VBFHToCC_PU200': r"$VBF H \to c\bar{c}$",
    'VBFHToInvisible_PU200': r"$VBF \to invisible$",
    'QCD_Pt15To3000_PU200': r"QCD $(p_T: 15-3000 GeV)$",
    'QCD_PtAll_PU200': r"QCD",
    'XtoHH_MX_500To1000_PU200': r"$X \to HH$",
}

# Regression styles

LABELS_DICT = {
    'scPuppiL1TSC4NGJetJets': 'SC4 NN',
    'scPuppiExtendedJets': 'SC4',
    'scPuppiL1TSC4NGJetJets_raw': 'SC4 NN',
    'scPuppiL1TSC4NGJetJets_jecs': 'SC4 NN JECs',
    'scPuppiExtendedJets_raw': 'SC4 Raw',
    'scPuppiExtendedJets_jecs': 'SC4 JECs',
    'genjets': 'GenJets',
    'ht': r'$HT^{Gen}$',
    'jet1': r'Leading $p_{T}^{Gen}$ Jet',
    'jet2': r'Subleading $p_{T}^{Gen}$ Jet',
    'ht15': r'$HT^{Gen}$ ($p_{T}^{Gen}$ > 15 GeV)',
    'ht30': r'$HT^{Gen}$ ($p_{T}^{Gen}$ > 30 GeV)',
    'mjj': r'$m_{jj}^{Gen}$',
    'max_mjj': r'$max(m_{jj}^{Gen})$',
    'dijet': r'Leading two $p_{T}^{Gen}$ Jets',
    'quadjet': r'Leading four $p_{T}^{Gen}$ Jets',
}

COLORS_DICT = {
    'scPuppiL1TSC4NGJetJets_raw': '#964a8b',
    'scPuppiL1TSC4NGJetJets_jecs': '#7a21dd',
    'scPuppiExtendedJets_raw': '#e42536',
    'scPuppiExtendedJets_jecs': '#f89c20',
    'genjets': 'gray',
}

LINESTYLES_DICT = {
    'scPuppiL1TSC4NGJetJets_raw': 'solid',
    'scPuppiL1TSC4NGJetJets_jecs': 'dashdot',
    'scPuppiExtendedJets_raw': 'dashed',
    'scPuppiExtendedJets_jecs': 'dotted',
}


def set_style():
    # Setup plotting to CMS style
    hep.cms.label()
    hep.cms.text("Simulation")
    plt.style.use(hep.style.CMS)

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE + 5)  # fontsize of the x and y labels
    plt.rc('axes', linewidth=LINEWIDTH + 2)  # thickness of axes
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE - 2)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    # line thickness
    matplotlib.rcParams['lines.linewidth'] = 5

    matplotlib.rcParams['xtick.major.size'] = 20
    matplotlib.rcParams['xtick.major.width'] = 5
    matplotlib.rcParams['xtick.minor.size'] = 10
    matplotlib.rcParams['xtick.minor.width'] = 4

    matplotlib.rcParams['ytick.major.size'] = 20
    matplotlib.rcParams['ytick.major.width'] = 5
    matplotlib.rcParams['ytick.minor.size'] = 10
    matplotlib.rcParams['ytick.minor.width'] = 4

# Input feature ranges of the old plotting scripts
INPUT_FEATURE_RANGES = {
    'pt': (0, 150),
    'pt_rel': (0, 1),
    'pt_log': (0, 5),
    'eta': (-580, 580),
    'deta': (-150, 150),
    'dphi': (-150, 150),
    'phi': (-3.5, 3.5),
    'mass': (0, 10),
    'isPhoton': (0, 1),
    'isElectronPlus': (0, 1),
    'isElectronMinus': (0, 1),
    'isMuonPlus': (0, 1),
    'isMuonMinus': (0, 1),
    'isNeutralHadron': (0, 1),
    'isChargedHadronPlus': (0, 1),
    'isChargedHadronMinus': (0, 1),
    'z0': (-420, 420),
    'dxy': (0, 80),
    'isfilled': (0, 1),
    'puppiweight': (0, 1),
    'quality': (0, 7.1),
    'emid': (0, 7.1),
    'charge': (-2, 2),
    'id': (-1, 10),
    'eta_phys': (-3, 3),
    'phi_phys': (-3.5, 3.5),
}
