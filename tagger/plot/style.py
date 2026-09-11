import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

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
    ] # for future reproducibility of color and class combinations (style of first DP note)

SMALL_SIZE = 25
MEDIUM_SIZE = 28
BIGGER_SIZE = 35

LEGEND_WIDTH = 20
LINEWIDTH = 5
ELINEWIDTH = 3
MARKERSIZE = 10
CAPSIZE = 4

FIGURE_SIZE = (17, 17)

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

INPUT_FEATURE_RANGES = {
    'pt': (0, 150),
    'pt_rel': (0, 1),
    'pt_log': (0, 5),
    'eta': (-3, 3),
    'deta': (-3, 3),
    'dphi': (-3.5, 3.5),
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
    'z0': (-0.5, 0.5),
    'dxy': (-0.2, 0.2),
    'isfilled': (0, 1),
    'puppiweight': (0, 1),
    'quality': (0, 15),
    'emid': (0, 1),
    'charge': (-2, 2),
    'id': (-1, 10),
    'eta_phys': (-3, 3),
    'phi_phys': (-3.5, 3.5),
}

PROCESS_STYLE = {
        'TT_PU200': r't$\bar{t}$',
        'ggHHbbbb_PU200': r'gg $\rightarrow$ HH $\rightarrow$ b$\bar{b}$b$\bar{b}$',
        'VBFHtt_PU200': r'VBF $\rightarrow$ H $\rightarrow$ t$\bar{t}$',
        'ggHHbbtt_PU200': r'gg $\rightarrow$ HH $\rightarrow$ b$\bar{b}$t$\bar{t}$',
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
COLLECTION_KEYS = ['scPuppiL1TSC4NGJetJets', 'scPuppiExtendedJets'] # add collection keys here if you want to plot them in regression plots

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
    import matplotlib as mpl

    mpl.rcParams['lines.linewidth'] = 5

    import matplotlib

    matplotlib.rcParams['xtick.major.size'] = 20
    matplotlib.rcParams['xtick.major.width'] = 5
    matplotlib.rcParams['xtick.minor.size'] = 10
    matplotlib.rcParams['xtick.minor.width'] = 4

    matplotlib.rcParams['ytick.major.size'] = 20
    matplotlib.rcParams['ytick.major.width'] = 5
    matplotlib.rcParams['ytick.minor.size'] = 10
    matplotlib.rcParams['ytick.minor.width'] = 4
