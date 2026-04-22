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

SMALL_SIZE = 25
MEDIUM_SIZE = 28
BIGGER_SIZE = 35

LEGEND_WIDTH = 20
LINEWIDTH = 5
MARKERSIZE = 20

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
    'pt': 'Const. $p_T$',
    'pt_rel': 'Const. relative $p_T$',
    'pt_log': 'Const. $log(p_T)$',
    'eta': 'Const. |$\\eta$|',
    'deta': 'Const. $\\Delta\\eta$',
    'dphi': 'Const. $\\Delta\\phi$',
    'phi': 'Const. $\\phi$',
    'mass': 'Const. mass',
    'isPhoton': 'Const. PID: photon',
    'isElectronPlus': 'Const. PID: electron +',
    'isElectronMinus': 'Const. PID: electron -',
    'isMuonPlus': 'Const. PID: muon +',
    'isMuonMinus': 'Const. PID: muon -',
    'isNeutralHadron': 'Const. PID: hadron neutral',
    'isChargedHadronPlus': 'Const. PID: hadron +',
    'isChargedHadronMinus': 'Const. PID: hadron -',
    'z0': 'Const. $z_0$',
    'dxy': 'Const. $d_{xy}$',
    'isfilled': 'Const. record filled',
    'puppiweight': 'Const. PUPPI Weight',
    'quality': 'Const. Track Quality',
    'emid': 'Const. ElectroMagnetic ID',
    'jet_eta': 'Jet |$\\eta$|',
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
    'log_pt': (0, 5),
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
    'is_filled': (0, 1),
    'puppiweight': (0, 1),
    'puppi_weight': (0, 1),
    'puppi_weight': (0, 1),
    'quality': (0, 15),
    'emid': (0, 1),
    'charge': (-2, 2),
    'id': (-1, 10),
    'eta_phys': (-3, 3),
    'phi_phys': (-3.5, 3.5),
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
