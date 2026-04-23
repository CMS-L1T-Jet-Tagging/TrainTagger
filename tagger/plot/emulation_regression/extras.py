import numpy as np

N_BUNCHES = 2760
REVOLUTION_FREQUENCY = 11246
MINBIAS_RATE = N_BUNCHES * REVOLUTION_FREQUENCY / 1000  # in kHz

PT_BINS = np.array([15, 17, 19, 22, 25, 30, 35, 40, 45, 50, 60, 76, 97, 122, 154, 195, 246, 311, 393, 496, 627, 792, 1000])

COLLECTION_KEYS = ['scPuppiL1TSC4NGJetJets', 'scPuppiExtendedJets']

LABELS_DICT = {
    'scPuppiL1TSC4NGJetJets': 'NG',
    'scPuppiExtendedJets': 'SC4',
    'ht': r'$HT^{Gen}$',
    'jet1': r'Leading $p_{T}$^{Gen} Jet',
    'jet2': r'Subleading $p_{T}$^{Gen} Jet',
    'ht15': r'HT^{Gen} ($p_{T}$ > 15 GeV)',
    'ht30': r'HT^{Gen} ($p_{T}$ > 30 GeV)',
    'mjj': r'$m_{jj}^{Gen}$',
    'max_mjj': r'$max(m_{jj}^{Gen})$',
    'scPuppiL1TSC4NGJetJets_raw': 'NG',
    'scPuppiL1TSC4NGJetJets_jecs': 'NG JECs',
    'scPuppiExtendedJets_raw': 'SC4 Raw',
    'scPuppiExtendedJets_jecs': 'SC4 JECs',
}

COLORS_DICT = {
    'scPuppiL1TSC4NGJetJets_raw': 'mediumpurple',
    'scPuppiL1TSC4NGJetJets_jecs': 'indigo',
    'scPuppiExtendedJets_raw': 'coral',
    'scPuppiExtendedJets_jecs': 'orangered'
}

PROCS_DICT = {
    'TT_PU200_151X_v1': r"$t\bar{t}$",
    'QCD_Pt15To3000_PU200_151Xv0': r"$QCD (p_T: 15-3000 GeV)$",
    'MinBias_PU200_151X_v1': 'MinBias',
    'VBFHToBB_PU200_151Xv0': r"$VBF H \to b\bar{b}$",
    'VBFHToCC_PU200_151Xv0': r"$VBF H \to c\bar{c}$",
    'TT_PU200': r"$t\bar{t}$",
    'QCD_Pt15To3000_PU200': r"QCD $(p_T: 15-3000 GeV)$",
    'QCD_PtAll_PU200': r"QCD",
    'XtoHH_MX_500To1000_PU200': r"$X \to HH$",
    'MinBias_PU200': 'MinBias',
    'VBFHToBB_PU200': r"$VBF H \to b\bar{b}$",
    'VBFHToCC_PU200': r"$VBF H \to c\bar{c}$",
    'VBFHToInvisible_PU200': r"$VBF \to invisible$",
}
