import numpy as np

COLLECTION_KEYS = ['scPuppiL1TSC4NGJetJets', 'scPuppiExtendedJets']

LABELS_DICT = {
    'scPuppiL1TSC4NGJetJets': 'SC4 NG',
    'scPuppiExtendedJets': 'SC4',
    'ht': r'$HT^{Gen}$',
    'jet1': r'Leading $p_{T}^{Gen}$ Jet',
    'jet2': r'Subleading $p_{T}^{Gen}$ Jet',
    'ht15': r'$HT^{Gen}$ ($p_{T}^{Gen}$ > 15 GeV)',
    'ht30': r'$HT^{Gen}$ ($p_{T}^{Gen}$ > 30 GeV)',
    'mjj': r'$m_{jj}^{Gen}$',
    'max_mjj': r'$max(m_{jj}^{Gen})$',
    'dijet': r'Leading two $p_{T}^{Gen}$ Jets',
    'quadjet': r'Leading four $p_{T}^{Gen}$ Jets',
    'scPuppiL1TSC4NGJetJets_raw': 'SC4 NG',
    'scPuppiL1TSC4NGJetJets_jecs': 'SC4 NG JECs',
    'scPuppiExtendedJets_raw': 'SC4 Raw',
    'scPuppiExtendedJets_jecs': 'SC4 JECs',
    'genjets': 'GenJets',
}

COLORS_DICT = {
    'scPuppiL1TSC4NGJetJets_raw': '#964a8b',
    'scPuppiL1TSC4NGJetJets_jecs': '#7a21dd',
    'scPuppiExtendedJets_raw': '#f89c20',
    'scPuppiExtendedJets_jecs': '#e42536',
    'genjets': 'gray',
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

ETA_BINS = [0, 1.3, 1.7, 1.9, 2.1, 2.4, 2.8, 3.0, 3.3, 3.6, 4.0, 4.8]
