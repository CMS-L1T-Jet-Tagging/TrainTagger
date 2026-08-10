
from turtle import color


JET_FEATURE_FIELDS = [
    'event',
    'jet_pt', 'jet_eta', 'jet_phi', 'jet_npuppicand',
    'jet_pt_phys', 'jet_eta_phys', 'jet_phi_phys',
    'jet_genmatch_pt', 'jet_genmatch_eta', 'jet_genmatch_phi',
    'jet_genmatch_hflav', 'jet_genmatch_pflav',
    'target_pt', 'target_pt_phys', 'jet_reject',
    'genHT',
]
SAVE_FIELDS = ['nn_inputs', 'class_label', 'target_pt', 'target_pt_phys'] + JET_FEATURE_FIELDS

PROCESS_INFO = {
    'MinBias':             {'id':  0, 'path': 'MinBias_PU200.root',                 'fraction': 20, 'class':  0, 'label': 'MinBias',      'color': "#000000", 'type': 'SM',  'where': 'both'},

    # Standard Model processes
    'QCD_Pt15To3000':      {'id':  1, 'path': 'QCD_Pt15To3000_PU200.root',          'fraction': 20, 'class':  1, 'label': 'QCD',          'color': '#7f7f7f', 'type': 'SM',  'where': 'test'},
    'QCD_Pt20To30':        {'id':  2, 'path': 'QCD_Pt20To30_PU200.root',            'fraction': 20, 'class':  1, 'label': 'QCD',          'color': '#7f7f7f', 'type': 'SM',  'where': 'train'},
    'QCD_Pt30To50':        {'id':  3, 'path': 'QCD_Pt30To50_PU200.root',            'fraction': 20, 'class':  1, 'label': 'QCD',          'color': '#7f7f7f', 'type': 'SM',  'where': 'train'},
    'QCD_Pt50To80':        {'id':  4, 'path': 'QCD_Pt50To80_PU200.root',            'fraction': 20, 'class':  1, 'label': 'QCD',          'color': '#7f7f7f', 'type': 'SM',  'where': 'train'},
    'QCD_Pt80To120':       {'id':  5, 'path': 'QCD_Pt80To120_PU200.root',           'fraction': 20, 'class':  1, 'label': 'QCD',          'color': '#7f7f7f', 'type': 'SM',  'where': 'train'},
    'QCD_Pt120To170':      {'id':  6, 'path': 'QCD_Pt120To170_PU200.root',          'fraction': 20, 'class':  1, 'label': 'QCD',          'color': '#7f7f7f', 'type': 'SM',  'where': 'train'},
    'QCD_Pt170To300':      {'id':  7, 'path': 'QCD_Pt170To300_PU200.root',          'fraction': 20, 'class':  1, 'label': 'QCD',          'color': '#7f7f7f', 'type': 'SM',  'where': 'train'},
    'QCD_Pt300To470':      {'id':  8, 'path': 'QCD_Pt300To470_PU200.root',          'fraction': 20, 'class':  1, 'label': 'QCD',          'color': '#7f7f7f', 'type': 'SM',  'where': 'train'},
    'QCD_Pt470To600':      {'id':  9, 'path': 'QCD_Pt470To600_PU200.root',          'fraction': 20, 'class':  1, 'label': 'QCD',          'color': '#7f7f7f', 'type': 'SM',  'where': 'train'},
    'QCD_Pt600ToInf':      {'id': 10, 'path': 'QCD_Pt600ToInf_PU200.root',          'fraction': 20, 'class':  1, 'label': 'QCD',          'color': '#7f7f7f', 'type': 'SM',  'where': 'train'},
    'DY_M10To50':          {'id': 11, 'path': 'DYToLL_M10To50_PU200.root',          'fraction': 20, 'class':  2, 'label': 'DY',           'color': '#7f7f7f', 'type': 'SM',  'where': 'train'},
    'DY_M50':              {'id': 12, 'path': 'DYToLL_M50_PU200.root',              'fraction': 20, 'class':  2, 'label': 'DY',           'color': '#7f7f7f', 'type': 'SM',  'where': 'train'},
    'DY_M50_PTLL200To400': {'id': 13, 'path': 'DYToLL_M50_PTLL200To400_PU200.root', 'fraction': 20, 'class':  2, 'label': 'DY',           'color': '#ff7f0e', 'type': 'SM',  'where': 'train'},
    'DY_M50_PTLL400To600': {'id': 14, 'path': 'DYToLL_M50_PTLL400To600_PU200.root', 'fraction': 20, 'class':  2, 'label': 'DY',           'color': '#ff7f0e', 'type': 'SM',  'where': 'train'},
    'DY_M50_PTLL600ToInf': {'id': 15, 'path': 'DYToLL_M50_PTLL600ToInf_PU200.root', 'fraction': 20, 'class':  2, 'label': 'DY',           'color': '#ff7f0e', 'type': 'SM',  'where': 'train'},
    'Wjets':               {'id': 16, 'path': 'WJetsToLNu_PU200.root',              'fraction': 20, 'class':  3, 'label': 'Wjets',        'color': '#1b9e77', 'type': 'SM',  'where': 'train'},
    'TT':                  {'id': 17, 'path': 'TT_PU200.root',                      'fraction': 20, 'class':  4, 'label': 'TTbar',        'color': '#2ca02c', 'type': 'SM',  'where': 'both'},

    # Signals
    'HH_4b':               {'id': 18, 'path': 'GluGluHHTo4B_PU200.root',            'fraction': 20, 'class': 10, 'label': 'HH4b',              'color': '#1f77b4', 'type': 'SM',  'where': 'test'},
    'HH_2b2tau':           {'id': 19, 'path': 'GluGluHHTo2B2Tau_PU200.root',        'fraction': 20, 'class': 11, 'label': 'HH2b2tau',          'color': '#17becf', 'type': 'SM',  'where': 'test'},
    'VBFHToInvisible':     {'id': 20, 'path': 'VBFHToInvisible_PU200.root',         'fraction': 20, 'class': 12, 'label': 'VBF_HtoInv',        'color': '#7f7f7f', 'type': 'BSM', 'where': 'test'},

    # BSM
    # 'XtoHH_MX_140To780':   {'id': 21, 'path': 'XtoHH_MX_140To780_PU200.root',       'fraction': 20, 'class': 13, 'label': 'XtoHH_Mlow',     'color': '#e377c2', 'type': 'BSM', 'where': 'test'},
    # 'XtoHH_MX_500To1000':  {'id': 22, 'path': 'XtoHH_MX_500To1000_PU200.root',      'fraction': 20, 'class': 14, 'label': 'XtoHH_M500to1000',  'color': '#9467bd', 'type': 'BSM', 'where': 'test'},
    # 'XtoHH_MX_1000To4000': {'id': 23, 'path': 'XtoHH_MX_1000To4000_PU200.root',     'fraction': 20, 'class': 15, 'label': 'XtoHH_Mhigh',    'color': '#9467bd', 'type': 'BSM', 'where': 'test'},
    'SMJ_cascadeA':        {'id': 24, 'path': 'SMJ_cascadeA.root',                  'fraction': 20, 'class': 16, 'label': 'SMJ_cascadeA',   'color': '#ffbb78', 'type': 'BSM', 'where': 'test'},
    'SMJ_cascadeC':        {'id': 25, 'path': 'SMJ_cascadeC.root',                  'fraction': 20, 'class': 17, 'label': 'SMJ_cascadeC',      'color': '#ff7f0e', 'type': 'BSM', 'where': 'test'},
    'SVJ_250':             {'id': 26, 'path': 'SVJ_250.root',                       'fraction': 20, 'class': 18, 'label': 'SVJ_250',           'color': '#ff9896', 'type': 'BSM', 'where': 'test'},
    'SVJ_500':             {'id': 27, 'path': 'SVJ_500.root',                       'fraction': 20, 'class': 19, 'label': 'SVJ_500',           'color': '#d62728', 'type': 'BSM', 'where': 'test'},
    'SUEP':                {'id': 28, 'path': 'SUEP.root',                          'fraction': 20, 'class': 20, 'label': 'SUEP',           'color': '#8E44AD', 'type': 'BSM', 'where': 'test'},
    # 'ZprimeToTauTau_M500': {'id': 29, 'path': 'ZprimeToTauTau_M500_PU200.root',     'fraction': 20, 'class': 21, 'label': 'ZpTo2tau_M500',  'color': '#8c564b', 'type': 'BSM', 'where': 'test'},
    # 'ZprimeToTauTau_M1500':{'id': 30, 'path': 'ZprimeToTauTau_M1500_PU200.root',    'fraction': 20, 'class': 22, 'label': 'ZpTo2tau_M1500', 'color': '#c49c94', 'type': 'BSM', 'where': 'test'},
}

process_info_order = [
    'MinBias', 'QCD', 'DY', 'Wjets', 'TT', 
    'HH_4b', 'HH_2b2tau', 'VBFHToInvisible', 
    'XtoHH_MX_140To780', 'XtoHH_MX_500To1000', 'XtoHH_MX_1000To4000',
    'SMJ_cascadeA', 'SMJ_cascadeC', 'SVJ_250', 'SVJ_500', 'SUEP',
    'ZprimeToTauTau_M500', 'ZprimeToTauTau_M1500'
]
def load_process_info(order=process_info_order):
    """
    Load the process information from the configuration.
    Returns a dictionary with process names as keys and their corresponding information as values.
    """
    process_info = PROCESS_INFO.copy()

    # Samples with multiple files need to be combined into a single entry for each process type for plotting
    if any(k.startswith('QCD_') for k in process_info.keys()):
        process_info['QCD'] = {'path': None, 'fraction': 0, 'class': 1, 'label': 'QCD', 'color': '#7f7f7f', 'type': 'SM', 'where': 'test'} 
    if any(k.startswith('DY_') for k in process_info.keys()):
        process_info['DY'] = {'path': None, 'fraction': 0, 'class': 2, 'label': 'DY', 'color': '#1f77b4', 'type': 'SM', 'where': 'test'}
    process_info = {k: v for k, v in process_info.items() if 'QCD_' not in k and 'DY_' not in k}
    
    process_info = {k: process_info[k] for k in order if k in process_info}
    return process_info

JET_INFO = {
    'unmatched': {'class': -1, 'label': 'unmatched', 'color': '#7f7f7f'},
    'b': {'class': 0, 'label': 'b', 'color': '#d62728'},
    'charm': {'class': 1, 'label': 'charm', 'color': '#1f77b4'},
    'light': {'class': 2, 'label': 'light', 'color': '#2ca02c'},
    'gluon': {'class': 3, 'label': 'gluon', 'color': '#9467bd'},
    'taup': {'class': 4, 'label': 'tau+', 'color': '#8c564b'},
    'taum': {'class': 5, 'label': 'tau-', 'color': '#c49c94'},
    'muon': {'class': 6, 'label': 'muon', 'color': '#e377c2'},
    'electron': {'class': 7, 'label': 'electron', 'color': '#17becf'},
}