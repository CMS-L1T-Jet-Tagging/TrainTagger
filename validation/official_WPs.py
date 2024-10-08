#To store the models' official calulated working points
from types import SimpleNamespace

WPs = {
    #Working points when pT is corrected
    'tau_ptCorrected': 0.17,
    'tau_l1_pt_ptCorrected': 34, #GeV

    'tau_ptCorrected_mult': 0.008,
    'tau_l1_pt_ptCorrected_mult': 26, #GeV

    #Working points when pT is not corrected
    'tau_ptUncorrected': 0.12,
    'tau_l1_pt_ptUncorrected': 34, #GeV

    'tau_ptUncorrected_mult': 0.003,
    'tau_l1_pt_ptUncorrected_mult': 26, #GeV

    'btag':0.95,
    'btag_l1_ht': 220,
}

WPs_CMSSW = {
    'tau': 0.22,
    'tau_l1_pt': 34,

    #Seededcone reco pt cut
    #From these slides: https://indico.cern.ch/event/1380964/contributions/5852368/attachments/2841655/4973190/AnnualReview_2024.pdf
    'l1_pt_sc_barrel': 164, #GeV
    'l1_pt_sc_endcap':121, #GeV

    'btag': 2.32 ,
    'btag_l1_ht': 220,
}