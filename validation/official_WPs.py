#To store the models' official calulated working points
from types import SimpleNamespace

WPs = {
    'tau': 0.15,
    'tau_l1_pt': 38, #GeV
}

WPs_CMSSW = {
    'tau': 0.22,
    'tau_l1_pt': 34,

    #Seededcone reco pt cut
    #From these slides: https://indico.cern.ch/event/1380964/contributions/5852368/attachments/2841655/4973190/AnnualReview_2024.pdf
    'l1_pt_sc_barrel': 164, #GeV
    'l1_pt_sc_endcap':121, #GeV
}