#Plotting
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import mplhep as hep
plt.style.use(hep.style.ROOT)
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'medium',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'medium',
         'ytick.labelsize':'medium'}
pylab.rcParams.update(params)

#GLOBAL VARIABLES TO USE ACROSS PLOTTING TOOLS
MINBIAS_RATE = 32e+3 #32 kHZ

# Define pT bins
PT_BINS = np.array([
    15, 17, 19, 22, 25, 30, 35, 40, 45, 50,
    60, 76, 97, 122, 154, 195, 246, 311,
    393, 496, 627, 792, 1000
])

WPs_CMSSW = {
    #Tau working points as defined here
    #https://github.com/cms-sw/cmssw/blob/e9b58bef8b37e6113ba31a03429ffc4c300adb12/DataFormats/L1TParticleFlow/interface/PFTau.h#L16-L17
    'tau': 0.22,
    'tau_l1_pt': 34,

    #Seededcone reco pt cut
    #From these slides: https://indico.cern.ch/event/1380964/contributions/5852368/attachments/2841655/4973190/AnnualReview_2024.pdf
    'l1_pt_sc_barrel': 164, #GeV
    'l1_pt_sc_endcap':121, #GeV

    #Slide 19 here: https://indico.cern.ch/event/1380964/contributions/5852368/attachments/2841655/4973190/AnnualReview_2024.pdf
    'btag': 2.32,
    'btag_l1_ht': 220,
}

#FUNCTIONS
def find_rate(rate_list, target_rate = 14, RateRange = 0.05):
    
    idx_list = []
    
    for i in range(len(rate_list)):
        if target_rate-RateRange <= rate_list[i] <= target_rate + RateRange:
            idx_list.append(i)
            
    return idx_list    

def plot_ratio(all_events, selected_events, plot=False):
    fig = plt.figure(figsize=(10, 12))
    _, eff = selected_events.plot_ratio(all_events,
                                        rp_num_label="Selected events", rp_denom_label=r"All",
                                        rp_uncert_draw_type="bar", rp_uncertainty_type="efficiency")

    plt.show(block=False)
    return eff

def get_bar_patch_data(artists):
    x_data = [artists.bar.patches[i].get_x() for i in range(len(artists.bar.patches))]
    y_data = [artists.bar.patches[i].get_y() for i in range(len(artists.bar.patches))]
    err_data = [artists.bar.patches[i].get_height() for i in range(len(artists.bar.patches))]
    return x_data, y_data, err_data
