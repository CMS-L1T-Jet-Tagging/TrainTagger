#Plotting
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
import mplhep as hep
import style 

style.set_style()
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

def plot_2d(variable_one,variable_two,range_one,range_two,name_one,name_two,title):
    fig,ax = plt.subplots(1,1,figsize=(18,15))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
    
    hist2d = ax.hist2d(variable_one, variable_two, range=(range_one,range_two), bins=50, norm=matplotlib.colors.LogNorm(),cmap=style.colormap)
    ax.set_xlabel(name_one, horizontalalignment='right', x=1.0)
    ax.set_ylabel(name_two, horizontalalignment='right', y=1.0)
    cbar = plt.colorbar(hist2d[3] , ax=ax)
    cbar.set_label('# Tracks')
    plt.suptitle(title)
    plt.tight_layout()
    return fig

def plot_histo(variable,name,title,xlabel,ylabel,range=(0,1)):
    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(18,15))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
    for i,histo in enumerate(variable):

        ax.hist(histo,bins=50,range=range,histtype="step",
                    color = colours[i],
                    label=name[i],
                    density=True)    
    ax.grid(True)
    ax.set_xlabel(xlabel,ha="right",x=1)
    ax.set_ylabel(ylabel,ha="right",y=1)
    ax.legend(loc='best')

    plt.suptitle(title)
    plt.tight_layout()
    return fig

def plot_roc(modelsAndNames,truthclass,keys = ["Emulation","Tensorflow","hls4ml"],labels = ["CMSSW Emulation", "Tensorflow", "hls4ml"],title="None",colours=style.colours):
    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(18,15))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)

    for i,key in enumerate(keys):
        tpr = modelsAndNames[key]["ROCs"]["tpr"]
        fpr = modelsAndNames[key]["ROCs"]["fpr"]
        auc1 = modelsAndNames[key]["ROCs"]["auc"]
        ax.plot(tpr[truthclass],fpr[truthclass],label='%s Tagger, AUC = %.2f%%'%(labels[i], auc1[truthclass]*100.),color=colours[i])
    ax.semilogy()
    ax.set_xlabel("Signal efficiency")
    ax.set_ylabel("Mistag rate")
    ax.set_xlim(0.,1.)
    ax.set_ylim(0.001,1)
    ax.grid(True)
    ax.legend(loc='best')
    plt.suptitle(title)
    plt.tight_layout()
    return fig
