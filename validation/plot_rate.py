import uproot4
import numpy as np
from argparse import ArgumentParser
from qkeras.utils import load_qmodel

#Import the calculated working points
from official_WPs import WPs, WPs_CMSSW

#Plotting
import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.ROOT)
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'medium',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'medium',
         'ytick.labelsize':'medium'}
pylab.rcParams.update(params)

#line thickness
import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 5

def plot_bkg_rate_tau(model, minbias_path, model_WP=WPs.tau, tree='jetntuple/Jets'):
    """
    Plot the background (mimbias) rate w.r.t pT cuts.
    """

    minbias_rate = 32e+3 #32 kHZ
    pt_cuts = list(np.arange(0,250,10)) 

    #Load the minbias data
    minbias = uproot4.open(minbias_path)[tree]

    #Emulator tau score
    cmssw_tau = np.asarray(minbias['jet_tauscore'].array())

    #Use event id to track which jets belong to which event.
    event_id = np.asarray(minbias['event'].array())
    event_id_cmssw = event_id[cmssw_tau > WPs_CMSSW.tau]

    #Cut on jet pT to extract the rate
    jet_pt = np.asarray(minbias['jet_pt'].array())
    jet_pt_cmssw = np.asarray(minbias['jet_taupt'].array())[cmssw_tau > WPs_CMSSW.tau]

    #Total number of unique event
    n_event = np.unique(event_id).shape[0]

    minbias_rate_no_nn = []
    minbias_rate_cmssw = []

    for pt_cut in pt_cuts:

        print("pT Cut: ", pt_cut)
        n_pass_no_nn = np.unique(event_id[jet_pt > pt_cut]).shape[0]
        n_pass_cmssw = np.unique(event_id_cmssw[jet_pt_cmssw > pt_cut]).shape[0]
        print('------------')

        minbias_rate_no_nn.append((n_pass_no_nn/n_event)*minbias_rate)
        minbias_rate_cmssw.append((n_pass_cmssw/n_event)*minbias_rate)

    plt.plot(pt_cuts, minbias_rate_no_nn, label=r'No ID/$p_T$ correction', linewidth = 5)
    plt.plot(pt_cuts, minbias_rate_cmssw, label=r'CMSSW PuppiTau Emulator', linewidth = 5)
    hep.cms.text("Phase 2 Simulation")
    hep.cms.lumitext("PU 200 (14 TeV)")
    plt.yscale('log')
    plt.ylabel(r"$\tau_h$ trigger rate [kHz]")
    plt.xlabel(r"Offline $p_T$ [GeV]")
    plt.legend(loc = 'upper right',fontsize = 15)
    plt.savefig('plots/bkg_rate_tau.pdf', bbox_inches='tight')


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('-m','--model', default='../models/model_QDeepSets_PermutationInv_nconst_16_nfeatures_8_nbits_8_pruned.h5' , help = 'Input model for plotting')    
    args = parser.parse_args()

    model=load_qmodel(args.model)

    #These paths are default to evaluate some of the rate
    minbias_path = '/eos/user/s/sewuchte/L1Trigger/ForDuc/nTuples/MinBias_PU200.root'

    plot_bkg_rate_tau(model, minbias_path)
