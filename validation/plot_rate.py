import sys
import uproot4
import numpy as np
from argparse import ArgumentParser
from qkeras.utils import load_qmodel

#Import the calculated working points
from official_WPs import WPs, WPs_CMSSW
sys.path.append('../')
from datatools.createDataset import dict_fields
from datatools import helpers

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

def plot_bkg_rate_tau(model, minbias_path, model_WP=WPs["tau"], tree='jetntuple/Jets', n_entries=500000):
    """
    Plot the background (mimbias) rate w.r.t pT cuts.
    """

    minbias_rate = 32e+3 #32 kHZ
    pt_cuts = list(np.arange(0,250,10)) 

    #Load the minbias data
    minbias = uproot4.open(minbias_path)[tree]

    #Prepare the nn inputs, and flip the last two axes
    nn_inputs = np.asarray(helpers.extract_nn_inputs(minbias, input_fields_tag='ext3', nconstit=16, n_entries=n_entries)).transpose(0, 2, 1)
    
    #Get the NN predictions
    tau_index = [2,3]
    pred_score, ratio = model.predict(nn_inputs)
    model_tau = pred_score[:, tau_index[0]] + pred_score[:, tau_index[1]]

    #Emulator tau score
    cmssw_tau = helpers.extract_array(minbias, 'jet_tauscore', n_entries)

    #Use event id to track which jets belong to which event.
    event_id = helpers.extract_array(minbias, 'event', n_entries)
    event_id_cmssw = event_id[cmssw_tau > WPs_CMSSW["tau"]]
    event_id_model = event_id[model_tau > WPs["tau"]]

    #Cut on jet pT to extract the rate
    jet_pt = helpers.extract_array(minbias, 'jet_pt', n_entries)
    jet_pt_cmssw = helpers.extract_array(minbias, 'jet_taupt', n_entries)[cmssw_tau > WPs_CMSSW["tau"]]
    jet_pt_model = (jet_pt*ratio.flatten())[model_tau > WPs["tau"]]

    #Total number of unique event
    n_event = len(np.unique(event_id))
    minbias_rate_no_nn = []
    minbias_rate_cmssw = []
    minbias_rate_model = []

    for pt_cut in pt_cuts:

        print("pT Cut: ", pt_cut)
        n_pass_no_nn = len(np.unique(event_id[jet_pt > pt_cut]))
        n_pass_cmssw = len(np.unique(event_id_cmssw[jet_pt_cmssw > pt_cut]))
        n_pass_model = len(np.unique(event_id_model[jet_pt_model > pt_cut]))
        print('------------')

        minbias_rate_no_nn.append((n_pass_no_nn/n_event)*minbias_rate)
        minbias_rate_cmssw.append((n_pass_cmssw/n_event)*minbias_rate)
        minbias_rate_model.append((n_pass_model/n_event)*minbias_rate)

    plt.plot(pt_cuts, minbias_rate_no_nn, label=r'No ID/$p_T$ correction', linewidth = 5)
    plt.plot(pt_cuts, minbias_rate_cmssw, label=r'CMSSW PuppiTau Emulator', linewidth = 5)
    plt.plot(pt_cuts, minbias_rate_model, label=r'SeedCone Tau', linewidth = 5)
    hep.cms.text("Phase 2 Simulation")
    hep.cms.lumitext("PU 200 (14 TeV)")
    plt.yscale('log')
    plt.ylabel(r"$\tau_h$ trigger rate [kHz]")
    plt.xlabel(r"Offline $p_T$ [GeV]")
    plt.legend(loc = 'upper right',fontsize = 15)
    plt.savefig('plots/bkg_rate_tau.pdf', bbox_inches='tight')


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('-m','--model', default='/eos/user/s/sewuchte/L1Trigger/ForDuc/trainings_regression_weighted/2024_07_25_v10_extendedAll200_btgc_ext3_QDeepSets_PermutationInv_nconst_16_nfeatures_21_nbits_8_pruned/model_QDeepSets_PermutationInv_nconst_16_nfeatures_21_nbits_8_pruned.h5' , help = 'Input model for plotting')    
    args = parser.parse_args()

    model=load_qmodel(args.model)
    print(model.summary())

    #These paths are default to evaluate some of the rate
    minbias_path = '/eos/user/s/sewuchte/L1Trigger/ForDuc/nTuples/MinBias_PU200.root'

    plot_bkg_rate_tau(model, minbias_path)
