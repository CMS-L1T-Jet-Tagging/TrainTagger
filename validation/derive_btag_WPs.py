import sys, gc
import uproot4
import numpy as np
import awkward as ak
from argparse import ArgumentParser
from qkeras.utils import load_qmodel

# Add path so the script sees other modules
sys.path.append('../')
from datatools.createDataset import dict_fields
from datatools import helpers
from hist import Hist
import hist

#Plotting
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

def extract_nn_inputs(minbias, nconstit=16, n_entries=None):
    """
    Extract NN inputs based on input_fields_tag
    """

    features = [
    'pt_rel_phys','deta_phys','dphi_phys',
    'pt_log','eta_phys','phi_phys', 'mass',
    'isPhoton', 'isElectronPlus', 'isElectronMinus', 'isMuonPlus', 'isMuonMinus', 'isNeutralHadron', 'isChargedHadronPlus', 'isChargedHadronMinus',
    'z0', 'dxySqrt',
    'isfilled',
    'puppiweight', 'emid', 'quality']

    #Concatenate all the inputs
    inputs_list = []

    #Vertically stacked them to create input sets
    #https://awkward-array.org/doc/main/user-guide/how-to-restructure-concatenate.html
    #Also pad and fill them with 0 to the number of constituents we are using (nconstit)
    for i in range(len(features)):

        if features[i] != "dxySqrt":
            field = f"jet_pfcand_{features[i]}"
            field_array = helpers.extract_array(minbias, field, n_entries)
        else:
            field = f"jet_pfcand_dxy_custom"
            dxy_custom = helpers.extract_array(minbias, field, n_entries)
            field_array = np.nan_to_num(np.sqrt(dxy_custom), nan=0.0, posinf=0., neginf=0.)

        padded_filled_array = helpers.pad_fill(field_array, nconstit)
        inputs_list.append(padded_filled_array[:, np.newaxis])

    inputs = ak.concatenate(inputs_list, axis=1)

    return inputs

def find_rate(rate_list, target_rate = 14, RateRange = 0.05):
    
    idx_list = []
    
    for i in range(len(rate_list)):
        if target_rate-RateRange <= rate_list[i] <= target_rate + RateRange:
            idx_list.append(i)
            
    return idx_list

def plot_rate(rate_list, ht_list, nn_list, target_rate = 14, plot_name="btag_rate_scan_cmwssw.pdf", correct_pt=True):
    
    fig, ax = plt.subplots()
    im = ax.scatter(nn_list, ht_list, c=rate_list, s=500, marker='s',
                    cmap='Spectral_r',
                    linewidths=0,
                    norm=matplotlib.colors.LogNorm())

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r'4-b rate [kHZ]')

    plt.ylabel(r"HT [GeV]")
    plt.xlabel(r"$\sum_{4~leading~jets}$ b scores")
    
    plt.xlim([0,1.5])
    plt.ylim([10,500])

    #plus, minus range
    RateRange = 0.5

    #Find the target rate points, plot them and print out some info as well
    target_rate_idx = find_rate(rate_list, target_rate = target_rate, RateRange=RateRange)
    
    legend_count = 0
    for i in target_rate_idx:
        print("Rate: ", rate_list[i])
        print("NN Cut: ", nn_list[i])
        print("ht Cut: ", ht_list[i])
        print("------")
        
        if legend_count == 0:
            plt.scatter(nn_list[i], ht_list[i], s=600, marker='*',
                        color ='firebrick', label = r"${} \pm {}$ kHz".format(target_rate, RateRange))
        else:
            plt.scatter(nn_list[i], ht_list[i], s=600, marker='*',
                        color ='firebrick')
            
        legend_count += 1
    
    plt.legend(loc='upper right')
    plt.savefig(f'plots/{plot_name}', bbox_inches='tight')

def derive_btag_rate_cmssw(minbias_path, n_entries=500000, target_rate=14, tree='jetntuple/Jets'):
    """
    Derive btag rate in cmssw emulator using n_entries.
    """

    minbias_rate = 32e+3 #32 kHZ

    #Load the minbias data
    minbias = uproot4.open(minbias_path)[tree]

    raw_event_id = helpers.extract_array(minbias, 'event', n_entries)
    raw_jet_pt = helpers.extract_array(minbias, 'jet_pt', n_entries)
    raw_b_score = helpers.extract_array(minbias, 'jet_bjetscore', n_entries)

    #Count number of total event
    n_events = len(np.unique(raw_event_id))
    print("Total number of minbias events: ", n_events)

    #Group these attributes by event id, and filter out groups that don't have at least 2 elements
    event_id, grouped_arrays  = helpers.group_id_values(raw_event_id, raw_jet_pt, raw_b_score, num_elements = 4)

    # Extract the grouped arrays
    # Jet pt is already sorted in the producer, no need to do it here
    jet_pt, b_score = grouped_arrays

    ht = ak.sum(jet_pt, axis=1)
    bscore_sum = ak.sum(b_score[:,:4], axis=1) #Only sum up the first four

    #Define the histograms (pT edge and NN Score edge)
    ht_edges = list(np.arange(0,500,2)) + [10000] #Make sure to capture everything
    NN_edges = list([round(i,2) for i in np.arange(0, 4.01, 0.02)])

    RateHist = Hist(hist.axis.Variable(ht_edges, name="ht", label="ht"),
                    hist.axis.Variable(NN_edges, name="nn", label="nn"))

    RateHist.fill(ht = ht, nn = bscore_sum)

    #Derive the rate
    rate_list = []
    ht_list = []
    nn_list = []

    #Loop through the edges and integrate
    for ht in ht_edges[:-1]:
        for NN in NN_edges[:-1]:
            
            #Calculate the rate
            rate = RateHist[{"ht": slice(ht*1j, ht_edges[-1]*1.0j, sum)}][{"nn": slice(NN*1.0j,4.0j, sum)}]/n_events
            rate_list.append(rate*minbias_rate)

            #Append the results   
            ht_list.append(ht)
            nn_list.append(NN)

    plot_rate(rate_list, ht_list, nn_list, target_rate=target_rate, correct_pt=False)

def derive_btag_rate(model, minbias_path, n_entries=500000, target_rate=14, tree='jetntuple/Jets'):
    """
    Derive btag-rate for model.
    """

    minbias_rate = 32e+3 #32 kHZ

    #Load the minbias data
    minbias = uproot4.open(minbias_path)[tree]

    raw_event_id = helpers.extract_array(minbias, 'event', n_entries)
    raw_jet_pt = helpers.extract_array(minbias, 'jet_pt', n_entries)
    raw_inputs = extract_nn_inputs(minbias, nconstit=16, n_entries=n_entries)

    #Count number of total event
    n_events = len(np.unique(raw_event_id))
    print("Total number of minbias events: ", n_events)

    #Group these attributes by event id, and filter out groups that don't have at least 2 elements
    event_id, grouped_arrays  = helpers.group_id_values(raw_event_id, raw_jet_pt, raw_inputs, num_elements=4)

    # Extract the grouped arrays
    # Jet pt is already sorted in the producer, no need to do it here
    jet_pt, jet_nn_inputs = grouped_arrays

    #Btag input list for first 4 jets
    btag_inputs = [np.asarray(jet_nn_inputs[:, i]).transpose(0, 2, 1) for i in range(0,4)]
    nn_outputs = [model.predict(nn_input) for nn_input in btag_inputs]

    b_index = 1

    bscore_sum = sum([pred_score[0][:, b_index] for pred_score in nn_outputs])
    ht = ak.sum(jet_pt, axis=1)

    assert(len(bscore_sum) == len(ht))

        #Define the histograms (pT edge and NN Score edge)
    ht_edges = list(np.arange(0,500,2)) + [10000] #Make sure to capture everything
    NN_edges = list([round(i,2) for i in np.arange(0, 1.51, 0.01)])

    RateHist = Hist(hist.axis.Variable(ht_edges, name="ht", label="ht"),
                    hist.axis.Variable(NN_edges, name="nn", label="nn"))

    RateHist.fill(ht = ht, nn = bscore_sum)

    #Derive the rate
    rate_list = []
    ht_list = []
    nn_list = []

    #Loop through the edges and integrate
    for ht in ht_edges[:-1]:
        for NN in NN_edges[:-1]:
            
            #Calculate the rate
            rate = RateHist[{"ht": slice(ht*1j, ht_edges[-1]*1.0j, sum)}][{"nn": slice(NN*1.0j,4.0j, sum)}]/n_events
            rate_list.append(rate*minbias_rate)

            #Append the results   
            ht_list.append(ht)
            nn_list.append(NN)

    plot_rate(rate_list, ht_list, nn_list, target_rate=target_rate,  plot_name="btag_rate_scan.pdf", correct_pt=False)



if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('-m','--model', default='/eos/user/s/sewuchte/L1Trigger/ForDuc/trainings_regression_weighted/2024_08_17_v6_extendedAll200_btgc_ext7_QDeepSets_PermutationInv_nconst_16_nfeatures_21_nbits_8_pruned/model_QDeepSets_PermutationInv_nconst_16_nfeatures_21_nbits_8_pruned.h5' , help = 'Input model for plotting')    
    args = parser.parse_args()

    model=load_qmodel(args.model)
    print(model.summary())

    #These paths are default to evaluate some of the rate
    minbias_path = '/eos/user/s/sewuchte/L1Trigger/ForDuc/nTuples/MinBias_PU200.root'

    derive_btag_rate(model, minbias_path, n_entries=None)