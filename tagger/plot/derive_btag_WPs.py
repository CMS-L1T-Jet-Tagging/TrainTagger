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

def find_rate(rate_list, target_rate = 14, RateRange = 0.05):
    
    idx_list = []
    
    for i in range(len(rate_list)):
        if target_rate-RateRange <= rate_list[i] <= target_rate + RateRange:
            idx_list.append(i)
            
    return idx_list

def plot_rate(rate_list, ht_list, nn_list, target_rate = 14, plot_name="btag_rate_scan_cmwssw.pdf", correct_pt=True, cmssw=False):
    
    fig, ax = plt.subplots()
    im = ax.scatter(nn_list, ht_list, c=rate_list, s=500, marker='s',
                    cmap='Spectral_r',
                    linewidths=0,
                    norm=matplotlib.colors.LogNorm())

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r'4-b rate [kHZ]')

    plt.ylabel(r"HT [GeV]")
    plt.xlabel(r"$\sum_{4~leading~jets}$ b scores")
    
    if cmssw:
        plt.xlim([0,3.5])
        plt.ylim([10,500])
    else:
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

    plot_rate(rate_list, ht_list, nn_list, target_rate=target_rate, correct_pt=False, cmssw=True)

def derive_btag_rate(model, minbias_path, n_entries=500000, target_rate=14, tree='jetntuple/Jets'):
    """
    Derive btag-rate for model.
    """

    minbias_rate = 32e+3 #32 kHZ

    #Load the minbias data
    minbias = uproot4.open(minbias_path)[tree]

    raw_event_id = helpers.extract_array(minbias, 'event', n_entries)
    raw_jet_pt = helpers.extract_array(minbias, 'jet_pt', n_entries)
    raw_inputs = helpers.extract_nn_inputs(minbias, input_fields_tag='ext7', nconstit=16, n_entries=n_entries)

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
    parser.add_argument('-m','--model', default='/eos/home-s/sewuchte/www/L1T/trainings_regression_weighted/2024_08_27_v4_extendedAll200_btgc_ext7_QDeepSets_PermutationInv_nconst_16_nfeatures_21_nbits_8_pruned/model_QDeepSets_PermutationInv_nconst_16_nfeatures_21_nbits_8_pruned.h5' , help = 'Input model for plotting')    
    parser.add_argument('--cmssw', action='store_true', help='Derive the btag rate for CMSSW')
    args = parser.parse_args()

    model=load_qmodel(args.model)
    print(model.summary())

    #These paths are default to evaluate some of the rate
    minbias_path = '/eos/cms/store/cmst3/group/l1tr/sewuchte/l1teg/fp_ntuples_v131Xv9/extendedTRK_HW_260824/MinBias_PU200.root'

    if args.cmssw: derive_btag_rate_cmssw(minbias_path, n_entries=3000000)
    else: derive_btag_rate(model, minbias_path, n_entries=3000000)