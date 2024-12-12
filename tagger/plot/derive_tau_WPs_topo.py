'''
Scripts for deriving the working points for taus.

Usage: 

python derive_tau_WPs.py
'''
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
import mplhep as hep
from tagger.plot.style import *

set_style()

def calculate_topo_score(tau_plus, tau_minus):
    #2=tau positive, 3=tau_negative

    p_pos = tau_plus[:,0] + tau_plus[:,1]
    p_neg = tau_minus[:,0] + tau_minus[:,1]

    return np.multiply(p_pos, p_neg)

    
def delta_r(eta1, phi1, eta2, phi2):
    """
    Calculate the delta R between two sets of eta and phi values.
    """
    delta_eta = eta1 - eta2
    delta_phi = phi1 - phi2

    # Ensure delta_phi is within -pi to pi
    delta_phi = (delta_phi + np.pi) % (2 * np.pi) - np.pi
    return np.sqrt(delta_eta**2 + delta_phi**2)

def find_rate(rate_list, target_rate = 28):
    
    RateRange = 1.5 #kHz
    
    idx_list = []
    
    for i in range(len(rate_list)):
        if target_rate-RateRange <= rate_list[i] <= target_rate + RateRange:
            idx_list.append(i)
            
    return idx_list

def plot_rate(rate_list, pt_list, nn_list, target_rate = 28, correct_pt=True):
    
    fig, ax = plt.subplots()
    im = ax.scatter(nn_list, pt_list, c=rate_list, s=500, marker='s',
                    cmap='Spectral_r',
                    linewidths=0,
                    norm=matplotlib.colors.LogNorm())

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r'Di-$\tau_h$ rate [kHZ]')

    plt.ylabel(r"Min reco $p_T$ [GeV]")
    plt.xlabel(r"Tau Topology Score")

    plt.xlim([0,0.6])
    plt.ylim([10,100])
    
    #Find the target rate points, plot them and print out some info as well
    target_rate_idx = find_rate(rate_list, target_rate = target_rate)
    
    legend_count = 0
    for i in target_rate_idx:
        print("Rate: ", rate_list[i])
        print("NN Cut: ", nn_list[i])
        print("pt Cut: ", pt_list[i])
        print("------")
        
        if legend_count == 0:
            plt.scatter(nn_list[i], pt_list[i], s=600, marker='*',
                        color ='firebrick', label = r"${} \pm 1.5$ kHz".format(target_rate))
        else:
            plt.scatter(nn_list[i], pt_list[i], s=600, marker='*',
                        color ='firebrick')
            
        legend_count += 1
    
    plt.legend(loc='upper right')

    plot_name = 'tau_rate_scan_topo_ptcorrected.pdf' if correct_pt else 'tau_rate_scan_topo_ptuncorrected.pdf' 
    plt.savefig(f'plots/{plot_name}', bbox_inches='tight')

def group_id_values(event_id, raw_tau_score_sum, *arrays, num_elements = 2):
    '''
    Group values according to event id.
    Filter out events that has less than num_elements
    '''

    # Use ak.argsort to sort based on event_id
    sorted_indices = ak.argsort(event_id)
    sorted_event_id = event_id[sorted_indices]

    # Find unique event_ids and counts manually
    unique_event_id, counts = np.unique(sorted_event_id, return_counts=True)
    
    # Use ak.unflatten to group the arrays by counts
    grouped_id = ak.unflatten(sorted_event_id, counts)
    grouped_arrays = [ak.unflatten(arr[sorted_indices], counts) for arr in arrays]

    #Sort by tau score
    tau_score = ak.unflatten(raw_tau_score_sum[sorted_indices],counts)
    tau_sort_index = ak.argsort(tau_score, ascending=False)
    grouped_arrays_sorted = [arr[tau_sort_index] for arr in grouped_arrays]

    #Filter out groups that don't have at least 2 elements
    mask = ak.num(grouped_id) >= num_elements
    filtered_grouped_arrays = [arr[mask] for arr in grouped_arrays_sorted]

    return grouped_id[mask], filtered_grouped_arrays

def derive_tau_rate(model, minbias_path, input_tag='ext7', tree='jetntuple/Jets', n_entries=500000, correct_pt=True):
    '''
    Derive the tau rate, using n_entries minbias events 
    '''

    minbias_rate = 32e+3 #32 kHZ

    #Load the minbias data
    minbias = uproot4.open(minbias_path)[tree]

    raw_event_id = helpers.extract_array(minbias, 'event', n_entries)
    raw_jet_pt = helpers.extract_array(minbias, 'jet_pt', n_entries)
    raw_jet_eta = helpers.extract_array(minbias, 'jet_eta_phys', n_entries)
    raw_jet_phi = helpers.extract_array(minbias, 'jet_phi_phys', n_entries)
    raw_inputs = np.asarray(helpers.extract_nn_inputs(minbias, input_fields_tag=input_tag, nconstit=16, n_entries=n_entries)).transpose(0, 2, 1)
    raw_pred_score, raw_pt_correction = model.predict(raw_inputs)

    tau_index = [2,3] #2=tau positive, 3=tau_negative
    raw_tau_score_sum = raw_pred_score[:,tau_index[0]] + raw_pred_score[:,tau_index[1]]
    raw_tau_plus = raw_pred_score[:,tau_index[0]]
    raw_tau_minus = raw_pred_score[:,tau_index[1]]

    #Count number of total event
    n_events = len(np.unique(raw_event_id))
    print("Total number of minbias events: ", n_events)
    
    #Group these attributes by event id, and filter out groups that don't have at least 2 elements
    event_id, grouped_arrays  = group_id_values(raw_event_id, raw_tau_score_sum, raw_tau_plus, raw_tau_minus, raw_jet_pt, raw_pt_correction.flatten(), raw_jet_eta, raw_jet_phi, num_elements=2)

    # Extract the grouped arrays
    tau_plus, tau_minus, jet_pt, jet_pt_correction, jet_eta, jet_phi = grouped_arrays

    #calculate delta_r
    eta1, eta2 = jet_eta[:, 0], jet_eta[:, 1]
    phi1, phi2 = jet_phi[:, 0], jet_phi[:, 1]
    delta_r_values = delta_r(eta1, phi1, eta2, phi2)

    # Additional cuts recommended here:
    # https://indico.cern.ch/event/1380964/contributions/5852368/attachments/2841655/4973190/AnnualReview_2024.pdf
    # Slide 7
    cuts = (np.abs(eta1) < 2.172) & (np.abs(eta2) < 2.172) & (delta_r_values > 0.5)

    tau_topo_score = calculate_topo_score(tau_plus, tau_minus)

    #correct for pt
    pt1_uncorrected, pt2_uncorrected = np.asarray(jet_pt[:,0][cuts]), np.asarray(jet_pt[:,1][cuts])
    ratio1, ratio2 = np.asarray(jet_pt_correction[:,0][cuts]), np.asarray(jet_pt_correction[:,1][cuts])

    pt1 = pt1_uncorrected*ratio1
    pt2 = pt2_uncorrected*ratio2

    #Put them together
    NN_score_min = tau_topo_score[cuts]

    pt = np.vstack([pt1, pt2]).transpose()
    pt_min = np.min(pt, axis=1)

    #Define the histograms (pT edge and NN Score edge)
    pT_edges = list(np.arange(0,100,2)) + [1500] #Make sure to capture everything
    NN_edges = list([round(i,4) for i in np.arange(0, 0.6, 0.0005)])

    RateHist = Hist(hist.axis.Variable(pT_edges, name="pt", label="pt"),
                    hist.axis.Variable(NN_edges, name="nn", label="nn"))

    RateHist.fill(pt = pt_min, nn = NN_score_min)

    #Derive the rate
    rate_list = []
    pt_list = []
    nn_list = []

    #Loop through the edges and integrate
    for pt in pT_edges[:-1]:
        for NN in NN_edges[:-1]:
            
            #Calculate the rate
            rate = RateHist[{"pt": slice(pt*1j, pT_edges[-1]*1.0j, sum)}][{"nn": slice(NN*1.0j,1.0j, sum)}]/n_events
            rate_list.append(rate*minbias_rate)

            #Append the results   
            pt_list.append(pt)
            nn_list.append(NN)

    plot_rate(rate_list, pt_list, nn_list, target_rate=28, correct_pt=correct_pt)

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('-m','--model', default='/eos/home-s/sewuchte/www/L1T/trainings_regression_weighted/2024_08_27_v4_extendedAll200_btgc_ext7_QDeepSets_PermutationInv_nconst_16_nfeatures_21_nbits_8_pruned/model_QDeepSets_PermutationInv_nconst_16_nfeatures_21_nbits_8_pruned.h5' , help = 'Input model for plotting')    
    parser.add_argument('--uncorrect_pt', action='store_true', help='Enable pt correction in plot_bkg_rate_tau')

    args = parser.parse_args()

    model=load_qmodel(args.model)
    print(model.summary())

    #These paths are default to evaluate some of the rate
    minbias_path = '/eos/cms/store/cmst3/group/l1tr/sewuchte/l1teg/fp_ntuples_v131Xv9/extendedTRK_HW_260824/MinBias_PU200.root'

    #Parse the options a bit here
    correct_pt = not args.uncorrect_pt

    derive_tau_rate(model, minbias_path, n_entries=3000000, input_tag='ext7', correct_pt=correct_pt)