'''
Scripts for deriving the working points for various heads of the tagger.
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

def delta_r(eta1, phi1, eta2, phi2):
    """
    Calculate the delta R between two sets of eta and phi values.
    """
    delta_eta = eta1 - eta2
    delta_phi = phi1 - phi2

    # Ensure delta_phi is within -pi to pi
    delta_phi = (delta_phi + np.pi) % (2 * np.pi) - np.pi
    return np.sqrt(delta_eta**2 + delta_phi**2)

def pad_fill(array, target):
    '''
    pad an array to target length and then fill it with 0s
    '''
    return ak.fill_none(ak.pad_none(array, target, axis=1, clip=True), 0)

def extract_array(tree, field, entry_stop):
    """
    Extracts an array from the tree with a limit on the number of entries.
    """
    return tree[field].array(entry_stop=entry_stop)

def group_id_values(event_id, *arrays):
    '''
    Group values according to event id.
    '''

    # Use ak.argsort to sort based on event_id
    sorted_indices = ak.argsort(event_id)
    sorted_event_id = event_id[sorted_indices]

    # Find unique event_ids and counts manually
    unique_event_id, counts = np.unique(sorted_event_id, return_counts=True)
    
    # Use ak.unflatten to group the arrays by counts
    grouped_id = ak.unflatten(sorted_event_id, counts)
    grouped_arrays = [ak.unflatten(arr[sorted_indices], counts) for arr in arrays]

    #Filter out groups that don't have at least 2 elements
    mask = ak.num(grouped_id) >= 2
    filtered_grouped_arrays = [arr[mask] for arr in grouped_arrays]

    return grouped_id[mask], filtered_grouped_arrays

def extract_nn_inputs(minbias, input_fields_tag='ext3', nconstit=16, n_entries=500000):
    """
    Extract NN inputs based on input_fields_tag
    """

    #The complete input sets are defined in utils/createDataset.py
    features = dict_fields[input_fields_tag]

    #Concatenate all the inputs
    inputs_list = []

    #Vertically stacked them to create input sets
    #https://awkward-array.org/doc/main/user-guide/how-to-restructure-concatenate.html
    #Also pad and fill them with 0 to the number of constituents we are using (nconstit)
    for i in range(len(features)):
        field = f"jet_pfcand_{features[i]}"
        field_array = extract_array(minbias, field, n_entries)
        padded_filled_array = pad_fill(field_array, nconstit)
        inputs_list.append(padded_filled_array[:, np.newaxis])

    inputs = ak.concatenate(inputs_list, axis=1)

    return inputs

def find_rate(rate_list, target_rate = 28):
    
    RateRange = 0.5 #kHz
    
    idx_list = []
    
    for i in range(len(rate_list)):
        if target_rate-RateRange <= rate_list[i] <= target_rate + RateRange:
            idx_list.append(i)
            
    return idx_list

def plot_rate(rate_list, pt_list, nn_list, target_rate = 28):
    
    fig, ax = plt.subplots()
    im = ax.scatter(nn_list, pt_list, c=rate_list, s=500, marker='s',
                    cmap='Spectral_r',
                    linewidths=0,
                    norm=matplotlib.colors.LogNorm())

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r'Di-$\tau_h$ rate [kHZ]')

    plt.ylabel(r"Min reco $p_T$ [GeV]")
    plt.xlabel(r"Min NN Score")
    
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
                        color ='firebrick', label = r"${} \pm 0.5$ kHz".format(target_rate))
        else:
            plt.scatter(nn_list[i], pt_list[i], s=600, marker='*',
                        color ='firebrick')
            
        legend_count += 1
    
    plt.legend(loc='upper right')
    plt.savefig('plots/tau_rate_scan.pdf', bbox_inches='tight')

def derive_tau_rate(model, minbias_path, tree='jetntuple/Jets', n_entries=500000):
    '''
    Derive the tau rate, using n_entries minbias events 
    '''

    minbias_rate = 32e+3 #32 kHZ

    #Load the minbias data
    minbias = uproot4.open(minbias_path)[tree]

    raw_event_id = extract_array(minbias, 'event', n_entries)
    raw_jet_pt = extract_array(minbias, 'jet_pt', n_entries)
    raw_jet_eta = extract_array(minbias, 'jet_eta_phys', n_entries)
    raw_jet_phi = extract_array(minbias, 'jet_phi_phys', n_entries)
    raw_inputs = extract_nn_inputs(minbias, input_fields_tag='ext3', nconstit=16, n_entries=n_entries)

    #Count number of total event
    n_events = len(np.unique(raw_event_id))
    print("Total number of minbias events: ", n_events)

    #Group these attributes by event id, and filter out groups that don't have at least 2 elements
    event_id, grouped_arrays  = group_id_values(raw_event_id, raw_jet_pt, raw_jet_eta, raw_jet_phi, raw_inputs)

    # Extract the grouped arrays
    # Jet pt is already sorted in the producer, no need to do it here
    jet_pt, jet_eta, jet_phi, jet_nn_inputs = grouped_arrays

    #calculate delta_r
    eta1, eta2 = jet_eta[:, 0], jet_eta[:, 1]
    phi1, phi2 = jet_phi[:, 0], jet_phi[:, 1]
    delta_r_values = delta_r(eta1, phi1, eta2, phi2)

    # Additional cuts recommended here:
    # https://indico.cern.ch/event/1380964/contributions/5852368/attachments/2841655/4973190/AnnualReview_2024.pdf
    # Slide 7
    cuts = (np.abs(eta1) < 2.172) & (np.abs(eta2) < 2.172) & (delta_r_values > 0.5)

    #Get inputs and pts for processing
    pt1_uncorrected, pt2_uncorrected = np.asarray(jet_pt[:,0][cuts]), np.asarray(jet_pt[:,1][cuts])
    input1, input2 = np.asarray(jet_nn_inputs[:, 0][cuts]).transpose(0, 2, 1), np.asarray(jet_nn_inputs[:, 1][cuts]).transpose(0, 2, 1) #Flip the last two axes

    #Get the NN predictions
    tau_index = 4
    pred_score1, ratio1 = model.predict(input1)
    pred_score2, ratio2 = model.predict(input2)

    pt1 = pt1_uncorrected*ratio1
    pt2 = pt2_uncorrected*ratio2

    tau_score1=pred_score1[:,tau_index]
    tau_score2=pred_score2[:,tau_index]
    
    # #Put them together
    NN_score = np.vstack([tau_score1, tau_score2]).transpose()
    NN_score_min = np.min(NN_score, axis=1)

    pt = np.vstack([pt1, pt2]).transpose()
    pt_min = np.min(pt, axis=1)

    #Define the histograms (pT edge and NN Score edge)
    pT_edges = list(np.arange(0,100,2)) + [1500] #Make sure to capture everything
    NN_edges = list([round(i,2) for i in np.arange(0, 1.01, 0.01)])

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

    plot_rate(rate_list, pt_list, nn_list)

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('-m','--model', default='/eos/user/s/sewuchte/L1Trigger/ForDuc/trainings_regression_weighted/2024_07_25_v10_extendedAll200_btgc_ext3_QDeepSets_PermutationInv_nconst_16_nfeatures_21_nbits_8_pruned/model_QDeepSets_PermutationInv_nconst_16_nfeatures_21_nbits_8_pruned.h5' , help = 'Input model for plotting')    
    args = parser.parse_args()

    model=load_qmodel(args.model)
    print(model.summary())

    #These paths are default to evaluate some of the rate
    minbias_path = '/eos/user/s/sewuchte/L1Trigger/ForDuc/nTuples/MinBias_PU200.root'

    derive_tau_rate(model, minbias_path, n_entries=100000)