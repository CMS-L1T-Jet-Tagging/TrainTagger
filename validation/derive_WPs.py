'''
Scripts for deriving the working points for various heads of the tagger.
'''
import sys
import uproot4
import numpy as np
import awkward as ak
from argparse import ArgumentParser
from qkeras.utils import load_qmodel

# Add path so the script sees other modules
sys.path.append('../')
from datatools.createDataset import dict_fields

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
    return ak.fill_none(ak.pad_none(array, target, axis=1), 0)

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


def extract_nn_inputs(minbias, input_fields_tag='ext3', nconstit=16):
    """
    Extract NN inputs based on input_set
    """

    #The complete input sets are defined in utils/createDataset.py
    features = dict_fields[input_fields_tag]

    #Concatenate all the inputs
    inputs_list = []

    #Vertically stacked them to create input sets
    #https://awkward-array.org/doc/main/user-guide/how-to-restructure-concatenate.html
    for i in range(len(features)): inputs_list.append(minbias[f"jet_pfcand_{features[i]}"].array()[:, np.newaxis])

    inputs = ak.concatenate(inputs_list, axis=1)

    return

def derive_tau_rate(model, minbias_path, tree='jetntuple/Jets'):

    #Load the minbias data
    minbias = uproot4.open(minbias_path)[tree]

    raw_event_id = minbias['event'].array()
    raw_jet_pt = minbias['jet_pt'].array()
    raw_jet_eta = minbias['jet_eta_phys'].array()
    raw_jet_phi = minbias['jet_phi_phys'].array()

    #Count number of total event
    n_event = np.unique(raw_event_id).shape[0]

    #Group these attributes by event id, and filter out groups that don't have at least 2 elements
    event_id, grouped_arrays  = group_id_values(raw_event_id, raw_jet_pt, raw_jet_eta, raw_jet_phi)

    # Extract the grouped arrays
    # Jet pt is already sorted in the producer, no need to do it here
    jet_pt, jet_eta, jet_phi = grouped_arrays

    pt1, pt2 = jet_pt[:,0], jet_pt[:,1]
    eta1, eta2 = jet_eta[:, 0], jet_eta[:, 1]
    phi1, phi2 = jet_phi[:, 0], jet_phi[:, 1]

    #calculate delta_r
    delta_r_values = delta_r(eta1, phi1, eta2, phi2)

    # Additional cuts recommended here:
    # https://indico.cern.ch/event/1380964/contributions/5852368/attachments/2841655/4973190/AnnualReview_2024.pdf
    # Slide 7
    cuts = (np.abs(eta1) < 2.172) & (np.abs(eta2) < 2.172) & (delta_r_values > 0.5)

    #Extract the NN inputs


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('-m','--model', default='/eos/user/s/sewuchte/L1Trigger/ForDuc/trainings_regression_weighted/2024_07_25_v10_extendedAll200_btgc_ext3_QDeepSets_PermutationInv_nconst_16_nfeatures_21_nbits_8_pruned/model_QDeepSets_PermutationInv_nconst_16_nfeatures_21_nbits_8_pruned.h5' , help = 'Input model for plotting')    
    args = parser.parse_args()

    model=load_qmodel(args.model)
    print(model.summary())

    #These paths are default to evaluate some of the rate
    minbias_path = '/eos/user/s/sewuchte/L1Trigger/ForDuc/nTuples/MinBias_PU200.root'

    minbias = uproot4.open(minbias_path)['jetntuple/Jets']
    extract_nn_inputs(minbias)

    # derive_tau_rate(model, minbias_path)