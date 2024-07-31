'''
Scripts for deriving the working points for various heads of the tagger.
'''

import uproot4
import numpy as np
import awkward as ak
from argparse import ArgumentParser
from qkeras.utils import load_qmodel

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

def group_id_values(event_id, values):
    '''
    Group values according to event id.
    '''

    # Use ak.argsort to sort based on event_id
    sorted_indices = ak.argsort(event_id)
    sorted_event_id = event_id[sorted_indices]
    sorted_values = values[sorted_indices]

    # Find unique event_ids and counts manually
    unique_event_id = np.unique(sorted_event_id)
    counts = [np.sum(sorted_event_id == eid) for eid in unique_event_id]

    # Use ak.unflatten to group the tau_pt by counts
    grouped_id = ak.unflatten(sorted_event_id, counts)
    grouped_values = ak.unflatten(sorted_values, counts)

    return grouped_id, grouped_values


def derive_tau_rate():

    

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('-m','--model', default='../models/model_QDeepSets_PermutationInv_nconst_16_nfeatures_8_nbits_8_pruned.h5' , help = 'Input model for plotting')    
    args = parser.parse_args()

    model=load_qmodel(args.model)

    #These paths are default to evaluate some of the rate
    minbias_path = '/eos/user/s/sewuchte/L1Trigger/ForDuc/nTuples/MinBias_PU200.root'

    derive_tau_rate(model, minbias_path)