"""
Plot HH efficiencies, usage:

python HH_eff.py <see more arguments below>
"""
import sys
import uproot4
import numpy as np
from argparse import ArgumentParser
from qkeras.utils import load_qmodel

# Add path so the script sees other modules
sys.path.append('../')
from datatools.createDataset import dict_fields
from datatools import helpers
from hist import Hist
import hist

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
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

####### Define functions
def HH_eff_HT(model, hh_file_path, n_entries=100000, tree='jetntuple/Jets'):
    """
    Main efficiency funtion that pulls everything together
    """

    signal = uproot4.open(hh_file_path)[tree]

    print(signal.keys())



if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('-m','--model', default='/eos/user/s/sewuchte/L1Trigger/ForDuc/trainings_regression_weighted/2024_08_17_v6_extendedAll200_btgc_ext7_QDeepSets_PermutationInv_nconst_16_nfeatures_21_nbits_8_pruned/model_QDeepSets_PermutationInv_nconst_16_nfeatures_21_nbits_8_pruned.h5', help = 'Input model for plotting')    

    args = parser.parse_args()

    #Load the model defined above
    model=load_qmodel(args.model)

    #These paths are default to evaluate some of the efficiency
    hh_file_path = '/eos/cms/store/cmst3/group/l1tr/sewuchte/l1teg/fp_ntuples_v131Xv9/extendedTRK_HW_260824/ggHHbbbb_PU200.root'

    #Barrel
    HH_eff(model, hh_file_path, n_entries=100000)
