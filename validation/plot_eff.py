import uproot4
import numpy as np
from argparse import ArgumentParser
from qkeras.utils import load_qmodel

#Import the calculated working points
from official_WPs import WPs

def plot_eff_tau(model, signal_path, cut_value, tree='jetntuple/Jets'):

    signal = uproot4.open(signal_path)[tree]

    #Plot the emulator tau rates
    print(signal.keys())


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('-m','--model', default='../models/model_QDeepSets_PermutationInv_nconst_16_nfeatures_8_nbits_8_pruned.h5' , help = 'Input model for plotting')    
    args = parser.parse_args()

    model=load_qmodel(args.model)

    #These paths are default to evaluate some of the efficiency
    tau_eff_filepath = '/eos/user/s/sewuchte/L1Trigger/ForDuc/nTuples/VBFHtt_PU200.root'

    plot_eff_tau(model, tau_eff_filepath, cut_value=WPs.tau)

    print(WPs.tau)
