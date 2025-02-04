"""
Script to plot all di-taus related physics performance plot
"""
import os, json
from argparse import ArgumentParser

from qkeras.utils import load_qmodel
import awkward as ak
import numpy as np
import uproot
import hist
from hist import Hist

import matplotlib.pyplot as plt
import matplotlib
import mplhep as hep
import tagger.plot.style as style

style.set_style()

#Interpolation of working point
from scipy.interpolate import interp1d

#Imports from other modules
from tagger.data.tools import extract_array, extract_nn_inputs, group_id_values
from common import MINBIAS_RATE, WPs_CMSSW, find_rate, plot_ratio, get_bar_patch_data, delta_r

def pick_and_plot_ditau(rate_list, pt_list, nn_list, model_dir, target_rate = 28):
    """
    Pick the working points and plot
    """

    plot_dir = os.path.join(model_dir, 'plots/physics/tautau')
    os.makedirs(plot_dir, exist_ok=True)
    
    fig,ax = plt.subplots(1,1,figsize=style.FIGURE_SIZE)
    hep.cms.label(llabel=style.CMSHEADER_LEFT,rlabel=style.CMSHEADER_RIGHT,ax=ax,fontsize=style.MEDIUM_SIZE-2)
    im = ax.scatter(nn_list, pt_list, c=rate_list, s=500, marker='s',
                    cmap='Spectral_r',
                    linewidths=0,
                    norm=matplotlib.colors.LogNorm())

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r'Di-tau rate [kHZ]')

    ax.set_ylabel(r"Min L1 $p_T$ [GeV]")
    ax.set_xlabel(r"Min Tau NN ($\tau^{+} + \tau^{-}$) Score")
    
    ax.set_xlim([0,0.4])
    ax.set_ylim([10,100])

    #plus, minus range
    RateRange = 1.0

    #Find the target rate points, plot them and print out some info as well
    target_rate_idx = find_rate(rate_list, target_rate = target_rate, RateRange=RateRange)

    #Get the coordinates
    target_rate_NN = [nn_list[i] for i in target_rate_idx] # NN cut dimension
    target_rate_PT = [pt_list[i] for i in target_rate_idx] # HT cut dimension

    # Create an interpolation function
    interp_func = interp1d(target_rate_PT, target_rate_NN, kind='linear', fill_value='extrapolate')

    # Interpolate the NN value for the desired HT
    working_point_NN = interp_func(WPs_CMSSW['tau_l1_pt'])

    # Export the working point
    working_point = {"PT": WPs_CMSSW['tau_l1_pt'], "NN": float(working_point_NN)}

    with open(os.path.join(plot_dir, "working_point.json"), "w") as f:
        json.dump(working_point, f, indent=4)
    
    # Generate 100 points spanning the entire pT range visible on the plot.
    pT_full = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 100)
    
    # Evaluate the interpolation function to obtain NN values for these pT points.
    NN_full = interp_func(pT_full)
    ax.plot(NN_full, pT_full, linewidth=style.LINEWIDTH, color ='firebrick', label = r"${} \pm {}$ kHz".format(target_rate, RateRange))

    #Just plot the points instead of the interpolation
    #ax.plot(target_rate_NN, target_rate_PT, linewidth=style.LINEWIDTH, color ='firebrick', label = r"${} \pm {}$ kHz".format(target_rate, RateRange))
    
    ax.legend(loc='upper right', fontsize=style.SMALL_SIZE-3)
    plt.savefig(f"{plot_dir}/tautau_WPs.pdf", bbox_inches='tight')
    plt.savefig(f"{plot_dir}/tautau_WPs.png", bbox_inches='tight')

def derive_diTaus_WPs(model_dir, minbias_path, target_rate=28, n_entries=100, tree='jetntuple/Jets'):
    """
    Derive the di-tau rate. 
    Seed definition can be found here (2024 Annual Review):

    https://indico.cern.ch/event/1380964/contributions/5852368/attachments/2841655/4973190/AnnualReview_2024.pdf

    Double Puppi Tau Seed, same NN cut and pT (52 GeV) on both taus to give 28 kHZ based on the definition above.  
    """

    model=load_qmodel(os.path.join(model_dir, "model/saved_model.h5"))

    #Load the minbias data
    minbias = uproot.open(minbias_path)[tree]

    # Load the inputs
    with open(os.path.join(model_dir, "input_vars.json"), "r") as f: input_vars = json.load(f)
    with open(os.path.join(model_dir, "class_label.json"), "r") as f: class_labels = json.load(f)

    raw_event_id = extract_array(minbias, 'event', n_entries)
    raw_jet_pt = extract_array(minbias, 'jet_pt', n_entries)
    raw_jet_eta = extract_array(minbias, 'jet_eta_phys', n_entries)
    raw_jet_phi = extract_array(minbias, 'jet_phi_phys', n_entries)
    raw_inputs = extract_nn_inputs(minbias, input_vars, n_entries=n_entries)

    #Count number of total event
    n_events = len(np.unique(raw_event_id))
    print("Total number of minbias events: ", n_events)

    #Group these attributes by event id, and filter out groups that don't have at least 2 elements
    event_id, grouped_arrays  = group_id_values(raw_event_id, raw_jet_pt, raw_jet_eta, raw_jet_phi, raw_inputs, num_elements=2)

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
    pt1_uncorrected, pt2_uncorrected = np.asarray(jet_pt[:, 0][cuts]), np.asarray(jet_pt[:,1][cuts])
    input1, input2 = np.asarray(jet_nn_inputs[:, 0][cuts]), np.asarray(jet_nn_inputs[:, 1][cuts])

    #Get the NN predictions
    tau_index = [class_labels['taup'], class_labels['taum']] #Tau positives and tau negatives
    pred_score1, ratio1 = model.predict(input1)
    pred_score2, ratio2 = model.predict(input2)

    #Correct the pT and add the score
    pt1 = pt1_uncorrected*(ratio1.flatten())
    pt2 = pt2_uncorrected*(ratio2.flatten())

    tau_score1=pred_score1[:,tau_index[0]] + pred_score1[:,tau_index[1]]
    tau_score2=pred_score2[:,tau_index[0]] + pred_score2[:,tau_index[1]]

    #Put them together
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
            rate = RateHist[{"pt": slice(pt*1j, None, sum)}][{"nn": slice(NN*1.0j, None, sum)}]/n_events
            rate_list.append(rate*MINBIAS_RATE)

            #Append the results   
            pt_list.append(pt)
            nn_list.append(NN)

    #Pick target rate and plot it
    pick_and_plot_ditau(rate_list, pt_list, nn_list, model_dir, target_rate=target_rate)

    return

def plot_bkg_rate_ditau(model_dir, minbias_path, n_entries=500000, tree='jetntuple/Jets'):

    return

if __name__ == "__main__":
    """
    2 steps:

    1. Derive working points: python diTaus.py --deriveWPs
    2. Run efficiency based on the derived working points: python diTaus.py --eff
    """

    parser = ArgumentParser()
    parser.add_argument('-m','--model_dir', default='output/baseline', help = 'Input model')
    parser.add_argument('-v', '--vbf_sample', default='/eos/cms/store/cmst3/group/l1tr/sewuchte/l1teg/fp_ntuples_v131Xv9/extendedTRK_5param_221124/VBFHtt_PU200.root' , help = 'Signal sample for VBF -> ditaus') 
    parser.add_argument('--minbias', default='/eos/cms/store/cmst3/group/l1tr/sewuchte/l1teg/fp_ntuples_v131Xv9/extendedTRK_5param_221124/MinBias_PU200.root' , help = 'Minbias sample for deriving rates')    

    #Different modes
    parser.add_argument('--deriveWPs', action='store_true', help='derive the working points for di-taus')
    parser.add_argument('--eff', action='store_true', help='plot efficiency for VBF-> tautau')
    parser.add_argument('--BkgRate', action='store_true', help='plot background rate for VBF->tautau')

    #Other controls
    parser.add_argument('-n','--n_entries', type=int, default=500000, help = 'Number of data entries in root file to run over, can speed up run time, set to None to run on all data entries')
    args = parser.parse_args()

    if args.deriveWPs:
        derive_diTaus_WPs(args.model_dir, args.minbias, n_entries=args.n_entries)
    elif args.BkgRate:
        plot_bkg_rate_ditau(args.model_dir, args.minbias, n_entries=args.n_entries)
