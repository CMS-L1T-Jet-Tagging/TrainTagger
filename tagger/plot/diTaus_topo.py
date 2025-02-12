"""
Script to plot all di-taus topology trigger related physics performance plot
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
from common import MINBIAS_RATE, WPs_CMSSW, find_rate, plot_ratio, delta_r, eta_region_selection, get_bar_patch_data

def calculate_topo_score(tau_plus, tau_minus):
    #2=tau positive, 3=tau_negative

    p_pos = tau_plus[:,0] + tau_plus[:,1]
    p_neg = tau_minus[:,0] + tau_minus[:,1]

    return np.multiply(p_pos, p_neg)

def group_id_values_topo(event_id, raw_tau_score_sum, *arrays, num_elements = 2):
    '''
    Group values according to event id specifically for topology di tau codes, since we also want to sort by tau scores
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
 
def pick_and_plot_topo(rate_list, pt_list, nn_list, model_dir, target_rate = 28, RateRange=1.0):

    plot_dir = os.path.join(model_dir, 'plots/physics/tautau_topo')
    os.makedirs(plot_dir, exist_ok=True)
    
    fig,ax = plt.subplots(1,1,figsize=style.FIGURE_SIZE)
    hep.cms.label(llabel=style.CMSHEADER_LEFT,rlabel=style.CMSHEADER_RIGHT,ax=ax,fontsize=style.MEDIUM_SIZE-2)
    im = ax.scatter(nn_list, pt_list, c=rate_list, s=500, marker='s',
                    cmap='Spectral_r',
                    linewidths=0,
                    norm=matplotlib.colors.LogNorm())

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r'Di-tau rate [kHZ]')

    ax.set_ylabel(r"Min reco $p_T$ [GeV]")
    ax.set_xlabel(r"Tau Topology Score")

    ax.set_xlim([0,0.2])
    ax.set_ylim([10,100])
    
    #Find the target rate points, plot them and print out some info as well
    target_rate_idx = find_rate(rate_list, target_rate = target_rate, RateRange=RateRange)
    
    #Get the coordinates
    target_rate_NN = [nn_list[i] for i in target_rate_idx] # NN cut dimension
    target_rate_PT = [pt_list[i] for i in target_rate_idx] # HT cut dimension

    # Create an interpolation function
    interp_func = interp1d(target_rate_PT, target_rate_NN, kind='linear', fill_value='extrapolate')

    # Interpolate the NN value for the desired HT
    working_point_NN = interp_func(WPs_CMSSW['tau_l1_pt']) #+ 0.02 #WP looks a bit too loose for taus using interpolation so just a quick hack

    # Export the working point
    working_point = {"PT": WPs_CMSSW['tau_l1_pt'], "NN": float(working_point_NN)}

    with open(os.path.join(plot_dir, "working_point.json"), "w") as f:
        json.dump(working_point, f, indent=4)
    
    # Generate 100 points spanning the entire pT range visible on the plot.
    pT_full = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 100)
    
    # Evaluate the interpolation function to obtain NN values for these pT points.
    NN_full = interp_func(pT_full)
    ax.plot(NN_full, pT_full, linewidth=style.LINEWIDTH, color ='firebrick', label = r"${} \pm {}$ kHz".format(target_rate, RateRange))

    ax.legend(loc='upper right', fontsize=style.MEDIUM_SIZE)
    plt.savefig(f"{plot_dir}/tautau_topo_WPs.pdf", bbox_inches='tight')
    plt.savefig(f"{plot_dir}/tautau_topo_WPs.png", bbox_inches='tight')
    
    
def derive_diTaus_topo_WPs(model_dir, minbias_path, n_entries=100, tree='jetntuple/Jets', target_rate=28):
    """
    Derive ditau topology working points. 
    Using a new score that uses the charge definition in the jet tagger. 

    topology_score = (tau_p_1 + tau_p_2)*(tau_m_1 + tau_m_2)
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
    raw_inputs = np.asarray(extract_nn_inputs(minbias, input_vars, n_entries=n_entries))
    raw_pred_score, raw_pt_correction = model.predict(raw_inputs)

    tau_index = [class_labels['taup'], class_labels['taum']] #2=tau positive, 3=tau_negative
    raw_tau_score_sum = raw_pred_score[:,tau_index[0]] + raw_pred_score[:,tau_index[1]]
    raw_tau_plus = raw_pred_score[:,tau_index[0]]
    raw_tau_minus = raw_pred_score[:,tau_index[1]]

    #Count number of total event
    n_events = len(np.unique(raw_event_id))
    print("Total number of minbias events: ", n_events)

    #Group these attributes by event id, and filter out groups that don't have at least 2 elements
    event_id, grouped_arrays  = group_id_values_topo(raw_event_id, raw_tau_score_sum, raw_tau_plus, raw_tau_minus, raw_jet_pt, raw_pt_correction.flatten(), raw_jet_eta, raw_jet_phi, num_elements=2)

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
            rate_list.append(rate*MINBIAS_RATE)

            #Append the results   
            pt_list.append(pt)
            nn_list.append(NN)

    pick_and_plot_topo(rate_list, pt_list, nn_list, model_dir, target_rate=28)

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
    parser.add_argument('--tree', default='jetntuple/Jets', help='Tree within the ntuple containing the jets')

    args = parser.parse_args()

    if args.deriveWPs:
        derive_diTaus_topo_WPs(args.model_dir, args.minbias, n_entries=args.n_entries, tree=args.tree)
    # elif args.BkgRate:
    #     plot_bkg_rate_ditau(args.model_dir, args.minbias, n_entries=args.n_entries, tree=args.tree)
    # elif args.eff:
    #     eff_ditau(args.model_dir, args.vbf_sample, n_entries=args.n_entries, eta_region='barrel', tree=args.tree)
    #     eff_ditau(args.model_dir, args.vbf_sample, n_entries=args.n_entries, eta_region='endcap', tree=args.tree)