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
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

def calculate_topo_score(tau_plus, tau_minus):
    #2=tau positive, 3=tau_negative

    p_pos = tau_plus[:,0] + tau_plus[:,1]
    p_neg = tau_minus[:,0] + tau_minus[:,1]

    return np.multiply(p_pos, p_neg)

def apply_mask(arrays, mask):
    masked_arrays = [array[mask] for array in arrays]
    return masked_arrays

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
    PT_WP = WPs_CMSSW['tau_l1_pt']
    working_point_NN = interp_func(PT_WP)

    # Export the working point
    working_point = {"PT": PT_WP, "NN": float(working_point_NN)}

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

    raw_tau_score_sum = raw_pred_score[:,class_labels['taup']] + raw_pred_score[:, class_labels['taum']]
    raw_tau_plus = raw_pred_score[:,class_labels['taup']]
    raw_tau_minus = raw_pred_score[:, class_labels['taum']]

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

#-------- Plot the background rate
def cmssw_pt_score(raw_event_id, raw_jet_pt, raw_jet_eta, raw_jet_phi, raw_cmssw_tau, raw_cmssw_taupt):

    #Group these attributes by event id, and filter out groups that don't have at least 2 elements
    event_id, grouped_arrays  = group_id_values(raw_event_id, raw_jet_pt, raw_jet_eta, raw_jet_phi, raw_cmssw_tau, raw_cmssw_taupt, num_elements=2)

    # Extract the grouped arrays
    # Jet pt is already sorted in the producer, no need to do it here
    jet_pt, jet_eta, jet_phi, cmssw_tau, cmssw_taupt = grouped_arrays

    #calculate delta_r
    eta1, eta2 = jet_eta[:, 0], jet_eta[:, 1]
    phi1, phi2 = jet_phi[:, 0], jet_phi[:, 1]
    delta_r_values = delta_r(eta1, phi1, eta2, phi2)

    # Additional cuts recommended here:
    # https://indico.cern.ch/event/1380964/contributions/5852368/attachments/2841655/4973190/AnnualReview_2024.pdf
    # Slide 7
    cuts = (np.abs(eta1) < 2.172) & (np.abs(eta2) < 2.172) & (delta_r_values > 0.5)

    #Get cmssw attribubtes to calculate the rate
    cmssw_pt1, cmssw_pt2 = np.asarray(cmssw_taupt[:,0][cuts]), np.asarray(cmssw_taupt[:,1][cuts])
    cmssw_pt = np.vstack([cmssw_pt1, cmssw_pt2]).transpose()
    cmssw_pt_min = np.min(cmssw_pt, axis=1)

    #Do similar thing for the tau score
    cmssw_tau1, cmssw_tau2 = np.asarray(cmssw_tau[:,0][cuts]), np.asarray(cmssw_tau[:,1][cuts])
    cmssw_tau = np.vstack([cmssw_tau1, cmssw_tau2]).transpose()
    cmssw_tau_min =  np.min(cmssw_tau, axis=1)

    pt1, pt2 = np.asarray(jet_pt[:,0][cuts]), np.asarray(jet_pt[:,1][cuts])
    pt = np.vstack([pt1, pt2]).transpose()
    pt_min = np.min(pt, axis=1)

    return event_id[cuts], pt_min, cmssw_pt_min, cmssw_tau_min

def model_pt_score(raw_event_id, raw_tau_score_sum, raw_tau_plus, raw_tau_minus, raw_jet_pt, raw_pt_correction, raw_jet_eta, raw_jet_phi):

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

    pt = np.vstack([pt1, pt2]).transpose()
    pt_min = np.min(pt, axis=1)

    return event_id[cuts], pt_min, tau_topo_score[cuts]

def plot_bkg_rate_ditau_topo(model_dir, minbias_path, n_entries=100, tree='jetntuple/Jets'):

    model=load_qmodel(os.path.join(model_dir, "model/saved_model.h5"))

    #Load the minbias data
    minbias = uproot.open(minbias_path)[tree]

    # Load the inputs
    with open(os.path.join(model_dir, "input_vars.json"), "r") as f: input_vars = json.load(f)
    with open(os.path.join(model_dir, "class_label.json"), "r") as f: class_labels = json.load(f)

    #Check if the working point have been derived
    WP_path = os.path.join(model_dir, "plots/physics/tautau_topo/working_point.json")

    #Get derived working points
    if os.path.exists(WP_path):
        with open(WP_path, "r") as f:  WPs = json.load(f)
        model_NN_WP = WPs['NN']
        model_pt_WP = WPs['PT']
    else:
        raise Exception("Working point does not exist. Run with --deriveWPs first.")

    raw_event_id = extract_array(minbias, 'event', n_entries)
    raw_jet_pt = extract_array(minbias, 'jet_pt', n_entries)
    raw_jet_eta = extract_array(minbias, 'jet_eta_phys', n_entries)
    raw_jet_phi = extract_array(minbias, 'jet_phi_phys', n_entries)
    raw_cmssw_tau = extract_array(minbias, 'jet_tauscore', n_entries)
    raw_cmssw_taupt = extract_array(minbias, 'jet_taupt', n_entries)

    raw_inputs = np.asarray(extract_nn_inputs(minbias, input_vars, n_entries=n_entries))
    raw_pred_score, raw_pt_correction = model.predict(raw_inputs)

    raw_tau_score_sum = raw_pred_score[:,class_labels['taup']] + raw_pred_score[:, class_labels['taum']]
    raw_tau_plus = raw_pred_score[:,class_labels['taup']]
    raw_tau_minus = raw_pred_score[:, class_labels['taum']]

    #Count number of total event
    n_events = len(np.unique(raw_event_id))
    print("Total number of minbias events: ", n_events)

    #Extract the minpt and tau score from cmssw
    cmssw_event_id, pt_min, cmssw_pt_min, cmssw_tau_min = cmssw_pt_score(raw_event_id, raw_jet_pt, raw_jet_eta, raw_jet_phi, raw_cmssw_tau, raw_cmssw_taupt)
    model_event_id, model_pt_min, model_tau_topo = model_pt_score(raw_event_id, raw_tau_score_sum, raw_tau_plus, raw_tau_minus, raw_jet_pt, raw_pt_correction, raw_jet_eta, raw_jet_phi)

    event_id_cmssw = cmssw_event_id[cmssw_tau_min > WPs_CMSSW["tau"]]

    #Load the working points for tau topo
    event_id_model = model_event_id[model_tau_topo > model_NN_WP]

    #Total number of unique event
    n_event = len(np.unique(raw_event_id))
    minbias_rate_no_nn = []
    minbias_rate_cmssw = []
    minbias_rate_model = []

    # Initialize lists for uncertainties (Poisson)
    uncertainty_no_nn = []
    uncertainty_cmssw = []
    uncertainty_model = []

    pt_cuts =  list(np.arange(0,100,2))
    for pt_cut in pt_cuts:

        print("pT Cut: ", pt_cut)
        n_pass_no_nn = len(np.unique(ak.flatten(cmssw_event_id[pt_min > pt_cut])))
        n_pass_cmssw = len(np.unique(ak.flatten(event_id_cmssw[cmssw_pt_min[cmssw_tau_min > WPs_CMSSW["tau"]] > pt_cut])))
        n_pass_model = len(np.unique(ak.flatten(event_id_model[model_pt_min[model_tau_topo > model_NN_WP] > pt_cut])))
        print('------------')

        minbias_rate_no_nn.append((n_pass_no_nn/n_event)*MINBIAS_RATE)
        minbias_rate_cmssw.append((n_pass_cmssw/n_event)*MINBIAS_RATE)
        minbias_rate_model.append((n_pass_model/n_event)*MINBIAS_RATE)

        # Poisson uncertainty is sqrt(N) where N is the number of events passing the cut
        uncertainty_no_nn.append(np.sqrt(n_pass_no_nn) / n_event * MINBIAS_RATE)
        uncertainty_cmssw.append(np.sqrt(n_pass_cmssw) / n_event * MINBIAS_RATE)
        uncertainty_model.append(np.sqrt(n_pass_model) / n_event * MINBIAS_RATE)

    #Plot
    fig, ax = plt.subplots(1, 1, figsize=style.FIGURE_SIZE)
    hep.cms.label(llabel=style.CMSHEADER_LEFT,rlabel=style.CMSHEADER_RIGHT,ax=ax,fontsize=style.MEDIUM_SIZE)

    # Plot the trigger rates
    ax.plot(pt_cuts, minbias_rate_no_nn, c=style.color_cycle[2], label=r'Dijet trigger without $\tau_{h}$ idenficitation', linewidth=style.LINEWIDTH)
    ax.plot(pt_cuts, minbias_rate_cmssw, c=style.color_cycle[0], label=r'NN PUPPI $\tau^\pm$ trigger', linewidth=style.LINEWIDTH)
    ax.plot(pt_cuts, minbias_rate_model, c=style.color_cycle[1],label=r'Multiclass $\tau$ topology trigger', linewidth=style.LINEWIDTH)

    # Add uncertainty bands
    ax.fill_between(pt_cuts,
                    np.array(minbias_rate_no_nn) - np.array(uncertainty_no_nn),
                    np.array(minbias_rate_no_nn) + np.array(uncertainty_no_nn),
                    color=style.color_cycle[2],
                    alpha=0.3)
    ax.fill_between(pt_cuts,
                    np.array(minbias_rate_cmssw) - np.array(uncertainty_cmssw),
                    np.array(minbias_rate_cmssw) + np.array(uncertainty_cmssw),
                    color=style.color_cycle[0],
                    alpha=0.3)
    ax.fill_between(pt_cuts,
                    np.array(minbias_rate_model) - np.array(uncertainty_model),
                    np.array(minbias_rate_model) + np.array(uncertainty_model),
                    color=style.color_cycle[1],
                    alpha=0.3)

    # Set plot properties
    ax.set_yscale('log')
    ax.set_ylabel(r"VBF H$\to \tau_h \tau_h$ trigger rate [kHz]")
    ax.set_xlabel(r"Min($p^1_T$,$p^2_T$) [GeV]")
    ax.legend(loc='upper right', fontsize=style.MEDIUM_SIZE)

    # Save the plot
    plot_dir = os.path.join(model_dir, 'plots/physics/tautau_topo')
    fig.savefig(os.path.join(plot_dir, "tautau_topo_BkgRate.pdf"), bbox_inches='tight')
    fig.savefig(os.path.join(plot_dir, "tautau_topo_BkgRate.png"), bbox_inches='tight')
    return

#------ Plot efficiency
def ratio_2D(nume, deno):
    ratio = np.divide(nume, deno, where=deno != 0)
    ratio[deno == 0] = np.nan
    return ratio

def plot_2D_ratio(ratio, pt_edges, plot_dir, figname="VBF_eff_CMSSW"):
    extent = [pt_edges[0], pt_edges[-1], pt_edges[0], pt_edges[-1]]

    fig,ax = plt.subplots(1,1,figsize=style.FIGURE_SIZE)
    hep.cms.label(llabel=style.CMSHEADER_LEFT,rlabel=style.CMSHEADER_RIGHT,ax=ax,fontsize=style.MEDIUM_SIZE-2)

    # Use ax.imshow and save the returned image for the colorbar
    im = ax.imshow(ratio.T, origin='lower', extent=extent, vmin=0, vmax=0.5, aspect='auto')
    fig.colorbar(im, ax=ax)

    ax.set_xlabel(r"Gen. $\tau_h$ $p_T^1$ [GeV]")
    ax.set_ylabel(r"Gen. $\tau_h$ $p_T^2$ [GeV]")

    fig.savefig(f'{plot_dir}/{figname}.png', bbox_inches='tight')
    fig.savefig(f'{plot_dir}/{figname}.pdf', bbox_inches='tight')

def topo_eff(model_dir, tau_eff_filepath, target_rate=28, tree='jetntuple/Jets', n_entries=100000):

    model=load_qmodel(os.path.join(model_dir, "model/saved_model.h5"))

    #Load the signal data
    signal = uproot.open(tau_eff_filepath)[tree]

    # Load the inputs
    with open(os.path.join(model_dir, "input_vars.json"), "r") as f: input_vars = json.load(f)
    with open(os.path.join(model_dir, "class_label.json"), "r") as f: class_labels = json.load(f)

    # mask non visible gen taus
    raw_jet_genpt = extract_array(signal, 'jet_genmatch_pt', n_entries)
    pt_mask = (raw_jet_genpt != 0)

    raw_jet_genpt = raw_jet_genpt[pt_mask]
    raw_event_id = extract_array(signal, 'event', n_entries)[pt_mask]
    raw_jet_pt = extract_array(signal, 'jet_pt', n_entries)[pt_mask]
    raw_jet_genmass = extract_array(signal, 'jet_genmatch_mass', n_entries)[pt_mask]
    raw_jet_geneta = extract_array(signal, 'jet_genmatch_eta', n_entries)[pt_mask]
    raw_jet_genphi = extract_array(signal, 'jet_genmatch_phi', n_entries)[pt_mask]
    raw_jet_eta = extract_array(signal, 'jet_eta_phys', n_entries)[pt_mask]
    raw_jet_phi = extract_array(signal, 'jet_phi_phys', n_entries)[pt_mask]

    raw_cmssw_tau = extract_array(signal, 'jet_tauscore', n_entries)[pt_mask]
    raw_cmssw_taupt = extract_array(signal, 'jet_taupt', n_entries)[pt_mask]

    #NN related
    raw_inputs = np.asarray(extract_nn_inputs(signal, input_vars, n_entries=n_entries))[pt_mask]
    raw_pred_score, raw_pt_correction = model.predict(raw_inputs)

    #Check if the working point have been derived
    WP_path = os.path.join(model_dir, "plots/physics/tautau_topo/working_point.json")

    #Get derived working points
    if os.path.exists(WP_path):
        with open(WP_path, "r") as f:  WPs = json.load(f)
        model_NN_WP = WPs['NN']
        model_PT_WP = WPs['PT']
    else:
        raise Exception("Working point does not exist. Run with --deriveWPs first.")

    raw_tau_score_sum = raw_pred_score[:,class_labels['taup']] + raw_pred_score[:,class_labels['taum']]
    raw_tau_plus = raw_pred_score[:,class_labels['taup']]
    raw_tau_minus = raw_pred_score[:,class_labels['taum']]

    #Count number of total event
    n_events = len(np.unique(raw_event_id))
    print("Total number of signal events: ", n_events)

    #Group these attributes by event id, and filter out groups that don't have at least 2 elements
    event_id, grouped_arrays  = group_id_values_topo(raw_event_id, raw_tau_score_sum, raw_tau_plus, raw_tau_minus, raw_jet_pt, raw_jet_genmass, raw_jet_genpt, raw_jet_geneta, raw_jet_genphi, raw_pt_correction.flatten(), raw_jet_eta, raw_jet_phi, raw_cmssw_tau, raw_cmssw_taupt, num_elements=2)

    # Extract the grouped arrays
    tau_plus, tau_minus, jet_pt, jet_genmass, jet_genpt, jet_geneta, jet_genphi, jet_pt_correction, jet_eta, jet_phi, cmssw_tau, cmssw_taupt = grouped_arrays
    genpt1, genpt2 = np.asarray(jet_genpt[:,0]), np.asarray(jet_genpt[:,1])

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

    pt = np.vstack([pt1, pt2]).transpose()
    pt_min_model = np.min(pt, axis=1)

    #Get cmssw attribubtes to calculate the rate
    cmssw_pt1, cmssw_pt2 = np.asarray(cmssw_taupt[:,0][cuts]), np.asarray(cmssw_taupt[:,1][cuts])
    cmssw_pt = np.vstack([cmssw_pt1, cmssw_pt2]).transpose()
    cmssw_pt_min = np.min(cmssw_pt, axis=1)

    #Do similar thing for the tau score
    cmssw_tau1, cmssw_tau2 = np.asarray(cmssw_tau[:,0][cuts]), np.asarray(cmssw_tau[:,1][cuts])
    cmssw_tau = np.vstack([cmssw_tau1, cmssw_tau2]).transpose()
    cmssw_tau_min =  np.min(cmssw_tau, axis=1)

    #Create histograms to contain the gen pts
    pt_edges = np.arange(0, 210, 15).tolist()
    pt_edges = np.concatenate((np.arange(0, 100, 10), np.arange(100, 160, 20), [200]))

    all_genpt = Hist(hist.axis.Variable(pt_edges, name="genpt1", label="genpt1"),
                    hist.axis.Variable(pt_edges, name="genpt2", label="genpt2"))
    cmssw_pt = Hist(hist.axis.Variable(pt_edges, name="genpt1", label="genpt1"),
                    hist.axis.Variable(pt_edges, name="genpt2", label="genpt2"))
    model_pt = Hist(hist.axis.Variable(pt_edges, name="genpt1", label="genpt1"),
                    hist.axis.Variable(pt_edges, name="genpt2", label="genpt2"))

    all_genpt.fill(genpt1=genpt1, genpt2=genpt2)

    cmssw_selection = (cmssw_tau_min > WPs_CMSSW['tau']) & (cmssw_pt_min > WPs_CMSSW['tau_l1_pt'])
    cmssw_pt.fill(genpt1=genpt1[cuts][cmssw_selection], genpt2=genpt2[cuts][cmssw_selection])

    model_selection = (pt_min_model > model_PT_WP) & (tau_topo_score[cuts] > model_NN_WP)
    model_pt.fill(genpt1=genpt1[cuts][model_selection], genpt2=genpt2[cuts][model_selection])

    cmssw_ratio = ratio_2D(cmssw_pt, all_genpt)
    model_ratio = ratio_2D(model_pt, all_genpt)
    model_vs_cmssw_ratio = ratio_2D(model_pt, cmssw_pt)

    plot_dir = os.path.join(model_dir, 'plots/physics/tautau_topo')

    #Plot them side by side
    fig_width = 2.5 * style.FIGURE_SIZE[0]
    fig_height = style.FIGURE_SIZE[1]
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height))

    extent = [pt_edges[0], pt_edges[-1], pt_edges[0], pt_edges[-1]]

    # Plot first efficiency ratio (e.g., CMSSW efficiency)
    im0 = axes[0].pcolormesh(pt_edges, pt_edges, cmssw_ratio.T, vmin=0, vmax=0.5)
    axes[0].set_xlabel(r"Gen. $\tau_h$ $p_T^1$ [GeV]")
    axes[0].set_ylabel(r"Gen. $\tau_h$ $p_T^2$ [GeV]")
    axes[0].set_title(f"NN PUPPI Tau", pad=45)
    hep.cms.label(llabel=style.CMSHEADER_LEFT, rlabel=style.CMSHEADER_RIGHT, ax=axes[0], fontsize=style.MEDIUM_SIZE-2)

    # Plot second efficiency ratio (e.g., Model efficiency)
    im1 = axes[1].pcolormesh(pt_edges, pt_edges, model_ratio.T, vmin=0, vmax=0.5)
    axes[1].set_xlabel(r"Gen. $\tau_h$ $p_T^1$ [GeV]")
    axes[1].set_ylabel(r"Gen. $\tau_h$ $p_T^2$ [GeV]")
    axes[1].set_title(f"Multiclass Tagger", pad=45)
    hep.cms.label(llabel=style.CMSHEADER_LEFT, rlabel=style.CMSHEADER_RIGHT, ax=axes[1], fontsize=style.MEDIUM_SIZE-2)

    # Add a common colorbar
    cbar = fig.colorbar(im1, ax=axes.ravel().tolist(), label=f'Efficiency @ {target_rate} kHz', orientation='vertical')

    # Save and show the plot
    fig.savefig(f'{plot_dir}/topo_vbf_eff.pdf', bbox_inches='tight')
    fig.savefig(f'{plot_dir}/topo_vbf_eff.png', bbox_inches='tight')

    # Ratio plot model vs CMSSW
    fig_height = style.FIGURE_SIZE[1] * 1.1
    fig_width = style.FIGURE_SIZE[0] * 1.4
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = GridSpec(2, 5, width_ratios=[0.1, 0.2, 0.5, 4, 1], height_ratios=[1.05, 4], hspace=0.05, wspace=0.11)

    # Main heatmap
    ax_title = fig.add_subplot(gs[1, 0])
    ax_title.axis("off")
    ax_title.text(0, 0.62, r"$\epsilon$ Jet Tagger / $\epsilon$ PUPPI $\tau$", fontsize=style.MEDIUM_SIZE, rotation=90)

    ax_main = fig.add_subplot(gs[1, 3])
    model_vs_cmssw_ratio[np.isinf(model_vs_cmssw_ratio)] = np.nan
    divnorm = matplotlib.colors.TwoSlopeNorm(vmin=0., vcenter=1., vmax=5.)
    im = ax_main.pcolormesh(pt_edges, pt_edges, model_vs_cmssw_ratio.T, norm=divnorm, cmap='coolwarm')
    ax_main.set_xlabel(r"Gen. $\tau_h$ $p_T^1$ [GeV]")
    ax_main.set_ylabel(r"Gen. $\tau_h$ $p_T^2$ [GeV]")
    ax_main.set_xticks(ax_main.get_xticks()[:-1])

    # Top histogram
    ax_top = fig.add_subplot(gs[0, 3], sharex=ax_main)
    counts_pt1, _ =  np.histogram(genpt1, pt_edges)
    counts_pt1_normalized = counts_pt1 / np.sum(counts_pt1)
    ax_top.bar(pt_edges[:-1], counts_pt1_normalized, width=np.diff(pt_edges), align='edge', color='gray', alpha=0.7)
    ax_top.set_yticks([0, .15, .3])
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: '{:.0f}'.format(y) if y.is_integer() else '{:.2f}'.format(y)))
    hep.cms.label(llabel=style.CMSHEADER_LEFT, rlabel=style.CMSHEADER_RIGHT, ax=ax_top, fontsize=style.MEDIUM_SIZE-2)

    # Right histogram
    ax_right = fig.add_subplot(gs[1, 4], sharey=ax_main)
    counts_pt2, _ =  np.histogram(genpt2, pt_edges)
    counts_pt2_normalized = counts_pt2 / np.sum(counts_pt2)
    ax_right.barh(pt_edges[:-1], counts_pt2_normalized, height=np.diff(pt_edges), align='edge', color='gray', alpha=0.7)
    ax_right.set_xticks([0, .15, .3])
    ax_right.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: '{:.0f}'.format(x) if x.is_integer() else '{:.2f}'.format(x)))
    ax_right.tick_params(axis="y", labelleft=False)

    # Add colorbar
    ax_bar = fig.add_subplot(gs[1, 1])
    fig.colorbar(im, cax=ax_bar, aspect=10)
    fig.savefig(f'{plot_dir}/topo_vbf_eff_model_cmssw_ratio.pdf', bbox_inches='tight')
    fig.savefig(f'{plot_dir}/topo_vbf_eff_model_cmssw_ratio.png', bbox_inches='tight')

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
    parser.add_argument('--BkgRate', action='store_true', help='plot background rate for VBF->tautau')
    parser.add_argument('--eff', action='store_true', help='plot efficiency for VBF-> tautau')

    #Other controls
    parser.add_argument('-n','--n_entries', type=int, default=10000, help = 'Number of data entries in root file to run over, can speed up run time, set to None to run on all data entries')
    parser.add_argument('--tree', default='jetntuple/Jets', help='Tree within the ntuple containing the jets')

    args = parser.parse_args()

    if args.deriveWPs:
        derive_diTaus_topo_WPs(args.model_dir, args.minbias, n_entries=args.n_entries, tree=args.tree)
    elif args.BkgRate:
        plot_bkg_rate_ditau_topo(args.model_dir, args.minbias, n_entries=args.n_entries, tree=args.tree)
    elif args.eff:
        topo_eff(args.model_dir, args.vbf_sample, n_entries=args.n_entries, tree=args.tree)
    #     eff_ditau(args.model_dir, args.vbf_sample, n_entries=args.n_entries, eta_region='endcap', tree=args.tree)
