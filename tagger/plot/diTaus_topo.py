"""
Script to plot all di-taus topology trigger related physics performance plot
"""

import os, json
from argparse import ArgumentParser

import awkward as ak
import numpy as np
import uproot
import hist
from hist import Hist

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import mplhep as hep
import tagger.plot.style as style

style.set_style()

#Interpolation of working point
from scipy.interpolate import interp1d

#Imports from other modules
from tagger.data.tools import extract_array, extract_nn_inputs, group_id_values, sort_arrays
from common import MINBIAS_RATE, WPs_CMSSW, find_rate, plot_ratio, delta_r, eta_region_selection, get_bar_patch_data, x_vs_y
from tagger.model.common import fromFolder

def calculate_topo_score(tau_plus, tau_minus, bkg, apply_light=False):
    #2=tau positive, 3=tau_negative

    p1 = np.array(tau_plus[:,0]*tau_minus[:,1])
    p2 = np.array(tau_minus[:,0]*tau_plus[:,1])

    b =  np.array(bkg[:,0]*bkg[:,1])

    out = x_vs_y(p1+p2, b, apply_light)

    return out

def pick_and_plot_topo(rate_list, pt_list, nn_list, model, target_rate = 28, RateRange=1.0):

    plot_dir = os.path.join(model.output_directory, 'plots/physics/tautau_topo')
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

def derive_diTaus_topo_WPs(model, minbias_path, n_entries=100, tree='jetntuple/Jets', target_rate=28):
    """
    Derive ditau topology working points.
    Using a new score that uses the charge definition in the jet tagger.

    topology_score = (tau_p_1 + tau_p_2)*(tau_m_1 + tau_m_2)
    """

    #Load the minbias data
    minbias = uproot.open(minbias_path)[tree]

    raw_event_id = extract_array(minbias, 'event', n_entries)
    raw_jet_pt = extract_array(minbias, 'jet_pt', n_entries)
    raw_jet_eta = extract_array(minbias, 'jet_eta_phys', n_entries)
    raw_jet_phi = extract_array(minbias, 'jet_phi_phys', n_entries)
    raw_inputs = np.asarray(extract_nn_inputs(minbias, model.input_vars, n_entries=n_entries))
    raw_pred_score, raw_pt_correction = model.predict(raw_inputs)

    apply_light = True
    raw_tau_score_sum = raw_pred_score[:,model.class_labels['taup']] + raw_pred_score[:, model.class_labels['taum']]
    raw_tau_plus = raw_pred_score[:,model.class_labels['taup']]
    raw_tau_minus = raw_pred_score[:, model.class_labels['taum']]
    raw_bkg =  raw_pred_score[:, model.class_labels['gluon']] + raw_pred_score[:, model.class_labels['light']] + raw_pred_score[:, model.class_labels['pileup']]

    #Group these attributes by event id, and filter out groups that don't have at least 2 elements
    event_id, grouped_arrays = group_id_values(raw_event_id, raw_tau_plus, raw_tau_minus, raw_bkg, raw_jet_pt, raw_pt_correction.flatten(), raw_jet_eta, raw_jet_phi, num_elements=2)

    # Extract the grouped arrays
    tau_plus, tau_minus, bkg, jet_pt, jet_pt_correction, jet_eta, jet_phi = grouped_arrays

    #Count number of total event
    n_events = len(np.unique(raw_event_id))
    print("Total number of minbias events: ", n_events)

    if(apply_light):
        base_tau_score = (tau_plus + tau_minus) / (bkg + tau_plus + tau_minus)
    else:
        base_tau_score = (tau_plus + tau_minus)

    corrected_jet_pt = jet_pt * jet_pt_correction 

    eta_cut = 2.172
    eta_pass = np.abs(jet_eta) < eta_cut

    #Define the histograms (pT edge and NN Score edge)
    pT_edges = list(np.arange(20,100,2)) + [1500] #Make sure to capture everything
    NN_edges = list([round(i,4) for i in np.arange(0, 0.6, 0.0005)])

    RateHist = Hist(hist.axis.Variable(pT_edges, name="pt", label="pt"),
                    hist.axis.Variable(NN_edges, name="nn", label="nn"))

    #Derive the rate
    rate_list = []
    pt_list = []
    nn_list = []

    #Loop through the edges and integrate
    #For each pt thresh pick two highest tau-score jets to compute the topo score 
    for pt in pT_edges[:-1]:

        #zero out tau score for jets below pt thresh so we don't pick them
        base_tau_score = ak.where((corrected_jet_pt < pt) | (~eta_pass), 0., base_tau_score)

        #sort arrays by tau score
        tau_plus_s, tau_minus_s, bkg_s, corrected_jet_pt_s, jet_eta_s, jet_phi_s = sort_arrays(base_tau_score, tau_plus, tau_minus, bkg, corrected_jet_pt, jet_eta, jet_phi)

        #calculate delta_r
        eta1, eta2 = jet_eta_s[:, 0], jet_eta_s[:, 1]
        phi1, phi2 = jet_phi_s[:, 0], jet_phi_s[:, 1]
        delta_r_values = delta_r(eta1, phi1, eta2, phi2)

        # Additional cuts recommended here:
        # https://indico.cern.ch/event/1380964/contributions/5852368/attachments/2841655/4973190/AnnualReview_2024.pdf
        # Slide 7
        cuts = (np.abs(eta1) < eta_cut) & (np.abs(eta2) < eta_cut) & (delta_r_values > 0.5)

        tau_topo_score = calculate_topo_score(tau_plus_s, tau_minus_s, bkg_s, apply_light)
        NN_score_min = np.asarray(tau_topo_score[cuts])

        #correct for pt
        pt1, pt2 = np.asarray(corrected_jet_pt_s[:,0][cuts]), np.asarray(corrected_jet_pt_s[:,1][cuts])
        pt_stack = np.vstack([pt1, pt2]).transpose()
        pt_min = np.min(pt_stack, axis=1)

        #fill hist for these tau choices
        RateHist.reset()
        RateHist.fill(pt = pt_min, nn = NN_score_min)


        for NN in NN_edges[:-1]:

            #Calculate the rate
            rate = RateHist[{"pt": slice(pt*1j, pT_edges[-1]*1.0j, sum)}][{"nn": slice(NN*1.0j,1.0j, sum)}]/n_events
            rate_list.append(rate*MINBIAS_RATE)

            #Append the results
            pt_list.append(pt)
            nn_list.append(NN)

    pick_and_plot_topo(rate_list, pt_list, nn_list, model, target_rate=28)

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

def model_pt_score(raw_event_id, raw_tau_score_sum, raw_tau_plus, raw_tau_minus, raw_bkg, raw_jet_pt, raw_pt_correction, raw_jet_eta, raw_jet_phi, apply_light):

    event_id, grouped_arrays  = group_id_values(raw_event_id, raw_tau_plus, raw_tau_minus, raw_bkg, raw_jet_pt, raw_pt_correction.flatten(), raw_jet_eta, raw_jet_phi, num_elements=2, ordering_var=raw_tau_score_sum)

    # Extract the grouped arrays
    tau_plus, tau_minus, bkg, jet_pt, jet_pt_correction, jet_eta, jet_phi = grouped_arrays

    #calculate delta_r
    eta1, eta2 = jet_eta[:, 0], jet_eta[:, 1]
    phi1, phi2 = jet_phi[:, 0], jet_phi[:, 1]
    delta_r_values = delta_r(eta1, phi1, eta2, phi2)

    # Additional cuts recommended here:
    # https://indico.cern.ch/event/1380964/contributions/5852368/attachments/2841655/4973190/AnnualReview_2024.pdf
    # Slide 7
    cuts = (np.abs(eta1) < 2.172) & (np.abs(eta2) < 2.172) & (delta_r_values > 0.5)

    tau_topo_score = calculate_topo_score(tau_plus, tau_minus, bkg, apply_light)

    #correct for pt
    pt1_uncorrected, pt2_uncorrected = np.asarray(jet_pt[:,0][cuts]), np.asarray(jet_pt[:,1][cuts])
    ratio1, ratio2 = np.asarray(jet_pt_correction[:,0][cuts]), np.asarray(jet_pt_correction[:,1][cuts])

    pt1 = pt1_uncorrected*ratio1
    pt2 = pt2_uncorrected*ratio2

    pt = np.vstack([pt1, pt2]).transpose()
    pt_min = np.min(pt, axis=1)

    return event_id[cuts], pt_min, tau_topo_score[cuts]

def plot_bkg_rate_ditau_topo(model, minbias_path, n_entries=100, tree='jetntuple/Jets'):

    #Load the minbias data
    minbias = uproot.open(minbias_path)[tree]
    #Check if the working point have been derived
    WP_path = os.path.join(model.output_directory, "plots/physics/tautau_topo/working_point.json")

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

    raw_inputs = np.asarray(extract_nn_inputs(minbias, model.input_vars, n_entries=n_entries))
    raw_pred_score, raw_pt_correction = model.predict(raw_inputs)

    apply_light = True
    raw_tau_score_sum = raw_pred_score[:,model.class_labels['taup']] + raw_pred_score[:, model.class_labels['taum']]
    raw_tau_plus = raw_pred_score[:,model.class_labels['taup']]
    raw_tau_minus = raw_pred_score[:, model.class_labels['taum']]
    raw_bkg =  raw_pred_score[:, model.class_labels['gluon']] + raw_pred_score[:, model.class_labels['light']] + raw_pred_score[:, model.class_labels['pileup']]

    if(apply_light): raw_tau_score = raw_tau_score_sum / (raw_bkg + raw_tau_score_sum)
    else: raw_tau_escore = raw_tau_score_sum


    #Count number of total event
    n_events = len(np.unique(raw_event_id))
    print("Total number of minbias events: ", n_events)

    #Extract the minpt and tau score from cmssw
    cmssw_event_id, pt_min, cmssw_pt_min, cmssw_tau_min = cmssw_pt_score(raw_event_id, raw_jet_pt, raw_jet_eta, raw_jet_phi, raw_cmssw_tau, raw_cmssw_taupt)
    model_event_id, model_pt_min, model_tau_topo = model_pt_score(raw_event_id, raw_tau_score, raw_tau_plus, raw_tau_minus, raw_bkg, raw_jet_pt, raw_pt_correction, raw_jet_eta, raw_jet_phi, apply_light)

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
    ax.plot(pt_cuts, minbias_rate_cmssw, c=style.color_cycle[0], label=r'NN PUPPI Tau $\tau^\pm$ trigger', linewidth=style.LINEWIDTH)
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
    plot_dir = os.path.join(model.output_directory, 'plots/physics/tautau_topo')
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

def topo_eff(model, tau_eff_filepath, target_rate=28, tree='jetntuple/Jets', tag='vbf', n_entries=100000):

    #Load the signal data
    signal = uproot.open(tau_eff_filepath)[tree]


    raw_event_id = extract_array(signal, 'event', n_entries)
    raw_jet_pt = extract_array(signal, 'jet_pt', n_entries)
    raw_jet_tauflav = extract_array(signal, 'jet_tauflav', n_entries)
    raw_jet_genpt = extract_array(signal, 'jet_genmatch_pt', n_entries)
    raw_jet_geneta = extract_array(signal, 'jet_genmatch_eta', n_entries)
    raw_jet_genphi = extract_array(signal, 'jet_genmatch_phi', n_entries)
    raw_jet_eta = extract_array(signal, 'jet_eta_phys', n_entries)
    raw_jet_phi = extract_array(signal, 'jet_phi_phys', n_entries)

    raw_cmssw_tau = extract_array(signal, 'jet_tauscore', n_entries)
    raw_cmssw_taupt = extract_array(signal, 'jet_taupt', n_entries)

    #NN related
    raw_inputs = np.asarray(extract_nn_inputs(signal, model.input_vars, n_entries=n_entries))
    raw_pred_score, raw_pt_correction = model.predict(raw_inputs)

    #Count number of total event
    n_events = len(np.unique(raw_event_id))
    print("Total number of signal events: ", n_events)


    #Check if the working point have been derived
    WP_path = os.path.join(model.output_directory, "plots/physics/tautau_topo/working_point.json")

    #Get derived working points
    if os.path.exists(WP_path):
        with open(WP_path, "r") as f:  WPs = json.load(f)
        model_NN_WP = WPs['NN']
        model_PT_WP = WPs['PT']
    else:
        raise Exception("Working point does not exist. Run with --deriveWPs first.")

    apply_light = True
    raw_tau_score_sum = raw_pred_score[:,model.class_labels['taup']] + raw_pred_score[:, model.class_labels['taum']]
    raw_tau_plus = raw_pred_score[:,model.class_labels['taup']]
    raw_tau_minus = raw_pred_score[:, model.class_labels['taum']]
    raw_bkg =  raw_pred_score[:, model.class_labels['gluon']] + raw_pred_score[:, model.class_labels['light']]  + raw_pred_score[:, model.class_labels['pileup']]

    #select the two tau candidates
    #want two jets with highest tau score of those jets with pt above threshold
    if(apply_light): raw_tau_score = raw_tau_score_sum / (raw_bkg + raw_tau_score_sum)
    else: raw_tau_score = raw_tau_score_sum

    corrected_jet_pt = raw_jet_pt * raw_pt_correction 

    #zero out tau score for jets below pt thresh so we don't accidentally pick them
    raw_tau_score = ak.where(corrected_jet_pt < model_PT_WP, 0., raw_tau_score)

    #Do same zeroing for CMSSW socres (for testing only)
    #raw_cmssw_tau = ak.where(raw_cmssw_taupt < WPs_CMSSW['tau_l1_pt'], 0., raw_cmssw_tau)

    #Select highest two tau score jets
    event_id, grouped_arrays  = group_id_values(raw_event_id, raw_tau_plus, raw_tau_minus, raw_bkg, raw_jet_pt, raw_pt_correction.flatten(), raw_jet_eta, raw_jet_phi, num_elements=2, ordering_var = raw_tau_score)

    tau_plus, tau_minus, bkg, jet_pt, jet_pt_correction, jet_eta, jet_phi, = grouped_arrays

    #extract separately the highest two CMSSW-score taus
    event_id, grouped_arrays  = group_id_values(raw_event_id, raw_cmssw_tau, raw_cmssw_taupt, raw_jet_eta, raw_jet_phi,  num_elements=2, ordering_var=raw_cmssw_tau)
    cmssw_tau, cmssw_taupt, cmssw_jet_eta, cmssw_jet_phi = grouped_arrays

    #extract separately the two gen taus
    #using gen tau ID to pick the correct ones
    event_id, grouped_arrays  = group_id_values(raw_event_id, raw_jet_tauflav, raw_jet_genpt, raw_jet_geneta, raw_jet_genphi, num_elements=2, ordering_var=raw_jet_tauflav)
    jet_tauflav, jet_genpt, jet_geneta, jet_genphi = grouped_arrays
    genpt1, genpt2 = np.asarray(jet_genpt[:,0]), np.asarray(jet_genpt[:,1])

    #mask to ensure well matched ditau event
    gen_mask = (jet_tauflav[:,0] == 1) & (jet_tauflav[:,0] == 1) & (genpt1 > 1.) & (genpt2 > 1.)
    print("Gen mask eff: %.4f" % np.mean(gen_mask))

    #calculate delta_r
    eta1, eta2 = jet_eta[:, 0], jet_eta[:, 1]
    phi1, phi2 = jet_phi[:, 0], jet_phi[:, 1]
    delta_r_values = delta_r(eta1, phi1, eta2, phi2)

    # Additional cuts recommended here:
    # https://indico.cern.ch/event/1380964/contributions/5852368/attachments/2841655/4973190/AnnualReview_2024.pdf
    # Slide 7
    cuts = gen_mask & (np.abs(eta1) < 2.172) & (np.abs(eta2) < 2.172) & (delta_r_values > 0.5)


    #calculate cuts for CMSSW taus
    cmssw_eta1, cmssw_eta2 = cmssw_jet_eta[:, 0], cmssw_jet_eta[:, 1]
    cmssw_phi1, cmssw_phi2 = cmssw_jet_phi[:, 0], cmssw_jet_phi[:, 1]
    cmssw_delta_r_values = delta_r(cmssw_eta1, cmssw_phi1, cmssw_eta2, cmssw_phi2)

    cmssw_cuts = gen_mask & (np.abs(cmssw_eta1) < 2.172) & (np.abs(cmssw_eta2) < 2.172) & (cmssw_delta_r_values > 0.5)


    tau_topo_score = calculate_topo_score(tau_plus, tau_minus, bkg, apply_light)

    #correct for pt
    pt1_uncorrected, pt2_uncorrected = np.asarray(jet_pt[:,0][cuts]), np.asarray(jet_pt[:,1][cuts])
    ratio1, ratio2 = np.asarray(jet_pt_correction[:,0][cuts]), np.asarray(jet_pt_correction[:,1][cuts])

    pt1 = pt1_uncorrected*ratio1
    pt2 = pt2_uncorrected*ratio2

    pt = np.vstack([pt1, pt2]).transpose()
    pt_min_model = np.min(pt, axis=1)

    #Get cmssw attribubtes to calculate the rate
    cmssw_pt1, cmssw_pt2 = np.asarray(cmssw_taupt[:,0][cmssw_cuts]), np.asarray(cmssw_taupt[:,1][cmssw_cuts])
    cmssw_pt = np.vstack([cmssw_pt1, cmssw_pt2]).transpose()
    cmssw_pt_min = np.min(cmssw_pt, axis=1)


    #Do similar thing for the tau score
    cmssw_tau1, cmssw_tau2 = np.asarray(cmssw_tau[:,0][cmssw_cuts]), np.asarray(cmssw_tau[:,1][cmssw_cuts])
    cmssw_tau_both = np.vstack([cmssw_tau1, cmssw_tau2]).transpose()
    cmssw_tau_min =  np.min(cmssw_tau_both, axis=1)

    #Create histograms to contain the gen pts
    pt_edges = np.arange(0, 210, 15).tolist()
    pt_edges = np.concatenate((np.arange(0, 100, 10), np.arange(100, 160, 20), [200]))

    all_genpt = Hist(hist.axis.Variable(pt_edges, name="genpt1", label="genpt1"),
                    hist.axis.Variable(pt_edges, name="genpt2", label="genpt2"))
    cmssw_pt = Hist(hist.axis.Variable(pt_edges, name="genpt1", label="genpt1"),
                    hist.axis.Variable(pt_edges, name="genpt2", label="genpt2"))
    model_pt = Hist(hist.axis.Variable(pt_edges, name="genpt1", label="genpt1"),
                    hist.axis.Variable(pt_edges, name="genpt2", label="genpt2"))

    all_genpt.fill(genpt1=genpt1[gen_mask], genpt2=genpt2[gen_mask])

    cmssw_selection = (cmssw_tau_min > WPs_CMSSW['tau']) & (cmssw_pt_min > WPs_CMSSW['tau_l1_pt'])
    cmssw_pt.fill(genpt1=genpt1[cmssw_cuts][cmssw_selection], genpt2=genpt2[cmssw_cuts][cmssw_selection])

    model_selection = (pt_min_model > model_PT_WP) & (tau_topo_score[cuts] > model_NN_WP)
    model_pt.fill(genpt1=genpt1[cuts][model_selection], genpt2=genpt2[cuts][model_selection])

    gen_pt = np.vstack([genpt1[gen_mask], genpt2[gen_mask]]).transpose()
    gen_pt_min = np.min(gen_pt, axis=1)
    denom = gen_mask

    #write out total eff to text file
    total_eff_model = np.sum(model_selection) / np.sum(denom)
    total_eff_cmssw = np.sum(cmssw_selection) / np.sum(denom)

    total_eff_model_unc = np.sqrt(np.sum(model_selection)) / np.sum(denom)
    total_eff_cmssw_unc = np.sqrt(np.sum(cmssw_selection)) / np.sum(denom)

    plot_dir = os.path.join(model.output_directory, 'plots/physics/tautau_topo')

    outname = plot_dir + "/TotalEff.txt"
    with open(outname, "w") as outfile:
        outfile.write("Total diTau Eff \n")
        outfile.write("Multiclass NN %.4f +/- %.4f \n" % (total_eff_model, total_eff_model_unc))
        outfile.write("CMSSW  %.4f +/- %.4f \n" % (total_eff_cmssw, total_eff_cmssw_unc))

    print("Total diTau Eff")
    print("Multiclass NN %.4f +/- %.4f" % (total_eff_model, total_eff_model_unc))
    print("CMSSW  %.4f +/- %.4f"  % (total_eff_cmssw, total_eff_cmssw_unc))

    #make 2D plots of efficiency
    cmssw_ratio = ratio_2D(cmssw_pt, all_genpt)
    model_ratio = ratio_2D(model_pt, all_genpt)
    model_vs_cmssw_ratio = ratio_2D(model_pt, cmssw_pt)



    #write out total eff to text file
    total_eff_model = np.mean(model_selection)
    total_eff_cmssw = np.mean(cmssw_selection)

    outname = plot_dir + "/TotalEff.txt"
    with open(outname, "w") as outfile:
        outfile.write("Total diTau Eff \n")
        outfile.write("Multiclass NN %.4f \n" % total_eff_model)
        outfile.write("CMSSW  %.4f \n" % total_eff_cmssw)

    print("Total diTau Eff")
    print("Multiclass NN %.4f" % total_eff_model)
    print("CMSSW  %.4f" % total_eff_cmssw)


    #Plot them side by side
    fig_width = 2.5 * style.FIGURE_SIZE[0]
    fig_height = style.FIGURE_SIZE[1]
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height))

    extent = [pt_edges[0], pt_edges[-1], pt_edges[0], pt_edges[-1]]

    eff_str = r"$\int \epsilon$"
    model_eff = np.round(total_eff_model, 2)
    cmssw_eff = np.round(total_eff_cmssw, 2)

    # Plot first efficiency ratio (e.g., CMSSW efficiency)
    im0 = axes[0].pcolormesh(pt_edges, pt_edges, cmssw_ratio.T, vmin=0, vmax=0.5)
    axes[0].set_xlabel(r"Gen. $\tau_h$ $p_T^1$ [GeV]")
    axes[0].set_ylabel(r"Gen. $\tau_h$ $p_T^2$ [GeV]")
    axes[0].set_title(f"NN PUPPI Tau Efficiency @ {target_rate}kHz, {eff_str}={cmssw_eff}", pad=45)
    hep.cms.label(llabel=style.CMSHEADER_LEFT, rlabel=style.CMSHEADER_RIGHT, ax=axes[0], fontsize=style.MEDIUM_SIZE-2)

    # Plot second efficiency ratio (e.g., Model efficiency)
    im1 = axes[1].pcolormesh(pt_edges, pt_edges, model_ratio.T, vmin=0, vmax=0.5)
    axes[1].set_xlabel(r"Gen. $\tau_h$ $p_T^1$ [GeV]")
    axes[1].set_ylabel(r"Gen. $\tau_h$ $p_T^2$ [GeV]")
    axes[1].set_title(f"Multiclass Tagger Efficiency @ {target_rate}kHz, {eff_str}={model_eff}", pad=45)
    hep.cms.label(llabel=style.CMSHEADER_LEFT, rlabel=style.CMSHEADER_RIGHT, ax=axes[1], fontsize=style.MEDIUM_SIZE-2)

    # Add a common colorbar
    cbar = fig.colorbar(im1, ax=axes.ravel().tolist())

    # Save and show the plot
    fig.savefig(f'{plot_dir}/topo_{tag}_eff.pdf', bbox_inches='tight')
    fig.savefig(f'{plot_dir}/topo_{tag}_eff.png', bbox_inches='tight')

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
    fig.savefig(f'{plot_dir}/topo_{tag}_eff_model_cmssw_ratio.pdf', bbox_inches='tight')
    fig.savefig(f'{plot_dir}/topo_{tag}_eff_model_cmssw_ratio.png', bbox_inches='tight')

    return

if __name__ == "__main__":
    """
    2 steps:

    1. Derive working points: python diTaus.py --deriveWPs
    2. Run efficiency based on the derived working points: python diTaus.py --eff
    """
    parser = ArgumentParser()
    parser.add_argument('-m','--model_dir', default='output/baseline', help = 'Input model')
    parser.add_argument('-v', '--vbf_sample', default='/eos/cms/store/cmst3/group/l1tr/sewuchte/l1teg/fp_jettuples_090125_addGenH/VBFHToTauTau_PU200.root' , help = 'Signal sample for VBF -> ditaus')
    parser.add_argument('--minbias', default='/eos/cms/store/cmst3/group/l1tr/sewuchte/l1teg/fp_jettuples_090125/MinBias_PU200.root' , help = 'Minbias sample for deriving rates')
    parser.add_argument('--tag', default='vbf' , help = 'Tag for plot label')

    #Different modes
    parser.add_argument('--deriveWPs', action='store_true', help='derive the working points for di-taus')
    parser.add_argument('--BkgRate', action='store_true', help='plot background rate for VBF->tautau')
    parser.add_argument('--eff', action='store_true', help='plot efficiency for VBF-> tautau')

    #Other controls
    parser.add_argument('-n','--n_entries', type=int, default=10000, help = 'Number of data entries in root file to run over, can speed up run time, set to None to run on all data entries')
    parser.add_argument('--tree', default='jetntuple/Jets', help='Tree within the ntuple containing the jets')

    args = parser.parse_args()

    model = fromFolder(args.model_dir)

    if args.deriveWPs:
        derive_diTaus_topo_WPs(model, args.minbias, n_entries=args.n_entries, tree=args.tree)
    elif args.BkgRate:
        plot_bkg_rate_ditau_topo(model, args.minbias, n_entries=args.n_entries, tree=args.tree)
    elif args.eff:
        topo_eff(model, args.vbf_sample, n_entries=args.n_entries, tree=args.tree, tag=args.tag)
