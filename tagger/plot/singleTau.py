"""
Script to plot all di-taus related physics performance plot
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
import mplhep as hep
import tagger.plot.style as style

style.set_style()

#Interpolation of working point
from scipy.interpolate import interp1d

#Imports from other modules
from tagger.data.tools import extract_array, extract_nn_inputs, group_id_values
from tagger.model.common import fromFolder
from common import MINBIAS_RATE, WPs_CMSSW, find_rate, plot_ratio, delta_r, eta_region_selection, get_bar_patch_data

def get_interp_func(WP_path):
    #Get derived working points
    if os.path.exists(WP_path):
        with open(WP_path, "r") as f:  WPs = json.load(f)
        model_NN_WP = WPs['NNs']
        model_pt_WP = WPs['PTs']
    else:
        raise Exception("Model working point does not exist. Run with --deriveWPs first.")

    #There are repeated WP entries per pt because of rate range
    #Take conservative threshold -> max highest per pt
    pruned_pts = []
    pruned_WPs = []

    for i in range(len(model_NN_WP)):

        if(model_pt_WP[i] not in pruned_pts):
            pruned_pts.append(model_pt_WP[i])
            pruned_WPs.append(model_NN_WP[i])
        else:
            pruned_WPs[-1] = max(pruned_WPs[-1], model_NN_WP[i])


    minWP, maxWP = np.min(model_NN_WP), np.max(model_NN_WP)
    interp_func = interp1d(model_pt_WP, model_NN_WP, kind='linear', fill_value=(maxWP, minWP), bounds_error=False)

    return interp_func

def tau_score(preds, class_labels):
    tau_index = [class_labels['taup'], class_labels['taum'], class_labels['electron']]

    tau = sum([preds[:,idx] for idx in tau_index] )
    bkg = preds[:,class_labels['pileup']] + preds[:,class_labels['gluon']] + preds[:,class_labels['light']]
    #bkg = preds[:,class_labels['gluon']] + preds[:,class_labels['light']]

    return tau / (tau + bkg)



def pick_and_plot_tau(rate_list, pt_list, nn_list, model, target_rate = 31, RateRange = 1.0, label=""):
    """
    Pick the working points and plot
    """

    plot_dir = os.path.join(model.output_directory, 'plots/physics/tau')
    os.makedirs(plot_dir, exist_ok=True)

    fig,ax = plt.subplots(1,1,figsize=style.FIGURE_SIZE)
    hep.cms.label(llabel=style.CMSHEADER_LEFT,rlabel=style.CMSHEADER_RIGHT,ax=ax,fontsize=style.MEDIUM_SIZE-2)
    im = ax.scatter(nn_list, pt_list, c=rate_list, s=500, marker='s',
                    cmap='Spectral_r',
                    linewidths=0,
                    norm=matplotlib.colors.LogNorm())

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r'Single-tau rate [kHZ]')

    ax.set_ylabel(r"Min L1 $p_T$ [GeV]")
    ax.set_xlabel(r"$\tau$ Score Threshold")

    ax.set_xlim([0,1.0])
    ax.set_ylim([50,200])

    #Find the target rate points, plot them and print out some info as well
    target_rate_idx = find_rate(rate_list, target_rate = target_rate, RateRange=RateRange)

    #Get the coordinates
    target_rate_NN = [float(nn_list[i]) for i in target_rate_idx] # NN cut dimension
    target_rate_PT = [float(pt_list[i]) for i in target_rate_idx] # HT cut dimension


    # Create an interpolation function
    interp_func = interp1d(target_rate_PT, target_rate_NN, kind='linear', fill_value='extrapolate')

    # Export the working point
    working_point = {"PTs": target_rate_PT, "NNs": target_rate_NN}

    with open(os.path.join(plot_dir, label+ "working_point.json"), "w") as f:
        json.dump(working_point, f, indent=4)

    # Generate 100 points spanning the entire pT range visible on the plot.
    pT_full = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 100)

    # Evaluate the interpolation function to obtain NN values for these pT points.
    NN_full = interp_func(pT_full)
    ax.plot(NN_full, pT_full, linewidth=style.LINEWIDTH, color ='firebrick', label = r"${} \pm {}$ kHz".format(target_rate, RateRange))

    #Just plot the points instead of the interpolation
    #ax.plot(target_rate_NN, target_rate_PT, linewidth=style.LINEWIDTH, color ='firebrick', label = r"${} \pm {}$ kHz".format(target_rate, RateRange))

    ax.legend(loc='upper right', fontsize=style.MEDIUM_SIZE)
    plt.savefig(f"{plot_dir}/{label}tau_WPs.pdf", bbox_inches='tight')
    plt.savefig(f"{plot_dir}/{label}tau_WPs.png", bbox_inches='tight')

def derive_tau_WPs(model, minbias_path, target_rate=31, cmssw_model=False, n_entries=100, tree='jetntuple/Jets'):
    """
    Derive the single tau rate.
    Seed definition can be found here (2024 Annual Review):

    https://indico.cern.ch/event/1380964/contributions/5852368/attachments/2841655/4973190/AnnualReview_2024.pdf

    Single Puppi Tau Seed, 31 kHZ based on the definition above.
    """


    #Load the minbias data
    minbias = uproot.open(minbias_path)[tree]

    raw_event_id = extract_array(minbias, 'event', n_entries)
    raw_jet_pt = extract_array(minbias, 'jet_pt', n_entries)
    raw_jet_eta = extract_array(minbias, 'jet_eta_phys', n_entries)
    raw_jet_phi = extract_array(minbias, 'jet_phi_phys', n_entries)

    raw_inputs = extract_nn_inputs(minbias, model.input_vars, n_entries=n_entries)



    #Count number of total event
    n_events = len(np.unique(raw_event_id))
    print("Total number of minbias events: ", n_events)

    #Group these attributes by event id, and filter out groups that don't have at least 1 element
    event_id, grouped_arrays  = group_id_values(raw_event_id, raw_jet_pt, raw_jet_eta, raw_jet_phi, raw_inputs, num_elements=1)

    # Extract the grouped arrays
    # Jet pt is already sorted in the producer, no need to do it here
    jet_pt, jet_eta, jet_phi, jet_nn_inputs = grouped_arrays

    # Additional cuts recommended here:
    # https://indico.cern.ch/event/1380964/contributions/5852368/attachments/2841655/4973190/AnnualReview_2024.pdf
    # Slide 7
    eta_cut = 2.172
    pt_cut = 30.

    #flatten for NN eval
    jet_pts = np.asarray(ak.flatten(jet_pt))
    jet_etas = np.asarray(ak.flatten(jet_eta))
    jet_inputs = np.asarray(ak.flatten(jet_nn_inputs))

    cuts = (jet_pts > pt_cut) & (np.abs(jet_etas) < eta_cut)


    all_scores = np.zeros_like(jet_pts)
    all_corr_pts = np.zeros_like(jet_pts)

    if(cmssw_model): #scores from CMSSW model
        all_corr_pts = extract_array(minbias, 'jet_taupt', n_entries)
        all_scores = extract_array(minbias, 'jet_tauscore', n_entries)

        all_scores = ak.where(~cuts, all_scores, 0.)

    else: #scores from new model

        pred_scores, pt_ratios = model.predict(jet_inputs[cuts])
        all_scores[cuts] = tau_score(pred_scores, model.class_labels)
        all_corr_pts[cuts] = pt_ratios.flatten() * jet_pts[cuts]

    #Reshape to orig shape
    all_scores = ak.unflatten(all_scores, ak.num(jet_pt))
    all_corr_pts = ak.unflatten(all_corr_pts, ak.num(jet_pt))

    #find highest tau score per event
    highest_score = ak.argmax(all_scores, axis=1, keepdims=True)

    NN_scores = all_scores[highest_score]
    NN_pt = all_corr_pts[highest_score]

    NN_scores = np.asarray(NN_scores).flatten()
    NN_pt = np.asarray(NN_pt).flatten()



    #Define the histograms (pT edge and NN Score edge)
    pT_edges = list(np.arange(30,150,2)) + [1500] #Make sure to capture everything
    NN_edges = list([round(i,4) for i in np.arange(0, 0.5, 0.002)]) + list([round(i,4) for i in np.arange(0.5, 1.01, 0.005)])

    RateHist = Hist(hist.axis.Variable(pT_edges, name="pt", label="pt"),
                    hist.axis.Variable(NN_edges, name="nn", label="nn"))

    RateHist.fill(pt = NN_pt, nn = NN_scores)

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
    label = "cmssw_" if cmssw_model else ""
    pick_and_plot_tau(rate_list, pt_list, nn_list, model, target_rate=target_rate, label=label)

    return

def plot_bkg_rate_tau(model, minbias_path, n_entries=500000, tree='jetntuple/Jets' ):
    """
    Plot the background (mimbias) rate w.r.t pT cuts.
    """

    pt_cuts = list(np.arange(0,250,10))

    #Load the minbias data
    minbias = uproot.open(minbias_path)[tree]

    #Impose eta cuts
    jet_eta =  extract_array(minbias, 'jet_eta_phys', n_entries)
    eta_selection = np.abs(jet_eta) < 2.5

    #
    nn_inputs = np.asarray(extract_nn_inputs(minbias, model.input_vars, n_entries=n_entries))

    #Get the NN predictions
    pred_score, ratio = model.predict(nn_inputs[eta_selection])
    model_tau = tau_score(pred_score, model.class_labels )

    #Emulator tau score
    cmssw_tau = extract_array(minbias, 'jet_tauscore', n_entries)[eta_selection]

    #Use event id to track which jets belong to which event.
    event_id = extract_array(minbias, 'event', n_entries)[eta_selection]
    event_id_cmssw = event_id[cmssw_tau > WPs_CMSSW["tau"]]

    #Load the working point from json file
    #Check if the working point have been derived
    WP_path = os.path.join(model.output_directory, "plots/physics/tau/working_point.json")

    #Get derived working points
    if os.path.exists(WP_path):
        with open(WP_path, "r") as f:  WPs = json.load(f)
        tau_wp = WPs['NN']
        tau_pt_wp = WPs['PT']
    else:
        raise Exception("Working point does not exist. Run with --deriveWPs first.")

    event_id_model = event_id[model_tau > tau_wp]

    #Cut on jet pT to extract the rate
    jet_pt = extract_array(minbias, 'jet_pt', n_entries)[eta_selection]


    jet_pt_cmssw = extract_array(minbias, 'jet_taupt', n_entries)[eta_selection][cmssw_tau > WPs_CMSSW["tau"]]
    jet_pt_model = (jet_pt*ratio.flatten())[model_tau > tau_wp]

    #Total number of unique event
    n_event = len(np.unique(event_id))
    minbias_rate_no_nn = []
    minbias_rate_cmssw = []
    minbias_rate_model = []

    # Initialize lists for uncertainties (Poisson)
    uncertainty_no_nn = []
    uncertainty_cmssw = []
    uncertainty_model = []

    for pt_cut in pt_cuts:

        print("pT Cut: ", pt_cut)
        n_pass_no_nn = len(np.unique(event_id[jet_pt > pt_cut]))
        n_pass_cmssw = len(np.unique(event_id_cmssw[jet_pt_cmssw > pt_cut]))
        n_pass_model = len(np.unique(event_id_model[jet_pt_model > pt_cut]))
        print('------------')

        minbias_rate_no_nn.append((n_pass_no_nn/n_event)* MINBIAS_RATE)
        minbias_rate_cmssw.append((n_pass_cmssw/n_event)* MINBIAS_RATE)
        minbias_rate_model.append((n_pass_model/n_event)* MINBIAS_RATE)

        # Poisson uncertainty is sqrt(N) where N is the number of events passing the cut
        uncertainty_no_nn.append(np.sqrt(n_pass_no_nn) / n_event * MINBIAS_RATE)
        uncertainty_cmssw.append(np.sqrt(n_pass_cmssw) / n_event * MINBIAS_RATE)
        uncertainty_model.append(np.sqrt(n_pass_model) / n_event * MINBIAS_RATE)

    fig, ax = plt.subplots(1, 1, figsize=style.FIGURE_SIZE)
    hep.cms.label(llabel=style.CMSHEADER_LEFT,rlabel=style.CMSHEADER_RIGHT,ax=ax,fontsize=style.MEDIUM_SIZE)

    # Plot the trigger rates
    ax.plot([],[], linestyle='none', label=r'$|\eta| < 2.5$')
    ax.plot(pt_cuts, minbias_rate_no_nn, c=style.color_cycle[2], label=r'No ID/$p_T$ correction', linewidth=style.LINEWIDTH)
    ax.plot(pt_cuts, minbias_rate_cmssw, c=style.color_cycle[0], label=r'CMSSW PuppiTau Emulator', linewidth=style.LINEWIDTH)
    ax.plot(pt_cuts, minbias_rate_model, c=style.color_cycle[1],label=r'SeedCone Tau', linewidth=style.LINEWIDTH)

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
    ax.set_ylabel(r"$\tau_h$ trigger rate [kHz]")
    ax.set_xlabel(r"L1 $p_T$ [GeV]")
    ax.legend(loc='upper right', fontsize=style.MEDIUM_SIZE)

    # Save the plot
    plot_dir = os.path.join(model.output_directory, 'plots/physics/tau')
    fig.savefig(os.path.join(plot_dir, "tau_BkgRate.pdf"), bbox_inches='tight')
    fig.savefig(os.path.join(plot_dir, "tau_BkgRate.png"), bbox_inches='tight')

    return

def eff_tau(model, signal_path, tree='jetntuple/Jets', n_entries=10000 ):
    """
    Plot the single tau efficiency for signal in signal_path w.r.t pt
    eta range for barrel: |eta| < 1.5
    eta range for endcap: 1.5 < |eta| < 2.172
    """

    plot_dir = os.path.join(model.output_directory, 'plots/physics/tau')

    #Check if the working point have been derived
    WP_path = os.path.join(model.output_directory, "plots/physics/tau/working_point.json")
    cmssw_WP_path = os.path.join(model.output_directory, "plots/physics/tau/cmssw_working_point.json")

    model_WP_interp = get_interp_func(WP_path)
    cmssw_WP_interp = get_interp_func(cmssw_WP_path)

    signal = uproot.open(signal_path)[tree]

    #Select out the taus
    tau_flav = extract_array(signal, 'jet_tauflav', n_entries)
    gen_pt_raw = extract_array(signal, 'jet_genmatch_pt', n_entries)
    gen_eta_raw = extract_array(signal, 'jet_genmatch_eta', n_entries)
    gen_dr_raw = extract_array(signal, 'jet_genmatch_dR', n_entries)

    l1_pt_raw = extract_array(signal, 'jet_pt', n_entries)
    jet_taupt_raw= extract_array(signal, 'jet_taupt', n_entries)
    jet_tauscore_raw = extract_array(signal, 'jet_tauscore', n_entries)

    #Get the model prediction
    nn_inputs = np.asarray(extract_nn_inputs(signal, model.input_vars, n_entries=n_entries))
    pred_score, ratio = model.predict(nn_inputs)

    nn_tauscore_raw = tau_score(pred_score, model.class_labels )
    nn_taupt_raw = np.multiply(l1_pt_raw, ratio.flatten())

    cut = nn_taupt_raw > 30.


    debug_plots = False

    if(debug_plots):
        plt.figure()
        plt.hist(nn_taupt_raw, bins=30)
        plt.xlabel("Tau pT")
        plt.xlim([0., 200.])
        plt.savefig(f"{plot_dir}/tau_pt.png")

        plt.figure()
        plt.hist(nn_tauscore_raw[cut], bins=30)
        plt.xlabel("Model Tau Score")
        plt.savefig(f"{plot_dir}/model_tau_score.png")


        plt.figure()
        plt.hist(jet_tauscore_raw[cut], bins=30)
        plt.xlabel("CMSSW Tau Score")
        plt.savefig(f"{plot_dir}/cmssw_tau_score.png")

    pT_edges = [0] + np.arange(30, 150, 2)

    #Denominator & numerator selection for efficiency
    eta_selection = np.abs(gen_eta_raw) < 2.172


    seeded_cone_effs = [0.]
    model_effs =[0.]
    cmssw_effs =[0.]
    denoms = [1e-6]

    #min number of events to compute a reliable eff
    min_mc = 50

    min_NN_cut = 0.00

    for pt_cut in pT_edges[1:]:
        model_cut = max(model_WP_interp(pt_cut), min_NN_cut)
        cmssw_cut = max(cmssw_WP_interp(pt_cut), min_NN_cut)


        tau_deno = (tau_flav==1) & (gen_pt_raw > pt_cut) & eta_selection
        denom = np.sum(tau_deno)

        denoms.append(denom)


        if(np.sum(tau_deno) < min_mc):
            #too few events, just put 0
            seeded_cone_effs.append(0.)
            model_effs.append(0.)
            cmssw_effs.append(0.)
        else:
            tau_nume_seedcone = tau_deno & (np.abs(gen_dr_raw) < 0.4) & (l1_pt_raw > pt_cut)
            tau_nume_nn = tau_deno & (np.abs(gen_dr_raw) < 0.4) & (nn_taupt_raw > pt_cut) & (nn_tauscore_raw > model_cut)
            tau_nume_cmssw = tau_deno & (np.abs(gen_dr_raw) < 0.4) & (jet_taupt_raw > pt_cut) & (jet_tauscore_raw > cmssw_cut)

            seeded_cone_effs.append(np.sum(tau_nume_seedcone))
            model_effs.append(np.sum(tau_nume_nn))
            cmssw_effs.append(np.sum(tau_nume_cmssw))

            #print(pt_cut, np.sum(tau_deno), np.sum(cmssw_cut), np.sum(model_cut))


    denoms = np.array(denoms)
    seeded_cone_uncs = np.sqrt(seeded_cone_effs) / denoms
    model_uncs = np.sqrt(model_effs) / denoms
    cmssw_uncs = np.sqrt(cmssw_effs) / denoms

    seeded_cone_effs = np.array(seeded_cone_effs) / denoms
    model_effs = np.array(model_effs) / denoms
    cmssw_effs = np.array(cmssw_effs) / denoms

    fig, ax = plt.subplots(1, 1, figsize=style.FIGURE_SIZE)
    hep.cms.label(llabel=style.CMSHEADER_LEFT,rlabel=style.CMSHEADER_RIGHT,ax=ax,fontsize=style.MEDIUM_SIZE)

    ax.plot([],[], linestyle='none', label=r'$|\eta| < 2.172$')
    ax.plot(pT_edges, seeded_cone_effs, c=style.color_cycle[2], label=r'Raw Seeded Cone Eff.', linewidth=style.LINEWIDTH)
    ax.plot(pT_edges, model_effs, c=style.color_cycle[0], label=r'CMSSW PuppiTau Emulator Eff., 31 kHz Rate', linewidth=style.LINEWIDTH)
    ax.plot(pT_edges, cmssw_effs, c=style.color_cycle[1],label=r'SeedCone Tau Eff., 31 kHz Rate', linewidth=style.LINEWIDTH)


    # Add uncertainty bands
    ax.fill_between(pT_edges,
                    seeded_cone_effs + seeded_cone_uncs,
                    seeded_cone_effs - seeded_cone_uncs,
                    color=style.color_cycle[2],
                    alpha=0.3)


    ax.fill_between(pT_edges,
                    model_effs + model_uncs,
                    model_effs - model_uncs,
                    color=style.color_cycle[0],
                    alpha=0.3)

    ax.fill_between(pT_edges,
                    cmssw_effs + cmssw_uncs,
                    cmssw_effs - cmssw_uncs,
                    color=style.color_cycle[1],
                    alpha=0.3)

    figname = f'tau_eff_pt_comparison'
    plt.legend()

    ax.set_xlabel(r"Single $\tau_h$ $p_T$ Threshold [GeV]")
    ax.set_ylabel(r"Trigger Efficiency")

    ax.legend(loc='upper left', fontsize=style.LEGEND_WIDTH)
    fig.savefig(f'{plot_dir}/{figname}.png', bbox_inches='tight')
    fig.savefig(f'{plot_dir}/{figname}.pdf', bbox_inches='tight')





    for eta_region in ['barrel', 'tau_endcap']:
        #selecting the eta region
        gen_eta_selection = eta_region_selection(gen_eta_raw, eta_region)

        #Pick a single pt WP for comparison
        pt_WP = 75.

        tau_deno = (tau_flav==1) & (gen_pt_raw > 1.) & eta_selection

        model_cut = model_WP_interp(pt_WP)
        cmssw_cut = cmssw_WP_interp(pt_WP)

        tau_nume_seedcone = tau_deno & (np.abs(gen_dr_raw) < 0.4) & (l1_pt_raw > pt_WP)
        tau_nume_nn = tau_deno & (np.abs(gen_dr_raw) < 0.4) & (nn_taupt_raw > pt_WP) & (nn_tauscore_raw > model_cut)
        tau_nume_cmssw = tau_deno & (np.abs(gen_dr_raw) < 0.4) & (jet_taupt_raw > pt_WP) & (jet_tauscore_raw > cmssw_cut)

        ##write out total eff to text file
        total_eff_nn = np.mean(tau_nume_nn) / np.mean(tau_deno)
        total_eff_seedcone = np.mean(tau_nume_seedcone) / np.mean(tau_deno)
        total_eff_cmssw = np.mean(tau_nume_cmssw) / np.mean(tau_deno)

        outname = plot_dir + "/TotalEff_%s.txt" % eta_region
        with open(outname, "w") as outfile:
            outfile.write("Total Tau Eff \n")
            outfile.write("SeededCone Inclusive (Eff Upper Limit) %.4f \n" % total_eff_seedcone)
            outfile.write("Multiclass NN %.4f \n" % total_eff_nn)
            outfile.write("CMSSW  %.4f \n" % total_eff_cmssw)



        #Get the needed attributes
        #Basically we want to bin the selected truth pt and divide it by the overall count
        gen_pt = gen_pt_raw[tau_deno]
        seedcone_pt = gen_pt_raw[tau_nume_seedcone]
        cmssw_pt = gen_pt_raw[tau_nume_cmssw]
        nn_pt = gen_pt_raw[tau_nume_nn]

        #Constructing the histograms
        pT_axis = hist.axis.Variable(pT_edges, name = r"$ \tau_h$ $p_T^{gen}$")

        all_tau = Hist(pT_axis)
        seedcone_tau = Hist(pT_axis)
        cmssw_tau = Hist(pT_axis)
        nn_tau = Hist(pT_axis)

        #Fill the histogram using the values above
        all_tau.fill(gen_pt)
        seedcone_tau.fill(seedcone_pt)
        cmssw_tau.fill(cmssw_pt)
        nn_tau.fill(nn_pt)

        #Plot and get the artist objects
        eff_seedcone = plot_ratio(all_tau, seedcone_tau)
        eff_cmssw = plot_ratio(all_tau, cmssw_tau)
        eff_nn = plot_ratio(all_tau, nn_tau)

        #Extract data from the artists
        sc_x, sc_y, sc_err = get_bar_patch_data(eff_seedcone)
        cmssw_x, cmssw_y, cmssw_err = get_bar_patch_data(eff_cmssw)
        nn_x, nn_y, nn_err = get_bar_patch_data(eff_nn)

        # Create figure and axis
        fig, ax = plt.subplots(1, 1, figsize=style.FIGURE_SIZE)
        hep.cms.label(llabel=style.CMSHEADER_LEFT,rlabel=style.CMSHEADER_RIGHT,ax=ax,fontsize=style.MEDIUM_SIZE)

        # Set the eta label if needed
        vbf_label = r"VBF H $\rightarrow$ $\tau\tau$, "
        pt_wp_label = "$p_T$ > %i GeV Trigger WP " % pt_WP
        eta_label = r'Barrel ($|\eta| < 1.5$)' if eta_region == 'barrel' else r'EndCap (1.5 < $|\eta|$ < 2.172)'
        if eta_region != 'none':
            # Add an invisible plot to include the eta label in the legend
            ax.plot([], [], 'none', label= pt_wp_label + eta_label)

        sc_err = np.nan_to_num(sc_err, nan=0.)
        cmssw_err = np.nan_to_num(cmssw_err, nan=0.)
        nn_err = np.nan_to_num(nn_err, nan=0.)


        # Plot errorbars for both sets of efficiencies
        ax.errorbar(sc_x, sc_y, yerr=sc_err, fmt='o', c=style.color_cycle[2], markersize=style.LINEWIDTH, linewidth=2, label=r'SeededCone PuppiJet Efficiency Limit') #Theoretical limit, uncomment for common sense check.
        ax.errorbar(cmssw_x, cmssw_y, yerr=cmssw_err, fmt='o', c=style.color_cycle[0], markersize=style.LINEWIDTH, linewidth=2, label=r'Tau CMSSW Emulator @ 31kHz')
        ax.errorbar(nn_x, nn_y, yerr=nn_err, fmt='o', c=style.color_cycle[1], markersize=style.LINEWIDTH, linewidth=2, label=r'SeededCone Tau Tagger @ 31kHz')

        # Plot a horizontal dashed line at y=1
        ax.axhline(1, xmin=0, xmax=150, linestyle='dashed', color='black', linewidth=3)

        # Set plot limits and labels
        ax.set_ylim([0., 1.1])
        ax.set_xlim([0, 150])
        ax.set_xlabel(r"$\tau_h$ $p_T^{gen}$ [GeV]")
        ax.set_ylabel(r"Efficiency")

        # Add legend
        ax.legend(loc='lower right', fontsize=style.LEGEND_WIDTH)

        # Save and show the plot
        figname = f'sc_and_tau_eff_{eta_region}'
        fig.savefig(f'{plot_dir}/{figname}.pdf', bbox_inches='tight')
        fig.savefig(f'{plot_dir}/{figname}.png', bbox_inches='tight')

    return

if __name__ == "__main__":
    """
    2 steps:

    1. Derive working points: python singleTau.py --deriveWPs
    2. Run efficiency based on the derived working points: python singleTau.py --eff
    """

    parser = ArgumentParser()
    parser.add_argument('-m','--model_dir', default='output/baseline', help = 'Input model')
    parser.add_argument('-v', '--vbf_sample', default='/eos/cms/store/cmst3/group/l1tr/sewuchte/l1teg/fp_jettuples_090125_addGenH/VBFHToTauTau_PU200.root' , help = 'Signal sample for VBF -> ditaus')
    parser.add_argument('--minbias', default='/eos/cms/store/cmst3/group/l1tr/sewuchte/l1teg/fp_jettuples_090125/MinBias_PU200.root' , help = 'Minbias sample for deriving rates')

    #Different modes
    parser.add_argument('--deriveWPs', action='store_true', help='derive the working points for di-taus')
    parser.add_argument('--eff', action='store_true', help='plot efficiency for VBF-> tautau')
    parser.add_argument('--BkgRate', action='store_true', help='plot background rate for VBF->tautau')

    #Other controls
    parser.add_argument('-n','--n_entries', type=int, default=500000, help = 'Number of data entries in root file to run over, can speed up run time, set to None to run on all data entries')
    parser.add_argument('--tree', default='jetntuple/Jets', help='Tree within the ntuple containing the jets')

    args = parser.parse_args()

    model = fromFolder(args.model_dir)

    if args.deriveWPs:
        derive_tau_WPs(model, args.minbias, n_entries=args.n_entries, tree=args.tree, cmssw_model=True)
        derive_tau_WPs(model, args.minbias, n_entries=args.n_entries, tree=args.tree)
    elif args.BkgRate:
        plot_bkg_rate_tau(model, args.minbias, n_entries=args.n_entries, tree=args.tree)
    elif args.eff:
        eff_tau(model, args.vbf_sample, n_entries=args.n_entries, tree=args.tree)
