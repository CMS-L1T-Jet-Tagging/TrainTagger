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
from common import MINBIAS_RATE, WPs_CMSSW, find_rate, plot_ratio, delta_r, eta_region_selection, get_bar_patch_data


def tau_score(preds, class_labels):
    tau_index = [class_labels['taup'], class_labels['taum']] #Tau positives and tau negatives

    tau = preds[:,tau_index[0]] + preds[:,tau_index[1]]
    #denom of just gluon seems to work best
    #bkg = preds[:,class_labels['gluon']]
    bkg = preds[:,class_labels['unmatched']]

    return tau / (tau + bkg)
    


def pick_and_plot_ditau(rate_list, pt_list, nn_list, model_dir, target_rate = 28, RateRange = 1.0):
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

    #Just plot the points instead of the interpolation
    #ax.plot(target_rate_NN, target_rate_PT, linewidth=style.LINEWIDTH, color ='firebrick', label = r"${} \pm {}$ kHz".format(target_rate, RateRange))

    ax.legend(loc='upper right', fontsize=style.MEDIUM_SIZE)
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
    pred_score1, ratio1 = model.predict(input1)
    pred_score2, ratio2 = model.predict(input2)

    #Correct the pT and add the score
    pt1 = pt1_uncorrected*(ratio1.flatten())
    pt2 = pt2_uncorrected*(ratio2.flatten())

    tau_score1 = tau_score(pred_score1, class_labels)
    tau_score2 = tau_score(pred_score2, class_labels)

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
    """
    Plot the background (mimbias) rate w.r.t pT cuts.
    """

    #Load metadata from the model directory
    model=load_qmodel(os.path.join(model_dir, "model/saved_model.h5"))
    with open(os.path.join(model_dir, "input_vars.json"), "r") as f: input_vars = json.load(f)
    with open(os.path.join(model_dir, "class_label.json"), "r") as f: class_labels = json.load(f)

    pt_cuts = list(np.arange(0,250,10))

    #Load the minbias data
    minbias = uproot.open(minbias_path)[tree]

    #Impose eta cuts
    jet_eta =  extract_array(minbias, 'jet_eta_phys', n_entries)
    eta_selection = np.abs(jet_eta) < 2.5

    #
    nn_inputs = np.asarray(extract_nn_inputs(minbias, input_vars, n_entries=n_entries))

    #Get the NN predictions
    pred_score, ratio = model.predict(nn_inputs[eta_selection])
    model_tau = tau_score(pred_score, class_labels)

    #Emulator tau score
    cmssw_tau = extract_array(minbias, 'jet_tauscore', n_entries)[eta_selection]

    #Use event id to track which jets belong to which event.
    event_id = extract_array(minbias, 'event', n_entries)[eta_selection]
    event_id_cmssw = event_id[cmssw_tau > WPs_CMSSW["tau"]]

    #Load the working point from json file
    #Check if the working point have been derived
    WP_path = os.path.join(model_dir, "plots/physics/tautau/working_point.json")

    #Get derived working points
    if os.path.exists(WP_path):
        with open(WP_path, "r") as f:  WPs = json.load(f)
        tautau_wp = WPs['NN']
        tautau_pt_wp = WPs['PT']
    else:
        raise Exception("Working point does not exist. Run with --deriveWPs first.")

    event_id_model = event_id[model_tau > tautau_wp]

    #Cut on jet pT to extract the rate
    jet_pt = extract_array(minbias, 'jet_pt', n_entries)[eta_selection]


    jet_pt_cmssw = extract_array(minbias, 'jet_taupt', n_entries)[eta_selection][cmssw_tau > WPs_CMSSW["tau"]]
    jet_pt_model = (jet_pt*ratio.flatten())[model_tau > tautau_wp]

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
    plot_dir = os.path.join(model_dir, 'plots/physics/tautau')
    fig.savefig(os.path.join(plot_dir, "tautau_BkgRate.pdf"), bbox_inches='tight')
    fig.savefig(os.path.join(plot_dir, "tautau_BkgRate.png"), bbox_inches='tight')

    return

def eff_ditau(model_dir, signal_path, eta_region='barrel', tree='jetntuple/Jets', n_entries=10000):
    """
    Plot the single tau efficiency for signal in signal_path w.r.t pt
    eta range for barrel: |eta| < 1.5
    eta range for endcap: 1.5 < |eta| < 2.5
    """

    plot_dir = os.path.join(model_dir, 'plots/physics/tautau')

    #Load metadata from the model directory
    model=load_qmodel(os.path.join(model_dir, "model/saved_model.h5"))
    with open(os.path.join(model_dir, "input_vars.json"), "r") as f: input_vars = json.load(f)
    with open(os.path.join(model_dir, "class_label.json"), "r") as f: class_labels = json.load(f)

    #Check if the working point have been derived
    WP_path = os.path.join(model_dir, "plots/physics/tautau/working_point.json")

    #Get derived working points
    if os.path.exists(WP_path):
        with open(WP_path, "r") as f:  WPs = json.load(f)
        model_NN_WP = WPs['NN']
        model_pt_WP = WPs['PT']
    else:
        raise Exception("Working point does not exist. Run with --deriveWPs first.")

    pT_egdes = [0,10,15,20,25,30,35,40,45,50,55,60,70,80,100,125,150,175,200]

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
    nn_inputs = np.asarray(extract_nn_inputs(signal, input_vars, n_entries=n_entries))
    pred_score, ratio = model.predict(nn_inputs)

    nn_tauscore_raw = tau_score(pred_score, class_labels) 
    nn_taupt_raw = np.multiply(l1_pt_raw, ratio.flatten())

    #selecting the eta region
    gen_eta_selection = eta_region_selection(gen_eta_raw, eta_region)

    #Denominator & numerator selection for efficiency
    tau_deno = (tau_flav==1) & (gen_pt_raw > 1.) & gen_eta_selection
    tau_nume_seedcone = tau_deno & (np.abs(gen_dr_raw) < 0.4) & (l1_pt_raw > model_pt_WP)
    tau_nume_nn = tau_deno & (np.abs(gen_dr_raw) < 0.4) & (nn_taupt_raw > model_pt_WP) & (nn_tauscore_raw > model_NN_WP)
    tau_nume_cmssw = tau_deno & (np.abs(gen_dr_raw) < 0.4) & (jet_taupt_raw > WPs_CMSSW['tau_l1_pt']) & (jet_tauscore_raw > WPs_CMSSW['tau'])

    #Get the needed attributes
    #Basically we want to bin the selected truth pt and divide it by the overall count
    gen_pt = gen_pt_raw[tau_deno]
    seedcone_pt = gen_pt_raw[tau_nume_seedcone]
    cmssw_pt = gen_pt_raw[tau_nume_cmssw]
    nn_pt = gen_pt_raw[tau_nume_nn]

    #Constructing the histograms
    pT_axis = hist.axis.Variable(pT_egdes, name = r"$ \tau_h$ $p_T^{gen}$")

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
    eta_label = r'Barrel ($|\eta| < 1.5$)' if eta_region == 'barrel' else r'EndCap (1.5 < $|\eta|$ < 2.5)'
    if eta_region != 'none':
        # Add an invisible plot to include the eta label in the legend
        ax.plot([], [], 'none', label=eta_label)

    # Plot errorbars for both sets of efficiencies
    ax.errorbar(sc_x, sc_y, yerr=sc_err, fmt='o', c=style.color_cycle[2], markersize=style.LINEWIDTH, linewidth=2, label=r'SeededCone PuppiJet Efficiency Limit') #Theoretical limit, uncomment for common sense check.
    ax.errorbar(cmssw_x, cmssw_y, yerr=cmssw_err, fmt='o', c=style.color_cycle[0], markersize=style.LINEWIDTH, linewidth=2, label=r'Tau CMSSW Emulator @ 28kHz')
    ax.errorbar(nn_x, nn_y, yerr=nn_err, fmt='o', c=style.color_cycle[1], markersize=style.LINEWIDTH, linewidth=2, label=r'SeededCone Tau Tagger @ 28kHz')

    # Plot a horizontal dashed line at y=1
    ax.axhline(1, xmin=0, xmax=150, linestyle='dashed', color='black', linewidth=3)

    # Set plot limits and labels
    ax.set_ylim([0., 1.1])
    ax.set_xlim([0, 150])
    ax.set_xlabel(r"$\tau_h$ $p_T^{gen}$ [GeV]")
    ax.set_ylabel(r"$\epsilon$ (VBF H $\rightarrow$ $\tau\tau$)")

    # Add legend
    ax.legend(loc='lower right', fontsize=style.LEGEND_WIDTH)

    # Save and show the plot
    figname = f'sc_and_tau_eff_{eta_region}'
    fig.savefig(f'{plot_dir}/{figname}.pdf', bbox_inches='tight')
    fig.savefig(f'{plot_dir}/{figname}.png', bbox_inches='tight')
    plt.show(block=False)

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

    #Different modes
    parser.add_argument('--deriveWPs', action='store_true', help='derive the working points for di-taus')
    parser.add_argument('--eff', action='store_true', help='plot efficiency for VBF-> tautau')
    parser.add_argument('--BkgRate', action='store_true', help='plot background rate for VBF->tautau')

    #Other controls
    parser.add_argument('-n','--n_entries', type=int, default=500000, help = 'Number of data entries in root file to run over, can speed up run time, set to None to run on all data entries')
    parser.add_argument('--tree', default='jetntuple/Jets', help='Tree within the ntuple containing the jets')

    args = parser.parse_args()

    if args.deriveWPs:
        derive_diTaus_WPs(args.model_dir, args.minbias, n_entries=args.n_entries, tree=args.tree)
    elif args.BkgRate:
        plot_bkg_rate_ditau(args.model_dir, args.minbias, n_entries=args.n_entries, tree=args.tree)
    elif args.eff:
        eff_ditau(args.model_dir, args.vbf_sample, n_entries=args.n_entries, eta_region='barrel', tree=args.tree)
        eff_ditau(args.model_dir, args.vbf_sample, n_entries=args.n_entries, eta_region='endcap', tree=args.tree)
