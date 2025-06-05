# flake8: noqa
import gc
import json
import os
from argparse import ArgumentParser

import awkward as ak
import hist
import matplotlib
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import uproot
from hist import Hist
from scipy.interpolate import interp1d

from tagger.data.tools import extract_array, extract_nn_inputs, group_id_values
from tagger.model.common import fromFolder
from tagger.plot import style
from tagger.plot.common import MINBIAS_RATE, WPs_CMSSW, find_rate, get_bar_patch_data, plot_ratio

style.set_style()


def x_vs_y(x, y, apply_light):
    if apply_light:
        s = x / (x + y)
        return s
    else:
        return x


def default_selection(jet_pt, jet_eta, apply_sel):
    if apply_sel:
        mask = (jet_pt[:, :4] > 10) & (np.abs(jet_eta[:, :4]) < 2.4)
        event_mask = np.sum(mask, axis=1) == 4
    else:
        event_mask = np.ones(len(jet_pt), dtype=bool)

    return event_mask


def nn_bscore_sum(model, jet_nn_inputs, jet_pt, jet_eta, apply_light, n_jets=4, b_index=0, l_index=2, g_index=3):

    # Get the inputs for the first n_jets
    btag_inputs = [np.asarray(jet_nn_inputs[:, i]) for i in range(0, n_jets)]

    # Get the nn outputs
    nn_outputs = [model.jet_model.predict(nn_input)[0] for nn_input in btag_inputs]

    # Sum them together
    bscore_sum = sum(
        [
            x_vs_y(pred_score[:, b_index], pred_score[:, l_index] + pred_score[:, g_index], apply_light)
            for pred_score in nn_outputs
        ]
    )

    return bscore_sum


def pick_and_plot(rate_list, ht_list, nn_list, model_dir, apply_sel, apply_light, target_rate=14):
    """
    Pick the working points and plot
    """

    plot_dir = os.path.join(model_dir, 'plots/physics/bbbb')
    os.makedirs(plot_dir, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=style.FIGURE_SIZE)
    hep.cms.label(llabel=style.CMSHEADER_LEFT, rlabel=style.CMSHEADER_RIGHT, ax=ax, fontsize=style.MEDIUM_SIZE - 2)
    im = ax.scatter(
        nn_list, ht_list, c=rate_list, s=500, marker='s', cmap='Spectral_r', linewidths=0, norm=matplotlib.colors.LogNorm()
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r'4-b rate [kHZ]')

    ax.set_ylabel(r"HT [GeV]")
    ax.set_xlabel(r"$\sum_{4~leading~jets}$ b scores")

    ax.set_xlim([0, 1.3])
    ax.set_ylim([10, 500])

    # plus, minus range
    RateRange = 0.5

    # Find the target rate points, plot them and print out some info as well
    target_rate_idx = find_rate(rate_list, target_rate=target_rate, RateRange=RateRange)

    # Get the coordinates
    target_rate_NN = [nn_list[i] for i in target_rate_idx]  # NN cut dimension
    target_rate_HT = [ht_list[i] for i in target_rate_idx]  # HT cut dimension

    # Create an interpolation function
    interp_func = interp1d(target_rate_HT, target_rate_NN, kind='linear', fill_value='extrapolate')

    # Interpolate the NN value for the desired HT
    working_point_NN = interp_func(WPs_CMSSW['btag_l1_ht'])
    working_point_low_HT = interp_func(WPs_CMSSW['btag_l1_ht'] - 40)

    # Export the working point
    score_type = "vs_qg" if apply_light else "raw"
    sel_type = "sel" if apply_sel else "all"
    working_point = {"HT": WPs_CMSSW['btag_l1_ht'], "NN": float(working_point_NN)}
    with open(os.path.join(plot_dir, f"working_point_{score_type}_{sel_type}.json"), "w") as f:
        json.dump(working_point, f, indent=4)

    ax.plot(
        target_rate_NN,
        target_rate_HT,
        linewidth=5,
        color='firebrick',
        label=r"${} \pm {}$ kHz".format(target_rate, RateRange),
    )

    ax.legend(loc='upper right')
    plt.savefig(f"{plot_dir}/bbbb_rate_{score_type}_{sel_type}.pdf", bbox_inches='tight')
    plt.savefig(f"{plot_dir}/bbbb_rate_{score_type}_{sel_type}.png", bbox_inches='tight')


def derive_HT_WP(RateHist, ht_edges, n_events, model_dir, target_rate=14, RateRange=0.8):
    """
    Derive the HT only working points (without bb cuts)
    """

    plot_dir = os.path.join(model_dir, 'plots/physics/bbbb')

    # Derive the rate
    rate_list = []
    ht_list = []

    # Loop through the edges and integrate
    for ht in ht_edges[:-1]:

        # Calculate the rate
        rate = RateHist[{"ht": slice(ht * 1j, None, sum)}][{"nn": slice(0.0j, None, sum)}] / n_events
        rate_list.append(rate * MINBIAS_RATE)

        # Append the results
        ht_list.append(ht)

    target_rate_idx = find_rate(rate_list, target_rate=target_rate, RateRange=RateRange)

    # write HT cut
    WP_json = os.path.join(plot_dir, "ht_working_point.json")
    working_point = {"ht_only_cut": float(ht_list[target_rate_idx[0]])}
    json.dump(working_point, open(WP_json, "w"), indent=4)


# WPs


def derive_bbbb_WPs(model, minbias_path, apply_sel, apply_light, target_rate=14, n_entries=100, tree='outnano/Jets'):
    """
    Derive the HH->4b working points
    """

    # Load the minbias data
    minbias = uproot.open(minbias_path)[tree]

    raw_event_id = extract_array(minbias, 'event', n_entries)
    raw_jet_pt = extract_array(minbias, 'jet_pt', n_entries)
    raw_jet_eta = extract_array(minbias, 'jet_eta_phys', n_entries)
    raw_inputs = extract_nn_inputs(minbias, model.input_vars, n_entries=n_entries)

    # Count number of total event
    n_events = len(np.unique(raw_event_id))
    print("Total number of minbias events: ", n_events)

    # Group these attributes by event id, and filter out groups that don't have at least 2 elements
    event_id, grouped_arrays = group_id_values(raw_event_id, raw_jet_pt, raw_jet_eta, raw_inputs, num_elements=4)

    # Extract the grouped arrays
    # Jet pt is already sorted in the producer, no need to do it here
    jet_pt, jet_eta, jet_nn_inputs = grouped_arrays
    jet_nn_inputs = jet_nn_inputs[default_selection(jet_pt, jet_eta, apply_sel)]

    # Btag input list for first 4 jets
    nn_outputs = [model.jet_model.predict(np.asarray(jet_nn_inputs[:, i]))[0] for i in range(0, 4)]

    # Calculate the output sum
    b_index = model.class_labels['b']
    l_index = model.class_labels['light']
    g_index = model.class_labels['gluon']
    bscore_sum = sum(
        [
            x_vs_y(pred_score[:, b_index], pred_score[:, l_index] + pred_score[:, g_index], apply_light)
            for pred_score in nn_outputs
        ]
    )
    sel_ht = ak.sum(jet_pt, axis=1)[default_selection(jet_pt, jet_eta, apply_sel)]
    jet_ht = ak.sum(jet_pt, axis=1)

    assert len(bscore_sum) == len(sel_ht)

    # Define the histograms (pT edge and NN Score edge)
    # Make sure to capture everything
    ht_edges = list(np.arange(0, 500, 2)) + [10000]
    NN_edges = list([round(i, 2) for i in np.arange(0, 2.5, 0.01)]) + [4.0]

    RateHist = Hist(hist.axis.Variable(ht_edges, name="ht", label="ht"), hist.axis.Variable(NN_edges, name="nn", label="nn"))

    RateHist.fill(ht=sel_ht, nn=bscore_sum)

    # Derive the rate
    rate_list = []
    ht_list = []
    nn_list = []

    # Loop through the edges and integrate
    for ht in ht_edges[:-1]:
        for NN in NN_edges[:-1]:

            # Calculate the rate
            rate = RateHist[{"ht": slice(ht * 1j, None, sum)}][{"nn": slice(NN * 1.0j, None, sum)}] / n_events
            rate_list.append(rate * MINBIAS_RATE)

            # Append the results
            ht_list.append(ht)
            nn_list.append(NN)

    # Pick target rate and plot it
    pick_and_plot(rate_list, ht_list, nn_list, model.output_directory, apply_sel, apply_light, target_rate=target_rate)

    # refill with full ht for ht wp derivation
    RateHist = Hist(hist.axis.Variable(ht_edges, name="ht", label="ht"), hist.axis.Variable(NN_edges, name="nn", label="nn"))

    RateHist.fill(ht=jet_ht, nn=np.zeros(len(jet_ht)))
    derive_HT_WP(RateHist, ht_edges, n_events, model.output_directory, target_rate=target_rate)

    return


def load_bbbb_WPs(model_dir, apply_sel, apply_light):
    """
    Check and lodad all bbbb working points
    """
    # Check if the working point have been derived
    score_type = "vs_qg" if apply_light else "raw"
    sel_type = "sel" if apply_sel else "all"
    WP_path = os.path.join(model_dir, f"plots/physics/bbbb/working_point_{score_type}_{sel_type}.json")
    HT_WP_path = os.path.join(model_dir, f"plots/physics/bbbb/ht_working_point.json")

    # Get derived working points
    if os.path.exists(WP_path) & os.path.exists(HT_WP_path):
        with open(WP_path, "r") as f:
            WPs = json.load(f)
        btag_wp = WPs['NN']
        btag_ht_wp = int(WPs['HT'])
        ht_only_wp = int(json.load(open(HT_WP_path))["ht_only_cut"])
    else:
        raise Exception("Working point does not exist. Run with --deriveWPs first.")

    return btag_wp, btag_ht_wp, ht_only_wp


# Efficiency


def bbbb_eff(model, signal_path, apply_sel, apply_light, n_entries=100000, tree='outnano/Jets'):
    """
    Plot HH->4b efficiency w.r.t HT
    """

    ht_edges = list(np.arange(0, 800, 20))
    ht_axis = hist.axis.Variable(ht_edges, name=r"$HT^{gen}$")

    # Working points for CMSSW
    cmssw_btag = WPs_CMSSW['btag']
    cmssw_btag_ht = WPs_CMSSW['btag_l1_ht']

    btag_wp, btag_ht_wp, ht_only_wp = load_bbbb_WPs(model.output_directory, apply_sel, apply_light)

    # Load the signal data
    signal = uproot.open(signal_path)[tree]

    raw_event_id = extract_array(signal, 'event', n_entries)
    raw_jet_genpt = extract_array(signal, 'jet_genmatch_pt', n_entries)
    raw_jet_pt = extract_array(signal, 'jet_pt', n_entries)
    raw_jet_eta = extract_array(signal, 'jet_eta_phys', n_entries)
    raw_cmssw_bscore = extract_array(signal, 'jet_bjetscore', n_entries)

    n_events = len(np.unique(raw_event_id))

    # Try to extract genHH_mass, set to None if not found
    try:
        raw_gen_mHH = extract_array(signal, 'genHH_mass', n_entries)
    except (KeyError, ValueError, uproot.exceptions.KeyInFileError) as e:
        print(f"Warning: 'genHH_mass' not found in signal file: {e}")
        raw_gen_mHH = None

    # Load the inputs
    raw_inputs = extract_nn_inputs(signal, model.input_vars, n_entries=n_entries)

    # Group event_id, gen_mHH, and genpt separately
    if raw_gen_mHH is not None:
        event_id, grouped_gen_arrays = group_id_values(raw_event_id, raw_gen_mHH, raw_jet_genpt, num_elements=0)
        all_event_gen_mHH = ak.firsts(grouped_gen_arrays[0])
    else:
        # If genHH_mass doesn't exist, only group event_id and genpt
        event_id, grouped_gen_arrays = group_id_values(raw_event_id, raw_jet_genpt, num_elements=0)
        all_event_gen_mHH = None

    # Last element will always be genpt
    all_jet_genht = ak.sum(grouped_gen_arrays[-1], axis=1)

    # Group these attributes by event id, and filter out groups that don't have at least 4 elements
    if raw_gen_mHH is not None:
        event_id, grouped_arrays = group_id_values(
            raw_event_id, raw_gen_mHH, raw_jet_genpt, raw_jet_pt, raw_jet_eta, raw_cmssw_bscore, raw_inputs, num_elements=4
        )
        event_gen_mHH, jet_genpt, jet_pt, jet_eta, cmssw_bscore, jet_nn_inputs = grouped_arrays

        # Just pick the first entry of jet mHH arrays
        event_gen_mHH = ak.firsts(event_gen_mHH)
    else:
        # Handle case where genHH_mass doesn't exist
        event_id, grouped_arrays = group_id_values(
            raw_event_id, raw_jet_genpt, raw_jet_pt, raw_jet_eta, raw_cmssw_bscore, raw_inputs, num_elements=4
        )
        jet_genpt, jet_pt, jet_eta, cmssw_bscore, jet_nn_inputs = grouped_arrays
        event_gen_mHH = None

    # Calculate the ht
    jet_genht = ak.sum(jet_genpt, axis=1)
    jet_ht = ak.sum(jet_pt, axis=1)

    # B score from cmssw emulator
    # Only sum up the first four
    cmsssw_bscore_sum = ak.sum(cmssw_bscore[:, :4], axis=1)
    model_bscore_sum = nn_bscore_sum(
        model,
        jet_nn_inputs,
        jet_pt,
        jet_eta,
        apply_light,
        b_index=model.class_labels['b'],
        l_index=model.class_labels['light'],
        g_index=model.class_labels['gluon'],
    )

    cmssw_selection = (jet_ht > cmssw_btag_ht) & (cmsssw_bscore_sum > cmssw_btag)
    cmssw_efficiency = np.round(ak.sum(cmssw_selection) / n_events, 2)
    model_selection = (jet_ht > btag_ht_wp) & (model_bscore_sum > btag_wp) & default_selection(jet_pt, jet_eta, apply_sel)
    model_efficiency = np.round(ak.sum(model_selection) / n_events, 2)
    ht_only_selection = jet_ht > ht_only_wp
    ht_only_efficiency = np.round(ak.sum(ht_only_selection) / n_events, 2)

    # Plot the efficiencies w.r.t mHH, only if genHH_mass exists
    if all_event_gen_mHH is not None and event_gen_mHH is not None:
        bbbb_eff_mHH(
            model.output_directory,
            all_event_gen_mHH,
            event_gen_mHH,
            cmssw_selection,
            model_selection,
            ht_only_selection,
            n_events,
            apply_sel,
            apply_light,
        )
    else:
        print("Skipping mHH efficiency plots because 'genHH_mass' is not available")

    # PLot the efficiencies
    # Basically we want to bin the selected truth ht and divide it by the overall count
    all_events = Hist(ht_axis)
    cmssw_selected_events = Hist(ht_axis)
    model_selected_events = Hist(ht_axis)
    ht_only_selected_events = Hist(ht_axis)

    all_events.fill(all_jet_genht)
    cmssw_selected_events.fill(jet_genht[cmssw_selection])
    model_selected_events.fill(jet_genht[model_selection])
    ht_only_selected_events.fill(jet_genht[ht_only_selection])

    # Plot the ratio
    eff_cmssw = plot_ratio(all_events, cmssw_selected_events)
    eff_model = plot_ratio(all_events, model_selected_events)

    # Get data from handles
    cmssw_x, cmssw_y, cmssw_err = get_bar_patch_data(eff_cmssw)
    model_x, model_y, model_err = get_bar_patch_data(eff_model)
    eff_ht_only = plot_ratio(all_events, ht_only_selected_events)
    ht_only_x, ht_only_y, ht_only_err = get_bar_patch_data(eff_ht_only)

    # Plot ht distribution in the background
    counts, bin_edges = np.histogram(np.clip(jet_genht, 0, 800), bins=np.arange(0, 800, 40))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]
    normalized_counts = counts / np.sum(counts)

    # Now plot all
    eff_str = r"$\int \epsilon$"
    fig, ax = plt.subplots(1, 1, figsize=style.FIGURE_SIZE)
    hep.cms.label(llabel=style.CMSHEADER_LEFT, rlabel=style.CMSHEADER_RIGHT, ax=ax, fontsize=style.MEDIUM_SIZE - 2)
    hep.histplot((normalized_counts, bin_edges), ax=ax, histtype='step', color='grey', label=r"$HT^{gen}$")
    ax.errorbar(
        cmssw_x,
        cmssw_y,
        yerr=cmssw_err,
        c=style.color_cycle[0],
        fmt='o',
        linewidth=3,
        label=r'BTag CMSSW Emulator @ 14 kHz, {}={} (L1 $HT$ > {} GeV, $\sum$ 4b > {})'.format(
            eff_str, cmssw_efficiency, cmssw_btag_ht, cmssw_btag
        ),
    )
    ax.errorbar(
        model_x,
        model_y,
        yerr=model_err,
        c=style.color_cycle[1],
        fmt='o',
        linewidth=3,
        label=r'Multiclass @ 14 kHz, {}={} (L1 $HT$ > {} GeV, $\sum$ 4b > {})'.format(
            eff_str, model_efficiency, btag_ht_wp, round(btag_wp, 2)
        ),
    )

    # Plot other labels
    ax.hlines(1, 0, 800, linestyles='dashed', color='black', linewidth=4)
    ax.grid(True)
    ax.set_ylim([0.0, 1.15])
    ax.set_xlim([0, 800])
    ax.set_xlabel(r"$HT^{gen}$ [GeV]")
    ax.set_ylabel(r"$\epsilon$(HH $\to$ 4b)")
    plt.legend(loc='upper left')

    # Save plot
    score_type = "vs_qg" if apply_light else "raw"
    sel_type = "sel" if apply_sel else "all"
    plot_path = os.path.join(model.output_directory, f"plots/physics/bbbb/HH_eff_HT_{score_type}_{sel_type}")
    plt.savefig(f'{plot_path}.pdf', bbox_inches='tight')
    plt.savefig(f'{plot_path}.png', bbox_inches='tight')

    # Plot a different plot comparing the multiclass with ht only selection
    fig2, ax2 = plt.subplots(1, 1, figsize=style.FIGURE_SIZE)
    hep.cms.label(llabel=style.CMSHEADER_LEFT, rlabel=style.CMSHEADER_RIGHT, ax=ax2, fontsize=style.MEDIUM_SIZE - 2)
    hep.histplot((normalized_counts, bin_edges), ax=ax2, histtype='step', color='grey', label=r"$HT^{gen}$")
    ax2.errorbar(
        model_x,
        model_y,
        yerr=model_err,
        c=style.color_cycle[1],
        fmt='o',
        linewidth=3,
        label=r'Multiclass @ 14 kHz, {}={} (L1 $HT$ > {} GeV, $\sum$ 4b > {})'.format(
            eff_str, model_efficiency, btag_ht_wp, round(btag_wp, 2)
        ),
    )
    ax2.errorbar(
        ht_only_x,
        ht_only_y,
        yerr=ht_only_err,
        c=style.color_cycle[2],
        fmt='o',
        linewidth=3,
        label=r'HT + QuadJets @ 14 kHz, {}={} (L1 $HT$ > {} GeV)'.format(eff_str, ht_only_efficiency, ht_only_wp),
    )

    # Common plot settings for second plot
    ax2.hlines(1, 0, 800, linestyles='dashed', color='black', linewidth=4)
    ax2.grid(True)
    ax2.set_ylim([0.0, 1.15])
    ax2.set_xlim([0, 800])
    ax2.set_xlabel(r"$HT^{gen}$ [GeV]")
    ax2.set_ylabel(r"$\epsilon$(HH $\to$ 4b)")
    ax2.legend(loc='upper left')

    # Save second plot
    ht_compare_path = os.path.join(model.output_directory, f"plots/physics/bbbb/HH_eff_HT_vs_HTonly_{score_type}_{sel_type}")
    plt.savefig(f'{ht_compare_path}.pdf', bbox_inches='tight')
    plt.savefig(f'{ht_compare_path}.png', bbox_inches='tight')


def bbbb_eff_mHH(
    model_dir,
    all_event_gen_mHH,
    event_gen_mHH,
    cmssw_selection,
    model_selection,
    ht_only_selection,
    n_events,
    apply_sel,
    apply_light,
):
    """
    Plot HH->4b w.r.t gen m_HH
    """

    # Define the histogram edges
    mHH_edges = list(np.arange(0, 1000, 20))
    mHH_axis = hist.axis.Variable(mHH_edges, name=r"$HT^{gen}$")

    # Efficiencies
    cmssw_efficiency = np.round(ak.sum(cmssw_selection) / n_events, 2)
    model_efficiency = np.round(ak.sum(model_selection) / n_events, 2)
    ht_only_efficiency = np.round(ak.sum(ht_only_selection) / n_events, 2)

    # Create the histograms
    all_events = Hist(mHH_axis)
    cmssw_selected_events = Hist(mHH_axis)
    model_selected_events = Hist(mHH_axis)
    ht_only_selected_events = Hist(mHH_axis)

    all_events.fill(all_event_gen_mHH)
    cmssw_selected_events.fill(event_gen_mHH[cmssw_selection])
    model_selected_events.fill(event_gen_mHH[model_selection])
    ht_only_selected_events.fill(event_gen_mHH[ht_only_selection])

    # Plot the ratio
    eff_cmssw = plot_ratio(all_events, cmssw_selected_events)
    eff_model = plot_ratio(all_events, model_selected_events)

    # Get data from handles
    cmssw_x, cmssw_y, cmssw_err = get_bar_patch_data(eff_cmssw)
    model_x, model_y, model_err = get_bar_patch_data(eff_model)
    eff_ht_only = plot_ratio(all_events, ht_only_selected_events)
    ht_only_x, ht_only_y, ht_only_err = get_bar_patch_data(eff_ht_only)

    # Plot ht distribution in the background
    counts, bin_edges = np.histogram(np.clip(event_gen_mHH, 0, 800), bins=np.arange(0, 800, 40))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]
    normalized_counts = counts / np.sum(counts)

    # Load the working point from model directory
    btag_wp, btag_ht_wp, ht_only_wp = load_bbbb_WPs(model_dir, apply_sel, apply_light)

    # Plot a plot comparing the multiclass with ht only selection
    eff_str = r"$\int \epsilon$"
    fig, ax = plt.subplots(1, 1, figsize=style.FIGURE_SIZE)
    hep.cms.label(llabel=style.CMSHEADER_LEFT, rlabel=style.CMSHEADER_RIGHT, ax=ax, fontsize=style.MEDIUM_SIZE - 2)

    hep.histplot((normalized_counts, bin_edges), ax=ax, histtype='step', color='grey', label=r"$mHH^{gen}$")
    ax.errorbar(
        model_x,
        model_y,
        yerr=model_err,
        c=style.color_cycle[1],
        fmt='o',
        linewidth=3,
        label=r'Multiclass @ 14 kHz, {}={} (L1 $HT$ > {} GeV, $\sum$ 4b > {})'.format(
            eff_str, model_efficiency, btag_ht_wp, round(btag_wp, 2)
        ),
    )
    ax.errorbar(
        ht_only_x,
        ht_only_y,
        yerr=ht_only_err,
        c=style.color_cycle[2],
        fmt='o',
        linewidth=3,
        label=r'HT + QuadJets @ 14 kHz, {}={} (L1 $HT$ > {} GeV)'.format(eff_str, ht_only_efficiency, ht_only_wp),
    )

    # Common plot settings for second plot
    ax.hlines(1, 0, 1000, linestyles='dashed', color='black', linewidth=4)
    ax.grid(True)
    ax.set_ylim([0.0, 1.17])
    ax.set_xlim([0, 1000])
    ax.set_xlabel(r"$m_{HH}^{gen}$ [GeV]")
    ax.set_ylabel(r"$\epsilon$(HH $\to$ 4b)")
    ax.legend(loc='upper left')

    # Save second plot
    score_type = "vs_qg" if apply_light else "raw"
    sel_type = "sel" if apply_sel else "all"
    ht_compare_path = os.path.join(model_dir, f"plots/physics/bbbb/HH_eff_mHH_{score_type}_{sel_type}")
    plt.savefig(f'{ht_compare_path}.pdf', bbox_inches='tight')
    plt.savefig(f'{ht_compare_path}.png', bbox_inches='tight')

    return


if __name__ == "__main__":
    """
    2 steps:

    1. Derive working points: python bbbb.py --deriveWPs
    2. Run efficiency based on the derived working points: python bbbb.py --eff
    """

    parser = ArgumentParser()
    parser.add_argument('-m', '--model_dir', default='output/baseline', help='Input model')
    parser.add_argument(
        '-s',
        '--sample',
        default='/eos/cms/store/cmst3/group/l1tr/sewuchte/l1teg/fp_jettuples_090125_addGenH/GluGluHHTo4B_PU200.root',
        help='Signal sample for HH->bbbb',
    )
    parser.add_argument(
        '--minbias',
        default='/eos/cms/store/cmst3/group/l1tr/sewuchte/l1teg/fp_jettuples_090125/MinBias_PU200.root',
        help='Minbias sample for deriving rates',
    )

    # Different modes
    parser.add_argument('--deriveWPs', action='store_true', help='derive the working points for b-tagging')
    parser.add_argument('--eff', action='store_true', help='plot efficiency for HH->4b')

    parser.add_argument('--tree', default='outnano/Jets', help='Tree within the ntuple containing the jets')

    # Other controls
    parser.add_argument(
        '-n',
        '--n_entries',
        type=int,
        default=1000,
        help='Number of data entries in root file to run over, can speed up run time, set to None to run on all data entries',
    )

    args = parser.parse_args()
    # Load the model
    model = fromFolder(args.model_dir)

    if args.deriveWPs:
        derive_bbbb_WPs(model, args.minbias, True, True, n_entries=args.n_entries, tree=args.tree)
        gc.collect()
        derive_bbbb_WPs(model, args.minbias, True, False, n_entries=args.n_entries, tree=args.tree)
        gc.collect()
        derive_bbbb_WPs(model, args.minbias, False, True, n_entries=args.n_entries, tree=args.tree)
        gc.collect()
        derive_bbbb_WPs(model, args.minbias, False, False, n_entries=args.n_entries, tree=args.tree)
    elif args.eff:
        bbbb_eff(model, args.sample, True, True, n_entries=args.n_entries, tree=args.tree)
        gc.collect()
        bbbb_eff(model, args.sample, True, False, n_entries=args.n_entries, tree=args.tree)
        gc.collect()
        bbbb_eff(model, args.sample, False, True, n_entries=args.n_entries, tree=args.tree)
        gc.collect()
        bbbb_eff(model, args.sample, False, False, n_entries=args.n_entries, tree=args.tree)
