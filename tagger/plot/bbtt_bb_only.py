import os
import json
import gc
from argparse import ArgumentParser

import awkward as ak
import numpy as np
import uproot
import hist
from hist import Hist

import matplotlib.pyplot as plt
import mplhep as hep
from tagger.plot import style
from tagger.plot.common import MINBIAS_RATE, find_rate, plot_ratio, get_bar_patch_data
from tagger.data.tools import extract_array, extract_nn_inputs, group_id_values
from tagger.model.common import fromFolder

style.set_style()


def bbtt_seed(jet_pt, tau_pt):
    # tau pt thresholds L1_DoubleIsoTau34er2p1
    tau_pt = ak.sort(tau_pt, axis=1, ascending=False)
    tau_pt1 = (tau_pt[:, 0] >= 34)
    tau_pt2 = (tau_pt[:, 1] >= 34)
    tau_pt_mask = tau_pt1 & tau_pt2

    ht = ak.sum(jet_pt, axis=1)
    ht_mask = ht > 220

    mask = tau_pt_mask & ht_mask
    n_passed = np.sum(mask)

    return mask, n_passed


def x_vs_y(x, y, apply_light=True):
    if apply_light:
        s = x / (x + y)
        return s
    else:
        return x


def default_selection(jet_pt, jet_eta, indices, apply_sel):
    if apply_sel == "bb":
        rows = np.arange(len(jet_pt)).reshape((-1, 1))
        mask = (jet_pt[rows, indices] > 7) & (
            np.abs(jet_eta[rows, indices]) < 2.4)
        event_mask = np.sum(mask, axis=1) == 2
    elif apply_sel == "all":
        rows = np.arange(len(jet_pt)).reshape((-1, 1))
        mask = (jet_pt[:, :4] > 5) & (np.abs(jet_eta[:, :4]) < 2.4)
        event_mask = np.sum(mask, axis=1) == 4
    else:
        event_mask = np.ones(len(jet_pt), dtype=bool)
    return event_mask


def nn_score_sums(model, jet_nn_inputs, n_jets=4):
    # Btag input list for first 4 jets
    nn_outputs = [model.jet_model.predict(np.asarray(jet_nn_inputs[:, i]))[
        0] for i in range(0, n_jets)]

    # Calculate the output sum
    b_idx = model.class_labels['b']
    l_idx = model.class_labels['light']
    g_idx = model.class_labels['gluon']

    # get sums of 2 leading b scores
    rows = np.arange(len(nn_outputs[0])).reshape((-1, 1))

    # raw preds
    b_preds = np.transpose([pred_score[:, b_idx] for pred_score in nn_outputs])
    b_preds_arg = np.argsort(b_preds, axis=1)[:, -2:]
    b_preds_sums = np.sum(b_preds[rows, b_preds_arg], axis=1)

    # vs light preds
    b_vs_qg = np.transpose([x_vs_y(pred_score[:, b_idx], pred_score[:,
                           l_idx] + pred_score[:, g_idx]) for pred_score in nn_outputs])
    b_vs_qg_arg = np.argsort(b_vs_qg, axis=1)[:, -2:]
    b_vs_qg_sums = np.sum(b_vs_qg[rows, b_vs_qg_arg], axis=1)

    bscore_sums = [b_preds_sums, b_vs_qg_sums]
    bscore_idxs = [b_preds_arg, b_vs_qg_arg]

    return bscore_sums, bscore_idxs


def pick_and_plot(rate_list, ht_list, bb_list, ht, raw_score, apply_sel, model_dir, signal_path, n_entries, target_rate, tree):
    """
    Pick the working points and plot
    """
    # plus, minus range
    RateRange = 0.5

    # Find the target rate points, plot them and print out some info as well
    target_rate_idx = find_rate(
        rate_list, target_rate=target_rate, RateRange=RateRange)

    # Get the coordinates
    target_rate_bb = np.array([bb_list[i]
                              for i in target_rate_idx])  # NN cut dimension
    target_rate_ht = np.array([ht_list[i]
                              for i in target_rate_idx])  # HT cut dimension

    # Export the working point
    fixed_ht_wp = {"HT": float(
        target_rate_ht[0]), "BB": float(target_rate_bb[0])}

    # save WPs
    score_type = 'raw' if raw_score else 'qg'
    plot_dir = os.path.join(model_dir, 'plots/physics/bbtt_bb_only')
    os.makedirs(plot_dir, exist_ok=True)
    with open(os.path.join(plot_dir, f"bbtt_fixed_wp_{score_type}_{apply_sel}.json"), "w") as f:
        json.dump(fixed_ht_wp, f, indent=4)

# derive rate, wps and efficiency plotting functions


def derive_rate(model, minbias_path, n_entries=100000, tree='jetntuple/Jets'):

    minbias = uproot.open(minbias_path)[tree]

    raw_event_id = extract_array(minbias, 'event', n_entries)
    raw_jet_pt = extract_array(minbias, 'jet_pt', n_entries)
    raw_jet_eta = extract_array(minbias, 'jet_eta_phys', n_entries)
    raw_tau_pt = extract_array(minbias, 'jet_taupt', n_entries)

    array_fields = [raw_jet_pt, raw_jet_eta, raw_tau_pt]

    event_id, grouped_arrays = group_id_values(
        raw_event_id, *array_fields, num_elements=4)
    jet_pt, jet_eta, tau_pt = grouped_arrays
    n_events = len(np.unique(raw_event_id))

    # Apply the cuts
    _, n_passed = bbtt_seed(jet_pt, tau_pt)

    # convert to rate [kHz]
    rate = {'rate': (n_passed / n_events) * MINBIAS_RATE}

    rate_dir = os.path.join(model.output_directory, "plots/physics/bbtt")
    os.makedirs(rate_dir, exist_ok=True)
    with open(os.path.join(rate_dir, f"bbtt_seed_rate.json"), "w") as f:
        json.dump(rate, f, indent=4)
    print(f"Derived rate: {rate['rate']} kHz")

    return


def derive_HT_WP(RateHist, ht_edges, n_events, model_dir, target_rate, RateRange=0.85):
    """
    Derive the HT only working points (without bb cuts)
    """
    plot_dir = os.path.join(model_dir, 'plots/physics/bbtt_bb_only')

    # Derive the rate
    rate_list = []
    ht_list = []

    # Loop through the edges and integrate
    for ht in ht_edges[:-1]:

        # Calculate the rate
        rate = RateHist[{"ht": slice(ht*1j, None, sum)}
                        ][{"nn": slice(0.0j, None, sum)}]/n_events
        rate_list.append(rate*MINBIAS_RATE)

        # Append the results
        ht_list.append(ht)

    target_rate_idx = find_rate(
        rate_list, target_rate=target_rate, RateRange=RateRange)

    # Read WPs dict and add HT cut
    WP_json = os.path.join(plot_dir, "ht_working_point.json")
    working_point = {"ht_only_cut": float(ht_list[target_rate_idx[0]])}
    json.dump(working_point, open(WP_json, "w"), indent=4)


def derive_bbtt_WPs(model, minbias_path, ht_cut, apply_sel, signal_path, n_entries=100, tree='outnano/Jets'):
    """
    Derive the HH->4b working points
    """

    with open(os.path.join(model.output_directory, f"plots/physics/bbtt/bbtt_seed_rate.json"), "r") as f:
        rate = json.load(f)
    rate = np.round(rate['rate'], 1)

    # Load input/ouput variables of the NN
    with open(os.path.join(model.output_directory, "class_label.json"), "r") as f:
        class_labels = json.load(f)

    # Load the minbias data
    minbias = uproot.open(minbias_path)[tree]

    raw_event_id = extract_array(minbias, 'event', n_entries)
    raw_jet_pt = extract_array(minbias, 'jet_pt', n_entries)
    raw_jet_eta = extract_array(minbias, 'jet_eta_phys', n_entries)
    raw_inputs = extract_nn_inputs(
        minbias, model.input_vars, n_entries=n_entries)

    # Count number of total event
    n_events = len(np.unique(raw_event_id))
    print("Total number of minbias events: ", n_events)

    # Group these attributes by event id, and filter out groups that don't have at least 2 elements
    event_id, grouped_arrays = group_id_values(
        raw_event_id, raw_jet_pt, raw_jet_eta, raw_inputs, num_elements=4)

    # Extract the grouped arrays
    # Jet pt is already sorted in the producer, no need to do it here
    jet_pt, jet_eta, jet_nn_inputs = grouped_arrays

    bscore_sums, bscore_idxs = nn_score_sums(model, jet_nn_inputs)
    def_sels = [default_selection(jet_pt, jet_eta, bscore_idxs[0], apply_sel),
                default_selection(jet_pt, jet_eta, bscore_idxs[1], apply_sel)]

    # apply the kinemeatic default selelction
    bscore_sums = [b_sum[def_sel]
                   for b_sum, def_sel in zip(bscore_sums, def_sels)]

    jet_ht = ak.sum(jet_pt, axis=1)

    # Define the histograms (pT edge and NN Score edge)
    # Make sure to capture everything
    ht_edges = list(np.arange(150, 500, 1)) + [10000]
    NN_edges = list([round(i, 3) for i in np.arange(0, 1.2, 0.001)]) + [2.0]

    # for raw and vs light preds
    raw = True
    for bscore_sum, def_sel in zip(bscore_sums, def_sels):
        sel_ht = jet_ht[def_sel]
        RateHist = Hist(hist.axis.Variable(ht_edges, name="ht", label="ht"),
                        hist.axis.Variable(NN_edges, name="nn", label="nn_bb"))
        RateHist.fill(ht=sel_ht, nn=bscore_sum)

        assert (len(bscore_sum) == len(sel_ht))
        assert (len(bscore_sum) == len(sel_ht))

        # Derive the rate
        rate_list = []
        ht_list = []
        bb_list = []

        # Loop through the edges and integrate
        for bb in NN_edges[:-1]:
            # Calculate the rate
            counts = RateHist[{"ht": slice(
                ht_cut*1j, None, sum)}][{"nn": slice(bb*1.0j, None, sum)}]
            rate_list.append((counts / n_events)*MINBIAS_RATE)

            # Append the results
            ht_list.append(ht_cut)
            bb_list.append(bb)

        # Pick target rate and plot it
        pick_and_plot(rate_list, ht_list, bb_list, ht_cut, raw, apply_sel,
                      model.output_directory, signal_path, n_entries, rate, tree)
        gc.collect()
        raw = False

    # Derive the HT working point
    derive_HT_WP(RateHist, ht_edges, n_events, model.output_directory, rate)
    return


def bbtt_eff_HT(model, signal_path, score_type, apply_sel, n_entries=100000, tree='outnano/Jets'):
    """
    Plot HH->4b efficiency w.r.t HT
    """
    with open(os.path.join(model.output_directory, f"plots/physics/bbtt/bbtt_seed_rate.json"), "r") as f:
        rate = json.load(f)
    rate = np.round(rate['rate'], 1)

    ht_egdes = list(np.arange(0, 800, 20))
    ht_axis = hist.axis.Variable(ht_egdes, name=r"$HT^{gen}$")

    # Check if the working point have been derived
    WP_path = os.path.join(
        model.output_directory, f"plots/physics/bbtt_bb_only/bbtt_fixed_wp_{score_type}_{apply_sel}.json")
    HT_path = os.path.join(model.output_directory,
                           "plots/physics/bbtt_bb_only/ht_working_point.json")

    # Get derived working points
    if os.path.exists(WP_path) & os.path.exists(HT_path):
        # first WP Path
        with open(WP_path, "r") as f:
            WPs = json.load(f)
        btag_wp = WPs['BB']
        ht_wp = int(WPs['HT'])
        # HT only WP Path
        with open(HT_path, "r") as f:
            WPs = json.load(f)
        ht_only_wp = int(WPs['ht_only_cut'])
    else:
        raise Exception(
            "Working point does not exist. Run with --deriveWPs first.")

    # Load the signal data
    signal = uproot.open(signal_path)[tree]

    # Calculate the truth HT
    raw_event_id = extract_array(signal, 'event', n_entries)
    raw_jet_genpt = extract_array(signal, 'jet_genmatch_pt', n_entries)
    raw_jet_pt = extract_array(signal, 'jet_pt', n_entries)
    raw_jet_eta = extract_array(signal, 'jet_eta_phys', n_entries)
    raw_tau_pt = extract_array(signal, 'jet_taupt', n_entries)

    raw_inputs = extract_nn_inputs(
        signal, model.input_vars, n_entries=n_entries)

    # Group these attributes by event id, and filter out groups that don't have at least 4 elements
    event_id, grouped_arrays = group_id_values(
        raw_event_id, raw_jet_genpt, raw_jet_pt, raw_jet_eta, raw_tau_pt, raw_inputs, num_elements=4)
    jet_genpt, jet_pt, jet_eta, tau_pt, jet_nn_inputs = grouped_arrays

    # Calculate the ht
    jet_genht = ak.sum(jet_genpt, axis=1)
    jet_ht = ak.sum(jet_pt, axis=1)

    # Result from the baseline selection, multiclass tagger and ht only working point
    baseline_selection, _ = bbtt_seed(jet_pt, tau_pt)
    model_bscore_sums, bscore_indices = nn_score_sums(model, jet_nn_inputs)

    # use either raw or vs light scores
    if score_type == 'raw':
        model_bscore_sum = model_bscore_sums[0]
        default_sel = default_selection(
            jet_pt, jet_eta, bscore_indices[0], apply_sel)
    else:
        model_bscore_sum = model_bscore_sums[1]
        default_sel = default_selection(
            jet_pt, jet_eta, bscore_indices[1], apply_sel)

    model_selection = (jet_ht > ht_wp) & (
        model_bscore_sum > btag_wp) & default_sel
    ht_only_selection = jet_ht > ht_only_wp

    # PLot the efficiencies
    # Basically we want to bin the selected truth ht and divide it by the overall count
    all_events = Hist(ht_axis)
    baseline_selected_events = Hist(ht_axis)
    model_selected_events = Hist(ht_axis)
    ht_only_selected_events = Hist(ht_axis)

    all_events.fill(jet_genht)
    baseline_selected_events.fill(jet_genht[baseline_selection])
    model_selected_events.fill(jet_genht[model_selection])
    ht_only_selected_events.fill(jet_genht[ht_only_selection])

    # Plot the ratio
    eff_baseline = plot_ratio(all_events, baseline_selected_events)
    eff_model = plot_ratio(all_events, model_selected_events)

    # Get data from handles
    baseline_x, baseline_y, baseline_err = get_bar_patch_data(eff_baseline)
    model_x, model_y, model_err = get_bar_patch_data(eff_model)
    eff_ht_only = plot_ratio(all_events, ht_only_selected_events)
    ht_only_x, ht_only_y, ht_only_err = get_bar_patch_data(eff_ht_only)

    # Plot ht distribution in the background
    counts, bin_edges = np.histogram(
        np.clip(jet_genht, 0, 800), bins=np.arange(0, 800, 40))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]
    normalized_counts = counts / np.sum(counts)

    # Now plot all
    tau_str = r"$\tau_{1,2}$"
    fig, ax = plt.subplots(1, 1, figsize=style.FIGURE_SIZE)
    hep.cms.label(llabel=style.CMSHEADER_LEFT,
                  rlabel=style.CMSHEADER_RIGHT, ax=ax, fontsize=style.MEDIUM_SIZE-2)
    hep.histplot((normalized_counts, bin_edges), ax=ax,
                 histtype='step', color='grey', label=r"$HT^{gen}$")
    ax.errorbar(baseline_x, baseline_y, yerr=baseline_err, c=style.color_cycle[0], fmt='o', linewidth=3,
                label=r'bb$\tau \tau$ seed@ {} kHz (L1 $HT$ > {} GeV, {} $p_T$ > {} GeV)'.format(rate, 220, tau_str, 34))
    ax.errorbar(model_x, model_y, yerr=model_err, c=style.color_cycle[1], fmt='o', linewidth=3,
                label=r'Multiclass @ {} kHz (L1 $HT$ > {} GeV, $\sum$ bb > {})'.format(rate, ht_wp, round(btag_wp, 2)))

    # Plot other labels
    ax.hlines(1, 0, 800, linestyles='dashed', color='black', linewidth=4)
    ax.grid(True)
    ax.set_ylim([0., 1.15])
    ax.set_xlim([0, 800])
    ax.set_xlabel(r"$HT^{gen}$ [GeV]")
    ax.set_ylabel(
        r"$\epsilon$(HH $\to$ bb$\tau \tau$ trigger rate at {} kHz)".format(rate))
    plt.legend(loc='upper left')

    # Save plot
    plot_path = os.path.join(
        model_dir, f"plots/physics/bbtt_bb_only/HHbbtt_bbtt_seed_eff_{score_type}_{apply_sel}")
    plt.savefig(f'{plot_path}.pdf', bbox_inches='tight')
    plt.savefig(f'{plot_path}.png', bbox_inches='tight')
    plt.clf()

    # Now plot all (HT only comparison)
    fig2, ax2 = plt.subplots(1, 1, figsize=style.FIGURE_SIZE)
    hep.cms.label(llabel=style.CMSHEADER_LEFT,
                  rlabel=style.CMSHEADER_RIGHT, ax=ax, fontsize=style.MEDIUM_SIZE-2)
    hep.histplot((normalized_counts, bin_edges), ax=ax2,
                 histtype='step', color='grey', label=r"$HT^{gen}$")
    ax2.errorbar(model_x, model_y, yerr=model_err, c=style.color_cycle[1], fmt='o', linewidth=3,
                 label=r'Multiclass @ {} kHz (L1 $HT$ > {} GeV, $\sum$ bb > {})'.format(rate, ht_wp, round(btag_wp, 2)))
    ax2.errorbar(ht_only_x, ht_only_y, yerr=ht_only_err,
                 c=style.color_cycle[2], fmt='o', linewidth=3, label=r'HT-only @ {} kHz (L1 $HT$ > {} GeV)'.format(rate, ht_only_wp))

    # Plot other labels
    ax2.hlines(1, 0, 800, linestyles='dashed', color='black', linewidth=4)
    ax2.grid(True)
    ax2.set_ylim([0., 1.15])
    ax2.set_xlim([0, 800])
    ax2.set_xlabel(r"$HT^{gen}$ [GeV]")
    ax2.set_ylabel(
        r"$\epsilon$(HH $\to$ bb$\tau \tau$ trigger rate at {} kHz)".format(rate))
    plt.legend(loc='upper left')

    # Save plot
    plot_path = os.path.join(
        model.output_directory, f"plots/physics/bbtt_bb_only/HHbbtt_HT_only_eff_{score_type}_{apply_sel}")
    plt.savefig(f'{plot_path}.pdf', bbox_inches='tight')
    plt.savefig(f'{plot_path}.png', bbox_inches='tight')


if __name__ == "__main__":
    """
    2 steps:

    1. Derive working points: python bbtt.py --deriveWPs
    2. Run efficiency based on the derived working points: python bbtt.py --eff
    """

    parser = ArgumentParser()
    parser.add_argument('-m', '--model_dir',
                        default='output/baseline', help='Input model')
    parser.add_argument(
        '-s', '--signal', default='/eos/cms/store/cmst3/group/l1tr/sewuchte/l1teg/fp_jettuples_090125/GluGluHHTo2B2Tau_PU200.root', help='Signal sample for HH->bbtt')
    parser.add_argument('--minbias', default='/eos/cms/store/cmst3/group/l1tr/sewuchte/l1teg/fp_jettuples_090125/MinBias_PU200.root',
                        help='Minbias sample for deriving rates')
    parser.add_argument('--deriveRate', action='store_true',
                        help='derive the rate for the bbtt seed')

    # Different modes
    parser.add_argument('--deriveWPs', action='store_true',
                        help='derive the working points for b-tagging')
    parser.add_argument('--eff', action='store_true',
                        help='plot efficiency for HH->bbtt')

    parser.add_argument('--tree', default='outnano/Jets',
                        help='Tree within the ntuple containing the jets')

    # Other controls
    parser.add_argument('-n', '--n_entries', type=int, default=1000,
                        help='Number of data entries in root file to run over, can speed up run time, set to None to run on all data entries')
    args = parser.parse_args()

    model = fromFolder(args.model_dir)

    if args.deriveRate:
        derive_rate(model, args.minbias,
                    n_entries=args.n_entries, tree=args.tree)
    if args.deriveWPs:
        derive_bbtt_WPs(model, args.minbias, 220, 'bb',
                        args.signal, n_entries=args.n_entries, tree=args.tree)
        gc.collect()
        derive_bbtt_WPs(model, args.minbias, 220, 'all',
                        args.signal, n_entries=args.n_entries, tree=args.tree)
    elif args.eff:
        bbtt_eff_HT(model, args.signal, 'raw', 'bb',
                    n_entries=args.n_entries, tree=args.tree)
        gc.collect()
        bbtt_eff_HT(model, args.signal, 'raw', 'all',
                    n_entries=args.n_entries, tree=args.tree)
        gc.collect()
        bbtt_eff_HT(model, args.signal, 'qg', 'bb',
                    n_entries=args.n_entries, tree=args.tree)
        gc.collect()
        bbtt_eff_HT(model, args.signal, 'qg', 'all',
                    n_entries=args.n_entries, tree=args.tree)
