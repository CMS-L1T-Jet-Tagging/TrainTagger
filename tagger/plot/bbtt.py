import os, json
import gc
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
from joblib import Parallel, delayed, parallel_backend
import time
from tqdm import tqdm

style.set_style()

#Interpolation of working point
from scipy.interpolate import interp1d

#Imports from other modules
from tagger.data.tools import extract_array, extract_nn_inputs, group_id_values
from common import MINBIAS_RATE, WPs_CMSSW, find_rate, plot_ratio, get_bar_patch_data

# Helpers
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
    if apply_sel == "tau":
        rows = np.arange(len(jet_pt)).reshape((-1, 1))
        mask = (jet_pt[rows, indices] > 7) & (np.abs(jet_eta[rows, indices]) < 2.4)
        event_mask = np.sum(mask, axis=1) == 2
    elif apply_sel == "all":
        rows = np.arange(len(jet_pt)).reshape((-1, 1))
        mask = (jet_pt[:,:4] > 5) & (np.abs(jet_eta[:,:4]) < 2.4)
        event_mask = np.sum(mask, axis=1) == 4
    else:
        event_mask = np.ones(len(jet_pt), dtype=bool)
    return event_mask

def max_tau_sum(taup_preds, taum_preds):
    """
    Calculate the sum of the two highest tau scores
    """
    taup_argsort, taum_argsort = np.argsort(taup_preds, axis=1)[:,::-1][:,:2], np.argsort(taum_preds, axis=1)[:,::-1][:,:2]
    rows = np.arange(len(taup_preds)).reshape((-1, 1))
    alt_scores = taup_preds[rows, taup_argsort] + taum_preds[rows, taum_argsort][:,::-1]
    tau_alt_idxs = np.stack((taup_argsort[:,0], taum_argsort[:,1]), axis=-1)
    tau_alt2_idxs = np.stack((taup_argsort[:,1], taum_argsort[:,0]), axis=-1)
    tau_alt_idxs[ak.argmax(alt_scores, axis=1) == 1] = tau_alt2_idxs[ak.argmax(alt_scores, axis=1) == 1]
    tau_scores1 = taup_preds[rows, taup_argsort[:,0].reshape(-1,1)] + taum_preds[rows, taum_argsort[:,0].reshape(-1,1)]
    tau_scores1 = tau_scores1.flatten()
    tau_scores1[taup_argsort[:,0] == taum_argsort[:,0]] = ak.max(alt_scores, axis=1)[taup_argsort[:,0] == taum_argsort[:,0]]
    tau_idxs = np.stack((taup_argsort[:,0], taum_argsort[:,0]), axis=-1)
    tau_idxs[taup_argsort[:,0] == taum_argsort[:,0]] =  tau_alt_idxs[taup_argsort[:,0] == taum_argsort[:,0]]
    taup_sum = np.sum(taup_preds[rows,tau_idxs], axis=1)
    taum_sum = np.sum(taum_preds[rows,tau_idxs], axis=1)
    tau_scores2 = np.multiply(taup_sum, taum_sum)

    return tau_scores2, tau_idxs

def nn_score_sums(model, jet_nn_inputs, class_labels, n_jets=4):
    #Btag input list for first 4 jets
    nn_outputs = [model.predict(np.asarray(jet_nn_inputs[:, i]))[0] for i in range(0,n_jets)]

    #Calculate the output sum
    b_idx = class_labels['b']
    tp_idx = class_labels['taup']
    tm_idx = class_labels['taum']
    e_idx = class_labels['electron']
    l_idx = class_labels['light']
    g_idx = class_labels['gluon']
    u_idx = class_labels['pileup']

    # vs light preds
    taup_vs_qg = np.transpose([x_vs_y(pred_score[:, tp_idx] + pred_score[:,e_idx], pred_score[:, g_idx] + pred_score[:, u_idx]) for pred_score in nn_outputs])
    taum_vs_qg = np.transpose([x_vs_y(pred_score[:, tm_idx] + pred_score[:,e_idx], pred_score[:, g_idx] + pred_score[:, u_idx]) for pred_score in nn_outputs])

    b_vs_qg = np.transpose([x_vs_y(pred_score[:, b_idx], pred_score[:, u_idx]) for pred_score in nn_outputs])

    # raw preds
    taup_preds = np.transpose([pred_score[:, tp_idx] for pred_score in nn_outputs])
    taum_preds = np.transpose([pred_score[:, tm_idx] for pred_score in nn_outputs])
    b_preds = np.transpose([pred_score[:, b_idx] for pred_score in nn_outputs])

    bscore_sums, tscore_sums, tscore_idxs = [], [], []
    for b, taup, taum in zip([b_preds, b_vs_qg], [taup_preds, taup_vs_qg], [taum_preds, taum_vs_qg]):
        tscore_sum, tau_indices = max_tau_sum(taup, taum)
        tscore_idxs.append(tau_indices)
        tscore_sums.append(tscore_sum)

        # use b scores from the remaining 2 jets
        indices = np.tile(np.arange(0,b.shape[1]),(b.shape[0],1))
        indices[np.arange(len(indices)).reshape(-1, 1), tau_indices] = -1
        b_indices = np.sort(indices, axis=1)[:, -2:]
        bscore_sum = b[np.arange(len(b)).reshape(-1, 1), b_indices].sum(axis=1)
        bscore_sums.append(bscore_sum)

    return bscore_sums, tscore_sums, tau_indices

def pick_and_plot(rate_list, signal_eff, ht_list, bb_list, tt_list, score_type, apply_sel, model_dir, n_entries, rate, tree):
    """
    Pick the working points and plot
    """
    #plus, minus range
    RateRange = 1.

    #Find the target rate points, plot them and print out some info as well
    target_rate_idx = find_rate(rate_list, target_rate = rate, RateRange=RateRange)

    #Get the coordinates
    target_rate_bb = np.array([bb_list[i] for i in target_rate_idx]) # NN cut dimension
    target_rate_tt = np.array([tt_list[i] for i in target_rate_idx]) # NN cut dimension
    target_rate_ht = np.array([ht_list[i] for i in target_rate_idx]) # HT cut dimension
    target_rate_eff = np.array([signal_eff[i] for i in target_rate_idx]) # signal efficiency dimension

    # get max efficiency at target rate
    best_WP = np.argmax(target_rate_eff)
    ht = target_rate_ht[best_WP]

    # Efficiency at target rate and HT WP
    best_bb = target_rate_bb[best_WP] # target rate and HT
    best_tt = target_rate_tt[best_WP] # target rate and HT
    best_eff = target_rate_eff[best_WP]

    wp_ht_eff_idx = np.argmax(target_eff)

    # Export the working point
    fixed_ht_wp = {"HT": float(ht), "BB": float(best_bb), "TT": float(best_tt)}

    # save WPs
    plot_dir = os.path.join(model_dir, 'plots/physics/bbtt')
    os.makedirs(plot_dir, exist_ok=True)
    with open(os.path.join(plot_dir, f"bbtt_fixed_wp_{score_type}_{apply_sel}_{rate}.json"), "w") as f:
        json.dump(fixed_ht_wp, f, indent=4)

def make_predictions(data_path, model_dir, n_entries, tree='outnano/Jets', njets=4):
    data = uproot.open(data_path)[tree]

    model = load_qmodel(os.path.join(model_dir, "model/saved_model.h5"))

    #Load input/ouput variables of the NN
    with open(os.path.join(model_dir, "input_vars.json"), "r") as f: input_vars = json.load(f)
    with open(os.path.join(model_dir, "class_label.json"), "r") as f: class_labels = json.load(f)

    raw_event_id = extract_array(data, 'event', n_entries)
    raw_jet_pt = extract_array(data, 'jet_pt', n_entries)
    raw_jet_eta = extract_array(data, 'jet_eta_phys', n_entries)
    raw_inputs = extract_nn_inputs(data, input_vars, n_entries=n_entries)

    #Count number of total event
    n_events = len(np.unique(raw_event_id))

    #Group these attributes by event id, and filter out groups that don't have at least 2 elements
    event_id, grouped_arrays  = group_id_values(raw_event_id, raw_jet_pt, raw_jet_eta, raw_inputs, num_elements=4)

    # Extract the grouped arrays
    # Jet pt is already sorted in the producer, no need to do it here
    jet_pt, jet_eta, jet_nn_inputs = grouped_arrays

    with open(os.path.join(model_dir, "class_label.json"), "r") as f: class_labels = json.load(f)

    #Calculate the output sums
    bscore_sums, tscore_sums, tau_indices = nn_score_sums(model, jet_nn_inputs, class_labels, n_jets=4)

    return bscore_sums, tscore_sums, tau_indices, jet_pt, jet_eta, n_events


# Callables for studies
def derive_HT_WP(RateHist, ht_edges, n_events, model_dir, target_rate, RateRange=0.85):
    """
    Derive the HT only working points (without bb cuts)
    """

    plot_dir = os.path.join(model_dir, 'plots/physics/bbtt')

    #Derive the rate
    rate_list = []
    ht_list = []

    #Loop through the edges and integrate
    for ht in ht_edges[:-1]:

        #Calculate the rate
        rate = RateHist[{"ht": slice(ht*1j, None, sum)}][{"nn": slice(0.0j, None, sum)}]/n_events
        rate_list.append(rate*MINBIAS_RATE)

        #Append the results
        ht_list.append(ht)

    target_rate_idx = find_rate(rate_list, target_rate = target_rate, RateRange=RateRange)

    #Read WPs dict and add HT cut
    WP_json = os.path.join(plot_dir, "ht_working_point.json")
    working_point = {"ht_only_cut": float(ht_list[target_rate_idx[0]])}
    json.dump(working_point, open(WP_json, "w"), indent=4)

def derive_bbtt_WPs(model_dir, minbias_path, apply_sel, signal_path, n_entries=100, tree='outnano/Jets'):
    """
    Derive the HH->4b working points
    """
    model = load_qmodel(os.path.join(model_dir, "model/saved_model.h5"))

    with open(os.path.join(model_dir, f"plots/physics/bbtt/bbtt_seed_rate.json"), "r") as f: rate = json.load(f)
    rate = np.round(rate['rate'], 1)

    #Load input/ouput variables of the NN
    with open(os.path.join(model_dir, "input_vars.json"), "r") as f: input_vars = json.load(f)
    with open(os.path.join(model_dir, "class_label.json"), "r") as f: class_labels = json.load(f)

    #Load the minbias data
    minbias = uproot.open(minbias_path)[tree]

    raw_event_id = extract_array(minbias, 'event', n_entries)
    raw_jet_pt = extract_array(minbias, 'jet_pt', n_entries)
    raw_jet_eta = extract_array(minbias, 'jet_eta_phys', n_entries)
    raw_inputs = extract_nn_inputs(minbias, input_vars, n_entries=n_entries)

    #Count number of total event
    n_events = len(np.unique(raw_event_id))
    print("Total number of minbias events: ", n_events)

    #Group these attributes by event id, and filter out groups that don't have at least 2 elements
    event_id, grouped_arrays  = group_id_values(raw_event_id, raw_jet_pt, raw_jet_eta, raw_inputs, num_elements=4)

    # Extract the grouped arrays
    # Jet pt is already sorted in the producer, no need to do it here
    jet_pt, jet_eta, jet_nn_inputs = grouped_arrays

    bscore_sums, tscore_sums, tau_indices = nn_score_sums(model, jet_nn_inputs, class_labels)
    def_sels = [default_selection(jet_pt, jet_eta, tau_indices[0], apply_sel),
                default_selection(jet_pt, jet_eta, tau_indices[1], apply_sel)]

    # apply the kinemeatic default selelction
    bscore_sums = [b_sum[def_sel] for b_sum, def_sel in zip(bscore_sums, def_sels)]
    tscore_sums = [t_sum[def_sel] for t_sum, def_sel in zip(tscore_sums, def_sels)]

    jet_ht = ak.sum(jet_pt, axis=1)

    #Define the histograms (pT edge and NN Score edge)
    ht_edges = list(np.arange(150,500,1)) + [10000] #Make sure to capture everything
    NN_edges = list([round(i,4) for i in np.arange(0.01, .4, 0.0025)]) + [2.0]

    # Signal preds to pick the working point
    s_bscore_sums, s_tscore_sums, s_tau_indices, signal_pt, signal_eta, s_n_events = make_predictions(signal_path, model_dir, n_entries, tree=tree)
    signal_ht = ak.sum(signal_pt, axis=1)
    s_def_sels = [default_selection(signal_pt, signal_eta, s_tau_indices[0], apply_sel),
                  default_selection(signal_pt, signal_eta, s_tau_indices[1], apply_sel)]
    s_bscore_sums = [b_sum[def_sel] for b_sum, def_sel in zip(s_bscore_sums, s_def_sels)]
    s_tscore_sums = [t_sum[def_sel] for t_sum, def_sel in zip(s_tscore_sums, s_def_sels)]

    # Rate Hists for rate derivation
    RateHistRaw = Hist(hist.axis.Variable(ht_edges, name="ht", label="ht"),
                    hist.axis.Variable(NN_edges, name="nn_bb", label="nn_bb"),
                    hist.axis.Variable(NN_edges, name="nn_tt", label="nn_tt"))
    RateHistQG = Hist(hist.axis.Variable(ht_edges, name="ht", label="ht"),
                    hist.axis.Variable(NN_edges, name="nn_bb", label="nn_bb"),
                    hist.axis.Variable(NN_edges, name="nn_tt", label="nn_tt"))
    SHistRaw = Hist(hist.axis.Variable(ht_edges, name="ht", label="ht"),
                    hist.axis.Variable(NN_edges, name="nn_bb", label="nn_bb"),
                    hist.axis.Variable(NN_edges, name="nn_tt", label="nn_tt"))
    SHistQG = Hist(hist.axis.Variable(ht_edges, name="ht", label="ht"),
                    hist.axis.Variable(NN_edges, name="nn_bb", label="nn_bb"),
                    hist.axis.Variable(NN_edges, name="nn_tt", label="nn_tt"))

    # Fill the histograms
    assert(len(jet_ht[def_sels[0]]) == len(bscore_sums[0]))
    assert(len(jet_ht[def_sels[1]]) == len(bscore_sums[1]))
    RateHistRaw.fill(ht = jet_ht[def_sels[0]], nn_bb = bscore_sums[0], nn_tt = tscore_sums[0])
    RateHistQG.fill(ht = jet_ht[def_sels[1]], nn_bb = bscore_sums[1], nn_tt = tscore_sums[1])

    # Signal Rate Hists to pick the working point
    assert(len(signal_ht[s_def_sels[0]]) == len(s_bscore_sums[0]))
    assert(len(signal_ht[s_def_sels[1]]) == len(s_bscore_sums[1]))
    SHistRaw.fill(ht = signal_ht[s_def_sels[0]], nn_bb = s_bscore_sums[0], nn_tt = s_tscore_sums[0])
    SHistQG.fill(ht = signal_ht[s_def_sels[1]], nn_bb = s_bscore_sums[1], nn_tt = s_tscore_sums[1])

    def pick_rates(rate, target_rate=rate):
        if (rate > 16 and rate < target_rate-2) or rate < 12 or rate > target_rate+2:
            return True
        else:
            return False

    # Parallelized loop through the edges and integrate
    def parallel_in_parallel(tt, bb, ht_cut):
        #Calculate the rate
        counts_raw = RateHistRaw[{"ht": slice(ht_cut*1j, None, sum)}][{"nn_bb": slice(bb*1.0j, None, sum)}][{"nn_tt": slice(tt*1.0j, None, sum)}]
        rate_raw = (counts_raw / n_events)*MINBIAS_RATE

        counts_qg = RateHistQG[{"ht": slice(ht_cut*1j, None, sum)}][{"nn_bb": slice(bb*1.0j, None, sum)}][{"nn_tt": slice(tt*1.0j, None, sum)}]
        rate_qg = (counts_qg / n_events)*MINBIAS_RATE

        # check if the rate is in the range, skip if not
        if pick_rates(rate_raw) and pick_rates(rate_qg): return

        # get signal efficiencies
        counts_signal_raw = SHistRaw[{"ht": slice(ht_cut*1j, None, sum)}][{"nn_bb": slice(bb*1.0j, None, sum)}][{"nn_tt": slice(tt*1.0j, None, sum)}]
        eff_signal_raw = counts_signal_raw / s_n_events

        counts_signal_qg = SHistQG[{"ht": slice(ht_cut*1j, None, sum)}][{"nn_bb": slice(bb*1.0j, None, sum)}][{"nn_tt": slice(tt*1.0j, None, sum)}]
        eff_signal_qg = counts_signal_qg / s_n_events

        return np.array([rate_raw, rate_qg, eff_signal_raw, eff_signal_qg, ht_cut, bb, tt])

    def parallel_in_parallel_wrapper(bb, ht_cut, n_threads=2):
        with parallel_backend('loky', inner_max_num_threads=n_threads):
            intermediate_out = Parallel(n_jobs=n_threads)(delayed(parallel_in_parallel)(tt=tt, bb=bb, ht_cut=ht_cut) for tt in NN_edges[:-1])
        return intermediate_out

    #Parallelized the first loop
    start = time.time()
    parallel_out = Parallel(n_jobs=3)(delayed(parallel_in_parallel_wrapper)(bb, ht_cut) for bb,ht_cut in tqdm(zip(NN_edges[:-1], ht_edges[:-1])))
    parallel_out = ak.drop_none(parallel_out)
    end = time.time()
    print("Time taken for parallel loop: ", end-start)
    np_out = ak.to_numpy(ak.flatten(parallel_out, axis=1))

    # Unpack and convert back to lists
    rate_list_raw, rate_list_qg = np_out[:,0].tolist(), np_out[:,1].tolist()
    eff_list_raw, eff_list_qg = np_out[:,2].tolist(), np_out[:,3].tolist()
    ht_list, bb_list, tt_list = np_out[:,4].tolist(), np_out[:,5].tolist(), np_out[:,6].tolist()

    #Pick target rate and plot it
    pick_and_plot(rate_list_raw, eff_list_raw, ht_list, bb_list, tt_list, 'raw', apply_sel, model_dir, n_entries, rate, tree)
    gc.collect()
    pick_and_plot(rate_list_raw, eff_list_raw, ht_list, bb_list, tt_list, 'raw', apply_sel, model_dir, n_entries, 14, tree)
    gc.collect()
    pick_and_plot(rate_list_qg, eff_list_qg, ht_list, bb_list, tt_list, 'qg', apply_sel, model_dir, n_entries, rate, tree)
    gc.collect()
    pick_and_plot(rate_list_qg, eff_list_qg, ht_list, bb_list, tt_list, 'qg', apply_sel, model_dir, n_entries, 14, tree)
    gc.collect()

    # refill with full ht for ht wp derivation
    RateHist = Hist(hist.axis.Variable(ht_edges, name="ht", label="ht"),
                    hist.axis.Variable(NN_edges, name="nn", label="nn"))

    RateHist.fill(ht = jet_ht, nn = np.zeros(len(jet_ht)))
    derive_HT_WP(RateHist, ht_edges, n_events, model_dir, 14)
    return

def bbtt_eff_HT(model_dir, signal_path, score_type, apply_sel, n_entries=100000, tree='outnano/Jets'):
    """
    Plot HH->4b efficiency w.r.t HT
    """
    with open(os.path.join(model_dir, f"plots/physics/bbtt/bbtt_seed_rate.json"), "r") as f: rate = json.load(f)
    rate = np.round(rate['rate'], 1)

    model=load_qmodel(os.path.join(model_dir, "model/saved_model.h5"))

    ht_egdes = list(np.arange(0,800,20))
    ht_axis = hist.axis.Variable(ht_egdes, name = r"$HT^{gen}$")

    #Check if the working point have been derived
    WP_path = os.path.join(model_dir, f"plots/physics/bbtt/bbtt_fixed_wp_{score_type}_{apply_sel}_{rate}.json")
    WP_path_14 = os.path.join(model_dir, f"plots/physics/bbtt/bbtt_fixed_wp_{score_type}_{apply_sel}_14.json")
    HT_path = os.path.join(model_dir, "plots/physics/bbtt/ht_working_point.json")

    #Get derived working points
    if os.path.exists(WP_path) & os.path.exists(HT_path) & os.path.exists(WP_path_14):
        # first WP Path
        with open(WP_path, "r") as f:  WPs = json.load(f)
        btag_wp = WPs['BB']
        ttag_wp = WPs['TT']
        ht_wp = int(WPs['HT'])
        # HT only WP Path
        with open(HT_path, "r") as f:  WPs = json.load(f)
        ht_only_wp = int(WPs['ht_only_cut'])
        with open(WP_path_14, "r") as f:  WPs = json.load(f)
        btag_wp_14 = WPs['BB']
        ttag_wp_14 = WPs['TT']
        ht_wp_14 = int(WPs['HT'])
    else:
        raise Exception("Working point does not exist. Run with --deriveWPs first.")

    #Load the signal data
    signal = uproot.open(signal_path)[tree]

    # Calculate the truth HT
    raw_event_id = extract_array(signal, 'event', n_entries)
    raw_jet_genpt = extract_array(signal, 'jet_genmatch_pt', n_entries)
    raw_jet_pt = extract_array(signal, 'jet_pt', n_entries)
    raw_jet_eta = extract_array(signal, 'jet_eta_phys', n_entries)
    raw_tau_pt = extract_array(signal, 'jet_taupt', n_entries)
    n_events = len(np.unique(raw_event_id))

    # Load the inputs
    with open(os.path.join(model_dir, "input_vars.json"), "r") as f: input_vars = json.load(f)
    with open(os.path.join(model_dir, "class_label.json"), "r") as f: class_labels = json.load(f)

    raw_inputs = extract_nn_inputs(signal, input_vars, n_entries=n_entries)

    #Group these attributes by event id, and filter out groups that don't have at least 4 elements
    event_id, grouped_arrays  = group_id_values(raw_event_id, raw_jet_genpt, raw_jet_pt, raw_jet_eta, raw_tau_pt, raw_inputs, num_elements=4)
    jet_genpt, jet_pt, jet_eta, tau_pt, jet_nn_inputs = grouped_arrays

    #Calculate the ht
    jet_genht = ak.sum(jet_genpt, axis=1)
    jet_ht = ak.sum(jet_pt, axis=1)

    # Result from the baseline selection, multiclass tagger and ht only working point
    baseline_selection, _ = bbtt_seed(jet_pt, tau_pt)
    baseline_efficiency = np.round(np.sum(baseline_selection) / n_events, 2)
    model_bscore_sums, model_tscore_sums, tau_indices = nn_score_sums(model, jet_nn_inputs, class_labels)

    # use either raw or vs light scores
    if score_type == 'raw':
        model_bscore_sum = model_bscore_sums[0]
        model_tscore_sum = model_tscore_sums[0]
        default_sel = default_selection(jet_pt, jet_eta, tau_indices[0], apply_sel)
    else:
        model_bscore_sum = model_bscore_sums[1]
        model_tscore_sum = model_tscore_sums[1]
        default_sel = default_selection(jet_pt, jet_eta, tau_indices[1], apply_sel)

    model_selection = (jet_ht > ht_wp) & (model_bscore_sum > btag_wp) & (model_tscore_sum > ttag_wp) & default_sel
    model_efficiency = np.round(np.sum(model_selection) / n_events, 2)
    model_selection_14 = (jet_ht > ht_wp_14) & (model_bscore_sum > btag_wp_14) & (model_tscore_sum > ttag_wp_14) & default_sel
    model_14_efficiency = np.round(np.sum(model_selection_14) / n_events, 2)
    ht_only_selection = jet_ht > ht_only_wp
    ht_only_efficiency = np.round(np.sum(ht_only_selection) / n_events, 2)

    #PLot the efficiencies
    #Basically we want to bin the selected truth ht and divide it by the overall count
    all_events = Hist(ht_axis)
    baseline_selected_events = Hist(ht_axis)
    model_selected_events = Hist(ht_axis)
    model_selected_events_14 = Hist(ht_axis)
    ht_only_selected_events = Hist(ht_axis)

    all_events.fill(jet_genht)
    baseline_selected_events.fill(jet_genht[baseline_selection])
    model_selected_events.fill(jet_genht[model_selection])
    model_selected_events_14.fill(jet_genht[model_selection_14])
    ht_only_selected_events.fill(jet_genht[ht_only_selection])

    #Plot the ratio
    eff_baseline = plot_ratio(all_events, baseline_selected_events)
    eff_model = plot_ratio(all_events, model_selected_events)
    eff_model_14 = plot_ratio(all_events, model_selected_events_14)

    #Get data from handles
    baseline_x, baseline_y, baseline_err = get_bar_patch_data(eff_baseline)
    model_x, model_y, model_err = get_bar_patch_data(eff_model)
    model_x_14, model_y_14, model_err_14 = get_bar_patch_data(eff_model_14)
    eff_ht_only = plot_ratio(all_events, ht_only_selected_events)
    ht_only_x, ht_only_y, ht_only_err = get_bar_patch_data(eff_ht_only)

    # Plot ht distribution in the background
    counts, bin_edges = np.histogram(np.clip(jet_genht, 0, 800), bins=np.arange(0,800,40))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]
    normalized_counts = counts / np.sum(counts)

    #Now plot all
    tau_str = r"$\tau_{1,2}$"
    eff_str = r"$\int \epsilon$"
    fig,ax = plt.subplots(1,1,figsize=style.FIGURE_SIZE)
    hep.cms.label(llabel=style.CMSHEADER_LEFT,rlabel=style.CMSHEADER_RIGHT,ax=ax,fontsize=style.MEDIUM_SIZE-2)
    hep.histplot((normalized_counts, bin_edges), ax=ax, histtype='step', color='grey', label=r"$HT^{gen}$")
    ax.errorbar(baseline_x, baseline_y, yerr=baseline_err, c=style.color_cycle[0], fmt='o', linewidth=3, label=r'bb$\tau \tau$ seed@ {} kHz, {}={} (L1 $HT$ > {} GeV, {} > {} GeV)'.format(rate, eff_str, baseline_efficiency, 220, tau_str, 34))
    ax.errorbar(model_x, model_y, yerr=model_err, c=style.color_cycle[1], fmt='o', linewidth=3, label=r'Multiclass @ {} kHz, {}={} (L1 $HT$ > {} GeV, $\sum$ $\tau\tau$ > {}, $\sum$ bb > {})'.format(rate, eff_str, model_efficiency, ht_wp, round(ttag_wp,2), round(btag_wp,2)))

    #Plot other labels
    ax.hlines(1, 0, 800, linestyles='dashed', color='black', linewidth=4)
    ax.grid(True)
    ax.set_ylim([0., 1.15])
    ax.set_xlim([0, 800])
    ax.set_xlabel(r"$HT^{gen}$ [GeV]")
    ax.set_ylabel(r"$\epsilon$(HH $\to$ bb$\tau \tau$ trigger rate at {} kHz)".format(rate))
    plt.legend(loc='upper left')

    #Save plot
    plot_path = os.path.join(model_dir, f"plots/physics/bbtt/HHbbtt_eff_bbtt_seed_{score_type}_{apply_sel}")
    plt.savefig(f'{plot_path}.pdf', bbox_inches='tight')
    plt.savefig(f'{plot_path}.png', bbox_inches='tight')
    plt.clf()

    # Plot for the HT-only working point
    fig2,ax2 = plt.subplots(1,1,figsize=style.FIGURE_SIZE)
    hep.cms.label(llabel=style.CMSHEADER_LEFT,rlabel=style.CMSHEADER_RIGHT,ax=ax2,fontsize=style.MEDIUM_SIZE-2)
    hep.histplot((normalized_counts, bin_edges), ax=ax2, histtype='step', color='grey', label=r"$HT^{gen}$")
    ax2.errorbar(model_x_14, model_y_14, yerr=model_err, c=style.color_cycle[1], fmt='o', linewidth=3, label=r'Multiclass @ {} kHz, {}={} (L1 $HT$ > {} GeV, $\sum$ $\tau\tau$ > {}, $\sum$ bb > {})'.format(14, eff_str, model_14_efficiency, ht_wp, round(ttag_wp,2), round(btag_wp,2)))
    ax2.errorbar(ht_only_x, ht_only_y, yerr=ht_only_err, c=style.color_cycle[2], fmt='o', linewidth=3, label=r'HT-only + QuadJets @ {} kHz, {}={} (L1 $HT$ > {} GeV)'.format(14, eff_str, ht_only_efficiency, ht_only_wp))

    #Plot other labels
    ax2.hlines(1, 0, 800, linestyles='dashed', color='black', linewidth=4)
    ax2.grid(True)
    ax2.set_ylim([0., 1.15])
    ax2.set_xlim([0, 800])
    ax2.set_xlabel(r"$HT^{gen}$ [GeV]")
    ax2.set_ylabel(r"$\epsilon$(HH $\to$ bb$\tau \tau$ trigger rate at 14 kHz")
    plt.legend(loc='upper left')

    #Save plot
    plot_path = os.path.join(model_dir, f"plots/physics/bbtt/HHbbtt_eff_HT_only_{score_type}_{apply_sel}")
    plt.savefig(f'{plot_path}.pdf', bbox_inches='tight')
    plt.savefig(f'{plot_path}.png', bbox_inches='tight')


if __name__ == "__main__":
    """
    2 steps:

    1. Derive working points: python bbtt.py --deriveWPs
    2. Run efficiency based on the derived working points: python bbtt.py --eff
    """

    parser = ArgumentParser()
    parser.add_argument('-m','--model_dir', default='output/baseline', help = 'Input model')
    parser.add_argument('-s', '--signal', default='/eos/cms/store/cmst3/group/l1tr/sewuchte/l1teg/fp_jettuples_090125_addGenH/GluGluHHTo2B2Tau_PU200.root' , help = 'Signal sample for HH->bbtt')
    parser.add_argument('--minbias', default='/eos/cms/store/cmst3/group/l1tr/sewuchte/l1teg/fp_jettuples_090125/MinBias_PU200.root' , help = 'Minbias sample for deriving rates')

    #Different modes
    parser.add_argument('--deriveRate', action='store_true', help='derive the rate for the bbtt seed')
    parser.add_argument('--deriveWPs', action='store_true', help='derive the working points for b-tagging')
    parser.add_argument('--eff', action='store_true', help='plot efficiency for HH->bbtt')

    parser.add_argument('--tree', default='outnano/Jets', help='Tree within the ntuple containing the jets')

    #Other controls
    parser.add_argument('-n','--n_entries', type=int, default=1000, help = 'Number of data entries in root file to run over, can speed up run time, set to None to run on all data entries')
    args = parser.parse_args()

    if args.deriveRate:
        derive_rate(args.minbias, args.model_dir, n_entries=args.n_entries,tree=args.tree)
    if args.deriveWPs:
        derive_bbtt_WPs(args.model_dir, args.minbias, 'tau', args.signal, n_entries=args.n_entries,tree=args.tree)
        gc.collect()
        derive_bbtt_WPs(args.model_dir, args.minbias, 'all', args.signal, n_entries=args.n_entries,tree=args.tree)
    elif args.eff:
        bbtt_eff_HT(args.model_dir, args.signal, 'raw', 'tau', n_entries=args.n_entries,tree=args.tree)
        gc.collect()
        bbtt_eff_HT(args.model_dir, args.signal, 'qg', 'tau', n_entries=args.n_entries,tree=args.tree)
        gc.collect()
        bbtt_eff_HT(args.model_dir, args.signal, 'raw', 'all', n_entries=args.n_entries,tree=args.tree)
        gc.collect()
        bbtt_eff_HT(args.model_dir, args.signal, 'qg', 'all', n_entries=args.n_entries,tree=args.tree)

