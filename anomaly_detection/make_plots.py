import os

# Third parties
import numpy as np
from argparse import ArgumentParser
from sklearn.metrics import roc_auc_score

# Import from other modules
# from tagger.data.tools import load_data, to_ML, select_events
# from tagger.model.common import fromFolder
# from tagger.plot.basic import basic, plot_event_ROC, plot_PCA, plot_latent,plot_nontrained_event_ROC,plot_latent_vs_variable,plot_output_scores
# import tagger.plot.style as style 

import matplotlib.pyplot as plt
# import mplhep as hep

from config import PROCESS_INFO, JET_INFO, JET_FEATURE_FIELDS, load_process_info
from plotting.anomaly_detection import *
from plotting.pt_regression import *
from plotting.embedding import *
from plotting.triggers import *
from plotting.utils import generate_balanced_sample_indices

MINBIAS_RATE = 40e3 # 40 MHz in kHz
RATE_TARGET = 10.0 # kHz
AD_METHODS = ["auto", "mahl"]

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-m", "--model_folder", type=str, default="./output/deepset_contrastive_vicreg", help="Path to the model folder")
    parser.add_argument("-ad", "--ad_method", type=str, default="all", choices=AD_METHODS + ["all"] + ["all_pt"], help="Anomaly detection method: 'auto' for autoencoder scores, 'mahl' for Mahalanobis scores")
    parser.add_argument("-o", "--plot_dir", type=str, default="plots", help="Output directory for plots")
    args = parser.parse_args()

    # Load process and jet information
    process_info = load_process_info()
    jet_info = JET_INFO
    ad_method = args.ad_method
    metrics = {}
    if ad_method == "all":
        ad_methods = AD_METHODS
    elif ad_method == "all_pt":
        ad_methods = AD_METHODS + ["pt"]
    else:
        ad_methods = [ad_method]

    model_folder = args.model_folder
    test_results = np.load(os.path.join(model_folder, 'testing_data', 'anomaly_detection_test_results.npz'))
    print("Opened test results from:", os.path.join(model_folder, 'testing_data', 'anomaly_detection_test_results.npz'))

    # Load test results including all jets
    all_reco_scores = test_results['reco_score']
    all_mahalanobis_scores = test_results['mahalanobis_score']
    all_event_id = test_results['event_id']
    all_event_class = test_results['event_class']
    all_jet_class = test_results['jet_class']
    all_jet_class_pred = test_results['jet_class_pred']
    all_jet_btag = all_jet_class_pred[:, JET_INFO['b']['class']]
    all_jet_features = test_results['jet_features']
    all_jet_embedding = test_results['jet_embedding']
    all_jet_ncand = all_jet_features[:, JET_FEATURE_FIELDS.index('jet_npuppicand')]
    all_jet_pt = all_jet_features[:, JET_FEATURE_FIELDS.index('jet_pt_phys')]
    all_jet_eta = all_jet_features[:, JET_FEATURE_FIELDS.index('jet_eta_phys')]
    all_jet_reject = all_jet_features[:, JET_FEATURE_FIELDS.index('jet_reject')]
    all_genjet_pt = all_jet_features[:, JET_FEATURE_FIELDS.index('target_pt_phys')]
    all_jet_pt_pred = test_results['jet_pt_pred']
    all_jet_pt_pred = all_jet_pt_pred * all_jet_pt
    all_jet_class_unc = np.max(all_jet_class_pred, axis=1)
    print("Test results loaded from:", os.path.join(model_folder, 'testing_data', 'anomaly_detection_test_results.npz'))
    del test_results

    # Filter jets based on selection criteria
    valid_jet_mask = (
        (all_jet_pt > 15.0) & (np.abs(all_jet_eta) < 2.4) & (all_jet_reject == 0) #& (all_genjet_pt > 5.0)
    )
    all_reco_scores = np.where(valid_jet_mask, all_reco_scores, 0.0)
    all_mahalanobis_scores = np.where(valid_jet_mask, all_mahalanobis_scores, 0.0)

    reco_scores = all_reco_scores[valid_jet_mask]
    mahalanobis_scores = all_mahalanobis_scores[valid_jet_mask]
    event_id = all_event_id[valid_jet_mask]
    event_class = all_event_class[valid_jet_mask]
    jet_class = all_jet_class[valid_jet_mask]
    jet_class_pred = all_jet_class_pred[valid_jet_mask]
    jet_pt_pred = all_jet_pt_pred[valid_jet_mask]
    jet_ncand = all_jet_ncand[valid_jet_mask]
    jet_pt = all_jet_pt[valid_jet_mask]
    jet_eta = all_jet_eta[valid_jet_mask]
    genjet_pt = all_genjet_pt[valid_jet_mask]
    jet_btag = all_jet_btag[valid_jet_mask]
    jet_class_unc = all_jet_class_unc[valid_jet_mask]
    jet_embedding = all_jet_embedding[valid_jet_mask]

    # Print the number of valid jets for each process
    for proc, info in process_info.items():
        all_proc_mask = (all_event_class == info['class'])
        proc_mask = (event_class == info['class'])
        pct_valid = np.divide(np.sum(proc_mask), np.sum(all_proc_mask)) * 100 if np.sum(all_proc_mask) > 0 else 0.0
        print(f"Process {proc:<20} ({info['class']:>2}) contains {np.sum(proc_mask):>7} valid jets out of {np.sum(all_proc_mask):>7} total jets ({pct_valid:.2f}%)")

    minbias_mask = (event_class == process_info['MinBias']['class'])
    sm_mask = np.isin(event_class, [info['class'] for proc, info in process_info.items() if info['type'] == 'SM'])
    smrare_mask = np.isin(event_class, [info['class'] for proc, info in process_info.items() if info['type'] == 'SM' and proc not in ['MinBias', 'QCD', 'DY', 'Wjets', 'TT']])
    bsm_mask = np.isin(event_class, [info['class'] for proc, info in process_info.items() if info['type'] == 'BSM'])

    # Set default plotting parameters
    figsize = (5.5, 5)
    hist_bins = 50

    #####################################
    # ANOMALY DETECTION PLOTS
    #####################################

    AD_PROCESSES = ["MinBias", "QCD", "HH_4b", "HH_2b2tau", "SMJ_cascadeC", "SVJ_250", "SVJ_500"]
    process_info_ad = {proc: info for proc, info in process_info.items() if proc in AD_PROCESSES}

    for ad_method in ad_methods:

        outdir_ad = os.path.join(model_folder, args.plot_dir, f"anomaly_detection_{ad_method}")
        os.makedirs(outdir_ad, exist_ok=True)

        print(f"Generating plots for anomaly detection method: {ad_method}")
        if ad_method == "auto":
            all_anomaly_scores = all_reco_scores
            anomaly_scores = reco_scores
            hist_range = (0, 100)
            eps = 1e-4
            log_scale = True
        elif ad_method == "mahl":
            all_anomaly_scores = all_mahalanobis_scores
            anomaly_scores = mahalanobis_scores
            hist_range = (0, 500)
            eps = 3e-1
            log_scale = True
        elif ad_method == "pt":
            all_anomaly_scores = all_jet_pt
            anomaly_scores = jet_pt
            hist_range = (0, 3000)
            eps = 15
            log_scale = True
        else:
            raise ValueError(f"Unknown anomaly detection method: {ad_method}")

        if log_scale:
            hist_edges = np.logspace(np.log10(hist_range[0] + eps), np.log10(hist_range[1]), hist_bins + 1)

        plot_anomaly_score_per_class_hist(
            class_info = jet_info,
            class_array = jet_class[sm_mask],
            test_scores = anomaly_scores[sm_mask],
            hist_edges = hist_edges,
            output_file = os.path.join(outdir_ad, f"anomaly_score_per_jet_class_hist.png"),
            log_scale = log_scale
        )

        plot_anomaly_score_per_class_hist(
            class_info = process_info_ad,
            class_array = event_class,
            test_scores = anomaly_scores,
            hist_edges = hist_edges,
            output_file = os.path.join(outdir_ad, f"anomaly_score_per_event_class_hist.png"),
            log_scale = log_scale
        )

        plot_anomaly_score_vs_variable(
            class_info = process_info_ad,
            class_array = event_class,
            test_scores = anomaly_scores,
            variable = jet_pt,
            var_edges = np.logspace(np.log10(15), np.log10(3000), 20),
            var_name = "jet_pt",
            output_file = os.path.join(outdir_ad, f"anomaly_score_vs_jet_pt.png"),
            log_scale = log_scale,
            log_variable = True
        )

        plot_anomaly_score_vs_variable(
            class_info = process_info_ad,
            class_array = event_class,
            test_scores = anomaly_scores,
            variable = jet_eta,
            var_edges = np.linspace(-2.5, 2.5, 50),
            var_name = "jet_eta",
            output_file = os.path.join(outdir_ad, f"anomaly_score_vs_jet_eta.png"),
            log_scale = log_scale,
            log_variable = False
        )

        plot_anomaly_score_vs_variable(
            class_info = process_info_ad,
            class_array = event_class,
            test_scores = anomaly_scores,
            variable = jet_btag,
            var_edges = np.linspace(0, 1, 40),
            var_name = "jet_btag",
            output_file = os.path.join(outdir_ad, f"anomaly_score_vs_jet_btag.png"),
            log_scale = log_scale,
            log_variable = False
        )

        plot_anomaly_score_vs_variable(
            class_info = process_info_ad,
            class_array = event_class,
            test_scores = anomaly_scores,
            variable = jet_class_unc,
            var_edges = np.linspace(0, 1, 40),
            var_name = "Jet Class Uncertainty",
            output_file = os.path.join(outdir_ad, f"anomaly_score_vs_jet_class_unc.png"),
            log_scale = log_scale,
            log_variable = False
        )


        # Do trigger efficiency plots for each process
        unique_events, inverse = np.unique(all_event_id, return_inverse=True)
        event_level_event_class = np.zeros(len(unique_events), dtype=int)
        np.maximum.at(event_level_event_class, inverse, all_event_class)
        _, event_level_scores_max = get_event_scores(all_event_id, all_anomaly_scores, method="max")
        _, event_level_scores_ptmax = get_var1_of_max_var2_jet_per_event(all_event_id, all_anomaly_scores, all_jet_pt)
        _, event_level_scores_sum = get_event_scores(all_event_id, all_anomaly_scores, method="sum")
        _, event_level_pt_scoremax = get_var1_of_max_var2_jet_per_event(all_event_id, all_jet_pt, all_anomaly_scores)
        _, event_level_ptmax_phys = get_max_var_jet(all_event_id, all_jet_pt)
        _, event_level_ht_phys = get_event_ht(all_event_id, all_jet_pt)
        _, event_level_genmatch_pt_scoremax = get_var1_of_max_var2_jet_per_event(all_event_id, all_genjet_pt, all_anomaly_scores)
        _, event_level_genmatch_ptmax = get_var1_of_max_var2_jet_per_event(all_event_id, all_genjet_pt, all_jet_pt)
        _, event_level_genmatch_ht = get_event_ht(all_event_id, all_genjet_pt, pt_threshold=0.0)

        minbias_class = process_info['MinBias']['class']
        minbias_mask = (event_level_event_class == minbias_class)
        minbias_event_scores_max = event_level_scores_max[minbias_mask]
        minbias_event_scores_ptmax = event_level_scores_ptmax[minbias_mask]
        minbias_event_scores_sum = event_level_scores_sum[minbias_mask]
        minbias_event_pt_scoremax = event_level_pt_scoremax[minbias_mask]
        minbias_event_ptmax_phys = event_level_ptmax_phys[minbias_mask]
        minbias_event_ht_phys = event_level_ht_phys[minbias_mask]

        # Get reconstruction error to cut on to obtain a fpr of 1% on the minbias sample
        rate_target = RATE_TARGET
        fpr_target = rate_target / MINBIAS_RATE
        threshold_max = np.percentile(minbias_event_scores_max, 100 * (1 - fpr_target))
        threshold_ptmax = np.percentile(minbias_event_scores_ptmax, 100 * (1 - fpr_target))
        threshold_sum = np.percentile(minbias_event_scores_sum, 100 * (1 - fpr_target))
        print(f"Reconstruction error threshold for minbias rate of {rate_target:.0f} kHz: {threshold_max:.4f} (method: max)")
        print(f"Reconstruction error threshold for minbias rate of {rate_target:.0f} kHz: {threshold_ptmax:.4f} (method: ptmax)")
        print(f"Reconstruction error threshold for minbias rate of {rate_target:.0f} kHz: {threshold_sum:.4f} (method: sum)")

        save_efficiency_table(
            process_info = process_info_ad,
            event_classes = event_level_event_class,
            event_scores_max = event_level_scores_max,
            event_scores_ptmax = event_level_scores_ptmax,
            event_scores_sum = event_level_scores_sum,
            threshold_max = threshold_max,
            threshold_ptmax = threshold_ptmax,
            threshold_sum = threshold_sum,
            output_file = os.path.join(outdir_ad, f"trigger_efficiency_table_{ad_method}.csv"),
            print_to_console = False,
        )

        save_efficiency_table(
            process_info = process_info_ad,
            event_classes = event_level_event_class,
            event_scores_max = event_level_scores_max,
            event_scores_ptmax = event_level_scores_ptmax,
            event_scores_sum = event_level_scores_sum,
            threshold_max = threshold_max,
            threshold_ptmax = threshold_ptmax,
            threshold_sum = threshold_sum,
            output_file = os.path.join(outdir_ad, f"trigger_efficiency_table_{ad_method}.txt"),
        )

        # Plot roc and turn-on for max aggregation method
        plot_roc(
            process_info = process_info_ad,
            event_classes = event_level_event_class,
            event_scores = event_level_scores_max,
            output_file = os.path.join(outdir_ad, f"event_level_roc_max.png"),
            minbias_rate = MINBIAS_RATE,
            rate_target = rate_target,
        )

        plot_turn_on(
            process_info = process_info_ad,
            event_classes = event_level_event_class,
            event_scores = event_level_scores_max,
            threshold = threshold_max,
            turn_on_var = event_level_pt_scoremax,
            turn_on_var_name = r"Gen $p_T^{\mathrm{max}}$ (GeV)",
            output_file = os.path.join(outdir_ad, f"turnon_pt_{ad_method}_max.png")
        )

        # Plot roc and turn-on for ptmax aggregation method
        plot_roc(
            process_info = process_info_ad,
            event_classes = event_level_event_class,
            event_scores = event_level_scores_ptmax,
            output_file = os.path.join(outdir_ad, f"event_level_roc_ptmax.png"),
            minbias_rate = MINBIAS_RATE,
            rate_target = rate_target
        )

        plot_turn_on(
            process_info = process_info_ad,
            event_classes = event_level_event_class,
            event_scores = event_level_scores_ptmax,
            threshold = threshold_ptmax,
            turn_on_var = event_level_genmatch_ptmax,
            turn_on_var_name = r"Gen $p_T^{\mathrm{max}}$ (GeV)",
            output_file = os.path.join(outdir_ad, f"turnon_pt_{ad_method}_ptmax.png")
        )

        # Plot roc and turn-on for sum aggregation method
        plot_roc(
            process_info = process_info_ad,
            event_classes = event_level_event_class,
            event_scores = event_level_scores_sum,
            output_file = os.path.join(outdir_ad, f"event_level_roc_sum.png"),
            minbias_rate = MINBIAS_RATE,
            rate_target = rate_target
        )

        plot_turn_on(
            process_info = process_info_ad,
            event_classes = event_level_event_class,
            event_scores = event_level_scores_sum,
            threshold = threshold_sum,
            turn_on_var = event_level_genmatch_ht,
            turn_on_var_name = r"Gen $H_T$ (GeV)",
            output_file = os.path.join(outdir_ad, f"turnon_ht_{ad_method}_sum.png"),
            bins=np.array([0, 20, 40, 60, 80, 100, 125, 150, 175, 200, 225, 250, 275, 300, 350, 400, 450, 500, 600, 700, 800, 1000, 1200, 1400, 1600, 1800, 2000])
        )

        minbias_scores_max_accept = (minbias_event_scores_max > threshold_max).astype(int)
        minbias_scores_ptmax_accept = (minbias_event_scores_ptmax > threshold_ptmax).astype(int)
        minbias_scores_sum_accept = (minbias_event_scores_sum > threshold_sum).astype(int)
        
        minbias_singlejet_accept = singlejet_pt_trigger(minbias_event_ptmax_phys)
        minbias_ht_accept = ht_trigger(minbias_event_ht_phys)

        minbias_singlejet_rate = np.mean(minbias_singlejet_accept) * MINBIAS_RATE
        minbias_ht_rate = np.mean(minbias_ht_accept) * MINBIAS_RATE
        print(f"MinBias single jet trigger rate: {minbias_singlejet_rate:.2f} kHz")
        print(f"MinBias HT trigger rate: {minbias_ht_rate:.2f} kHz")

        # Plot turn on of SVJ and HH4b processes using anomaly score, or normal pt/ht triggers for comparison
        svj500_class = process_info['SVJ_500']['class']
        svj500_mask = (event_level_event_class == svj500_class)
        svj500_accept = (event_level_scores_max[svj500_mask] > threshold_max).astype(int)
        svj500_max_turnon, svj500_max_edges = get_trigger_turn_on(
            accept = svj500_accept,
            x_values = event_level_pt_scoremax[svj500_mask]
        )
        svj500_singlejet_accept = singlejet_pt_trigger(event_level_pt_scoremax[svj500_mask])
        svj500_singlejet_turnon, svj500_singlejet_edges = get_trigger_turn_on(
            accept = svj500_singlejet_accept,
            x_values = event_level_pt_scoremax[svj500_mask]
        )
        svj500_trigger_info = {
            "Anomaly Score (Max)": {
                "efficiencies": svj500_max_turnon,
                "x_edges": svj500_max_edges,
                "x_values": event_level_pt_scoremax[svj500_mask],
                "rate": np.mean(minbias_scores_max_accept) * MINBIAS_RATE,
                "color": process_info['SVJ_500']['color'],
                "linestyle": "-",
                "marker": "o"
            },
            "Single Jet Trigger": {
                "efficiencies": svj500_singlejet_turnon,
                "x_edges": svj500_singlejet_edges,
                "x_values": event_level_pt_scoremax[svj500_mask],
                "rate": minbias_singlejet_rate,
                "color": process_info['SVJ_500']['color'],
                "linestyle": "--",
                "marker": "d"
            }
        }
        plot_turn_ons(
            trigger_infos = svj500_trigger_info,
            x_values = event_level_pt_scoremax[svj500_mask],
            x_edges = svj500_max_edges,
            x_variable_name = "Gen $p_T^{\mathrm{max}}$ (GeV)",
            output_file = os.path.join(outdir_ad, f"turnon_comparison_svj500.png")
        )

        # svj250_class = process_info['SVJ_250']['class']
        # svj250_mask = (event_level_event_class == svj250_class)
        # svj250_accept = (event_level_scores_max[svj250_mask] > threshold_max).astype(int)
        # svj250_max = event_level_scores_max[svj250_mask]
        # svj250_singlejet_accept = singlejet_pt_trigger(event_level_pt_scoremax[svj250_mask])
        # svj250_singlejet_turnon = get_trigger_turn_on(
        #     accept = svj250_singlejet_accept,
        #     x_edges = np.array([0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]),
        #     x_values = event_level_pt_scoremax[svj250_mask]
        # )

        hh4b_class = process_info['HH_4b']['class']
        hh4b_mask = (event_level_event_class == hh4b_class)
        hh4b_genht = event_level_genmatch_ht[hh4b_mask]
        hh4b_sum_accept = (event_level_scores_sum[hh4b_mask] > threshold_sum).astype(int)
        hh4b_sum_turnon, edges = get_trigger_turn_on(
            accept = hh4b_sum_accept,
            x_values = hh4b_genht,
        )
        hh4b_sum = event_level_scores_sum[hh4b_mask]
        hh4b_ht_accept = ht_trigger(event_level_ht_phys[hh4b_mask])
        hh4b_ht_turnon, edges = get_trigger_turn_on(
            accept = hh4b_ht_accept,
            x_values = hh4b_genht,
        )
        hh4b_trigger_info = {
            "Anomaly Score (Sum)": {
                "efficiencies": hh4b_sum_turnon,
                "x_edges": edges,
                "x_values": hh4b_genht,
                "rate": np.mean(minbias_scores_sum_accept) * MINBIAS_RATE,
                "color": process_info['HH_4b']['color'],
                "linestyle": "-",
                "marker": "o"
            },
            "HT Trigger": {
                "efficiencies": hh4b_ht_turnon,
                "x_edges": edges,
                "x_values": hh4b_genht,
                "rate": minbias_ht_rate,
                "color": process_info['HH_4b']['color'],
                "linestyle": "--",
                "marker": "d"
            }
        }
        plot_turn_ons(
            trigger_infos = hh4b_trigger_info,
            x_values = hh4b_genht,
            x_edges = edges,
            output_file = os.path.join(outdir_ad, f"turnon_comparison_hh4b.png"),
            x_variable_name = "Gen $H_T$ (GeV)"
        )



        # Plot ROCs vs QCD for each BSM process
        process_info_qcd_roc = {proc: info for proc, info in process_info.items() if proc in ["QCD", "HH_4b", "SVJ_500"]}
        weight_edges = np.logspace(np.log10(15), np.log10(5000), 40)
        plot_roc_vs_qcd(
            process_info = process_info_qcd_roc,
            event_classes = event_class,
            event_scores = anomaly_scores,
            output_file = os.path.join(outdir_ad, f"roc_vs_qcd.png"),
            bkg_class = 'QCD',
            weight_variable = jet_pt,
            weight_bins = weight_edges,
        )

    #####################################
    # PT REGRESSION PLOTS
    #####################################

    reg_outdir = os.path.join(model_folder, args.plot_dir, f"pt_regression")
    os.makedirs(reg_outdir, exist_ok=True)
    print(f"Generating plots for pt regression in directory: {reg_outdir}")

    PROCESSES_REG = ["QCD", "TT"]
    process_info_reg = {proc: info for proc, info in process_info.items() if proc in PROCESSES_REG}

    # Save response and resolution metrics for each process
    genmatched = genjet_pt > 20
    for proc, info in process_info_reg.items():
        jet_response = jet_pt_pred[genmatched & (event_class == info['class'])] / genjet_pt[genmatched & (event_class == info['class'])]
        response = np.mean(jet_response)
        resolution = (np.percentile(jet_response, 84.1) - np.percentile(jet_response, 15.9)) / 2
        metrics[f'response_{proc}'] = response
        metrics[f'resolution_{proc}'] = resolution

    pt_edges = np.array([15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100, 120, 150, 200, 300, 500, 800, 5000])
    plot_response_resolution_vs_var(
        class_info = process_info_reg,
        jet_pt_pred = jet_pt_pred,
        jet_pt_reco = jet_pt,
        jet_pt_true = genjet_pt,
        class_array = event_class,
        var = genjet_pt,
        var_edges = pt_edges,
        var_name = "Gen Jet $p_T$ (GeV)",
        output_file = os.path.join(reg_outdir, f"response_resolution_vs_genpt.png"),
        logx = True
    )

    eta_edges = np.linspace(-2.4, 2.4, 20)
    plot_response_resolution_vs_var(
        class_info = process_info_reg,
        jet_pt_pred = jet_pt_pred,
        jet_pt_reco = jet_pt,
        jet_pt_true = genjet_pt,
        class_array = event_class,
        var = jet_eta,
        var_edges = eta_edges,
        var_name = r"Jet $\eta$",
        output_file = os.path.join(reg_outdir, f"response_resolution_vs_eta.png")
    )

    #####################################
    # VIZUALIZING EMBEDDING SPACE
    #####################################

    PROCESSES_EMB = ["MinBias", "QCD", "TT", "HH_4b", "SVJ_500"]
    process_info_emb = {proc: info for proc, info in process_info.items() if proc in PROCESSES_EMB}

    emb_outdir = os.path.join(model_folder, args.plot_dir, f"embedding")
    os.makedirs(emb_outdir, exist_ok=True)
    print(f"Generating plots for embedding space in directory: {emb_outdir}")

    # Compute embedding metrics and generate plots
    metrics['mean_embedding_norm'] = mean_embedding_norm(jet_embedding)
    metrics['cov_trace'] = covariance_trace(jet_embedding)
    metrics['effective_rank'] = effective_rank(jet_embedding)

    pca = PCA(n_components=10)
    jet_embedding_pca = pca.fit_transform(jet_embedding)

    #####################################
    # GROUPED BY SAMPLE
    #####################################

    sample_indices = generate_balanced_sample_indices(
        class_info = process_info_emb,
        class_array = event_class,
        njets_per_class = 2500
    )
    red_sm_mask = sm_mask[sample_indices]
    z = jet_embedding[sample_indices]
    z_pca = jet_embedding_pca[sample_indices]
    red_jet_reco = reco_scores[sample_indices]
    red_jet_mahalanobis = mahalanobis_scores[sample_indices]
    red_jet_pt = jet_pt[sample_indices]
    red_jet_class = jet_class[sample_indices]
    red_event_class = event_class[sample_indices]
    red_jet_btag = jet_btag[sample_indices]
    z_tsne = fit_tsne_embedding(z)  # fit once

    metrics['mean_cosine_sim'] = mean_cosine_similarity(z)
    metrics['uniformity'] = uniformity(z)

    # Plot 1D embedding metrics
    plot_eigenvalue_spectrum(z, output_file=os.path.join(emb_outdir, "eigenvalue_spectrum.png"))
    plot_pairwise_distance_vs_delta_pt(z, red_jet_pt, output_file=os.path.join(emb_outdir, "pairwise_distance_vs_delta_pt.png"))
    plot_pca_explained_variance(z_pca, output_file=os.path.join(emb_outdir, "pca_explained_variance.png"))
    plot_pairwise_distance_hist(z, red_event_class, output_file=os.path.join(emb_outdir, "pairwise_distance_distribution_per_process.png"))

    # Plot embedding space vizualizations
    plot_pca_embedding_2d(
        z_pca, 
        output_file=os.path.join(emb_outdir, "pca_embedding_2d_colorpt.png"), 
        variable_name="Jet log($p_T$) (GeV)",
        variable_array=red_jet_pt, 
        percentile=99,
        log_variable=True
    )
    plot_pca_embedding_2d_matrix(
        z_pca, 
        output_file=os.path.join(emb_outdir, "pca_embedding_2d_matrix_event_pt.png"), 
        n_components=3, 
        class_info=process_info_emb,
        class_array=red_event_class,
        variable_name=r"Jet log($p_T$) (GeV)",
        variable_array=red_jet_pt, 
        percentile=99, 
        log_variable=True
    )
    plot_pca_embedding_2d(
        z_pca, 
        output_file=os.path.join(emb_outdir, "pca_embedding_2d_colorreco.png"), 
        variable_name="Jet Reco score",
        variable_array=red_jet_reco,
        percentile=99,
        log_variable=True
    )
    plot_pca_embedding_2d(
        z_pca, 
        output_file=os.path.join(emb_outdir, "pca_embedding_2d_colormahalanobis.png"), 
        variable_name="Jet Mahalanobis score",
        variable_array=red_jet_mahalanobis,
        percentile=99,
        log_variable=True
    )
    plot_pca_embedding_2d(
        z_pca, 
        output_file=os.path.join(emb_outdir, "pca_embedding_2d_per_process.png"), 
        class_info=process_info_emb,
        class_array=red_event_class,
        percentile=99
    )
    plot_tsne_embedding_2d(
        z_tsne, 
        output_file=os.path.join(emb_outdir, "tsne_by_process.png"), 
        class_info=process_info_emb, 
        class_array=red_event_class,
    )
    plot_tsne_embedding_2d(
        z_tsne, 
        output_file=os.path.join(emb_outdir, "tsne_by_flav_colorpt.png"), 
        variable_name="Jet log($p_T$) (GeV)",
        variable_array=red_jet_pt,
        percentile=99,
        log_variable=True
    )
    plot_tsne_embedding_2d(
        z_tsne, 
        output_file=os.path.join(emb_outdir, "tsne_by_process_colorreco.png"), 
        variable_name="Jet Reco score",
        variable_array=red_jet_reco,
        percentile=99,
        log_variable=True
    )
    plot_tsne_embedding_2d(
        z_tsne, 
        output_file=os.path.join(emb_outdir, "tsne_by_process_colormahalanobis.png"), 
        variable_name="Jet Mahalanobis score",
        variable_array=red_jet_mahalanobis,
        percentile=99,
        log_variable=True
    )

    # plot_class_distance_matrix(process_info, z, red_event_class, output_file=os.path.join(emb_outdir, "class_distance_matrix_process.png"))
    
    # Now compute metrics and plots grouped by process
    # metrics['silhouette_proc'] = silhouette(z_bkg, red_event_class_bkg)
    # metrics['linear_probe_acc_proc'] = linear_probe_accuracy(z_bkg, red_event_class_bkg, cv=5)
    # metrics['knn_acc_proc'] = knn_accuracy(z_bkg, red_event_class_bkg, k=20, cv=5)
    # metrics['neighbour_pur_proc'] = neighbour_purity(z_bkg, red_event_class_bkg, k=20)


    
    #####################################
    # GROUPED BY JET FLAVOR
    #####################################
    # use only jets that are matched to a gen-level jet and from SM processes
    matched_mask = (jet_class >= 0) & sm_mask
    matched_jet_embedding = jet_embedding[matched_mask]
    matched_jet_embedding_pca = jet_embedding_pca[matched_mask]
    matched_jet_class = jet_class[matched_mask]
    matched_event_class = event_class[matched_mask]
    matched_jet_pt = jet_pt[matched_mask]
    matched_jet_reco = reco_scores[matched_mask]
    matched_jet_mahalanobis = mahalanobis_scores[matched_mask]
    matched_jet_btag = jet_btag[matched_mask]

    sample_indices = generate_balanced_sample_indices(
        class_info = jet_info,
        class_array = matched_jet_class,
        njets_per_class = 2500
    )
    z = matched_jet_embedding[sample_indices]
    z_pca = matched_jet_embedding_pca[sample_indices]
    red_jet_pt = matched_jet_pt[sample_indices]
    red_jet_class = matched_jet_class[sample_indices]
    red_event_class = matched_event_class[sample_indices]
    red_jet_btag = matched_jet_btag[sample_indices]
    red_jet_reco = matched_jet_reco[sample_indices]
    red_jet_mahalanobis = matched_jet_mahalanobis[sample_indices]
    z_tsne = fit_tsne_embedding(z)  # fit once

    plot_pairwise_distance_hist(z, red_jet_class, output_file=os.path.join(emb_outdir, "pairwise_distance_distribution_per_flav.png"))
    plot_pca_embedding_2d(  
        z_pca, 
        output_file=os.path.join(emb_outdir, "pca_embedding_2d_colorbtag.png"), 
        variable_name="Jet b-tag score",
        variable_array=red_jet_btag,
        percentile=99,
    )
    plot_pca_embedding_2d_matrix(
        z_pca, 
        output_file=os.path.join(emb_outdir, "pca_embedding_2d_matrix_jet_btag.png"), 
        n_components=3, 
        class_info=jet_info,
        class_array=red_jet_class, 
        variable_name="Jet b-tag score",
        variable_array=red_jet_btag, 
        percentile=99, 
    )
    plot_pca_embedding_2d(
        z_pca, 
        output_file=os.path.join(emb_outdir, "pca_embedding_2d_per_flav.png"), 
        class_info=jet_info,
        class_array=red_jet_class,
        percentile=99
    )
    plot_tsne_embedding_2d(
        z_tsne, 
        output_file=os.path.join(emb_outdir, "tsne_by_flav.png"), 
        class_info=jet_info,
        class_array=red_jet_class
    )
    plot_tsne_embedding_2d(
        z_tsne, 
        output_file=os.path.join(emb_outdir, "tsne_by_flav_colorbtag.png"), 
        variable_name="Jet b-tag score",
        variable_array=red_jet_btag,
        percentile=99,
    )
    plot_tsne_embedding_2d(
        z_tsne, 
        output_file=os.path.join(emb_outdir, "tsne_by_flav_colorreco.png"), 
        variable_name="Jet Reco score",
        variable_array=red_jet_reco,
        percentile=99,
        log_variable=True
    )
    plot_tsne_embedding_2d(
        z_tsne, 
        output_file=os.path.join(emb_outdir, "tsne_by_flav_colormahalanobis.png"), 
        variable_name="Jet Mahalanobis score",
        variable_array=red_jet_mahalanobis,
        percentile=99,
        log_variable=True
    )
    # plot_class_distance_matrix(jet_info, z, red_jet_class, output_file=os.path.join(emb_outdir, "class_distance_matrix_flav.png"))

    # metrics['silhouette_flav'] = silhouette(z_bkg, red_jet_class_bkg)
    # metrics['linear_probe_acc_flav'] = linear_probe_accuracy(z_bkg, red_jet_class_bkg, cv=5)
    # metrics['knn_acc_flav'] = knn_accuracy(z_bkg, red_jet_class_bkg, k=20, cv=5)
    # metrics['neighbour_pur_flav'] = neighbour_purity(z_bkg, red_jet_class_bkg, k=20)

    for metric_name, metric_value in metrics.items():
        print(f"{metric_name:<40}: {metric_value:.4f}")

    # Save metrics to a csv file for later analysis
    metric_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
    metric_df.to_csv(os.path.join(emb_outdir, "embedding_metrics.csv"), index=False)

    # And a text file for easy reading
    with open(os.path.join(emb_outdir, "embedding_metrics.txt"), 'w') as f:
        for metric_name, metric_value in metrics.items():
            f.write(f"{metric_name:<40}: {metric_value:.4f}\n")
    








