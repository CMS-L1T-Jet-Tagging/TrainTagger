import os

# Third parties
import numpy as np
from argparse import ArgumentParser
from sklearn.metrics import roc_auc_score
import json
# import mplhep as hep

from tagger.plot import style
from tagger.plot.common import MINBIAS_RATE, PT_BINS
from tagger.data.tools import get_valid_jets

from tagger.anomaly_detection.config import PROCESS_INFO, JET_INFO, load_process_info
from tagger.plot.anomaly_detection import *
# from tagger.anomaly_detection.plotting.pt_regression import *
# from tagger.anomaly_detection.plotting.embedding import *
# from tagger.anomaly_detection.plotting.triggers import *
from tagger.anomaly_detection.plotting.utils import generate_balanced_sample_indices

import matplotlib
import matplotlib.pyplot as plt

import mplhep as hep

# Third parties
import pandas as pd
import numpy as np
from sklearn.metrics import auc, roc_curve

from tagger.plot import style

from tagger.plot.common import PT_BINS, plot_histo

matplotlib.use('Agg')

plt.rcParams.update({'figure.max_open_warning': 0})
style.set_style()

import matplotlib.style as mplstyle
mplstyle.use('fast')

RATE_TARGET = 10.0 # kHz
AD_METHODS = ["auto", "mahl"]


def make_plots(model_folder: str, all_data: dict, ad_method='all', plot_dir='plots'):
    all_data = all_data.copy()  # Create a copy to avoid modifying the original data

    # Load the dataset IDs from the training data folder
    with open(os.path.join(model_folder, 'sample_labels.json'), 'r') as f:
        dataset_ids = json.load(f)

    # Load extra variables for plotting
    with open(os.path.join(model_folder, 'extra_vars.json'), 'r') as f:
        extra_vars = json.load(f)

    # Load process and jet information
    process_info = load_process_info(dataset_ids=dataset_ids)
    class_mapping = {info['id']: info['class'] for proc, info in process_info.items()}
    jet_info = JET_INFO
    
    if ad_method == "all":
        ad_methods = AD_METHODS
    elif ad_method == "all_pt":
        ad_methods = AD_METHODS + ["pt"]
    else:
        ad_methods = [ad_method]

    metrics = {}

    for k in ['jet_pt_phys', 'jet_eta_phys', 'jet_npuppicand', 'target_pt_phys', 'jet_reject']:
        all_data[k] = all_data['jet_features'][:, extra_vars.index(k)]

    all_data['event_class'] = np.vectorize(class_mapping.get)(all_data['event_class'])
    del all_data['jet_features']

    all_data['minbias_mask'] = (all_data['event_class'] == process_info['MinBias']['class'])
    all_data['sm_mask'] = np.isin(all_data['event_class'], [info['class'] for proc, info in process_info.items() if info['type'] == 'SM'])
    all_data['smrare_mask'] = np.isin(all_data['event_class'], [info['class'] for proc, info in process_info.items() if info['type'] == 'SM' and proc not in ['MinBias', 'QCD', 'DY', 'Wjets', 'TT']])
    all_data['bsm_mask'] = np.isin(all_data['event_class'], [info['class'] for proc, info in process_info.items() if info['type'] == 'BSM'])

    valid_mask = get_valid_jets(all_data['jet_pt_phys'], all_data['jet_eta_phys'], all_data['jet_reject'])
    data = {k: v[valid_mask] for k, v in all_data.items()}

    # Set default plotting parameters
    figsize = style.FIGURE_SIZE
    hist_bins = 50

    #####################################
    # ANOMALY DETECTION PLOTS
    #####################################

    AD_PROCESSES = ["MinBias", 'QCD', "HH_4b", "HH_2b2tau", "SUEP", "SVJ_250", "SVJ_500"]
    process_info_ad = {proc: info for proc, info in process_info.items() if proc in AD_PROCESSES}

    for ad_method in ad_methods:

        outdir_ad = os.path.join(model_folder, plot_dir, f"anomaly_detection_{ad_method}")
        os.makedirs(outdir_ad, exist_ok=True)

        print(f"Generating plots for anomaly detection method: {ad_method}")
        if ad_method == "auto":
            all_anomaly_scores = all_data['reco_score']
            anomaly_scores = data['reco_score']
            hist_range = (0, 10)
            eps = 1e-2
            log_scale = True
        elif ad_method == "mahl":
            all_anomaly_scores = all_data['mahalanobis_score']
            anomaly_scores = data['mahalanobis_score']
            hist_range = (0, 100)
            eps = 3e-1
            log_scale = True
        elif ad_method == "pt":
            all_anomaly_scores = all_data['jet_pt_phys']
            anomaly_scores = data['jet_pt_phys']
            hist_range = (0, 3000)
            eps = 15
            log_scale = True
        else:
            raise ValueError(f"Unknown anomaly detection method: {ad_method}")

        if log_scale:
            hist_edges = np.logspace(np.log10(hist_range[0] + eps), np.log10(hist_range[1]), hist_bins + 1)

        # Invalid jets get anomaly score of 0, so we can safely mask them out for plotting
        all_anomaly_scores[~valid_mask] = 0

        plot_anomaly_score_per_class_hist(
            class_info = jet_info,
            class_array = data['jet_class'][data['sm_mask']],
            test_scores = anomaly_scores[data['sm_mask']],
            hist_edges = hist_edges,
            output_file = os.path.join(outdir_ad, f"anomaly_score_per_jet_class_hist.png"),
            log_scale = log_scale
        )

        plot_anomaly_score_per_class_hist(
            class_info = process_info_ad,
            class_array = data['event_class'],
            test_scores = anomaly_scores,
            hist_edges = hist_edges,
            output_file = os.path.join(outdir_ad, f"anomaly_score_per_event_class_hist.png"),
            log_scale = log_scale
        )

        plot_anomaly_score_vs_variable(
            class_info = process_info_ad,
            class_array = data['event_class'],
            test_scores = anomaly_scores,
            variable = data['jet_pt_phys'],
            var_edges = np.logspace(np.log10(15), np.log10(3000), 20),
            var_name = "jet_pt",
            output_file = os.path.join(outdir_ad, f"anomaly_score_vs_jet_pt.png"),
            log_scale = log_scale,
            log_variable = True
        )

        plot_anomaly_score_vs_variable(
            class_info = process_info_ad,
            class_array = data['event_class'],
            test_scores = anomaly_scores,
            variable = data['jet_eta_phys'],
            var_edges = np.linspace(-2.5, 2.5, 50),
            var_name = "jet_eta",
            output_file = os.path.join(outdir_ad, f"anomaly_score_vs_jet_eta.png"),
            log_scale = log_scale,
            log_variable = False
        )

        # plot_anomaly_score_vs_variable(
        #     class_info = process_info_ad,
        #     class_array = event_class,
        #     test_scores = anomaly_scores,
        #     variable = jet_btag,
        #     var_edges = np.linspace(0, 1, 40),
        #     var_name = "jet_btag",
        #     output_file = os.path.join(outdir_ad, f"anomaly_score_vs_jet_btag.png"),
        #     log_scale = log_scale,
        #     log_variable = False
        # )

        # plot_anomaly_score_vs_variable(
        #     class_info = process_info_ad,
        #     class_array = event_class,
        #     test_scores = anomaly_scores,
        #     variable = jet_class_unc,
        #     var_edges = np.linspace(0, 1, 40),
        #     var_name = "Jet Class Uncertainty",
        #     output_file = os.path.join(outdir_ad, f"anomaly_score_vs_jet_class_unc.png"),
        #     log_scale = log_scale,
        #     log_variable = False
        # )

        unique_events, inverse = np.unique(all_data['event_id'], return_inverse=True)
        event_level_event_class = np.zeros(len(unique_events), dtype=int)
        np.maximum.at(event_level_event_class, inverse, all_data['event_class'])
        _, event_level_scores_max = get_event_scores(all_data['event_id'], all_anomaly_scores, method="max")
        _, event_level_scores_ptmax = get_var1_of_max_var2_jet_per_event(all_data['event_id'], all_anomaly_scores, all_data['jet_pt_phys'])
        _, event_level_scores_sum = get_event_scores(all_data['event_id'], all_anomaly_scores, method="sum")
        _, event_level_pt_scoremax = get_var1_of_max_var2_jet_per_event(all_data['event_id'], all_data['jet_pt_phys'], all_anomaly_scores)
        _, event_level_ptmax_phys = get_max_var_jet(all_data['event_id'], all_data['jet_pt_phys'])
        _, event_level_ht_phys = get_event_ht(all_data['event_id'], all_data['jet_pt_phys'], pt_threshold=0.0)
        _, event_level_genmatch_pt_scoremax = get_var1_of_max_var2_jet_per_event(all_data['event_id'], all_data['target_pt_phys'], all_anomaly_scores)
        _, event_level_genmatch_ptmax = get_var1_of_max_var2_jet_per_event(all_data['event_id'], all_data['target_pt_phys'], all_data['jet_pt_phys'])
        _, event_level_genmatch_ht = get_event_ht(all_data['event_id'], all_data['target_pt_phys'], pt_threshold=0.0)

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

        # minbias_scores_max_accept = (minbias_event_scores_max > threshold_max).astype(int)
        # minbias_scores_ptmax_accept = (minbias_event_scores_ptmax > threshold_ptmax).astype(int)
        # minbias_scores_sum_accept = (minbias_event_scores_sum > threshold_sum).astype(int)
        
        # minbias_singlejet_accept = singlejet_pt_trigger(minbias_event_ptmax_phys)
        # minbias_ht_accept = ht_trigger(minbias_event_ht_phys)

        # minbias_singlejet_rate = np.mean(minbias_singlejet_accept) * MINBIAS_RATE
        # minbias_ht_rate = np.mean(minbias_ht_accept) * MINBIAS_RATE
        # print(f"MinBias single jet trigger rate: {minbias_singlejet_rate:.2f} kHz")
        # print(f"MinBias HT trigger rate: {minbias_ht_rate:.2f} kHz")


    #####################################
    # PT REGRESSION PLOTS
    #####################################

    # reg_outdir = os.path.join(model_folder, args.plot_dir, f"pt_regression")
    # os.makedirs(reg_outdir, exist_ok=True)
    # print(f"Generating plots for pt regression in directory: {reg_outdir}")

    # PROCESSES_REG = ['QCD', "TT"]
    # process_info_reg = {proc: info for proc, info in process_info.items() if proc in PROCESSES_REG}

    # # Save response and resolution metrics for each process
    # genmatched = data['target_pt_phys'] > 20
    # for proc, info in process_info_reg.items():
    #     mask = (data['event_class'] == info['class']) & genmatched
    #     jet_response = data['pt_pred'][mask] / data['target_pt_phys'][mask]
    #     response = np.mean(jet_response)
    #     resolution = (np.percentile(jet_response, 84.1) - np.percentile(jet_response, 15.9)) / 2
    #     metrics[f'response_{proc}'] = response
    #     metrics[f'resolution_{proc}'] = resolution

    # pt_edges = np.array([15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100, 120, 150, 200, 300, 500, 800, 5000])
    # plot_response_resolution_vs_var(
    #     class_info = process_info_reg,
    #     jet_pt_pred = data['pt_pred'],
    #     jet_pt_reco = data['jet_pt_phys'],
    #     jet_pt_true = data['target_pt_phys'],
    #     class_array = data['event_class'],
    #     var = data['target_pt_phys'],
    #     var_edges = pt_edges,
    #     var_name = "Gen Jet $p_T$ (GeV)",
    #     output_file = os.path.join(reg_outdir, f"response_resolution_vs_genpt.png"),
    #     logx = True
    # )

    # eta_edges = np.linspace(-2.4, 2.4, 20)
    # plot_response_resolution_vs_var(
    #     class_info = process_info_reg,
    #     jet_pt_pred = data['pt_pred'],
    #     jet_pt_reco = data['jet_pt_phys'],
    #     jet_pt_true = data['target_pt_phys'],
    #     class_array = data['event_class'],
    #     var = data['jet_eta_phys'],
    #     var_edges = eta_edges,
    #     var_name = r"Jet $\eta$",
    #     output_file = os.path.join(reg_outdir, f"response_resolution_vs_eta.png")
    # )

    # #####################################
    # # VIZUALIZING EMBEDDING SPACE
    # #####################################

    # PROCESSES_EMB = ["MinBias", 'QCD', "TT", "HH_4b", "SVJ_500"]
    # process_info_emb = {proc: info for proc, info in process_info.items() if proc in PROCESSES_EMB}

    # emb_outdir = os.path.join(model_folder, args.plot_dir, f"embedding")
    # os.makedirs(emb_outdir, exist_ok=True)
    # print(f"Generating plots for embedding space in directory: {emb_outdir}")

    # # Compute embedding metrics and generate plots
    # metrics['mean_embedding_norm'] = mean_embedding_norm(data['embedding'])
    # metrics['cov_trace'] = covariance_trace(data['embedding'])
    # metrics['effective_rank'] = effective_rank(data['embedding'])

    # pca = PCA(n_components=10)
    # jet_embedding_pca = pca.fit_transform(data['embedding'])

    # #####################################
    # # GROUPED BY SAMPLE
    # #####################################

    # sample_indices = generate_balanced_sample_indices(
    #     class_info = process_info_emb,
    #     class_array = data['event_class'],
    #     njets_per_class = 2500
    # )
    # red_sm_mask = data['sm_mask'][sample_indices]
    # z = data['embedding'][sample_indices]
    # z_pca = jet_embedding_pca[sample_indices]
    # red_jet_reco = data['reco_score'][sample_indices]
    # red_jet_mahalanobis = data['mahalanobis_score'][sample_indices]
    # red_jet_pt = data['jet_pt_phys'][sample_indices]
    # red_jet_class = data['jet_class'][sample_indices]
    # red_event_class = data['event_class'][sample_indices]
    # # red_jet_btag = data['jet_btag'][sample_indices]
    # z_tsne = fit_tsne_embedding(z)  # fit once

    # metrics['mean_cosine_sim'] = mean_cosine_similarity(z)
    # metrics['uniformity'] = uniformity(z)

    # # Plot 1D embedding metrics
    # plot_eigenvalue_spectrum(z, output_file=os.path.join(emb_outdir, "eigenvalue_spectrum.png"))
    # plot_pairwise_distance_vs_delta_pt(z, red_jet_pt, output_file=os.path.join(emb_outdir, "pairwise_distance_vs_delta_pt.png"))
    # plot_pca_explained_variance(z_pca, output_file=os.path.join(emb_outdir, "pca_explained_variance.png"))
    # plot_pairwise_distance_hist(z, red_event_class, output_file=os.path.join(emb_outdir, "pairwise_distance_distribution_per_process.png"))

    # # Plot embedding space vizualizations
    # plot_pca_embedding_2d(
    #     z_pca, 
    #     output_file=os.path.join(emb_outdir, "pca_embedding_2d_colorpt.png"), 
    #     variable_name="Jet log($p_T$) (GeV)",
    #     variable_array=red_jet_pt, 
    #     percentile=99,
    #     log_variable=True
    # )
    # plot_pca_embedding_2d_matrix(
    #     z_pca, 
    #     output_file=os.path.join(emb_outdir, "pca_embedding_2d_matrix_event_pt.png"), 
    #     n_components=3, 
    #     class_info=process_info_emb,
    #     class_array=red_event_class,
    #     variable_name=r"Jet log($p_T$) (GeV)",
    #     variable_array=red_jet_pt, 
    #     percentile=99, 
    #     log_variable=True
    # )
    # plot_pca_embedding_2d(
    #     z_pca, 
    #     output_file=os.path.join(emb_outdir, "pca_embedding_2d_colorreco.png"), 
    #     variable_name="Jet Reco score",
    #     variable_array=red_jet_reco,
    #     percentile=99,
    #     log_variable=True
    # )
    # plot_pca_embedding_2d(
    #     z_pca, 
    #     output_file=os.path.join(emb_outdir, "pca_embedding_2d_colormahalanobis.png"), 
    #     variable_name="Jet Mahalanobis score",
    #     variable_array=red_jet_mahalanobis,
    #     percentile=99,
    #     log_variable=True
    # )
    # plot_pca_embedding_2d(
    #     z_pca, 
    #     output_file=os.path.join(emb_outdir, "pca_embedding_2d_per_process.png"), 
    #     class_info=process_info_emb,
    #     class_array=red_event_class,
    #     percentile=99
    # )
    # plot_tsne_embedding_2d(
    #     z_tsne, 
    #     output_file=os.path.join(emb_outdir, "tsne_by_process.png"), 
    #     class_info=process_info_emb, 
    #     class_array=red_event_class,
    # )
    # plot_tsne_embedding_2d(
    #     z_tsne, 
    #     output_file=os.path.join(emb_outdir, "tsne_by_flav_colorpt.png"), 
    #     variable_name="Jet log($p_T$) (GeV)",
    #     variable_array=red_jet_pt,
    #     percentile=99,
    #     log_variable=True
    # )
    # plot_tsne_embedding_2d(
    #     z_tsne, 
    #     output_file=os.path.join(emb_outdir, "tsne_by_process_colorreco.png"), 
    #     variable_name="Jet Reco score",
    #     variable_array=red_jet_reco,
    #     percentile=99,
    #     log_variable=True
    # )
    # plot_tsne_embedding_2d(
    #     z_tsne, 
    #     output_file=os.path.join(emb_outdir, "tsne_by_process_colormahalanobis.png"), 
    #     variable_name="Jet Mahalanobis score",
    #     variable_array=red_jet_mahalanobis,
    #     percentile=99,
    #     log_variable=True
    # )

    # # plot_class_distance_matrix(process_info, z, red_event_class, output_file=os.path.join(emb_outdir, "class_distance_matrix_process.png"))
    
    # # Now compute metrics and plots grouped by process
    # # metrics['silhouette_proc'] = silhouette(z_bkg, red_event_class_bkg)
    # # metrics['linear_probe_acc_proc'] = linear_probe_accuracy(z_bkg, red_event_class_bkg, cv=5)
    # # metrics['knn_acc_proc'] = knn_accuracy(z_bkg, red_event_class_bkg, k=20, cv=5)
    # # metrics['neighbour_pur_proc'] = neighbour_purity(z_bkg, red_event_class_bkg, k=20)


    
    # #####################################
    # # GROUPED BY JET FLAVOR
    # #####################################
    # # use only jets that are matched to a gen-level jet and from SM processes
    # matched_mask = (data['jet_class'] >= 0) & data['sm_mask']
    # matched_jet_embedding = data['embedding'][matched_mask]
    # matched_jet_embedding_pca = jet_embedding_pca[matched_mask]
    # matched_jet_class = data['jet_class'][matched_mask]
    # matched_event_class = data['event_class'][matched_mask]
    # matched_jet_pt = data['jet_pt_phys'][matched_mask]
    # matched_jet_reco = data['reco_score'][matched_mask]
    # matched_jet_mahalanobis = data['mahalanobis_score'][matched_mask]
    # # matched_jet_btag = data['jet_btag'][matched_mask]

    # sample_indices = generate_balanced_sample_indices(
    #     class_info = jet_info,
    #     class_array = matched_jet_class,
    #     njets_per_class = 2500
    # )
    # z = matched_jet_embedding[sample_indices]
    # z_pca = matched_jet_embedding_pca[sample_indices]
    # red_jet_pt = matched_jet_pt[sample_indices]
    # red_jet_class = matched_jet_class[sample_indices]
    # red_event_class = matched_event_class[sample_indices]
    # # red_jet_btag = matched_jet_btag[sample_indices]
    # red_jet_reco = matched_jet_reco[sample_indices]
    # red_jet_mahalanobis = matched_jet_mahalanobis[sample_indices]
    # z_tsne = fit_tsne_embedding(z)  # fit once

    # plot_pairwise_distance_hist(z, red_jet_class, output_file=os.path.join(emb_outdir, "pairwise_distance_distribution_per_flav.png"))
    # # plot_pca_embedding_2d(  
    # #     z_pca, 
    # #     output_file=os.path.join(emb_outdir, "pca_embedding_2d_colorbtag.png"), 
    # #     variable_name="Jet b-tag score",
    # #     variable_array=red_jet_btag,
    # #     percentile=99,
    # # )
    # # plot_pca_embedding_2d_matrix(
    # #     z_pca, 
    # #     output_file=os.path.join(emb_outdir, "pca_embedding_2d_matrix_jet_btag.png"), 
    # #     n_components=3, 
    # #     class_info=jet_info,
    # #     class_array=red_jet_class, 
    # #     variable_name="Jet b-tag score",
    # #     variable_array=red_jet_btag, 
    # #     percentile=99, 
    # # )
    # plot_pca_embedding_2d(
    #     z_pca, 
    #     output_file=os.path.join(emb_outdir, "pca_embedding_2d_per_flav.png"), 
    #     class_info=jet_info,
    #     class_array=red_jet_class,
    #     percentile=99
    # )
    # plot_tsne_embedding_2d(
    #     z_tsne, 
    #     output_file=os.path.join(emb_outdir, "tsne_by_flav.png"), 
    #     class_info=jet_info,
    #     class_array=red_jet_class
    # )
    # # plot_tsne_embedding_2d(
    # #     z_tsne, 
    # #     output_file=os.path.join(emb_outdir, "tsne_by_flav_colorbtag.png"), 
    # #     variable_name="Jet b-tag score",
    # #     variable_array=red_jet_btag,
    # #     percentile=99,
    # # )
    # plot_tsne_embedding_2d(
    #     z_tsne, 
    #     output_file=os.path.join(emb_outdir, "tsne_by_flav_colorreco.png"), 
    #     variable_name="Jet Reco score",
    #     variable_array=red_jet_reco,
    #     percentile=99,
    #     log_variable=True
    # )
    # plot_tsne_embedding_2d(
    #     z_tsne, 
    #     output_file=os.path.join(emb_outdir, "tsne_by_flav_colormahalanobis.png"), 
    #     variable_name="Jet Mahalanobis score",
    #     variable_array=red_jet_mahalanobis,
    #     percentile=99,
    #     log_variable=True
    # )
    # # plot_class_distance_matrix(jet_info, z, red_jet_class, output_file=os.path.join(emb_outdir, "class_distance_matrix_flav.png"))

    # # metrics['silhouette_flav'] = silhouette(z_bkg, red_jet_class_bkg)
    # # metrics['linear_probe_acc_flav'] = linear_probe_accuracy(z_bkg, red_jet_class_bkg, cv=5)
    # # metrics['knn_acc_flav'] = knn_accuracy(z_bkg, red_jet_class_bkg, k=20, cv=5)
    # # metrics['neighbour_pur_flav'] = neighbour_purity(z_bkg, red_jet_class_bkg, k=20)

    # for metric_name, metric_value in metrics.items():
    #     print(f"{metric_name:<40}: {metric_value:.4f}")

    # # Save metrics to a csv file for later analysis
    # metric_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
    # metric_df.to_csv(os.path.join(emb_outdir, "embedding_metrics.csv"), index=False)

    # # And a text file for easy reading
    # with open(os.path.join(emb_outdir, "embedding_metrics.txt"), 'w') as f:
    #     for metric_name, metric_value in metrics.items():
    #         f.write(f"{metric_name:<40}: {metric_value:.4f}\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-m", "--model_folder", type=str, default="./output/deepset_contrastive_vicreg", help="Path to the model folder")
    parser.add_argument("-ad", "--ad_method", type=str, default="all", choices=AD_METHODS + ["all"] + ["all_pt"], help="Anomaly detection method: 'auto' for autoencoder scores, 'mahl' for Mahalanobis scores")
    parser.add_argument("-o", "--plot_dir", type=str, default="plots", help="Output directory for plots")
    args = parser.parse_args()

    all_data = {}
    for f in os.listdir(os.path.join(args.model_folder, 'testing_data_ad')):
        if f.endswith('.npy'):
            print(f"Loading {f} from testing_data_ad...")
            data = np.load(os.path.join(args.model_folder, 'testing_data_ad', f), allow_pickle=True)
            all_data[f.replace('.npy', '')] = data

    make_plots(args.model_folder, all_data, ad_method=args.ad_method, plot_dir=args.plot_dir)
    








