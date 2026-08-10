import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
import pandas as pd
from collections import defaultdict
import os

def get_mahalanobis_score(embeddings: np.ndarray, classes: np.ndarray, bkg_class: list | int):
    from scipy.spatial import distance

    bkg_mask = np.isin(classes, bkg_class)
    background_embeddings = embeddings[bkg_mask]
    
    # Compute the mean and covariance of the background embeddings
    mean_bg = np.mean(background_embeddings, axis=0)
    cov_bg = np.cov(background_embeddings, rowvar=False)

    # Regularize the covariance matrix to avoid singularity
    cov_bg += np.eye(cov_bg.shape[0]) * 1e-6

    # Compute the Mahalanobis distance for each embedding
    mahalanobis_scores = distance.cdist(embeddings, [mean_bg], metric='mahalanobis', VI=np.linalg.inv(cov_bg))

    return mahalanobis_scores.flatten()

def get_msp_score(class_preds: np.ndarray) -> np.ndarray:
    return 1 - np.max(class_preds, axis=1)

def get_margin_score(class_preds: np.ndarray) -> np.ndarray:
    sorted_preds = np.sort(class_preds, axis=1)
    margin = sorted_preds[:, -1] - sorted_preds[:, -2]
    return 1 - margin

def get_gini_score(class_preds: np.ndarray) -> np.ndarray:
    return 1 - np.sum(class_preds ** 2, axis=1)

def get_entropy_score(class_preds: np.ndarray) -> np.ndarray:
    epsilon = 1e-10
    return -np.sum(class_preds * np.log(class_preds + epsilon), axis=1)

def get_energy_score(class_preds: np.ndarray) -> np.ndarray:
    return -np.log(np.sum(np.exp(class_preds), axis=1))

def get_event_scores(event_ids, jet_scores, method="max"):
    """Get the anomaly score for each event based on the scores of its jets."""
    unique_events, inverse = np.unique(event_ids, return_inverse=True)
    if method == "max":
        event_scores = np.zeros(len(unique_events))
        np.maximum.at(event_scores, inverse, jet_scores)
    elif method == "mean":
        event_scores_sum = np.bincount(inverse, weights=jet_scores)
        event_scores_count = np.bincount(inverse)
        event_scores = event_scores_sum / np.maximum(event_scores_count, 1)
    elif method == "sum":
        event_scores = np.bincount(inverse, weights=jet_scores)
    else:
        raise ValueError(f"Unknown method: {method}")
    return unique_events, event_scores

def get_var1_of_max_var2_jet_per_event(event_ids, jet_var1, jet_var2):
    """
    Get the variable1 value of the jet with the maximum variable2 for each event.
    Inputs:
        event_ids: array of event IDs for each jet (n_jets,)
        jet_var1: array of variable1 values for each jet (n_jets,)
        jet_var2: array of variable2 values for each jet (n_jets,)
    Outputs:
        unique_events: array of unique event IDs (n_events,)
        max_var_per_event: array of maximum variable values for each event (n_events,)
    """
    unique_events, inverse = np.unique(event_ids, return_inverse=True)
    max_var2_per_event = np.full(len(unique_events), -np.inf)
    max_var1_per_event = np.zeros(len(unique_events))

    for i, (var2, var1) in enumerate(zip(jet_var2, jet_var1)):
        ev = inverse[i]
        if var2 > max_var2_per_event[ev]:
            max_var2_per_event[ev] = var2
            max_var1_per_event[ev] = var1

    return unique_events, max_var1_per_event

def get_max_var_jet(event_ids, jet_var):
    """Get the maximum variable2 of jets for each event."""
    unique_events, inverse = np.unique(event_ids, return_inverse=True)
    max_var_per_event = np.zeros(len(unique_events))
    np.maximum.at(max_var_per_event, inverse, jet_var)
    return unique_events, max_var_per_event

def get_event_ht(event_ids, jet_pts, pt_threshold=30):
    """
    Get the HT (scalar sum of jet pt) for each event, considering only jets above a pt threshold.
    Events with no jets above threshold are assigned HT = 0.
    """
    # Get all unique events first, before any filtering
    unique_events, inverse = np.unique(event_ids, return_inverse=True)
    ht_per_event = np.zeros(len(unique_events))

    # Apply threshold mask and accumulate only passing jets
    mask = jet_pts > pt_threshold
    np.add.at(ht_per_event, inverse[mask], jet_pts[mask])

    return unique_events, ht_per_event


def get_threshold_for_fpr(scores, fpr=0.01):
    """
    Get the threshold for a given false positive rate (FPR) based on the scores of the background events.
    """
    threshold = np.percentile(scores, 100 * (1 - fpr))
    return threshold

def save_efficiency_table(
        process_info: dict,
        event_classes: np.ndarray,
        event_scores_max: np.ndarray,
        event_scores_ptmax: np.ndarray,
        event_scores_sum: np.ndarray,
        threshold_max: float,
        threshold_ptmax: float,
        threshold_sum: float,
        output_file: str,
        print_to_console: bool = True
    ):
    """
    Save the efficiency table for each process to a CSV file.
    """
    scores_max_minbias = event_scores_max[event_classes == process_info['MinBias']['class']]
    scores_ptmax_minbias = event_scores_ptmax[event_classes == process_info['MinBias']['class']]
    scores_sum_minbias = event_scores_sum[event_classes == process_info['MinBias']['class']]

    rows = []
    for proc, info in process_info.items():
        if proc.lower() == 'minbias':
            continue
        
        proc_mask = event_classes == info['class']
        scores_max = event_scores_max[proc_mask]
        scores_ptmax = event_scores_ptmax[proc_mask]
        scores_sum = event_scores_sum[proc_mask]
        n_events = proc_mask.sum()

        event_classes_combined = np.concatenate([event_classes[event_classes == process_info['MinBias']['class']], event_classes[proc_mask]])
        scores_combined_max = np.concatenate([scores_max_minbias, scores_max])
        scores_combined_ptmax = np.concatenate([scores_ptmax_minbias, scores_ptmax])
        scores_combined_sum = np.concatenate([scores_sum_minbias, scores_sum])

        roc_auc_max = roc_auc_score((event_classes_combined != process_info['MinBias']['class']).astype(int), scores_combined_max)
        roc_auc_ptmax = roc_auc_score((event_classes_combined != process_info['MinBias']['class']).astype(int), scores_combined_ptmax)
        roc_auc_sum = roc_auc_score((event_classes_combined != process_info['MinBias']['class']).astype(int), scores_combined_sum)

        rows.append({
            'process':        info['label'],
            'n_events':       n_events,
            'efficiency (max)': round(float((scores_max > threshold_max).mean() * 100), 2),
            'efficiency (ptmax)': round(float((scores_ptmax > threshold_ptmax).mean() * 100), 2),
            'efficiency (sum)': round(float((scores_sum > threshold_sum).mean() * 100), 2),
            'roc_auc (max)': round(float(roc_auc_max), 4),
            'roc_auc (ptmax)': round(float(roc_auc_ptmax), 4),
            'roc_auc (sum)': round(float(roc_auc_sum), 4),
        })

    df = pd.DataFrame(rows)
    format = output_file.split('.')[-1].lower()
    if format == 'csv':
        df.to_csv(output_file, index=False)
    elif format == 'latex':
        with open(output_file, 'w') as f:
            f.write(df.to_latex(index=False, float_format="%.3f"))
    elif format == 'html':
        with open(output_file, 'w') as f:
            f.write(df.to_html(index=False, float_format="%.3f"))
    elif format == 'json':
        df.to_json(output_file, orient='records', lines=True)
    elif format == 'txt':
        with open(output_file, 'w') as f:
            f.write(df.to_string(index=False, float_format="%.3f"))
    else:
        raise ValueError(f"Unsupported format: {format}. Supported formats are: csv, latex, html, json, txt.")
    
    if print_to_console:
        print(df.to_string(index=False, float_format="%.3f"))  # Print the table to console for quick reference


def plot_anomaly_score_per_class_hist(
        class_info: dict,
        class_array: np.ndarray,
        test_scores: np.ndarray,
        hist_edges: np.ndarray,
        output_file: str,
        log_scale: bool = False
    ):
    """
    Plot the histogram of anomaly scores for different classes.
    """
    # Set default plot configuration
    figsize = (5.5, 5)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    for cls, info in class_info.items():
        mask = (class_array == info['class'])
        scores = np.clip(test_scores[mask], hist_edges[0], hist_edges[-1])
        ax.hist(scores, bins=hist_edges, histtype='step', density=True, label=info['label'], color=info['color'])

    ax.set_xlabel('Anomaly Score')
    ax.set_ylabel('Density')
    ax.set_title("Jet Anomaly Score")
    ax.set_ylim(bottom=1e-5)  # Set a lower limit for the y-axis to avoid issues with log scale
    ax.set_yscale('log')
    ax.set_xlim(hist_edges[0], hist_edges[-1])
    if log_scale:
        ax.set_xscale('log')
    ax.legend(fontsize=8)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()


def plot_anomaly_score_vs_variable(
        class_info: dict,
        test_scores: np.ndarray, 
        class_array: np.ndarray,
        variable: np.ndarray,
        var_edges: np.ndarray,
        var_name: str,
        output_file: str,
        log_scale: bool = False,
        log_variable: bool = False
    ):
    """
    Plot the mean anomaly score as a function of a given variable for different processes.
    """
    # pt_edges and var_centers are now passed as arguments
    var_centers = [(var_edges[i] + var_edges[i+1]) / 2 for i in range(len(var_edges) - 1)]

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 5))

    for cls, info in class_info.items():
        label_mask = (class_array == info['class'])

        var_bin_means = []
        for i in range(len(var_edges) - 1):
            bin_mask = (variable[label_mask] >= var_edges[i]) & (variable[label_mask] < var_edges[i+1])
            if bin_mask.sum() > 0:
                mean_score = test_scores[label_mask][bin_mask].mean()
                var_bin_means.append(mean_score)
            else:
                var_bin_means.append(np.nan)

        ax.plot(var_centers, var_bin_means, marker='.', label=info['label'], color=info['color'])

    # Add label for correlation
    corr = np.corrcoef(variable, test_scores)[0, 1]
    ax.text(0.05, 0.95, f'Correlation: {corr:.2f}', transform=ax.transAxes, fontsize=10, verticalalignment='top')

    ax.set_xlabel(f'{var_name} Bin')
    ax.set_ylabel('Mean Anomaly Score')
    ax.set_title(f'Mean Anomaly Score vs {var_name}')
    if log_variable:
        ax.set_xscale('log')
    if log_scale:
        ax.set_yscale('log')
        # ax.set_ylim(bottom=1e-3, top=1e3)  # Set a lower limit for the y-axis to avoid issues with log scale
    ax.legend(fontsize=8)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()


def plot_roc(
        process_info: dict,
        event_classes: np.ndarray,
        event_scores: np.ndarray,
        output_file: str,
        minbias_rate: float = 40e3, # 40 MHz in kHz
        rate_target: float = 10.0, # kHz
        bkg_class: str = 'MinBias'
    ):
    """
    Plot ROC curves for each process based on event-level scores.
    """
    bkg_scores = event_scores[event_classes == process_info[bkg_class]['class']]

    fig, ax = plt.subplots(figsize=(5.5, 5))

    for proc, info in process_info.items():
        if proc == bkg_class:
            continue

        sig_scores = event_scores[event_classes == info['class']]

        y_true = np.concatenate([np.zeros(len(bkg_scores)), np.ones(len(sig_scores))])
        y_score = np.concatenate([bkg_scores, sig_scores])

        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        bkg_rate = fpr * minbias_rate
        ax.plot(tpr, bkg_rate, lw=2, label=f"{info['label']} (AUC={roc_auc:.2f})", color=info['color'])

    # Mark target operating point
    if rate_target is not None:
        ax.axhline(rate_target, color='red', linestyle=':', label=f'{rate_target:.0f} kHz')

    ax.set_yscale('log')
    ax.set_ylim(0.1, minbias_rate)
    ax.set_xlim(0, 1.1)
    ax.set_ylabel('MinBias Rate [kHz]')
    ax.set_xlabel('Efficiency')
    ax.grid(True, which='major', axis='both', alpha=0.25)
    ax.legend(fontsize=8, loc='lower right')
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()

def plot_roc_vs_qcd(
        process_info: dict,
        event_classes: np.ndarray,
        event_scores: np.ndarray,
        output_file: str,
        bkg_class: str = 'QCD',
        weight_variable: np.ndarray = None,
        weight_bins: np.ndarray = None,
    ):
    """
    Plot ROC curves for each process based on event-level scores.
    """
    bkg_scores = event_scores[event_classes == process_info[bkg_class]['class']]

    if weight_variable is not None:
        bkg_variable = weight_variable[event_classes == process_info[bkg_class]['class']]
        c_bkg, b = np.histogram(bkg_variable, bins=weight_bins)

    fig, ax = plt.subplots(figsize=(5.5, 5))

    for proc, info in process_info.items():
        if proc == bkg_class:
            continue

        sig_scores = event_scores[event_classes == info['class']]

        y_true = np.concatenate([np.zeros(len(bkg_scores)), np.ones(len(sig_scores))])
        y_score = np.concatenate([bkg_scores, sig_scores])
        if weight_variable is not None:
            sig_variable = weight_variable[event_classes == info['class']]
            c_sig, _ = np.histogram(sig_variable, bins=weight_bins)
            sig_bin_weights = np.divide(c_bkg, c_sig, out=np.zeros_like(c_bkg, dtype=float), where=c_sig > 0)
            sig_bin_indices = np.digitize(sig_variable, weight_bins) - 1
            sig_weights = np.zeros(len(sig_variable), dtype=float)
            valid = ((sig_bin_indices >= 0)& (sig_bin_indices < len(sig_bin_weights)))
            sig_weights[valid] = sig_bin_weights[sig_bin_indices[valid]]
            weights = np.concatenate([np.ones(len(bkg_scores), dtype=float), sig_weights,])
        else:
            weights = None

        fpr, tpr, _ = roc_curve(y_true, y_score, sample_weight=weights)
        roc_auc = auc(fpr, tpr)
        ax.plot(tpr, fpr, lw=2, label=f"{info['label']} (AUC={roc_auc:.2f})", color=info['color'])

    # Mark target operating point
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    ax.set_ylabel('FPR')
    ax.set_xlabel('TPR')
    ax.grid(True, which='major', axis='both', alpha=0.25)
    ax.legend(fontsize=8, loc='lower right')
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()


def plot_turn_on(
        process_info: dict,
        event_classes: np.ndarray,
        event_scores: np.ndarray,
        threshold: float,
        turn_on_var: np.ndarray,
        turn_on_var_name: str,
        output_file: str,
        bins: np.ndarray = np.array([0, 20, 40, 60, 80, 100, 125, 150, 175, 200, 225, 250, 275, 300, 350, 400, 450, 500, 600, 700, 800, 1000]),
    ):
    """
    Plot turn-on curves for each process based on event-level scores.
    """
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    fig, ax = plt.subplots(figsize=(5.5, 5))

    for proc, info in process_info.items():
        if proc == 'MinBias':
            continue

        mask = event_classes == info['class']
        _var = turn_on_var[mask]
        _scores = event_scores[mask]

        total, _ = np.histogram(_var, bins=bins)
        passed, _ = np.histogram(_var[_scores > threshold], bins=bins)
        efficiency = np.divide(passed, total, out=np.zeros_like(passed, dtype=float), where=total > 0)

        ax.plot(bin_centers, efficiency, label=f"{info['label']}", marker='.', color=info['color'])

    ax.set_xlabel(turn_on_var_name)
    ax.set_ylabel("Trigger Efficiency")
    ax.set_ylim(0, 1.1)
    ax.grid(True)
    ax.legend(loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()





