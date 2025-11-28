# flake8: noqa
import collections
import os

import awkward as ak
import matplotlib
import matplotlib.pyplot as plt
import mplhep as hep

# Third parties
import pandas
import tensorflow as tf
from matplotlib.pyplot import cm
from scipy.stats import norm

setattr(collections, "MutableMapping", collections.abc.MutableMapping)
import histbook
import numpy as np
import shap
from sklearn.metrics import auc, roc_curve

from tagger.data.tools import load_data, to_ML
from tagger.plot import style

from .common import PT_BINS, plot_histo

matplotlib.use('Agg')

plt.rcParams.update({'figure.max_open_warning': 0})

# some custom imports for efficiency plots
np.bool = np.bool_


style.set_style()

# DEFINE ALL THE PLOTTING FUNCTIONS HERE!!!! THEY WILL BE CALLED IN basic() function >>>>>>>


def loss_history(plot_dir, loss_names, history):
    for metric in loss_names:
        metric = metric + '_loss'

        fig, ax = plt.subplots(1, 1, figsize=style.FIGURE_SIZE)
        hep.cms.label(llabel=style.CMSHEADER_LEFT, rlabel=style.CMSHEADER_RIGHT, ax=ax, fontsize=style.CMSHEADER_SIZE)
        ax.plot(history[metric], label='Train Loss', linewidth=style.LINEWIDTH)
        ax.plot(history['val_' + metric], label='Validation Loss', linewidth=style.LINEWIDTH)
        ax.grid(True)
        # ax.set_ylabel('Loss')
        ax.set_ylabel('Loss ' + metric)
        ax.set_xlabel('Epoch')
        ax.legend(loc='upper right')

        save_path = os.path.join(plot_dir, "loss_" + metric + "_history")
        plt.savefig(f"{save_path}.png", bbox_inches='tight')
        plt.savefig(f"{save_path}.pdf", bbox_inches='tight')

        fig.clf()


def ROC_taus(y_pred, y_test, class_labels, plot_dir, signal_proc=None):
    """
    Plot ROC curves for taus vs jets and leptons
    """

    save_dir = os.path.join(plot_dir, 'roc_taus')
    os.makedirs(save_dir, exist_ok=True)

    # Define class label groups
    tau_indices = [class_labels['taup'], class_labels['taum']]
    jet_indices = [class_labels[key] for key in ['b', 'charm', 'light', 'gluon']]
    muon_indices = [class_labels['muon']]
    electron_indices = [class_labels['electron']]

    # Function to calculate the roc inputs for plotting
    def compute_roc_inputs(y_pred, y_test, signal_indices, background_indices):
        """
        Compute the true labels and scores for ROC curve inputs.
        """
        signal_mask = sum(y_test[:, idx] for idx in signal_indices) > 0
        total_mask = signal_mask | (sum(y_test[:, idx] for idx in background_indices) > 0)

        signal_scores = sum(y_pred[:, idx] for idx in signal_indices)
        background_scores = sum(y_pred[:, idx] for idx in background_indices)
        total_scores = signal_scores + background_scores

        return signal_mask[total_mask], (signal_scores / total_scores)[total_mask]

    # Compute ROC data
    roc_data = []

    for label, bkg_indices in [
        (r'$\tau_h^{\pm}$ vs Jets (b, c, light, gluon)', jet_indices),
        (r'$\tau_h^{\pm}$ vs Muons', muon_indices),
        (r'$\tau_h^{\pm}$ vs Electrons', electron_indices),
    ]:
        y_true, y_score = compute_roc_inputs(y_pred, y_test, tau_indices, bkg_indices)
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        roc_data.append((tpr, fpr, roc_auc, label))

    # Plot all ROC curves in one figure
    plt.figure(figsize=style.FIGURE_SIZE)
    hep.cms.label(llabel=style.CMSHEADER_LEFT, rlabel=style.CMSHEADER_RIGHT, fontsize=style.CMSHEADER_SIZE)

    for tpr, fpr, roc_auc, label in roc_data:
        plt.plot(tpr, fpr, label=f'{label} (AUC = {roc_auc:.2f})', linewidth=style.LINEWIDTH)

    plt.grid(True)
    plt.xlabel('Signal Efficiency')
    plt.ylabel('Mistag Rate')
    plt.yscale('log')
    plt.ylim(1e-3, 1.1)
    leg = plt.legend(loc='upper left', fontsize=style.SMALL_SIZE + 3, title=signal_proc)
    leg._legend_box.align = "left"

    save_path = os.path.join(save_dir, "ROC_taus_combined")
    plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
    plt.savefig(f"{save_path}.png", bbox_inches='tight')
    plt.close()


def ROC_binary(y_pred, y_test, class_labels, plot_dir, class_pair, signal_proc=None):
    """
    Generate ROC curves comparing between two specific class labels.
    """

    save_dir = os.path.join(plot_dir, 'roc_binary')
    os.makedirs(save_dir, exist_ok=True)

    # Ensure class_pair exists in class_labels
    assert (
        class_pair[0] in class_labels and class_pair[1] in class_labels
    ), "Both class_pair labels must exist in class_labels"

    # Get indices of the classes to compare
    idx1, idx2 = class_labels[class_pair[0]], class_labels[class_pair[1]]

    # Select true labels and predicted probabilities for the selected classes
    y_true1, y_true2 = y_test[:, idx1], y_test[:, idx2]
    y_score1, y_score2 = y_pred[:, idx1], y_pred[:, idx2]

    # Combine the labels and scores for binary classification
    selection = (y_true1 == 1) | (y_true2 == 1)
    y_true_binary = y_true1[selection]
    # Normalized probabilities
    y_score_binary = y_score1[selection] / (y_score1[selection] + y_score2[selection])

    # Compute FPR, TPR, and AUC
    fpr, tpr, _ = roc_curve(y_true_binary, y_score_binary)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    fig, ax = plt.subplots(1, 1, figsize=style.FIGURE_SIZE)
    hep.cms.label(llabel=style.CMSHEADER_LEFT, rlabel=style.CMSHEADER_RIGHT, ax=ax, fontsize=style.CMSHEADER_SIZE)
    ax.plot(
        tpr,
        fpr,
        label=f'{style.CLASS_LABEL_STYLE[class_pair[0]]} vs {style.CLASS_LABEL_STYLE[class_pair[1]]} (AUC = {roc_auc:.2f})',
        color='blue',
        linewidth=5,
    )
    ax.grid(True)
    ax.set_ylabel('Mistag Rate')
    ax.set_xlabel('Signal Efficiency')
    leg = ax.legend(loc='lower right', fontsize=style.SMALL_SIZE + 3, title=signal_proc)
    leg._legend_box.align = "left"
    ax.set_yscale('log')
    ax.set_ylim([1e-3, 1.1])

    # Save the plot
    save_path = os.path.join(save_dir, f"ROC_{class_pair[0]}_vs_{class_pair[1]}")
    plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
    plt.savefig(f"{save_path}.png", bbox_inches='tight')
    plt.close()


def ROC(y_pred, y_test, class_labels, plot_dir, ROC_dict):
    # Create a colormap for unique colors
    # Use 'tab10' with enough colors
    colormap = cm.get_cmap('Set1', len(class_labels))

    # Create a plot for ROC curves
    fig, ax = plt.subplots(1, 1, figsize=style.FIGURE_SIZE)
    hep.cms.label(llabel=style.CMSHEADER_LEFT, rlabel=style.CMSHEADER_RIGHT, ax=ax, fontsize=style.CMSHEADER_SIZE)
    for i, class_label in enumerate(class_labels):

        # Get true labels and predicted probabilities for the current class
        # Extract the one-hot column for the current class
        y_true = y_test[:, i]
        y_score = y_pred[:, i]  # Predicted probabilities for the current class

        # Compute FPR, TPR, and AUC
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        ROC_dict[class_label] = roc_auc
        # Plot the ROC curve for the current class
        ax.plot(
            tpr,
            fpr,
            label=f'{style.CLASS_LABEL_STYLE[class_label]} (AUC = {roc_auc:.2f})',
            color=colormap(i),
            linewidth=style.LINEWIDTH,
        )

    # Plot formatting
    ax.grid(True)
    ax.set_ylabel('Mistag Rate')
    ax.set_xlabel('Signal Efficiency')

    auc_list = [value for key, value in ROC_dict.items()]
    handles, labels = plt.gca().get_legend_handles_labels()
    order = np.argsort(auc_list)
    ax.legend(
        [handles[idx] for idx in order],
        [labels[idx] for idx in order],
        loc='upper left',
        ncol=2,
        fontsize=style.SMALL_SIZE - 3,
    )

    ax.set_yscale('log')
    ax.set_ylim([1e-3, 1.1])

    # Save the plot
    save_path = os.path.join(plot_dir, "basic_ROC")
    plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
    plt.savefig(f"{save_path}.png", bbox_inches='tight')
    plt.close()

    return ROC_dict


def confusion(y_pred, y_test, class_labels, plot_dir):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    cm = confusion_matrix(
        np.argmax(y_test, axis=1),
        np.argmax(y_pred, axis=1),
        normalize="true",
    )
    cm = np.round(cm, 3)
    class_labels = {v: k for k, v in class_labels.items()}
    labels = [style.CLASS_LABEL_STYLE[class_labels[i]] for i in range(y_test.shape[1])]

    # Create a plot of the confusion matrix
    fig, ax = plt.subplots(1, 1, figsize=style.FIGURE_SIZE)
    hep.cms.label(llabel=style.CMSHEADER_LEFT, rlabel=style.CMSHEADER_RIGHT, fontsize=style.CMSHEADER_SIZE)
    matrix_display = ConfusionMatrixDisplay(cm, display_labels=labels)

    matrix_display.plot(ax=ax)
    matrix_display.im_.set_clim(0, 1)

    # Remove default the colorbar
    matrix_display.im_.colorbar.remove()

    # Adjust colorbar height to match the plot
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.5)
    plt.colorbar(matrix_display.im_, cax=cax)

    # Save the plot
    plt.savefig(os.path.join(plot_dir, f"confusion_matrix.png"), bbox_inches='tight')
    plt.savefig(os.path.join(plot_dir, f"confusion_matrix.pdf"), bbox_inches='tight')


def pt_correction_hist(pt_ratio, truth_pt_test, reco_pt_test, plot_dir):
    """
    Plot the histograms of truth pt, reconstructed (uncorrected) pt, and corrected pt
    """

    plot_histo(
        [truth_pt_test, reco_pt_test, np.multiply(reco_pt_test, pt_ratio)],
        ['Truth', 'Reconstructed', 'NN Predicted'],
        '',
        r'$p_T$ [GeV]',
        'a.u',
        log = 'linear',
        x_range=(0, 300),
    )
    save_path = os.path.join(plot_dir, "pt_hist")
    plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
    plt.savefig(f"{save_path}.png", bbox_inches='tight')
    plt.close()

    return


def plot_input_vars(X_test, y_test, input_vars, class_labels, plot_dir):

    save_dir = os.path.join(plot_dir, 'inputs')
    os.makedirs(save_dir, exist_ok=True)

    is_filled = (X_test[:, :, 16] == 1)
    for i in range(len(input_vars)):
        inputs = []
        labels = []
        for iclass, class_label in enumerate(class_labels):
            labels.append(style.INPUT_FEATURE_STYLE[input_vars[i]] + " " + style.CLASS_LABEL_STYLE[class_label])
            # Filter by class (use [:,None] to ignore the candidate dimension) and by if is_filled is 1
            # don't want all 0 inputs in our plots but also want to preserve real 0s in the plots
            input_per_class = X_test[:, :, i][(y_test[:, iclass] == 1)[:,None] & is_filled ].flatten()
            inputs.append(input_per_class)
        plot_histo(
            inputs,
            labels,
            '',
            style.INPUT_FEATURE_STYLE[input_vars[i]],
            'a.u',
            log = 'log',
            x_range=(np.min(X_test[:, :, i]), np.max(X_test[:, :, i])),
        )
        save_path = os.path.join(save_dir, input_vars[i]+"_split")
        plt.savefig(f"{save_path}.png", bbox_inches='tight')
        plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
        plt.close()

    for i in range(len(input_vars)):
        plot_histo(
            [X_test[:, :, i][is_filled].flatten()],
            [style.INPUT_FEATURE_STYLE[input_vars[i]]],
            '',
            style.INPUT_FEATURE_STYLE[input_vars[i]],
            'a.u',
            log = 'log',
            x_range=(np.min(X_test[:, :, i]), np.max(X_test[:, :, i])),
        )
        save_path = os.path.join(save_dir, input_vars[i])
        plt.savefig(f"{save_path}.png", bbox_inches='tight')
        plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
        plt.close()

    labels = ["Multiplicities:  " + style.CLASS_LABEL_STYLE[class_label] for class_label in class_labels]

    multiplicities = {i : [] for i in range(len(class_labels)+1)}

    for ibatch,batch in enumerate(X_test):
        for iclass, class_label in enumerate(class_labels):
            num_candidates = (batch[(y_test[ibatch, iclass] == 1) & (batch[:,16] != 0)]).shape[0]
            if num_candidates > 0:
                multiplicities[iclass].append(num_candidates)
                multiplicities[len(class_labels)].append(num_candidates)

    plot_histo(
            [np.array(multiplicities[i]) for i in range(len(class_labels))],
            labels,
            '',
            'multiplicities',
            'a.u',
            log = 'log',
            x_range=(0,16),
            bins=17
    )
    save_path = os.path.join(save_dir, 'multiplicies_split')
    plt.savefig(f"{save_path}.png", bbox_inches='tight')
    plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
    plt.close()

    plot_histo(
            [np.array(multiplicities[len(class_labels)])],
            ['Multiplicity'],
            '',
            'Multiplicity',
            'a.u',
            log = 'log',
            x_range=(0,16),
            bins=17
    )
    save_path = os.path.join(save_dir, 'multiplicity')
    plt.savefig(f"{save_path}.png", bbox_inches='tight')
    plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
    plt.close()



def get_response(truth_pt, reco_pt, pt_ratio):

    # Calculate the regressed pt
    regressed_pt = np.multiply(reco_pt, pt_ratio)

    # to calculate response
    uncorrected_response = []
    regressed_response = []
    uncorrected_errors = []
    regressed_errors = []

    # Loop over the pT ranges
    for i in range(len(PT_BINS) - 1):
        pt_min = PT_BINS[i]
        pt_max = PT_BINS[i + 1]

        selection = (truth_pt > pt_min) & (truth_pt < pt_max)

        # Compute responses
        uncorrected_response_bin = reco_pt[selection] / truth_pt[selection]
        regressed_response_bin = regressed_pt[selection] / truth_pt[selection]

        # Append the mean response
        uncorrected_response.append(np.mean(uncorrected_response_bin))
        regressed_response.append(np.mean(regressed_response_bin))

        # Compute the standard deviation and uncertainty in the mean
        n_events = len(truth_pt[selection])

        if n_events > 0:
            uncorrected_std = np.std(uncorrected_response_bin)
            regressed_std = np.std(regressed_response_bin)

            uncorrected_errors.append(uncorrected_std / np.sqrt(n_events))
            regressed_errors.append(regressed_std / np.sqrt(n_events))
        else:
            # No events in bin
            uncorrected_errors.append(0)
            regressed_errors.append(0)

    return uncorrected_response, regressed_response, uncorrected_errors, regressed_errors


def response(class_labels, y_test, truth_pt_test, reco_pt_test, pt_ratio, plot_dir):
    save_dir = os.path.join(plot_dir, 'response')
    os.makedirs(save_dir, exist_ok=True)

    # pT coordinate points for plotting
    pt_points = [np.mean((PT_BINS[i], PT_BINS[i + 1])) for i in range(len(PT_BINS) - 1)]

    def plot_response(uncorrected_response, regressed_response, uncorrected_errors, regressed_errors, flavor, plot_name):

        # Plot the response
        fig, ax = plt.subplots(1, 1, figsize=style.FIGURE_SIZE)
        hep.cms.label(llabel=style.CMSHEADER_LEFT, rlabel=style.CMSHEADER_RIGHT, ax=ax, fontsize=style.CMSHEADER_SIZE)
        ax.errorbar(
            pt_points,
            uncorrected_response,
            yerr=uncorrected_errors,
            fmt='o',
            label=f"Uncorrected - {style.CLASS_LABEL_STYLE[flavor]}",
            capsize=4,
            ms=8,
            elinewidth=3,
        )
        ax.errorbar(
            pt_points,
            regressed_response,
            yerr=regressed_errors,
            fmt='o',
            label=f"Regressed - {style.CLASS_LABEL_STYLE[flavor]}",
            capsize=4,
            ms=8,
            elinewidth=3,
        )

        ax.set_xlabel(r"Jet $p_T^{Gen}$ [GeV]")
        ax.set_ylabel("Response (L1/Gen)")
        ax.legend()
        ax.grid()

        # Save the plot
        save_path = os.path.join(save_dir, plot_name)
        plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
        plt.savefig(f"{save_path}.png", bbox_inches='tight')
        plt.close()

    # Inclusive response
    uncorrected_response, regressed_response, uncorrected_errors, regressed_errors = get_response(
        truth_pt_test, reco_pt_test, pt_ratio
    )
    plot_response(
        uncorrected_response,
        regressed_response,
        uncorrected_errors,
        regressed_errors,
        flavor='inclusive',
        plot_name="inclusive_response",
    )

    # Flavor-wise response
    for flavor in class_labels.keys():
        idx = class_labels[flavor]
        flavor_selection = y_test[:, idx] == 1

        uncorrected_response, regressed_response, uncorrected_errors, regressed_errors = get_response(
            truth_pt_test[flavor_selection], reco_pt_test[flavor_selection], pt_ratio[flavor_selection]
        )
        plot_response(
            uncorrected_response,
            regressed_response,
            uncorrected_errors,
            regressed_errors,
            flavor=flavor,
            plot_name=f"{flavor}_response",
        )

    # Taus, jets, leptons rms
    rms_selection = {
        'taus': [class_labels['taup'], class_labels['taum']],
        'jets': [class_labels[key] for key in ['b', 'charm', 'light', 'gluon']],
        'leptons': [class_labels[key] for key in ['muon', 'electron']],
    }

    for key in rms_selection.keys():
        selection = sum(y_test[:, idx] for idx in rms_selection[key]) > 0

        uncorrected_response, regressed_response, uncorrected_errors, regressed_errors = get_response(
            truth_pt_test[selection], reco_pt_test[selection], pt_ratio[selection]
        )
        plot_response(
            uncorrected_response,
            regressed_response,
            uncorrected_errors,
            regressed_errors,
            flavor=key,
            plot_name=f"{key}_response",
        )

    return


def get_rms(truth_pt, reco_pt, pt_ratio):

    # Calculate the regressed pt
    regressed_pt = np.multiply(reco_pt, pt_ratio)

    # Get the residuals
    un_corrected_res = reco_pt - truth_pt
    regressed_res = regressed_pt - truth_pt

    rms_uncorr = []
    rms_reg = []
    rms_uncorr_err = []
    rms_reg_err = []

    # Loop over the pT ranges
    for i in range(len(PT_BINS) - 1):
        pt_min = PT_BINS[i]
        pt_max = PT_BINS[i + 1]
        pt_avg = np.mean((pt_min, pt_max))

        selection = (truth_pt > pt_min) & (truth_pt < pt_max)

        # Fit a Gaussian to the residuals and extract the standard deviation
        mu_uncorr, sigma_uncorr = norm.fit(un_corrected_res[selection] / pt_avg)
        mu_reg, sigma_reg = norm.fit(regressed_res[selection] / pt_avg)

        # Get the errors for the standard deviation
        # Standard error of the standard deviation for a normal distribution
        n_uncorr = len(un_corrected_res[selection])
        n_reg = len(regressed_res[selection])

        if n_uncorr <= 1 or n_reg <= 1:
            sigma_uncorr_err = sigma_uncorr
            sigma_reg_err = sigma_reg
        else:
            sigma_uncorr_err = sigma_uncorr / np.sqrt(2 * (n_uncorr - 1))
            sigma_reg_err = sigma_reg / np.sqrt(2 * (n_reg - 1))

        rms_uncorr.append(sigma_uncorr)
        rms_reg.append(sigma_reg)
        rms_uncorr_err.append(sigma_uncorr_err)
        rms_reg_err.append(sigma_reg_err)

    return rms_uncorr, rms_reg, rms_uncorr_err, rms_reg_err


def rms(class_labels, y_test, truth_pt_test, reco_pt_test, pt_ratio, plot_dir):

    save_dir = os.path.join(plot_dir, 'residual_rms')
    os.makedirs(save_dir, exist_ok=True)

    # pT coordinate points for plotting
    pt_points = [np.mean((PT_BINS[i], PT_BINS[i + 1])) for i in range(len(PT_BINS) - 1)]

    def plot_rms(uncorrected_rms, regressed_rms, uncorrected_rms_err, regressed_rms_err, flavor, plot_name):

        # Plot the response
        fig, ax = plt.subplots(1, 1, figsize=style.FIGURE_SIZE)
        hep.cms.label(llabel=style.CMSHEADER_LEFT, rlabel=style.CMSHEADER_RIGHT, ax=ax, fontsize=style.CMSHEADER_SIZE)
        ax.errorbar(
            pt_points,
            uncorrected_rms,
            yerr=uncorrected_rms_err,
            fmt='o',
            label=r"Uncorrected $\sigma$- {}".format(style.CLASS_LABEL_STYLE[flavor]),
            capsize=4,
            ms=8,
            elinewidth=3,
        )
        ax.errorbar(
            pt_points,
            regressed_rms,
            yerr=regressed_rms_err,
            fmt='o',
            label=r"Regressed $\sigma$ - {}".format(style.CLASS_LABEL_STYLE[flavor]),
            capsize=4,
            ms=8,
            elinewidth=3,
        )

        ax.set_xlabel(r"Jet $p_T^{Gen}$ [GeV]")
        ax.set_ylabel(r"$\sigma_{(p_T^{Gen} - p_T^{Reco})/p_T^{Gen}}$")
        ax.legend()
        ax.grid(True)

        # Save the plot
        save_path = os.path.join(save_dir, plot_name)
        plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
        plt.savefig(f"{save_path}.png", bbox_inches='tight')
        plt.close()

    # Inclusive rms
    uncorrected_rms, regressed_rms, uncorrected_rms_err, regressed_rms_err = get_rms(truth_pt_test, reco_pt_test, pt_ratio)
    plot_rms(
        uncorrected_rms, regressed_rms, uncorrected_rms_err, regressed_rms_err, flavor='inclusive', plot_name='inclusive'
    )

    # Flavor-wise rms
    for flavor in class_labels.keys():
        idx = class_labels[flavor]
        flavor_selection = y_test[:, idx] == 1

        uncorrected_rms, regressed_rms, uncorrected_rms_err, regressed_rms_err = get_rms(
            truth_pt_test[flavor_selection], reco_pt_test[flavor_selection], pt_ratio[flavor_selection]
        )
        plot_rms(
            uncorrected_rms, regressed_rms, uncorrected_rms_err, regressed_rms_err, flavor=flavor, plot_name=f"{flavor}_rms"
        )

    # Taus, jets, leptons rms
    rms_selection = {
        'taus': [class_labels['taup'], class_labels['taum']],
        'jets': [class_labels[key] for key in ['b', 'charm', 'light', 'gluon']],
        'leptons': [class_labels[key] for key in ['muon', 'electron']],
    }

    for key in rms_selection.keys():
        selection = sum(y_test[:, idx] for idx in rms_selection[key]) > 0

        uncorrected_rms, regressed_rms, uncorrected_rms_err, regressed_rms_err = get_rms(
            truth_pt_test[selection], reco_pt_test[selection], pt_ratio[selection]
        )
        plot_rms(uncorrected_rms, regressed_rms, uncorrected_rms_err, regressed_rms_err, flavor=key, plot_name=f"{key}_rms")

    return


def shapPlot(shap_values, feature_names, class_names):
    fig, ax = plt.subplots(1, 1, figsize=style.FIGURE_SIZE)
    feature_order = np.argsort(np.sum(np.mean(np.abs(shap_values), axis=1), axis=0))
    num_features = shap_values[0].shape[1]
    feature_inds = feature_order
    y_pos = np.arange(len(feature_inds))
    left_pos = np.zeros(len(feature_inds))

    axis_color = "#333333"
    class_inds = np.argsort([-np.abs(shap_values[i]).mean() for i in range(len(shap_values))])
    # Use 'tab10' with enough colors
    colormap = cm.get_cmap('Set1', len(class_names))

    for i, ind in enumerate(class_inds):
        global_shap_values = np.abs(shap_values[ind]).mean(0)
        label = style.CLASS_LABEL_STYLE[class_names[ind]]
        ax.barh(
            y_pos,
            global_shap_values[feature_inds],
            0.7,
            left=left_pos,
            align='center',
            label=label,
            color=colormap(class_inds[i]),
        )
        left_pos += global_shap_values[feature_inds]

    # ax.set_yticklabels([style.INPUT_FEATURE_STYLE[feature_names[i]] for i in feature_inds])
    ax.legend(loc='lower right', fontsize=30)

    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(color=axis_color, labelcolor=axis_color)
    ax.set_yticks(
        range(len(feature_order)), [style.INPUT_FEATURE_STYLE[feature_names[i]] for i in feature_order], fontsize=30
    )
    ax.set_xlabel("mean (Shapley value) - (average impact on model output magnitude)", fontsize=30)
    plt.tight_layout()


def plot_shaply(model, X_test, class_labels, input_vars, plot_dir):

    labels = list(class_labels.keys())
    try:
        model2 = tf.keras.Model(model.jet_model.input, model.jet_model.output[0])
        model3 = tf.keras.Model(model.jet_model.input, model.jet_model.output[1])

        for explainer, name in [
            (shap.GradientExplainer(model2, X_test[:1000]), "GradientExplainer"),
        ]:
            print("... {0}: explainer.shap_values(X)".format(name))
            shap_values = explainer.shap_values(X_test[:1000])
            new = np.sum(shap_values, axis=1)
            print("... shap summary_plot classification")
            plt.clf()
            new = np.transpose(new, (2, 0, 1))
            shapPlot(new, input_vars, labels)
            plt.savefig(plot_dir + "/shap_summary_class.pdf", bbox_inches='tight')
            plt.savefig(plot_dir + "/shap_summary_class.png", bbox_inches='tight')

        for explainer, name in [
            (shap.GradientExplainer(model3, X_test[:1000]), "GradientExplainer"),
        ]:
            print("... {0}: explainer.shap_values(X)".format(name))
            shap_values = explainer.shap_values(X_test[:1000])
            new = np.sum(shap_values, axis=1)
            print("... shap summary_plot regression")
            plt.clf()
            labels = ["Regression"]
            new = np.transpose(new, (2, 0, 1))
            shapPlot(new, input_vars, labels)
            plt.savefig(plot_dir + "/shap_summary_reg.pdf", bbox_inches='tight')
            plt.savefig(plot_dir + "/shap_summary_reg.png", bbox_inches='tight')
            
    except AttributeError:
        print('Shapely not yet implemented for non-keras models')



def efficiency(y_pred, y_test, reco_pt_test, class_labels, plot_dir):

    save_dir = os.path.join(plot_dir, 'efficiencies')
    os.makedirs(save_dir, exist_ok=True)

    # pT coordinate points for plotting
    pt_points = [np.mean((PT_BINS[i], PT_BINS[i + 1])) for i in range(len(PT_BINS) - 1)]

    def plot_efficiency(df_wp_loose, df_wp_medium, df_wp_tight, plot_name):
        # Plot the efficiency
        fig, ax = plt.subplots(1, 1, figsize=style.FIGURE_SIZE)
        hep.cms.label(llabel=style.CMSHEADER_LEFT, rlabel=style.CMSHEADER_RIGHT, ax=ax, fontsize=style.CMSHEADER_SIZE)

        ax = df_wp_loose.plot.line(
            x="midpoints", y="wp_loose", yerr="err(wp_loose)", label="Loose (mistag = 50%)", color="blue", linestyle='-'
        )
        df_wp_medium.plot.line(
            x="midpoints",
            y="wp_medium",
            yerr="err(wp_medium)",
            label="Medium (eff = 20%)",
            ax=ax,
            color="orange",
            linestyle='-',
        )
        df_wp_tight.plot.line(
            x="midpoints", y="wp_tight", yerr="err(wp_tight)", label="Tight (eff = 10%)", ax=ax, color="green", linestyle='-'
        )
        ax.set_xlabel(r'Jet $p_T$ [GeV]')
        ax.set_ylabel('Tagging efficiency')
        ax.set_ylim(0.0, 1.3)
        # plt.xlim(0., 1000.)
        ax.legend()
        # Save the plot
        save_path = os.path.join(save_dir, plot_name)
        plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
        plt.savefig(f"{save_path}.png", bbox_inches='tight')
        plt.cla()
        plt.close()

    eff_selection = {
        'taus': [class_labels['taup'], class_labels['taum']],
        'b': [class_labels['b']],
    }

    for key in eff_selection.keys():
        selection = sum(y_test[:, idx] for idx in eff_selection[key]) > 0
        summed_y_true = sum(y_test[:, idx] for idx in eff_selection[key])
        summed_y_score = sum(y_pred[:, idx] for idx in eff_selection[key])

        # Compute FPR, TPR, and AUC
        fpr, tpr, thres = roc_curve(summed_y_true, summed_y_score)
        roc_auc = auc(fpr, tpr)

        # get working points for 50%, 20%, 10%
        wp_loose = thres[np.argmin(np.abs(np.array(fpr) - 0.5))]
        wp_medium = thres[np.argmin(np.abs(np.array(fpr) - 0.2))]
        wp_tight = thres[np.argmin(np.abs(np.array(fpr) - 0.1))]

        # some tricks to get proper uncertainties in the efficiencies
        data_eff = ak.to_dataframe(
            ak.Array({"truth": summed_y_true[selection], "pt": reco_pt_test[selection], "pred": summed_y_score[selection]})
        )
        data_eff["wp_loose"] = data_eff["pred"] > wp_loose
        data_eff["wp_medium"] = data_eff["pred"] > wp_medium
        data_eff["wp_tight"] = data_eff["pred"] > wp_tight

        h_wp_loose = histbook.Hist(histbook.split("pt", pt_points), histbook.cut("wp_loose"))
        h_wp_medium = histbook.Hist(histbook.split("pt", pt_points), histbook.cut("wp_medium"))
        h_wp_tight = histbook.Hist(histbook.split("pt", pt_points), histbook.cut("wp_tight"))
        h_wp_loose.fill(data_eff)
        h_wp_medium.fill(data_eff)
        h_wp_tight.fill(data_eff)
        df_wp_loose = h_wp_loose.pandas("wp_loose", error="normal")
        df_wp_medium = h_wp_medium.pandas("wp_medium", error="normal")
        df_wp_tight = h_wp_tight.pandas("wp_tight", error="normal")
        df_wp_loose["midpoints"] = [x[0].mid if isinstance(x[0], pandas.Interval) else np.nan for x in df_wp_loose.index]
        df_wp_medium["midpoints"] = [x[0].mid if isinstance(x[0], pandas.Interval) else np.nan for x in df_wp_medium.index]
        df_wp_tight["midpoints"] = [x[0].mid if isinstance(x[0], pandas.Interval) else np.nan for x in df_wp_tight.index]

        plot_name = f"{key}_efficiency"
        plot_efficiency(df_wp_loose, df_wp_medium, df_wp_tight, plot_name)

    return


def ROC_jets(y_pred, y_test, class_labels, plot_dir, process_label=None):
    """
    Plot combined ROC for light vs b, charm, gluon in one plot.
    """

    save_dir = os.path.join(plot_dir, 'roc_jets')
    os.makedirs(save_dir, exist_ok=True)

    light_idx = [class_labels['light']]
    targets = {
        'b': [class_labels['b']],
        'charm': [class_labels['charm']],
        'gluon': [class_labels['gluon']],
    }

    def compute_roc_inputs(signal_indices, background_indices):
        signal_mask = sum(y_test[:, idx] for idx in signal_indices) > 0
        background_mask = sum(y_test[:, idx] for idx in background_indices) > 0
        total_mask = signal_mask | background_mask

        signal_scores = sum(y_pred[:, idx] for idx in signal_indices)
        background_scores = sum(y_pred[:, idx] for idx in background_indices)
        total_scores = signal_scores + background_scores

        y_true = signal_mask[total_mask]
        y_score = (signal_scores / total_scores)[total_mask]
        return y_true, y_score

    # Collect all ROC data
    roc_data = []
    for label, bkg_idx in targets.items():
        y_true, y_score = compute_roc_inputs(light_idx, bkg_idx)
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        roc_data.append((tpr, fpr, roc_auc, label))

    # Plot all in one
    plt.figure(figsize=style.FIGURE_SIZE)
    hep.cms.label(llabel=style.CMSHEADER_LEFT, rlabel=style.CMSHEADER_RIGHT, fontsize=style.CMSHEADER_SIZE)

    for tpr, fpr, roc_auc, label in roc_data:
        formatted = f"light vs {label} (AUC = {roc_auc:.2f})"
        plt.plot(tpr, fpr, label=formatted, linewidth=style.LINEWIDTH)

    plt.grid(True)
    plt.xlabel('Signal Efficiency')
    plt.ylabel('Mistag Rate')
    plt.yscale('log')
    plt.ylim(1e-3, 1.1)
    leg = plt.legend(loc='lower right', fontsize=style.SMALL_SIZE + 3, title=process_label)
    leg._legend_box.align = "left"

    save_path = os.path.join(save_dir, "ROC_light_vs_all_jets")
    plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
    plt.savefig(f"{save_path}.png", bbox_inches='tight')
    plt.close()


# Helper functions for signal specific plotting


def filter_process(test_data, process_dir):
    """
    Filter jets from specific signal process to create plots for specified signal processes.
    Comparison done through concatenation of sets to be compared and np unique to check for duplicates.
    """
    train, test, class_labels = load_data(os.path.join("signal_process_data", process_dir), percentage=100)[:3]
    train, test = to_ML(train, class_labels), to_ML(test, class_labels)

    # apply unique to sets to be compared, since there tend to be duplicates
    process_data = np.unique(np.concatenate((train[0], test[0]), axis=0), axis=0)
    unique_test_data, indices_unique_test_data = np.unique(test_data, axis=0, return_index=True)
    comparison_data = np.concatenate((unique_test_data, process_data), axis=0)
    u, index, counts = np.unique(comparison_data, axis=0, return_index=True, return_counts=True)
    process_indices = index[counts == 2]
    filtered_indices = indices_unique_test_data[process_indices]

    return filtered_indices, train, test


# fancy signal process labels


def process_labels(process_key):
    processes = {
        'TT_PU200': r't$\bar{t}$',
        'ggHHbbbb_PU200': r'gg $\rightarrow$ HH $\rightarrow$ b$\bar{b}$b$\bar{b}$',
        'VBFHtt_PU200': r'VBF $\rightarrow$ H $\rightarrow$ t$\bar{t}$',
        'ggHHbbtt_PU200': r'gg $\rightarrow$ HH $\rightarrow$ b$\bar{b}$t$\bar{t}$',
        'ggHtt_PU200': r'gg $\rightarrow$ HH $\rightarrow$ t$\bar{t}$',
    }

    return processes[process_key]


# <<<<<<<<<<<<<<<<< end of plotting functions, call basic to plot all of them


def basic(model, signal_dirs):
    """
    Plot the basic ROCs for different classes. Does not reflect L1 rate
    Returns a dictionary of ROCs for each class
    """

    plot_dir = os.path.join(model.output_directory, "plots/training")

    ROC_dict = {class_label: 0 for class_label in model.class_labels}

    # Load the testing data
    X_test = np.load(f"{model.output_directory}/testing_data/X_test.npy")
    y_test = np.load(f"{model.output_directory}/testing_data/y_test.npy")
    truth_pt_test = np.load(f"{model.output_directory}/testing_data/truth_pt_test.npy")
    reco_pt_test = np.load(f"{model.output_directory}/testing_data/reco_pt_test.npy")

    model_outputs = model.predict(X_test)

    # Get classification outputs
    y_pred = model_outputs[0]
    pt_ratio = model_outputs[1]

    # Plot ROC curves
    ROC_dict = ROC(y_pred, y_test, model.class_labels, plot_dir, ROC_dict)
    class_pairs = []
    # Generate all possible pairs of classes
    for i in model.class_labels.keys():
        for j in model.class_labels.keys():
            if i != j:
                class_pair = [i, j]
                class_pairs.append(class_pair)

    # Make ROC binaries for complete test set and each signal process
    for i in range(-1, len(signal_dirs), 1):
        sample_plot_dir = os.path.join(model.output_directory, "plots/physics", f"binary_rocs_{signal_dirs[i]}")
        if i == -1:
            y_p, y_t = y_pred, y_test
            process_label = None
        else:
            signal_indices, sample_train, sample_test = filter_process(X_test, signal_dirs[i])
            sample_data = np.concatenate((sample_train[0], sample_test[0]), axis=0)
            sample_labels = np.concatenate((sample_train[1], sample_test[1]), axis=0)
            sample_preds = model.predict(sample_data)[0]
            y_p, y_t = y_pred[signal_indices], y_test[signal_indices]
            process_label = process_labels(signal_dirs[i])
            os.makedirs(binary_dir, exist_ok=True)

        # Plot the binary ROCs for each class pair
        for class_pair in class_pairs:
            binary_dir = os.path.join(sample_plot_dir, f"test_set") if i != -1 else plot_dir
            ROC_binary(y_p, y_t, model.class_labels, binary_dir, class_pair, process_label)
            if i != -1:
                binary_dir = os.path.join(sample_plot_dir, "full_sample")
                ROC_binary(sample_preds, sample_labels, model.class_labels, binary_dir, class_pair, process_label)

        # Add light vs b/charm/gluon combined plot
        binary_dir_test = os.path.join(sample_plot_dir, "test_set") if i != -1 else plot_dir
        ROC_jets(y_p, y_t, model.class_labels, binary_dir_test, process_label)
        ROC_taus(y_p, y_t, model.class_labels, binary_dir_test, process_label)

        if i != -1:
            binary_dir_full = os.path.join(sample_plot_dir, "full_sample")
            ROC_jets(sample_preds, sample_labels, model.class_labels, binary_dir_full, process_label)
            ROC_taus(sample_preds, sample_labels, model.class_labels, binary_dir_full, process_label)

    # Efficiencies
    efficiency(y_pred, y_test, reco_pt_test, model.class_labels, plot_dir)

    # Confusion matrix
    confusion(y_pred, y_test, model.class_labels, plot_dir)

    # Plot pt corrections
    pt_correction_hist(pt_ratio, truth_pt_test, reco_pt_test, plot_dir)

    # Plot input distributions
    plot_input_vars(X_test, y_test, model.input_vars, model.class_labels, plot_dir)

    # Plot inclusive response and individual flavor
    response(model.class_labels, y_test, truth_pt_test, reco_pt_test, pt_ratio, plot_dir)

    # Plot the rms of the residuals vs pt
    rms(model.class_labels, y_test, truth_pt_test, reco_pt_test, pt_ratio, plot_dir)

    # Plot the shaply feature importance
    plot_shaply(model, X_test, model.class_labels, model.input_vars, plot_dir)

    return ROC_dict
