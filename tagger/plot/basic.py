import json

#Third parties
import shap
import numpy as np
from qkeras.utils import load_qmodel
from sklearn.metrics import roc_curve, auc
from itertools import combinations
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import mplhep as hep
import tensorflow as tf
import tagger.plot.style as style

import os
from .common import PT_BINS
from .common import plot_histo
from scipy.stats import norm

style.set_style()

###### DEFINE ALL THE PLOTTING FUNCTIONS HERE!!!! THEY WILL BE CALLED IN basic() function >>>>>>>
def loss_history(plot_dir, history):
    fig,ax = plt.subplots(1,1,figsize=style.FIGURE_SIZE)
    hep.cms.label(llabel=style.CMSHEADER_LEFT,rlabel=style.CMSHEADER_RIGHT,ax=ax, fontsize=style.CMSHEADER_SIZE)
    ax.plot(history.history['loss'], label='Train Loss', linewidth=style.LINEWIDTH)
    ax.plot(history.history['val_loss'], label='Validation Loss',linewidth=style.LINEWIDTH)
    ax.grid(True)
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    ax.legend(loc='upper right')

    save_path = os.path.join(plot_dir, "loss_history")
    plt.savefig(f"{save_path}.png", bbox_inches='tight')
    plt.savefig(f"{save_path}.pdf", bbox_inches='tight')

def ROC_taus(y_pred, y_test, class_labels, plot_dir):
    """
    Plot ROC curves for taus vs jets and leptons
    """

    save_dir = os.path.join(plot_dir, 'roc_taus')
    os.makedirs(save_dir, exist_ok=True)

    # Define class label groups
    tau_indices = [class_labels['taup'], class_labels['taum']]
    jet_indices = [class_labels[key] for key in ['b', 'charm', 'light', 'gluon']]
    lepton_indices = [class_labels[key] for key in ['muon', 'electron']]

    #Function to calculate the roc inputs for plotting
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

    def plot_roc(y_true, y_score, label, save_path):
        """
        Plot and save a single ROC curve.
        """
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=style.FIGURE_SIZE)
        hep.cms.label(
            llabel=style.CMSHEADER_LEFT,
            rlabel=style.CMSHEADER_RIGHT,
            fontsize=style.CMSHEADER_SIZE
        )
        plt.plot(tpr, fpr, label=f'{label} (AUC = {roc_auc:.2f})', linewidth=style.LINEWIDTH)
        plt.grid(True)
        plt.xlabel('Signal Efficiency')
        plt.ylabel('Mistag Rate')
        plt.yscale('log')
        plt.ylim(1e-3, 1.1)
        plt.legend(loc='lower right')

        plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
        plt.savefig(f"{save_path}.png", bbox_inches='tight')
        plt.close()

    #Plot taus versus jets ROC
    y_true_taus_vs_jets, y_score_taus_vs_jets = compute_roc_inputs(y_pred, y_test, tau_indices, jet_indices)
    plot_roc(y_true_taus_vs_jets, y_score_taus_vs_jets, r'$\tau = \tau_{{h}}^{{+}} + \tau_{{h}}^{{-}}$ vs Jets (b, charm, light, gluon)', os.path.join(save_dir, "ROC_taus_vs_jets"))

    #Plot taus versus leptons ROC
    y_true_taus_vs_leptons, y_score_taus_vs_leptons = compute_roc_inputs(y_pred, y_test, tau_indices, lepton_indices)
    plot_roc(y_true_taus_vs_leptons, y_score_taus_vs_leptons, r'$\tau = \tau_{{h}}^{{+}} + \tau_{{h}}^{{-}}$ vs Leptons (muon, electron)', os.path.join(save_dir, "ROC_taus_vs_leptons"))


def ROC_binary(y_pred, y_test, class_labels, plot_dir, class_pair):
    """
    Generate ROC curves comparing between two specific class labels.
    """

    save_dir = os.path.join(plot_dir, 'roc_binary')
    os.makedirs(save_dir, exist_ok=True)

    # Ensure class_pair exists in class_labels
    assert class_pair[0] in class_labels and class_pair[1] in class_labels, \
        "Both class_pair labels must exist in class_labels"

    # Get indices of the classes to compare
    idx1, idx2 = class_labels[class_pair[0]], class_labels[class_pair[1]]

    # Select true labels and predicted probabilities for the selected classes
    y_true1, y_true2 = y_test[:, idx1], y_test[:, idx2]
    y_score1, y_score2 = y_pred[:, idx1], y_pred[:, idx2]

    # Combine the labels and scores for binary classification
    selection = (y_true1 == 1) | (y_true2 == 1)
    y_true_binary = y_true1[selection]
    y_score_binary = y_score1[selection] / (y_score1[selection] + y_score2[selection])  # Normalized probabilities

    # Compute FPR, TPR, and AUC
    fpr, tpr, _ = roc_curve(y_true_binary, y_score_binary)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    fig,ax = plt.subplots(1,1,figsize=style.FIGURE_SIZE)
    hep.cms.label(llabel=style.CMSHEADER_LEFT,rlabel=style.CMSHEADER_RIGHT,ax=ax,fontsize=style.CMSHEADER_SIZE)
    ax.plot(tpr, fpr, label=f'{style.CLASS_LABEL_STYLE[class_pair[0]]} vs {style.CLASS_LABEL_STYLE[class_pair[1]]} (AUC = {roc_auc:.2f})',
             color='blue', linewidth=5)
    ax.grid(True)
    ax.set_ylabel('Mistag Rate')
    ax.set_xlabel('Signal Efficiency')
    ax.legend(loc='lower right')
    ax.set_yscale('log')
    ax.set_ylim([1e-3, 1.1])

    # Save the plot
    save_path = os.path.join(save_dir, f"ROC_{class_pair[0]}_vs_{class_pair[1]}")
    plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
    plt.savefig(f"{save_path}.png", bbox_inches='tight')
    plt.close()

def ROC(y_pred, y_test, class_labels, plot_dir, ROC_dict):
    # Create a colormap for unique colors
    colormap = cm.get_cmap('Set1', len(class_labels))  # Use 'tab10' with enough colors

    # Create a plot for ROC curves
    fig,ax = plt.subplots(1,1,figsize=style.FIGURE_SIZE)
    hep.cms.label(llabel=style.CMSHEADER_LEFT,rlabel=style.CMSHEADER_RIGHT,ax=ax, fontsize=style.CMSHEADER_SIZE)
    for i, class_label in enumerate(class_labels):

        # Get true labels and predicted probabilities for the current class
        y_true = y_test[:, i]  # Extract the one-hot column for the current class
        y_score = y_pred[:, i] # Predicted probabilities for the current class

        # Compute FPR, TPR, and AUC
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        ROC_dict[class_label] = roc_auc
        # Plot the ROC curve for the current class
        ax.plot(tpr, fpr, label=f'{style.CLASS_LABEL_STYLE[class_label]} (AUC = {roc_auc:.2f})',
                 color=colormap(i), linewidth=style.LINEWIDTH)

    # Plot formatting
    ax.grid(True)
    ax.set_ylabel('Mistag Rate')
    ax.set_xlabel('Signal Efficiency')

    auc_list = [value for key,value in ROC_dict.items()]
    handles, labels = plt.gca().get_legend_handles_labels()
    order = np.argsort(auc_list)
    ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc='upper left',ncol=2,fontsize=style.SMALL_SIZE-3)

    ax.set_yscale('log')
    ax.set_ylim([1e-3, 1.1])

    # Save the plot
    save_path = os.path.join(plot_dir, "basic_ROC")
    plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
    plt.savefig(f"{save_path}.png", bbox_inches='tight')
    plt.close()

    return ROC_dict

def confusion(y_pred, y_test, class_labels, plot_dir):
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    from mpl_toolkits.axes_grid1 import make_axes_locatable

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

    plot_histo([truth_pt_test,reco_pt_test,np.multiply(reco_pt_test,pt_ratio)],
                ['Truth','Reconstructed','NN Predicted'],'',r'$p_T$ [GeV]','a.u',range=(0,300))
    save_path = os.path.join(plot_dir, "pt_hist")
    plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
    plt.savefig(f"{save_path}.png", bbox_inches='tight')
    plt.close()

    return

def plot_input_vars(X_test, input_vars, plot_dir):

    save_dir = os.path.join(plot_dir,'inputs')
    os.makedirs(save_dir, exist_ok=True)

    for i in range(len(input_vars)):
        plot_histo([X_test[:,:,i].flatten()],
                [style.INPUT_FEATURE_STYLE[input_vars[i]]],'',style.INPUT_FEATURE_STYLE[input_vars[i]],'a.u',range=(np.min(X_test[:,:,i]),np.max(X_test[:,:,i])))
        save_path = os.path.join(save_dir, input_vars[i])
        plt.savefig(f"{save_path}.png", bbox_inches='tight')
        plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
        plt.close()

def get_response(truth_pt, reco_pt, pt_ratio):

    #Calculate the regressed pt
    regressed_pt = np.multiply(reco_pt, pt_ratio)

    #to calculate response
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

            uncorrected_errors.append(uncorrected_std/np.sqrt(n_events))
            regressed_errors.append(regressed_std/np.sqrt(n_events))
        else:
            # No events in bin
            uncorrected_errors.append(0)
            regressed_errors.append(0)

    return uncorrected_response, regressed_response, uncorrected_errors, regressed_errors

def response(class_labels, y_test, truth_pt_test, reco_pt_test, pt_ratio, plot_dir, tau_training=False):
    save_dir = os.path.join(plot_dir, 'response')
    os.makedirs(save_dir, exist_ok=True)

    # pT coordinate points for plotting
    pt_points = [np.mean((PT_BINS[i], PT_BINS[i + 1])) for i in range(len(PT_BINS) - 1)]

    def plot_response(uncorrected_response, regressed_response, uncorrected_errors, regressed_errors, flavor, plot_name):

        # Plot the response
        fig,ax = plt.subplots(1,1,figsize=style.FIGURE_SIZE)
        hep.cms.label(llabel=style.CMSHEADER_LEFT,rlabel=style.CMSHEADER_RIGHT,ax=ax,fontsize=style.CMSHEADER_SIZE)
        ax.errorbar(pt_points, uncorrected_response, yerr=uncorrected_errors, fmt='o', label=f"Uncorrected - {style.CLASS_LABEL_STYLE[flavor]}", capsize=4,ms=8,elinewidth=3)
        ax.errorbar(pt_points, regressed_response, yerr=regressed_errors, fmt='o', label=f"Regressed - {style.CLASS_LABEL_STYLE[flavor]}", capsize=4,ms=8,elinewidth=3)

        ax.set_xlabel(r"Jet $p_T^{Gen}$ [GeV]")
        ax.set_ylabel("Response (Reco/Gen)")
        ax.legend()
        ax.grid()

        # Save the plot
        save_path = os.path.join(save_dir, plot_name)
        plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
        plt.savefig(f"{save_path}.png", bbox_inches='tight')
        plt.close()


    # Inclusive response
    uncorrected_response, regressed_response, uncorrected_errors, regressed_errors = get_response(truth_pt_test, reco_pt_test, pt_ratio)
    plot_response(uncorrected_response, regressed_response, uncorrected_errors, regressed_errors, flavor='inclusive', plot_name="inclusive_response")

    #Flavor-wise response
    for flavor in class_labels.keys():
        idx = class_labels[flavor]
        flavor_selection = y_test[:,idx] == 1

        uncorrected_response, regressed_response, uncorrected_errors, regressed_errors = get_response(truth_pt_test[flavor_selection], reco_pt_test[flavor_selection], pt_ratio[flavor_selection])
        plot_response(uncorrected_response, regressed_response, uncorrected_errors, regressed_errors, flavor=flavor, plot_name=f"{flavor}_response")

    #Taus, jets, leptons rms
    if(not tau_training):
        rms_selection = {
            'taus': [class_labels['taup'], class_labels['taum']],
            'jets': [class_labels[key] for key in ['b', 'charm', 'light', 'gluon']],
            'leptons': [class_labels[key] for key in ['muon', 'electron']]
        }
    else:
        rms_selection = {
            'taus': [class_labels['taus']],
            'jets': [class_labels['light']],
            }


    for key in rms_selection.keys():
        selection = sum(y_test[:, idx] for idx in rms_selection[key]) > 0

        uncorrected_response, regressed_response, uncorrected_errors, regressed_errors = get_response(truth_pt_test[selection], reco_pt_test[selection], pt_ratio[selection])
        plot_response(uncorrected_response, regressed_response, uncorrected_errors, regressed_errors, flavor=key, plot_name=f"{key}_response")

    return

def get_rms(truth_pt, reco_pt, pt_ratio):

    #Calculate the regressed pt
    regressed_pt = np.multiply(reco_pt, pt_ratio)

    #Get the residuals
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
        mu_uncorr, sigma_uncorr = norm.fit(un_corrected_res[selection]/pt_avg)
        mu_reg, sigma_reg = norm.fit(regressed_res[selection]/pt_avg)

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

def rms(class_labels, y_test, truth_pt_test, reco_pt_test, pt_ratio, plot_dir, tau_training):

    save_dir = os.path.join(plot_dir, 'residual_rms')
    os.makedirs(save_dir, exist_ok=True)

    # pT coordinate points for plotting
    pt_points = [np.mean((PT_BINS[i], PT_BINS[i + 1])) for i in range(len(PT_BINS) - 1)]

    def plot_rms(uncorrected_rms, regressed_rms, uncorrected_rms_err, regressed_rms_err, flavor, plot_name):

        # Plot the response
        fig,ax = plt.subplots(1,1,figsize=style.FIGURE_SIZE)
        hep.cms.label(llabel=style.CMSHEADER_LEFT,rlabel=style.CMSHEADER_RIGHT,ax=ax,fontsize=style.CMSHEADER_SIZE)
        ax.errorbar(pt_points, uncorrected_rms, yerr=uncorrected_rms_err, fmt='o', label=r"Uncorrected $\sigma$- {}".format(style.CLASS_LABEL_STYLE[flavor]), capsize=4,ms=8,elinewidth=3)
        ax.errorbar(pt_points, regressed_rms, yerr=regressed_rms_err, fmt='o', label=r"Regressed $\sigma$ - {}".format(style.CLASS_LABEL_STYLE[flavor]), capsize=4,ms=8,elinewidth=3)

        ax.set_xlabel(r"Jet $p_T^{Gen}$ [GeV]")
        ax.set_ylabel(r"$\sigma_{(p_T^{Gen} - p_T^{Reco})/p_T^{Gen}}$")
        ax.legend()
        ax.grid(True)

        # Save the plot
        save_path = os.path.join(save_dir, plot_name)
        plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
        plt.savefig(f"{save_path}.png", bbox_inches='tight')
        plt.close()

    #Inclusive rms
    uncorrected_rms, regressed_rms, uncorrected_rms_err, regressed_rms_err = get_rms(truth_pt_test, reco_pt_test, pt_ratio)
    plot_rms(uncorrected_rms, regressed_rms, uncorrected_rms_err, regressed_rms_err, flavor='inclusive', plot_name='inclusive')

    #Flavor-wise rms
    for flavor in class_labels.keys():
        idx = class_labels[flavor]
        flavor_selection = y_test[:,idx] == 1

        uncorrected_rms, regressed_rms, uncorrected_rms_err, regressed_rms_err = get_rms(truth_pt_test[flavor_selection], reco_pt_test[flavor_selection], pt_ratio[flavor_selection])
        plot_rms(uncorrected_rms, regressed_rms, uncorrected_rms_err, regressed_rms_err, flavor=flavor, plot_name=f"{flavor}_rms")

    #Taus, jets, leptons rms
    if(not tau_training):
        rms_selection = {
            'taus': [class_labels['taup'], class_labels['taum']],
            'jets': [class_labels[key] for key in ['b', 'charm', 'light', 'gluon']],
            'leptons': [class_labels[key] for key in ['muon', 'electron']]
        }
    else:
        rms_selection = {
            'taus': [class_labels['taus']],
            'jets': [class_labels['light']],
            }

    for key in rms_selection.keys():
        selection = sum(y_test[:, idx] for idx in rms_selection[key]) > 0

        uncorrected_rms, regressed_rms, uncorrected_rms_err, regressed_rms_err = get_rms(truth_pt_test[selection], reco_pt_test[selection], pt_ratio[selection])
        plot_rms(uncorrected_rms, regressed_rms, uncorrected_rms_err, regressed_rms_err, flavor=key, plot_name=f"{key}_rms")

    return

def shapPlot(shap_values, feature_names, class_names):
    fig,ax = plt.subplots(1,1,figsize=style.FIGURE_SIZE)
    feature_order = np.argsort(np.sum(np.mean(np.abs(shap_values), axis=1), axis=0))
    num_features = (shap_values[0].shape[1])
    feature_inds = feature_order
    y_pos = np.arange(len(feature_inds))
    left_pos = np.zeros(len(feature_inds))

    axis_color="#333333"
    class_inds = np.argsort([-np.abs(shap_values[i]).mean() for i in range(len(shap_values))])
    colormap = cm.get_cmap('Set1', len(class_names))  # Use 'tab10' with enough colors

    for i, ind in enumerate(class_inds):
        global_shap_values = np.abs(shap_values[ind]).mean(0)
        label = style.CLASS_LABEL_STYLE[class_names[ind]]
        ax.barh(y_pos, global_shap_values[feature_inds], 0.7, left=left_pos, align='center',label=label,color=colormap(class_inds[i]))
        left_pos += global_shap_values[feature_inds]

    #ax.set_yticklabels([style.INPUT_FEATURE_STYLE[feature_names[i]] for i in feature_inds])
    ax.legend(loc='lower right',fontsize=30)

    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(color=axis_color, labelcolor=axis_color)
    ax.set_yticks(range(len(feature_order)), [style.INPUT_FEATURE_STYLE[feature_names[i]] for i in feature_order],fontsize=30)
    ax.set_xlabel("mean (shapley value) - (average impact on model output magnitude)",fontsize=30)
    plt.tight_layout()

def plot_shaply(model, X_test, class_labels, input_vars, plot_dir):

    labels = list(class_labels.keys())
    model2 = tf.keras.Model(model.input, model.output[0])
    model3 = tf.keras.Model(model.input, model.output[1])

    for explainer, name  in [(shap.GradientExplainer(model2, X_test[:1000]), "GradientExplainer"), ]:
        print("... {0}: explainer.shap_values(X)".format(name))
        shap_values = explainer.shap_values(X_test[:1000])
        new = np.sum(shap_values, axis = 1)
        print("... shap summary_plot classification")
        plt.clf()
        new = np.transpose(new, (2,0,1))
        shapPlot(new, input_vars, labels)
        plt.savefig(plot_dir+"/shap_summary_class.pdf",bbox_inches='tight')
        plt.savefig(plot_dir+"/shap_summary_class.png",bbox_inches='tight')

    for explainer, name  in [(shap.GradientExplainer(model3, X_test[:1000]), "GradientExplainer"), ]:
        print("... {0}: explainer.shap_values(X)".format(name))
        shap_values = explainer.shap_values(X_test[:1000])
        new = np.sum(shap_values, axis = 1)
        print("... shap summary_plot regression")
        plt.clf()
        labels = ["Regression"]
        new = np.transpose(new, (2,0,1))
        shapPlot(new, input_vars, labels)
        plt.savefig(plot_dir+"/shap_summary_reg.pdf",bbox_inches='tight')
        plt.savefig(plot_dir+"/shap_summary_reg.png",bbox_inches='tight')

# <<<<<<<<<<<<<<<<< end of plotting functions, call basic to plot all of them
def basic(model_dir, tau_training):
    """
    Plot the basic ROCs for different classes. Does not reflect L1 rate
    Returns a dictionary of ROCs for each class
    """

    plot_dir = os.path.join(model_dir, "plots/training")

    #Load the metadata for class_label
    with open(f"{model_dir}/class_label.json", 'r') as file: class_labels = json.load(file)
    with open(f"{model_dir}/input_vars.json", 'r') as file: input_vars = json.load(file)

    ROC_dict = {class_label : 0 for class_label in class_labels}

    #Load the testing data
    X_test = np.load(f"{model_dir}/testing_data/X_test.npy")
    y_test = np.load(f"{model_dir}/testing_data/y_test.npy")
    truth_pt_test = np.load(f"{model_dir}/testing_data/truth_pt_test.npy")
    reco_pt_test = np.load(f"{model_dir}/testing_data/reco_pt_test.npy")

    #Load model
    model = load_qmodel(f"{model_dir}/model/saved_model.h5")
    model_outputs = model.predict(X_test)

    #Get classification outputs
    y_pred = model_outputs[0]
    pt_ratio = model_outputs[1].flatten()

    #Plot ROC curves
    ROC_dict = ROC(y_pred, y_test, class_labels, plot_dir,ROC_dict)

    #Generate all possible pairs of classes
    for i in class_labels.keys():
        for j in class_labels.keys():
            if i != j:
                class_pair = (i,j)
                ROC_binary(y_pred, y_test, class_labels, plot_dir, class_pair)

    if(not tau_training):
        #ROC for taus versus jets and taus versus leptons
        ROC_taus(y_pred, y_test, class_labels, plot_dir)

    # Confusion matrix
    confusion(y_pred, y_test, class_labels, plot_dir)

    #Plot pt corrections
    pt_correction_hist(pt_ratio, truth_pt_test, reco_pt_test, plot_dir)

    #Plot input distributions
    plot_input_vars(X_test, input_vars, plot_dir)

    #Plot inclusive response and individual flavor
    response(class_labels, y_test, truth_pt_test, reco_pt_test, pt_ratio, plot_dir, tau_training=tau_training)

    #Plot the rms of the residuals vs pt
    rms(class_labels, y_test, truth_pt_test, reco_pt_test, pt_ratio, plot_dir, tau_training=tau_training)

    #Plot the shaply feature importance
    plot_shaply(model, X_test, class_labels, input_vars, plot_dir)

    return ROC_dict
