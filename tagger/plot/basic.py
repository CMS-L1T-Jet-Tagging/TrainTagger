import json

#Third parties
import numpy as np
from qkeras.utils import load_qmodel
from sklearn.metrics import roc_curve, auc
from itertools import combinations
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import mplhep as hep

from tagger.plot.style import *

import os
from .common import PT_BINS
from .common import plot_histo
from scipy.stats import norm

set_style()

###### DEFINE ALL THE PLOTTING FUNCTIONS HERE!!!! THEY WILL BE CALLED IN basic() function >>>>>>>
def loss_history(plot_dir, history):
    fig,ax = plt.subplots(1,1,figsize=FIGURE_SIZE)
    hep.cms.label(llabel=CMSHEADER_LEFT,rlabel=CMSHEADER_RIGHT,ax=ax)
    ax.plot(history.history['loss'], label='Train Loss', linewidth=3)
    ax.plot(history.history['val_loss'], label='Validation Loss',linewidth=3)
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    ax.legend(loc='upper right')

    save_path = os.path.join(plot_dir, "loss_history")
    plt.savefig(f"{save_path}.png", bbox_inches='tight')
    plt.savefig(f"{save_path}.pdf", bbox_inches='tight')

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
    fig,ax = plt.subplots(1,1,figsize=FIGURE_SIZE)
    hep.cms.label(llabel=CMSHEADER_LEFT,rlabel=CMSHEADER_RIGHT,ax=ax)
    ax.plot(tpr, fpr, label=f'{class_pair[0]} vs {class_pair[1]} (AUC = {roc_auc:.2f})',
             color='blue', linewidth=5)
    ax.grid(True)
    ax.set_ylabel('False Positive Rate')
    ax.set_xlabel('True Positive Rate')
    ax.legend(loc='lower right')
    ax.set_yscale('log')
    ax.set_ylim([1e-3, 1.1])

    # Save the plot
    save_path = os.path.join(save_dir, f"ROC_{class_pair[0]}_vs_{class_pair[1]}")
    plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
    plt.savefig(f"{save_path}.png", bbox_inches='tight')
    plt.close()

def ROC(y_pred, y_test, class_labels, plot_dir,ROC_dict):
    # Create a colormap for unique colors
    colormap = cm.get_cmap('Set1', len(class_labels))  # Use 'tab10' with enough colors

    # Create a plot for ROC curves
    fig,ax = plt.subplots(1,1,figsize=FIGURE_SIZE)
    hep.cms.label(llabel=CMSHEADER_LEFT,rlabel=CMSHEADER_RIGHT,ax=ax)
    for i, class_label in enumerate(class_labels):

        # Get true labels and predicted probabilities for the current class
        y_true = y_test[:, i]  # Extract the one-hot column for the current class
        y_score = y_pred[:, i] # Predicted probabilities for the current class

        # Compute FPR, TPR, and AUC
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        ROC_dict[class_label] = roc_auc
        # Plot the ROC curve for the current class
        ax.plot(tpr, fpr, label=f'{class_label} (AUC = {roc_auc:.2f})',
                 color=colormap(i), linewidth=5)

    # Plot formatting
    ax.grid(True)
    ax.set_ylabel('False Positive Rate')
    ax.set_xlabel('True Positive Rate')

    auc_list = [value for key,value in ROC_dict.items()]
    handles, labels = plt.gca().get_legend_handles_labels()
    order = np.argsort(auc_list)
    ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc='upper left',ncol=2)


    ax.set_yscale('log')
    ax.set_ylim([1e-3, 1.1])

    # Save the plot
    save_path = os.path.join(plot_dir, "basic_ROC")
    plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
    plt.savefig(f"{save_path}.png", bbox_inches='tight')
    plt.close()

    return ROC_dict

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
                [input_vars[i]],'',input_vars[i],'a.u',range=(np.min(X_test[:,:,i]),np.max(X_test[:,:,i])))
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

def response(class_labels, y_test, truth_pt_test, reco_pt_test, pt_ratio, plot_dir):
    save_dir = os.path.join(plot_dir, 'response')
    os.makedirs(save_dir, exist_ok=True)

    # pT coordinate points for plotting
    pt_points = [np.mean((PT_BINS[i], PT_BINS[i + 1])) for i in range(len(PT_BINS) - 1)]

    def plot_response(uncorrected_response, regressed_response, uncorrected_errors, regressed_errors, flavor, plot_name):

        # Plot the response
        fig,ax = plt.subplots(1,1,figsize=FIGURE_SIZE)
        hep.cms.label(llabel=CMSHEADER_LEFT,rlabel=CMSHEADER_RIGHT,ax=ax)
        ax.errorbar(pt_points, uncorrected_response, yerr=uncorrected_errors, fmt='o', label=f"Uncorrected - {flavor}", capsize=4,ms=8,elinewidth=3)
        ax.errorbar(pt_points, regressed_response, yerr=regressed_errors, fmt='o', label=f"Regressed - {flavor}", capsize=4,ms=8,elinewidth=3)

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

def rms(class_labels, y_test, truth_pt_test, reco_pt_test, pt_ratio, plot_dir):

    save_dir = os.path.join(plot_dir, 'residual_rms')
    os.makedirs(save_dir, exist_ok=True)

    # pT coordinate points for plotting
    pt_points = [np.mean((PT_BINS[i], PT_BINS[i + 1])) for i in range(len(PT_BINS) - 1)]

    def plot_rms(uncorrected_rms, regressed_rms, uncorrected_rms_err, regressed_rms_err, flavor, plot_name):

        # Plot the response
        fig,ax = plt.subplots(1,1,figsize=FIGURE_SIZE)
        hep.cms.label(llabel=CMSHEADER_LEFT,rlabel=CMSHEADER_RIGHT,ax=ax)
        ax.errorbar(pt_points, uncorrected_rms, yerr=uncorrected_rms_err, fmt='o', label=r"Uncorrected $\sigma$- {}".format(flavor), capsize=4,ms=8,elinewidth=3)
        ax.errorbar(pt_points, regressed_rms, yerr=regressed_rms_err, fmt='o', label=r"Regressed $\sigma$ - {}".format(flavor), capsize=4,ms=8,elinewidth=3)

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

    return
# <<<<<<<<<<<<<<<<< end of plotting functions, call basic to plot all of them

def basic(model_dir):
    """
    Plot the basic ROCs for different classes. Does not reflect L1 rate
    Returns a dictionary of ROCs for each class
    """
    
    plot_dir = os.path.join(model_dir, "plots/training")

    #Load the metada for class_label
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

    #Plot pt corrections
    pt_correction_hist(pt_ratio, truth_pt_test, reco_pt_test, plot_dir)

    #Plot input distributions
    plot_input_vars(X_test, input_vars, plot_dir)

    #Plot inclusive response and individual flavor
    response(class_labels, y_test, truth_pt_test, reco_pt_test, pt_ratio, plot_dir)
    
    #Plot the rms of the residuals vs pt
    rms(class_labels, y_test, truth_pt_test, reco_pt_test, pt_ratio, plot_dir)

    return ROC_dict