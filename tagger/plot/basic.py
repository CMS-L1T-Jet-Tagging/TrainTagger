import json

#Third parties
import numpy as np
from qkeras.utils import load_qmodel
from sklearn.metrics import roc_curve, auc

#For plotting
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.pyplot import cm
import mplhep as hep
plt.style.use(hep.style.ROOT)

#Plotting default config
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'medium',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'medium',
         'ytick.labelsize':'medium'}
pylab.rcParams.update(params)

import os

def loss_history(plot_dir, history):
    plt.plot(history.history['loss'], label='Train Loss', linewidth=3)
    plt.plot(history.history['val_loss'], label='Validation Loss',linewidth=3)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')

    save_path = os.path.join(plot_dir, "loss_history")
    plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
    plt.savefig(f"{save_path}.png", bbox_inches='tight')

def basic_ROC(model_dir):
    """
    Plot the basic ROCs for different classes. Does not reflect L1 rate
    """

    plot_dir = os.path.join(model_dir, "plots")

    #Load the metada for class_label
    with open(f"{model_dir}/class_label.json", 'r') as file: class_labels = json.load(file)

    #Load the testing data & model
    X_test = np.load(f"{model_dir}/X_test.npy")
    y_test = np.load(f"{model_dir}/y_test.npy")
    
    model = load_qmodel(f"{model_dir}/saved_model.h5")
    model_outputs = model.predict(X_test)

    #Get classification outputs
    y_pred = model_outputs[0]

    # Create a colormap for unique colors
    colormap = cm.get_cmap('Set1', len(class_labels))  # Use 'tab10' with enough colors

    # Create a plot for ROC curves
    plt.figure(figsize=(16, 16))
    for i, class_label in enumerate(class_labels):
        
        # Get true labels and predicted probabilities for the current class
        y_true = y_test[:, i]  # Extract the one-hot column for the current class
        y_score = y_pred[:, i] # Predicted probabilities for the current class

        # Compute FPR, TPR, and AUC
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        # Plot the ROC curve for the current class
        plt.plot(tpr, fpr, label=f'{class_label} (AUC = {roc_auc:.2f})',
                 color=colormap(i), linewidth=5)

    # Plot formatting
    plt.grid(True)
    plt.ylabel('False Positive Rate')
    plt.xlabel('True Positive Rate')
    hep.cms.text("Phase 2 Simulation")
    hep.cms.lumitext("PU 200 (14 TeV)")
    plt.legend(loc='lower right')

    plt.yscale('log')
    plt.ylim([1e-3, 1.1])

    # Save the plot
    save_path = os.path.join(plot_dir, "basic_ROC")
    plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
    plt.savefig(f"{save_path}.png", bbox_inches='tight')
    plt.close()

    return

def pt_correction_hist(plot_dir, model, data_test):
    """
    Plot the histograms of truth pt, reconstructed (uncorrected) pt, and corrected pt
    """

    return

def rms(plot_dir):
    """
    
    """
    return