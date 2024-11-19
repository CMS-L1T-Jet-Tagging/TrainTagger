import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.pyplot import cm
import mplhep as hep
plt.style.use(hep.style.ROOT)

import numpy as np
from qkeras.utils import load_qmodel

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

    #Load the testing data & model
    X_test = np.load(f"{model_dir}/X_test.npy")
    y_test = np.load(f"{model_dir}/y_test.npy")
    
    model = load_qmodel(f"{model_dir}/saved_model.h5")


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