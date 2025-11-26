import os, json
import gc
from argparse import ArgumentParser

import awkward as ak
import numpy as np
import uproot
import hist
from hist import Hist

import matplotlib.pyplot as plt
import matplotlib
import mplhep as hep
import tagger.plot.style as style
from tagger.data.tools import constituents_mask

from tensorflow.keras.models import Model

def get_pt_weights(model, jet_nn_inputs, jet_pt):
    pt_weights_model = Model(inputs=model.jet_model.input, outputs=model.jet_model.get_layer('pt_weights_output').output)

    #Get the pt weights from the model
    mask = constituents_mask(jet_nn_inputs, 10)
    pt_mask = mask[:,:, 0]
    constitunts_pt = jet_nn_inputs[:, :, 0]
    inverse_jet_pt = (1.0 / jet_pt).reshape(-1,1)

    pt_weights = pt_weights_model.predict([jet_nn_inputs,
        mask,
        pt_mask,
        constitunts_pt,
        inverse_jet_pt])

    return pt_weights

def plot_2D_histogram(pt_weights, x_var, var_name, binning, save_path):

    fig, ax = plt.subplots(1, 1, figsize=style.FIGURE_SIZE)

    # Histogram
    h = ax.hist2d(
        x_var,
        pt_weights,
        bins=binning,
        cmap="viridis"
    )

    ax.set_xlabel(var_name)
    ax.set_ylabel(r"$p_{T}$ weights")

    hep.cms.label(
        llabel=style.CMSHEADER_LEFT,
        rlabel=style.CMSHEADER_RIGHT,
        ax=ax,
        fontsize=style.MEDIUM_SIZE
    )

    # Colorbar
    cbar = plt.colorbar(h[3], ax=ax, label="Entries")

    # Reduce top padding since we handled spacing manually
    plt.tight_layout()

    plt.savefig(save_path + ".png")
    plt.savefig(save_path + ".pdf")

    return


def pt_weights_plotting(model, inputs, plot_path):

    # Unpack inputs
    X_test, y_test, reco_pt_test = inputs
    pt_weights = get_pt_weights(model, X_test, reco_pt_test)

    plot_path = os.path.join(plot_path, "pt_weights")
    os.makedirs(plot_path, exist_ok=True)

    plot_2D_histogram(
        pt_weights.flatten(),
        X_test[:, :, 0].flatten(),
        r"Constituents $p_{T}$ [GeV]",
        binning=(20, 20),
        save_path=os.path.join(plot_path, "pt_weights_vs_pt"),
        )
    plot_2D_histogram(
        pt_weights.flatten(),
        X_test[:, :, 3].flatten(),
        r"Constituents $\Delta \eta$",
        binning=(10, 10),
        save_path=os.path.join(plot_path, "pt_weights_vs_eta"),
        )

    class_labels = model.class_labels
    for flav, i in class_labels.items():
        class_mask = y_test == i
        plot_2D_histogram(
            pt_weights[class_mask].flatten(),
            X_test[:, :, 0][class_mask].flatten(),
            r"Constituents $p_{T}$ [GeV]",
            binning=(20, 20),
            save_path=os.path.join(plot_path, f"pt_weights_vs_pt_{flav}"),
            )
        plot_2D_histogram(
            pt_weights[class_mask].flatten(),
            X_test[:, :, 3][class_mask].flatten(),
            r"Constituents $\Delta \eta$",
            binning=(10, 10),
            save_path=os.path.join(plot_path, f"pt_weights_vs_eta_{flav}"),
            )
