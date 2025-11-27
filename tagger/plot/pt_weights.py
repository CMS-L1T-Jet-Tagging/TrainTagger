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

def get_pt_weights(model, jet_nn_inputs, jet_pt, layer_name):
    pt_weights_model = Model(inputs=model.jet_model.input, outputs=model.jet_model.get_layer(layer_name).output)

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

def plot_1D_histogram(pt_weights, pt_corretion, binning, save_path):
    # show distribution of pt weights
    fig, ax = plt.subplots(1, 1, figsize=style.FIGURE_SIZE)
    h = ax.hist(
        pt_weights,
        bins=binning,
        histtype='step',
        color='blue',
    )
    ax.set_xlabel(rf"$p_{T}$ {pt_corretion}")
    ax.set_ylabel("Entries")
    hep.cms.label(
        llabel=style.CMSHEADER_LEFT,
        rlabel=style.CMSHEADER_RIGHT,
        ax=ax,
        fontsize=style.MEDIUM_SIZE
    )

    # Reduce top padding since we handled spacing manually
    plt.tight_layout()

    plt.savefig(save_path + ".png")
    plt.savefig(save_path + ".pdf")
    return

def plot_2D_histogram(pt_weights, pt_corretion, x_var, var_name, caps, binning, save_path):

    fig, ax = plt.subplots(1, 1, figsize=style.FIGURE_SIZE)
    mask = (x_var != 0)
    x_var = np.clip(x_var[mask], caps[0], caps[1])
    pt_weights = pt_weights[mask]

    # Histogram
    h = ax.hist2d(
        x_var,
        pt_weights,
        bins=binning,
        cmap="viridis"
    )

    ax.set_xlabel(var_name)
    ax.set_ylabel(rf"$p_{T}$ {pt_corretion}")

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


def pt_weights_plotting(model, inputs, layer_name, plot_path):

    # Unpack inputs
    X_test, y_test, reco_pt_test = inputs
    pt_correction_type = layer_name.split("_")[1] # 'weights' or 'offsets'
    pt_weights = get_pt_weights(model, X_test, reco_pt_test, layer_name)

    plot_path = os.path.join(plot_path, "pt_weights")
    os.makedirs(plot_path, exist_ok=True)

    mask = X_test[:, :, 0].flatten() != 0
    plot_1D_histogram(
        pt_weights.flatten()[mask],
        pt_correction_type,
        20,
        os.path.join(plot_path, "pt_weights_distribution"),
        )

    plot_2D_histogram(
        pt_weights.flatten(),
        pt_correction_type,
        X_test[:, :, 0].flatten(),
        r"Constituents $p_{T}$ [GeV]",
        [0, 800],
        binning=(20, 20),
        save_path=os.path.join(plot_path, "pt_weights_vs_pt"),
        )
    plot_2D_histogram(
        pt_weights.flatten(),
        pt_correction_type,
        X_test[:, :, 3].flatten() * (np.pi / 720),
        r"Constituents $\Delta \eta$",
        [-5, 5],
        binning=(20, 10),
        save_path=os.path.join(plot_path, "pt_weights_vs_eta"),
        )

    class_labels = model.class_labels
    y_test = np.argmax(y_test, axis=1) # Convert one-hot to class indices
    for flav, i in class_labels.items():
        class_mask = y_test == i
        plot_2D_histogram(
            pt_weights[class_mask].flatten(),
            pt_correction_type,
            X_test[:, :, 0][class_mask].flatten(),
            r"Constituents $p_{T}$ [GeV]",
            [0, 800],
            binning=(20, 20),
            save_path=os.path.join(plot_path, f"pt_weights_vs_pt_{flav}"),
            )
        plot_2D_histogram(
            pt_weights[class_mask].flatten(),
            pt_correction_type,
            X_test[:, :, 3][class_mask].flatten() * (np.pi / 720),
            r"Constituents $\Delta \eta$",
            [-5, 5],
            binning=(10, 10),
            save_path=os.path.join(plot_path, f"pt_weights_vs_eta_{flav}"),
            )

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-m', '--model-dir', required=True, help='model directory')

    # Load model
    model = Model.fromFolder(args.model_dir)

    # Load testing data
    X_test = np.load(f"{model.output_directory}/testing_data/X_test.npy")
    y_test = np.load(f"{model.output_directory}/testing_data/y_test.npy")
    reco_pt_test = np.load(f"{model.output_directory}/testing_data/reco_pt_test.npy")

    inputs = (X_test, y_test, reco_pt_test)

    output_dir = os.path.join(model.output_directory, "plots/training")

    # Plot pt weights
    pt_weights_plotting(model, inputs, 'pt_weights_output', output_dir)
    try:
        pt_weights_plotting(model, inputs, 'pt_offsets_output', output_dir)
    except:
        print("No pt_offsets_output layer found in model, skipping offset weights plotting.")

