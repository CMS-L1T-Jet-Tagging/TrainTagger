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
from tagger.model.common import fromFolder

from tensorflow.keras.models import Model

from tagger.plot import style
style.set_style()

binning_dict = {
    'pt': [0, 100, 20],
    'pt_rel': [0, 1, 10],
    'pt_log': [0, 6, 20],
    'deta': [-0.5, 0.5, 10],
    'dphi': [-0.5, 0.5, 10],
    'mass': [0, 50, 20],
    'isPhoton': [-0.1, 1.1, 2],
    'isElectronPlus': [-0.1, 1.1, 2],
    'isElectronMinus': [-0.1, 1.1, 2],
    'isMuonPlus': [-0.1, 1.1, 2],
    'isMuonMinus': [-0.1, 1.1, 2],
    'isNeutralHadron': [-0.1, 1.1, 2],
    'isChargedHadronPlus': [-0.1, 1.1, 2],
    'isChargedHadronMinus': [-0.1, 1.1, 2],
    'z0': [-30, 30, 10],
    'dxy': [-0.5, 0.5, 10],
    'isfilled': [-0.1, 1.1, 2],
    'puppiweight': [-0.1, 1.1, 5],
    'quality': [-0.1, 1.1, 5],
    'emid': [-0.1, 1.1, 5],
}

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

def plot_1D_histogram(pt_weights, pt, pt_correction, binning, save_path):
    # show distribution of pt weights
    pt_bins = [0, 0, 5, 15, 30, 80, np.inf]
    colors = ['purple', 'blue', 'cyan', 'green', 'gold', 'red']
    pt_weights = np.clip(pt_weights, -np.inf, 200)
    fig, ax = plt.subplots(1, 1, figsize=style.FIGURE_SIZE)
    for i, (l, u) in enumerate(zip(pt_bins[:-1], pt_bins[1:])):
        mask = (pt >= l) & (pt < u) if i>0 else pt < np.inf
        label = f"{l} < pT < {u}" if u != np.inf else f"pT > {l}"
        h = ax.hist(
            pt_weights[mask],
            bins=binning,
            histtype='step',
            label=label,
            color=colors[i],
            density=True,
            linewidth=2.5,
            range=(pt_weights.min(), pt_weights.max()),
        )
        ax.set_ylabel("Entries")
        hep.cms.label(
            llabel=style.CMSHEADER_LEFT,
            rlabel=style.CMSHEADER_RIGHT,
            ax=ax,
            fontsize=style.MEDIUM_SIZE,
        )
    plt.yscale('log')
    plt.xlabel(rf"$p_T$ {pt_correction}")
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(save_path, f"pt_{pt_correction}_distribution_pT_binned")
    plt.savefig(plot_path + ".png")
    plt.savefig(plot_path + ".pdf")
    return

def plot_2D_histogram(pt_weights, pt_corretion, x_var, var_name, mask, plot_params, save_path):

    # apply mask to remove weights for padded constituents
    pt_weights = np.clip(pt_weights[mask], -np.inf, 200)
    x_var = x_var[mask]
    x_var *= np.pi/720 if var_name == "dphi" or var_name == "deta" else 1.0

    fig, ax = plt.subplots(1, 1, figsize=style.FIGURE_SIZE)
    caps, bins = plot_params[:-1], plot_params[-1]
    x_var = np.clip(x_var, caps[0], caps[1])

    # Histogram
    h = ax.hist2d(
        x_var,
        pt_weights,
        bins=[bins, 15],
        cmap="viridis"
    )

    hep.cms.label(
        llabel=style.CMSHEADER_LEFT,
        rlabel=style.CMSHEADER_RIGHT,
        ax=ax,
        fontsize=style.MEDIUM_SIZE
    )

    hist_counts = h[0]
    hist_counts_normalized = hist_counts / np.max(hist_counts)

    # Redraw heatmap with imshow
    im = ax.imshow(
        hist_counts_normalized.T,  # transpose to match axes
        origin='lower',
        aspect='auto',
        cmap='viridis',
    )

    ax.set_xlabel(style.INPUT_FEATURE_STYLE[var_name])
    ax.set_ylabel(rf"$p_T$ {pt_corretion}")

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Entries (normalized)")

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

    plot_path = os.path.join(plot_path, f"pt_weights")
    os.makedirs(plot_path, exist_ok=True)

    mask = X_test[:, :, 0].flatten() != 0
    plot_1D_histogram(
        pt_weights.flatten()[mask],
        X_test[:, :, 0].flatten()[mask],
        pt_correction_type,
        20,
        plot_path,
        )

    # class_labels = model.class_labels
    # y_test = np.argmax(y_test, axis=1) # Convert one-hot to class indices
    # for i, input_var in enumerate(model.input_vars):
    #     plot_2D_histogram(
    #         pt_weights.flatten(),
    #         pt_correction_type,
    #         X_test[:, :, i].flatten(),
    #         input_var,
    #         mask,
    #         binning_dict[input_var],
    #         save_path=os.path.join(plot_path, f"pt_{pt_correction_type}_vs_{input_var}"),
    #         )

    #     for flav, c in class_labels.items():
    #         class_mask = y_test == c
    #         sub_mask = X_test[:, :, 0][class_mask].flatten() != 0
    #         plot_2D_histogram(
    #             pt_weights[class_mask].flatten(),
    #             pt_correction_type,
    #             X_test[:, :, i][class_mask].flatten(),
    #             input_var,
    #             sub_mask,
    #             binning_dict[input_var],
    #             save_path=os.path.join(plot_path, f"pt_{pt_correction_type}_vs_{input_var}_{flav}"),
    #             )

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-m', '--model-dir', required=True, help='model directory')
    args = parser.parse_args()

    # Load model
    model = model = fromFolder(args.model_dir)

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

