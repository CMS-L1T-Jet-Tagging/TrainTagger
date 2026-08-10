# Python
import gc
import json
import os
import shutil

import awkward as ak

# Third party
import numpy as np
import tensorflow as tf
import uproot
import yaml
from tqdm import tqdm

# Dataset configuration
from ..config import EXTRA_FIELDS, FILTER_PATTERN, INPUT_TAG, N_PARTICLES

gc.set_threshold(0)

def get_valid_jets(pt, eta, reject):
    """Return a boolean mask for valid jets based on kinematic cuts."""
    return (pt > 15) & (np.abs(eta) < 2.4) & (reject == 0)


def split_flavor(data):
    """
    Splits data by particle flavor and applies conditions for each category. Also creates the pT target.

    Parameters:
        data (awkward array): The input data to split.

    Returns:
        dict: A dictionary containing the split data by label.
    """

    genmatch_pt_base = data['jet_genmatch_pt'] > 0

    # Define conditions for each label
    conditions = {
        "b": (
            genmatch_pt_base
            & (data['jet_muflav'] == 0)
            & (data['jet_tauflav'] == 0)
            & (data['jet_elflav'] == 0)
            & (data['jet_genmatch_hflav'] == 5)
        ),  # Bottom
        "charm": (
            genmatch_pt_base
            & (data['jet_muflav'] == 0)
            & (data['jet_tauflav'] == 0)
            & (data['jet_elflav'] == 0)
            & (data['jet_genmatch_hflav'] == 4)
        ),  # Charm
        "light": (
            genmatch_pt_base
            & (data['jet_muflav'] == 0)
            & (data['jet_tauflav'] == 0)
            & (data['jet_elflav'] == 0)
            & (data['jet_genmatch_hflav'] == 0)
            & (
                (abs(data['jet_genmatch_pflav']) == 0)
                | (abs(data['jet_genmatch_pflav']) == 1)
                | (abs(data['jet_genmatch_pflav']) == 2)
                | (abs(data['jet_genmatch_pflav']) == 3)
            )
        ),  # uds
        "gluon": (
            genmatch_pt_base
            & (data['jet_muflav'] == 0)
            & (data['jet_tauflav'] == 0)
            & (data['jet_elflav'] == 0)
            & (data['jet_genmatch_hflav'] == 0)
            & (data['jet_genmatch_pflav'] == 21)
        ),  # Gluon
        "taup": (
            genmatch_pt_base
            & (data['jet_muflav'] == 0)
            & (data['jet_tauflav'] == 1)
            & (data['jet_taucharge'] > 0)
            & (data['jet_elflav'] == 0)
        ),  # Tau +
        "taum": (
            genmatch_pt_base
            & (data['jet_muflav'] == 0)
            & (data['jet_tauflav'] == 1)
            & (data['jet_taucharge'] < 0)
            & (data['jet_elflav'] == 0)
        ),  # Tau -
        "muon": (
            genmatch_pt_base & (data['jet_muflav'] == 1) & (data['jet_tauflav'] == 0) & (data['jet_elflav'] == 0)
        ),  # muon
        "electron": (
            genmatch_pt_base & (data['jet_muflav'] == 0) & (data['jet_tauflav'] == 0) & (data['jet_elflav'] == 1)
        ),  # electron
    }

    # Automatically generate class labels based on the order of keys in conditions
    class_labels = {label: idx for idx, label in enumerate(conditions)}

    # Initialize the new array in data for numeric labels with default -1 for unmatched entries
    data['class_label'] = ak.full_like(data['jet_genmatch_pt'], -1)

    # Assign numeric values based on conditions using awkward's where function
    for label, condition in conditions.items():
        data['class_label'] = ak.where(condition, class_labels[label], data['class_label'])

    # Set pt regression target
    hadrons = conditions["b"] | conditions["charm"] | conditions["light"] | conditions["gluon"]
    leptons = conditions["taup"] | conditions["taum"] | conditions["muon"] | conditions["electron"]

    hadron_pt_ratio = ak.where(hadrons, data["jet_genmatch_pt"] / data["jet_pt_phys"], 0)
    lepton_pt_ratio = ak.where(leptons, data["jet_genmatch_lep_vis_pt"] / data["jet_pt_phys"], 0)
    hadron_pt_ratio = ak.nan_to_num(hadron_pt_ratio, nan=0, posinf=0, neginf=0)
    lepton_pt_ratio = ak.nan_to_num(lepton_pt_ratio, nan=0, posinf=0, neginf=0)

    hadron_pt = ak.nan_to_num(data["jet_genmatch_pt"], nan=0, posinf=0, neginf=0)
    lepton_pt = ak.nan_to_num((data["jet_genmatch_lep_vis_pt"]), nan=0, posinf=0, neginf=0)

    data['target_pt'] = np.clip(hadrons * hadron_pt_ratio + leptons * lepton_pt_ratio, 0.3, 2)
    data['target_pt_phys'] = hadrons * hadron_pt + leptons * lepton_pt

    data['target_pt'] = ak.where(data['class_label'] == -1, -1, data['target_pt'])
    data['target_pt_phys'] = ak.where(data['class_label'] == -1, -1, data['target_pt_phys'])

    return data, class_labels

def get_puppicand_fields(tag):

    # Get the directory of the current file (tools.py)
    current_dir = os.path.dirname(__file__)

    # Construct the path to puppicand_fields.yml relative to tools.py
    puppicand_fields_path = os.path.join(current_dir, "puppicand_fields.yml")

    # Load the YAML file as a dictionary
    with open(puppicand_fields_path, "r") as file:
        puppicand_fields = yaml.safe_load(file)

    return puppicand_fields[tag]



def pad_fill(array, target):
    '''
    pad an array to target length and then fill it with 0s
    '''
    return ak.fill_none(ak.pad_none(array, target, axis=1, clip=True), 0)


def make_nn_inputs(data_split, tag, n_parts):

    features = get_puppicand_fields(tag)
    inputs_list = []

    # Vertically stacked them to create input sets
    # https://awkward-array.org/doc/main/user-guide/how-to-restructure-concatenate.html
    # Also pad and fill them with 0 to the number of constituents we are using (nconstit)
    for field in features:
        field_array = data_split["jet_puppicand"][field]
        padded_filled_array = pad_fill(field_array, n_parts)
        inputs_list.append(padded_filled_array[:, :, np.newaxis])

    inputs = ak.concatenate(inputs_list, axis=2)
    data_split['nn_inputs'] = inputs
    return


def extract_array(tree, field, entry_stop):
    """
    Extracts an array from the tree with a limit on the number of entries.
    """
    return tree[field].array(entry_stop=entry_stop)


def to_ML(data, class_labels):
    """
    Take in the data from make_data (loaded by load_data) and make them ready for training.
    """

    X = np.asarray(data['nn_inputs'])
    y = tf.keras.utils.to_categorical(np.asarray(data['class_label']), num_classes=len(class_labels))
    pt_target = np.asarray(data['target_pt'])
    truth_pt = np.asarray(data['target_pt_phys'])
    reco_pt = np.asarray(data['jet_pt_phys'])
    reco_eta = np.asarray(data['jet_eta_phys'])
    reco_phi =np.asarray(data['jet_phi_phys'])
    event =np.asarray(data['event'])
    
    jet_features = np.stack([reco_pt,reco_eta,reco_phi])

    return X, y, pt_target, truth_pt, reco_pt, jet_features, event

