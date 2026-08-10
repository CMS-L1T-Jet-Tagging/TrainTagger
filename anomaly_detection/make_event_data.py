import os
from tqdm import tqdm
from collections import defaultdict
from argparse import ArgumentParser

import numpy as np
import uproot
import awkward as ak
import tensorflow as tf
import yaml

from tagger.data.config import FILTER_PATTERN, INPUT_TAG, N_PARTICLES

from config import PROCESS_INFO, JET_FEATURE_FIELDS, SAVE_FIELDS


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

    no_mu = (data['jet_muflav'] == 0)
    no_tau = (data['jet_tauflav'] == 0)
    no_ele = (data['jet_elflav'] == 0)
    hflav_b = (data['jet_genmatch_hflav'] == 5)
    hflav_c = (data['jet_genmatch_hflav'] == 4)
    hflav_light = (data['jet_genmatch_hflav'] == 0)
    pflav_uds = (abs(data['jet_genmatch_pflav']) <= 3) & (abs(data['jet_genmatch_pflav']) >= 0)
    pflav_gluon = (data['jet_genmatch_pflav'] == 21)

    # Define conditions for each label
    conditions = {
        "b": (
            genmatch_pt_base & no_mu & no_tau & no_ele & hflav_b
        ),  # Bottom
        "charm": (
            genmatch_pt_base & no_mu & no_tau & no_ele & hflav_c
        ),  # Charm
        "light": (
            genmatch_pt_base & no_mu & no_tau & no_ele & hflav_light & pflav_uds
        ),  # uds
        "gluon": (
            genmatch_pt_base & no_mu & no_tau & no_ele & hflav_light & pflav_gluon
        ),  # Gluon
        "taup": (
            genmatch_pt_base & no_mu & (data['jet_tauflav'] == 1) & (data['jet_taucharge'] > 0) & no_ele
        ),  # Tau +
        "taum": (
            genmatch_pt_base & no_mu & (data['jet_tauflav'] == 1) & (data['jet_taucharge'] < 0) & no_ele
        ),  # Tau -
        "muon": (
            genmatch_pt_base & (data['jet_muflav'] == 1) & no_tau & no_ele
        ),  # muon
        "electron": (
            genmatch_pt_base & no_mu & no_tau & (data['jet_elflav'] == 1)
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
    reco_phi = np.asarray(data['jet_phi_phys'])
    event = np.asarray(data['event'])
    
    jet_features = np.stack([reco_pt,reco_eta,reco_phi])

    return X, y, pt_target, truth_pt, reco_pt, jet_features, event


def _get_unique_events(infile: str, fraction: float) -> np.ndarray:
    """Read the event branch from a ROOT file and return the subset of unique event IDs to use."""
    with uproot.open(infile) as f:
        all_event_labels = f["outnano/Jets"]["event"].array(library="np")
    all_unique_events = np.unique(all_event_labels)
    n_events_to_use = max(1, int(len(all_unique_events) * fraction))
    total_jets = len(all_event_labels)
    return all_unique_events[:n_events_to_use], all_unique_events, total_jets


def _process_chunk(chunk_data, info, ievent, save_fields):
    """
    Apply selections, run model inference, and return a dict of arrays for one chunk.
    Returns None if the chunk is empty after selections.
    """
    # Apply kinematic cuts
    # valid_jet_mask = get_valid_jets(chunk_data['jet_pt_phys'], chunk_data['jet_eta_phys'], chunk_data['jet_reject'])
    # chunk_data = chunk_data[valid_jet_mask]
    if len(chunk_data) == 0:
        print(f"[{info['name']:<16}] Chunk is empty after kinematic cuts. Skipping...")
        return None, ievent

    # Prepare inputs
    chunk_data_split, class_labels = split_flavor(chunk_data)
    make_nn_inputs(chunk_data_split, INPUT_TAG, N_PARTICLES)
    filtered_data = {field: chunk_data_split[field] for field in save_fields}

    # Get jet class array
    jet_class = np.array(chunk_data_split['class_label'], dtype=int)

    # Assign raw event ID from original root file
    event_labels = ak.to_numpy(chunk_data_split['event'])

    # Clean up to free memory    
    del chunk_data, chunk_data_split

    # Stack jet features
    jet_features = np.stack(
        [np.asarray(filtered_data[f], dtype=np.float32) for f in JET_FEATURE_FIELDS],
        axis=-1
    )

    return {
        'event_id_raw':   event_labels,
        'dataset_id':     np.full(len(event_labels), info['id'], dtype=int),
        'event_class':    np.full(len(event_labels), info['class'], dtype=int),
        'jet_class':      jet_class,
        'jet_features':   jet_features,
        'nn_inputs':      np.asarray(filtered_data['nn_inputs'], dtype=np.float32)
    }


def process_data(process_info: dict, data_dir: str, save_fields: list) -> dict:
    """
    Iterate over all processes in process_info, apply selections and model inference
    chunk by chunk, and return a dict of concatenated arrays with one entry per jet.
    """
    ievent = 0

    for process, info in process_info.items():
        infile = os.path.join(data_dir, info['path'])
        frac = info['fraction'] / 100.0

        unique_events, all_unique_events, total_jets = _get_unique_events(infile, frac)
        n_events_to_use = len(unique_events)
        print(
            f"[{process:<16}] Start: {total_jets:<8} jets across {len(all_unique_events):<8} events. "
            f"Using {n_events_to_use} ({frac*100:.2f}%)."
        )

        n_jet_process = 0
        accumulator = defaultdict(list)
        with tqdm(desc=f"[{process:<16}] Chunk [0]") as pbar:
            for chunk_idx, chunk_data in enumerate(
                uproot.iterate(infile, filter_name=FILTER_PATTERN, how="zip", step_size="1GB", max_workers=8)
            ):
                print(f"[{process:<16}] Processing chunk {chunk_idx+1} with {len(chunk_data)} jets and {len(np.unique(chunk_data['event']))} events...")
                pbar.set_description(f"[{process:<16}] Chunk [{chunk_idx+1}]")

                # Fraction filter
                event_mask = np.isin(chunk_data['event'], unique_events)
                chunk_data = chunk_data[event_mask]
                if len(chunk_data) == 0:
                    pbar.update(1)
                    continue

                chunk_result = _process_chunk(chunk_data, info, ievent, save_fields)
                if chunk_result is None:
                    pbar.update(1)
                    continue

                for key, val in chunk_result.items():
                    accumulator[key].append(val)

                pbar.update(1)

        process_data = {key: np.concatenate(arrays, axis=0) for key, arrays in tqdm(accumulator.items(), desc="Concatenating")}
        n_jet_process = len(process_data['jet_class'])

        process_event_ids = process_data['event_id_raw']
        unique_event_ids, inverse_event_ids = np.unique(process_event_ids, return_inverse=True)
        n_event_process = len(unique_event_ids)
        new_event_ids = ievent + np.arange(n_event_process)
        process_data['event_id'] = new_event_ids[inverse_event_ids]
        ievent += n_event_process

        print(
            f"[{process:<16}] End : {n_jet_process:<8} jets ({100*n_jet_process/total_jets:.2f}%)", 
            f"across {n_event_process:<8} events ({100*n_event_process/len(all_unique_events):.2f}%)"
        )

        # Save the process data to a .npz file for later use
        save_path = os.path.join(f"{data_dir}_numpy", info['path'].replace('.root', '.npz'))
        np.savez(save_path, **process_data)

    return 0

# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
if __name__ == "__main__":

    parser = ArgumentParser(description="Train an autoencoder for anomaly detection on jet embeddings.")
    parser.add_argument("--data_dir", "-d", type=str, default="data", help="Directory containing the ROOT files.")
    parser.add_argument("--fraction", "-f", type=int, default=20, help="Fraction of events to use from each process (0-100).")

    args = parser.parse_args()

    data_dir = args.data_dir
    fraction = args.fraction
    print(f"Processing data from {data_dir} with fraction {fraction}%...")

    if os.path.exists(f"{data_dir}_numpy"):
        print(f"Directory {data_dir}_numpy already exists. Please remove it or choose a different data_dir.")
        exit(1)

    print(f"Creating directory {data_dir}_numpy for saving processed data...")
    os.makedirs(f"{data_dir}_numpy", exist_ok=True)

    process_info = PROCESS_INFO
    for info in process_info.values():
        info['fraction'] = fraction

    process_data(process_info, data_dir, SAVE_FIELDS)
    print(f"Data processing complete. Processed data saved to .npz files in {data_dir}_numpy.")