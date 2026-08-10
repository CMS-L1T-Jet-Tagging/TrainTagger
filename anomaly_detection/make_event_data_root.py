import os
from tqdm import tqdm
from collections import defaultdict
from argparse import ArgumentParser
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import numpy as np
import uproot
import awkward as ak

from tagger.data.tools import _split_flavor, _make_nn_inputs, to_ML
from tagger.data.config import FILTER_PATTERN, INPUT_TAG, N_PARTICLES

from config import PROCESS_INFO, JET_FEATURE_FIELDS, SAVE_FIELDS




class FileProcessor:
    def __init__(self, file: str, process_info: dict, save_fields: list, ievent: int = 0):
        self.file = file
        self.process_info = process_info
        self.save_fields = save_fields
        self.ievent = ievent

    def get_valid_mask(self, pt, eta, reject):
        """Apply kinematic cuts to the chunk data."""
        return ((pt > 15) & (np.abs(eta) < 2.4) & (reject == 0))
    

    def process_chunk(self, chunk_data):
        """
        Apply selections, run model inference, and return a dict of arrays for one chunk.
        Returns None if the chunk is empty after selections.
        """
        # valid_jet_mask = self.get_valid_mask(chunk_data['jet_pt_phys'], chunk_data['jet_eta_phys'], chunk_data['jet_reject'])
        # chunk_data = chunk_data[valid_jet_mask]
        if len(chunk_data) == 0:
            logger.warning(f"Chunk is empty after applying kinematic cuts. Skipping...")
            return None

        # Prepare inputs
        chunk_data_split, class_labels = _split_flavor(chunk_data)
        jet_class = np.array(chunk_data_split['class_label'], dtype=int)
        num_chunk = len(jet_class)
        _make_nn_inputs(chunk_data_split, INPUT_TAG, N_PARTICLES)
        filtered_data = {field: chunk_data_split[field] for field in self.save_fields}

        # Vectorised event ID assignment
        event_labels = ak.to_numpy(chunk_data_split['event'])
        event_ids_in_chunk = np.unique(event_labels)
        n_new_events = len(event_ids_in_chunk)
        ievent += n_new_events
        sorter = np.searchsorted(event_ids_in_chunk, event_labels)
        event_id = (ievent + sorter).astype(int)
        del chunk_data, chunk_data_split

        # Stack jet features
        jet_features = np.stack(
            [np.asarray(filtered_data[f], dtype=np.float32) for f in JET_FEATURE_FIELDS],
            axis=-1
        )

        # Model inference
        X, _, _, _, _, _, _ = to_ML(filtered_data, class_labels)
        del filtered_data

        return {
            'event_id':       event_id,
            'dataset_id':     np.full(num_chunk, self.process_info['id'], dtype=int),
            'event_class':    np.full(num_chunk, self.process_info['class'], dtype=int),
            'jet_class':      jet_class,
            'jet_features':   jet_features,
            'nn_inputs':      np.asarray(X, dtype=np.float32)
        }, ievent, n_new_events




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
    # valid_jet_mask = (
    #     (chunk_data['jet_pt_phys'] > 15) &
    #     (np.abs(chunk_data['jet_eta_phys']) < 2.4) &
    #     (chunk_data['jet_reject'] == 0)
    # )
    # chunk_data = chunk_data[valid_jet_mask]
    if len(chunk_data) == 0:
        return None, ievent

    # Prepare inputs
    chunk_data_split, class_labels = _split_flavor(chunk_data)
    jet_class = np.array(chunk_data_split['class_label'], dtype=int)
    num_chunk = len(jet_class)
    _make_nn_inputs(chunk_data_split, INPUT_TAG, N_PARTICLES)
    filtered_data = {field: chunk_data_split[field] for field in save_fields}

    # Vectorised event ID assignment
    event_labels = ak.to_numpy(chunk_data_split['event'])
    event_ids_in_chunk = np.unique(event_labels)
    n_new_events = len(event_ids_in_chunk)
    ievent += n_new_events
    sorter = np.searchsorted(event_ids_in_chunk, event_labels)
    event_id = (ievent + sorter).astype(int)
    del chunk_data, chunk_data_split

    # Stack jet features
    jet_features = np.stack(
        [np.asarray(filtered_data[f], dtype=np.float32) for f in JET_FEATURE_FIELDS],
        axis=-1
    )

    # Model inference
    X, _, _, _, _, _, _ = to_ML(filtered_data, class_labels)
    del filtered_data

    return {
        'event_id':       event_id,
        'dataset_id':     np.full(num_chunk, info['id'], dtype=int),
        'event_class':    np.full(num_chunk, info['class'], dtype=int),
        'jet_class':      jet_class,
        'jet_features':   jet_features,
        'nn_inputs':      np.asarray(X, dtype=np.float32)
    }, ievent, n_new_events


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

        n_event_process = 0
        n_jet_process = 0
        accumulator = defaultdict(list)
        with tqdm(desc=f"[{process:<16}] Chunk [0]") as pbar:
            for chunk_idx, chunk_data in enumerate(
                uproot.iterate(infile, filter_name=FILTER_PATTERN, how="zip", step_size="1GB", max_workers=8)
            ):
                pbar.set_description(f"[{process:<16}] Chunk [{chunk_idx+1}]")

                # Fraction filter
                event_mask = np.isin(chunk_data['event'], unique_events)
                chunk_data = chunk_data[event_mask]
                if len(chunk_data) == 0:
                    pbar.update(1)
                    continue

                chunk_result, ievent, n_new_events = _process_chunk(chunk_data, info, ievent, save_fields)
                if chunk_result is None:
                    pbar.update(1)
                    continue

                for key, val in chunk_result.items():
                    accumulator[key].append(val)

                n_event_process += n_new_events
                n_jet_process += len(chunk_result['event_id'])

                pbar.update(1)

        print(
            f"[{process:<16}] End : {n_jet_process:<8} jets ({100*n_jet_process/total_jets:.2f}%)", 
            f"across {n_event_process:<8} events ({100*n_event_process/len(all_unique_events):.2f}%)"
        )

        process_data = {key: np.concatenate(arrays, axis=0) for key, arrays in tqdm(accumulator.items(), desc="Concatenating")}

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