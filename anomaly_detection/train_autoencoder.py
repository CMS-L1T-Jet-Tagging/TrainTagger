import os
from sklearn.covariance import LedoitWolf
from tqdm import tqdm
from collections import defaultdict
from argparse import ArgumentParser

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

from anomaly_detection.plotting.anomaly_detection import get_mahalanobis_score
from tagger.model.common import fromFolder

from config import PROCESS_INFO, JET_FEATURE_FIELDS

def load_data(process_info: dict, model, data_dir: str) -> dict:
    """
    Iterate over all processes in process_info, apply selections and model inference
    chunk by chunk, and return a dict of concatenated arrays with one entry per jet.
    """
    accumulator = defaultdict(list)

    for process, info in process_info.items():
        infile = os.path.join(data_dir, info['path'].replace('.root', '.npz'))
        print(f"[ {process:<16} ] Loading data from {infile}...")

        file_data = np.load(infile, allow_pickle=True)
        X = file_data['nn_inputs']
        print(f"[ {process:<16} ] Performing model inference on {len(X)} jets...")
        # TEMPORARY: Remove pt_rel from inputs
        # X = np.delete(X, 1, axis=2)
        print(f"[ {process:<16} ] X shape: {X.shape} | dtype: {X.dtype}")
        jet_class_preds, jet_pt_preds = model.predict(X, batch_size=20_000, verbose=0)
        jet_embeddings = model.embedding_predict(X, batch_size=20_000, verbose=0)
        del X

        for key in ['event_id', 'dataset_id', 'event_class', 'jet_class', 'jet_features']:
            accumulator[key].append(file_data[key])
        del file_data

        accumulator['jet_class_preds'].append(np.asarray(jet_class_preds, dtype=np.float32))
        accumulator['jet_pt_preds'].append(np.asarray(jet_pt_preds, dtype=np.float32))
        accumulator['jet_embeddings'].append(np.asarray(jet_embeddings, dtype=np.float32))
        del jet_class_preds, jet_pt_preds, jet_embeddings

    print("Concatenating arrays...")
    return {
        key: np.concatenate(arrays, axis=0)
        for key, arrays in tqdm(accumulator.items(), desc="Concatenating")
    }


def split_data(data: dict, process_info: dict) -> tuple[dict, dict, dict]:
    """
    Split data into train, val, and test dicts based on event IDs.
    Processes with where=train are used in training and validations
    Processes with where=both are used in training and validation, but also included in the test set
    Processes with where=test are only used in the test set
    """
    train_classes = [info['id'] for info in process_info.values() if info['where'] == 'train']
    both_classes = [info['id'] for info in process_info.values() if info['where'] == 'both']
    test_classes = [info['id'] for info in process_info.values() if info['where'] == 'test']
    print(f"Train classes: {train_classes}\nBoth classes: {both_classes}\nTest classes: {test_classes}")

    train_mask = np.isin(data['dataset_id'], train_classes)
    both_mask = np.isin(data['dataset_id'], both_classes)
    test_mask = np.isin(data['dataset_id'], test_classes)
    print(f"Unique JETS - {len(data['dataset_id'])} - Train: {train_mask.sum()} | Both: {both_mask.sum()} | Test: {test_mask.sum()}")

    train_event_ids = np.unique(data['event_id'][train_mask])
    both_event_ids = np.unique(data['event_id'][both_mask])
    test_event_ids = np.unique(data['event_id'][test_mask])
    print(f"Unique EVENTS - {len(np.unique(data['event_id']))} - Train: {len(train_event_ids)} | Both: {len(both_event_ids)} | Test: {len(test_event_ids)}")

    # Split the both_event_ids into train and test subsets
    if both_event_ids.size > 0:
        both_train_event_ids, both_test_event_ids = train_test_split(both_event_ids, test_size=0.2, random_state=42)
        print(f"Split 'both' events into train: {len(both_train_event_ids)} | test: {len(both_test_event_ids)}")
    else:
        both_train_event_ids = np.array([], dtype=data['event_id'].dtype)
        both_test_event_ids = np.array([], dtype=data['event_id'].dtype)
        print(f"No 'both' events to split.")

    # Then merge with the train and test event IDs
    train_event_ids = np.concatenate([train_event_ids, both_train_event_ids])
    test_event_ids = np.concatenate([test_event_ids, both_test_event_ids])
    print(f"Final event counts - Train: {len(train_event_ids)} | Test: {len(test_event_ids)}")

    # Split the train_event_ids into train and validation subsets
    train_event_ids, val_event_ids  = train_test_split(train_event_ids, test_size=0.125, random_state=42)
    print(f"Split 'train' events into train: {len(train_event_ids)} | val: {len(val_event_ids)}")

    # Map event IDs to jet indices
    train_indices = np.isin(data['event_id'], train_event_ids)
    val_indices   = np.isin(data['event_id'], val_event_ids)
    test_indices  = np.isin(data['event_id'], test_event_ids)
    print(f"Final jet counts - Train: {train_indices.sum()} | Val: {val_indices.sum()} | Test: {test_indices.sum()}")

    n = len(data["event_id"])
    split_map = np.full(n, -1, dtype=np.int8)
    split_map[train_indices] = 0
    split_map[val_indices]   = 1
    split_map[test_indices]  = 2
    print(f"Split map counts - Train: {(split_map == 0).sum()} | Val: {(split_map == 1).sum()} | Test: {(split_map == 2).sum()}")

    assert (split_map >= 0).all(), "Some jets were not assigned to any split"
    train_data = {k: v[split_map == 0] for k, v in data.items()}
    val_data   = {k: v[split_map == 1] for k, v in data.items()}
    test_data  = {k: v[split_map == 2] for k, v in data.items()}
    print(f"FINAL - Train: {len(train_data['event_id'])} jets | Val: {len(val_data['event_id'])} | Test: {len(test_data['event_id'])}")

    # Sanity check - ensure no event ID appears in more than one split
    assert not set(train_data['event_id']) & set(val_data['event_id']),  "Train/val overlap"
    assert not set(train_data['event_id']) & set(test_data['event_id']), "Train/test overlap"
    assert not set(val_data['event_id'])   & set(test_data['event_id']), "Val/test overlap"

    return train_data, val_data, test_data


def calculate_reweighting_map(reweight_info: dict) -> np.ndarray:
    arrays = [np.asarray(info['array']) for info in reweight_info.values()]
    edges  = [np.asarray(info['edges']) for info in reweight_info.values()]

    # Build N-dim histogram: counts[i, j, ...] = number of events in that bin
    counts, _ = np.histogramdd(np.column_stack(arrays), bins=edges)

    # Target: uniform over occupied bins => weight per bin = 1 / count (0 for empty)
    with np.errstate(divide='ignore', invalid='ignore'):
        bin_weights = np.where(counts > 0, 1.0 / counts, 0.0)

    return bin_weights, edges

def apply_reweighting_map(bin_weights: np.ndarray, arrays: list, edges: list) -> np.ndarray:
    arrays = [np.asarray(arr) for arr in arrays]
    n_events = len(arrays[0])

    # Find which bin each event falls into (returns 1-based indices per axis)
    bin_indices = [
        np.clip(np.digitize(arr, edge) - 1, 0, len(edge) - 2)
        for arr, edge in zip(arrays, edges)
    ]

    # Look up the bin weight for each event
    weights = bin_weights[tuple(bin_indices)]

    # Normalise so weights sum to n_events (keeps overall scale sensible)
    total = weights.sum()
    if total > 0:
        weights = weights * (n_events / total)

    return weights

def build_autoencoder(input_dim=10, bottleneck_dim=4):

    inputs = keras.Input(shape=(input_dim,))

    # ---------- Encoder ----------
    x = keras.layers.Dense(8, activation='relu')(inputs)
    x = keras.layers.Dense(6, activation='relu')(x)
    x = keras.layers.Dense(bottleneck_dim, activation=None, name='bottleneck')(x)
    x = keras.layers.Dense(6, activation='relu')(x)
    x = keras.layers.Dense(8, activation='relu')(x)

    outputs = keras.layers.Dense(input_dim, activation=None)(x)
    autoencoder = keras.Model(inputs, outputs)
    autoencoder.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse')

    return autoencoder

def build_autoencoder_large(input_dim=64, bottleneck_dim=8):

    inputs = keras.Input(shape=(input_dim,))

    # ---------- Encoder ----------
    x = keras.layers.Dense(32, activation=tf.nn.gelu)(inputs)
    x = keras.layers.Dense(16, activation=tf.nn.gelu)(x)
    x = keras.layers.Dense(bottleneck_dim, activation=None, name='bottleneck')(x)
    x = keras.layers.Dense(16, activation=tf.nn.gelu)(x)
    x = keras.layers.Dense(32, activation=tf.nn.gelu)(x)

    outputs = keras.layers.Dense(input_dim, activation=None)(x)
    autoencoder = keras.Model(inputs, outputs)
    autoencoder.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse')

    return autoencoder


def get_reconstruction_score(autoencoder: keras.Model, X: np.ndarray) -> np.ndarray:
    X_recon = autoencoder.predict(X, batch_size=20_000, verbose=1)
    return np.mean((X - X_recon) ** 2, axis=1)

def get_centroid_score(X_train: np.ndarray, X_test: np.ndarray, metric: str = "cosine") -> np.ndarray:
    center = np.mean(X_train, axis=0)
    if metric == "cosine":
        Xn = X_test / np.linalg.norm(X_test, axis=1, keepdims=True)
        cn = center / np.linalg.norm(center)
        return 1 - Xn @ cn  # 0 = close, 2 = opposite
    return np.linalg.norm(X_test - center, axis=1)

def get_mahalanobis_score(X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """
    Mahalanobis distance of each event in X_test from the background distribution
    fit on X_train, using Ledoit-Wolf shrinkage for a well-conditioned covariance estimate.
    """
    cov_estimator = LedoitWolf().fit(X_train)
    
    # sanity check before trusting the inverse
    eigvals = np.linalg.eigvalsh(cov_estimator.covariance_)
    cond_number = eigvals.max() / eigvals.min()
    if cond_number > 1e8:
        print(f"Warning: covariance condition number is {cond_number:.2e} — distances may be unreliable")
    
    return np.sqrt(cov_estimator.mahalanobis(X_test))

def get_mahalanobis_score(X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    from scipy.spatial import distance
    
    # Compute the mean and covariance of the background embeddings
    mean_bg = np.mean(X_train, axis=0)
    cov_bg = np.cov(X_train, rowvar=False)

    # Regularize the covariance matrix to avoid singularity
    cov_bg += np.eye(cov_bg.shape[0]) * 1e-6

    # Compute the Mahalanobis distance for each embedding
    mahalanobis_scores = distance.cdist(X_test, [mean_bg], metric='mahalanobis', VI=np.linalg.inv(cov_bg))

    return mahalanobis_scores.flatten()

def get_prototype_score(X_train, y_train, X_test, metric="cosine") -> np.ndarray:
    """
    Calculate the distance of each test event to the nearest class prototype (mean of each class in training data).
    """
    classes = np.unique(y_train)
    protos = np.stack([X_train[y_train == c].mean(axis=0) for c in classes])
    if metric == "cosine":
        Xn = X_test / np.linalg.norm(X_test, axis=1, keepdims=True)
        pn = protos / np.linalg.norm(protos, axis=1, keepdims=True)
        sims = Xn @ pn.T
        return 1 - sims.max(axis=1)
    d = np.linalg.norm(X_test[:, None, :] - protos[None, :, :], axis=2)
    return d.min(axis=1)

def get_gmm_score(X_train: np.ndarray, X_test: np.ndarray, n_components: int = 1) -> np.ndarray:
    """
    Fit a Gaussian Mixture Model (GMM) to the training data and compute the log-likelihood of each test event.
    """
    from sklearn.mixture import GaussianMixture

    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    gmm.fit(X_train)
    log_likelihood = gmm.score_samples(X_test)
    return -log_likelihood  # Return negative log-likelihood as anomaly score

def get_kmeans_score(X_train: np.ndarray, X_test: np.ndarray, n_clusters: int = 5) -> np.ndarray:
    """
    Fit a KMeans model to the training data and compute the distance of each test event to the nearest cluster center.
    """
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_train)
    distances = kmeans.transform(X_test)
    min_distances = np.min(distances, axis=1)
    return min_distances  # Higher distances indicate more anomalous


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
if __name__ == "__main__":

    parser = ArgumentParser(description="Train an autoencoder for anomaly detection on jet embeddings.")
    parser.add_argument("--model_folder", "-m", type=str, help="Directory containing the pre-trained model and where to save results.")
    parser.add_argument("--data_dir", type=str, default="data_numpy", help="Directory containing the ROOT files.")

    args = parser.parse_args()

    data_dir = args.data_dir
    model_folder = args.model_folder

    print(f"Loading model from {model_folder} and data from {data_dir}...")

    process_info = PROCESS_INFO
    model = fromFolder(model_folder)

    # Load, split
    data = load_data(process_info, model, data_dir)
    train_data, val_data, test_data = split_data(data, process_info)
    print(f"Train: {len(train_data['event_id'])} | Val: {len(val_data['event_id'])} | Test: {len(test_data['event_id'])}")
    del data  # free the unsplit copy

    # Apply kinematic cuts to the training and validation sets, but not the test set 
    # We want to evaluate on the full distribution of jets in the test set, including those that would be rejected by the cuts
    # The cuts can be added back in at the analysis phase if desired
    jet_pt_idx = JET_FEATURE_FIELDS.index('jet_pt_phys')
    jet_eta_idx = JET_FEATURE_FIELDS.index('jet_eta_phys')
    jet_reject_idx = JET_FEATURE_FIELDS.index('jet_reject')
    valid_jet_mask_train = (
        (train_data['jet_features'][:, jet_pt_idx] > 15) &
        (np.abs(train_data['jet_features'][:, jet_eta_idx]) < 2.4) &
        (train_data['jet_features'][:, jet_reject_idx] == 0)
    )
    valid_jet_mask_val = (
        (val_data['jet_features'][:, jet_pt_idx] > 15) &
        (np.abs(val_data['jet_features'][:, jet_eta_idx]) < 2.4) &
        (val_data['jet_features'][:, jet_reject_idx] == 0)
    )
    train_data = {key: val[valid_jet_mask_train] for key, val in train_data.items()}
    val_data = {key: val[valid_jet_mask_val] for key, val in val_data.items()}

    print(f"Train: {len(train_data['event_id'])} | Val: {len(val_data['event_id'])} | Test: {len(test_data['event_id'])}")

    # Train
    X_train, X_val, X_test = train_data["jet_embeddings"], val_data["jet_embeddings"], test_data["jet_embeddings"]
    print(f"X_train: {X_train.shape} | X_val: {X_val.shape} | X_test: {X_test.shape}")

    # reweighting_info = {
    #     "jet_pt": {
    #         "edges": np.concatenate([np.logspace(np.log10(15), np.log10(1000), 11), [5000]]),
    #         "array": train_data['jet_features'][:, jet_pt_idx],
    #     },
    #     # "event_class": {
    #     #     "edges": [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5],
    #     #     "array": train_data['event_class'],
    #     # },
    #     "jet_class": {
    #         "edges": np.arange(-1.5, 8.5, 1),
    #         "array": train_data['jet_class'],
    #     },
    # }
    # print("Reweighting by: ", list(reweighting_info.keys()))
    # weight_hist, edges = calculate_reweighting_map(reweighting_info)
    # w_train = apply_reweighting_map(
    #     weight_hist, 
    #     [reweighting_info[key]['array'] for key in reweighting_info], 
    #     [reweighting_info[key]['edges'] for key in reweighting_info]
    # )
    w_train = np.ones_like(train_data['event_id'], dtype=np.float32)  # No reweighting when training on MinBias

    # autoencoder = keras.models.load_model(os.path.join(model_folder, 'model', 'anomaly_detection_autoencoder.keras'))
    autoencoder = build_autoencoder(input_dim=X_train.shape[1], bottleneck_dim=4)
    # autoencoder = build_autoencoder_large(input_dim=X_train.shape[1], bottleneck_dim=16)
    autoencoder.summary()

    # Standardize the embeddings before training the anomaly detectors
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0) + 1e-6
    X_train_std = (X_train - mean) / std
    X_val_std = (X_val - mean) / std
    X_test_std = (X_test - mean) / std

    #####
    # AUTOENCODER
    #####
    autoencoder.fit(
        X_train_std, X_train_std,
        sample_weight=w_train,
        epochs=200, batch_size=2048,
        validation_data=(X_val_std, X_val_std),
        shuffle=True,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6),
        ],
        verbose=0,
    )

    # ANOMAL DETECTION SCORES
    reco_scores = get_reconstruction_score(autoencoder, X_test_std)
    mahalanobis_scores = get_mahalanobis_score(X_train, X_test)
    # centroid_scores = get_centroid_score(X_train, X_test, metric="cosine")
    # prototype_scores = get_prototype_score(X_train, train_data['event_class'], X_test, metric="cosine")
    # gmm_scores = get_gmm_score(X_train, X_test, n_components=8)
    # kmeans_scores = get_kmeans_score(X_train, X_test, n_clusters=8)


    np.savez(
        os.path.join(model_folder, 'testing_data', 'anomaly_detection_test_results.npz'),
        event_id             = test_data['event_id'],
        event_class          = test_data['event_class'],
        jet_class            = test_data['jet_class'],
        jet_embedding        = test_data['jet_embeddings'],
        jet_class_pred       = test_data['jet_class_preds'],
        jet_pt_pred          = test_data['jet_pt_preds'],
        jet_features         = test_data['jet_features'],
        reco_score           = reco_scores,
        mahalanobis_score    = mahalanobis_scores,
        # centroid_score       = centroid_scores,
        # prototype_score      = prototype_scores,
        # gmm_score            = gmm_scores,
        # kmeans_score         = kmeans_scores,
    )
    autoencoder.save(os.path.join(model_folder, 'model', 'anomaly_detection_autoencoder.keras'))
    print("Done.")