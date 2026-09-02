import os
import numpy as np

def set_gpu(gpu_id: int):
    import tensorflow as tf
    if not tf.config.list_physical_devices('GPU'):
        print("No GPU found. Using CPU.")
    else:
        print(f"GPU found. Using {tf.config.list_physical_devices('GPU')[gpu_id].name}")
        tf.config.set_visible_devices(tf.config.list_physical_devices('GPU')[gpu_id], 'GPU')

def save_test_data(out_dir, X_test, y_test, truth_pt_test, reco_pt_test):

    os.makedirs(os.path.join(out_dir, 'testing_data'), exist_ok=True)

    np.save(os.path.join(out_dir, "testing_data/X_test.npy"), X_test)
    np.save(os.path.join(out_dir, "testing_data/y_test.npy"), y_test)
    np.save(os.path.join(out_dir, "testing_data/truth_pt_test.npy"), truth_pt_test)
    np.save(os.path.join(out_dir, "testing_data/reco_pt_test.npy"), reco_pt_test)

    print(f"Test data saved to {out_dir}")

def calculate_class_weights(class_labels_train: np.ndarray, class_mapping: dict[str, int],) -> dict[int, float]:
    """Calculate inverse-frequency weights for each class."""

    num_classes = len(class_mapping)
    class_counts = {}
    for cls in class_mapping.values():
        class_counts[cls] = np.sum(class_labels_train == cls)
    total_count = sum(class_counts.values())

    class_weights = {}
    for cls in class_mapping.values():
        if class_counts[cls] > 0:
            class_weights[cls] = (total_count / (num_classes * class_counts[cls]))
        else:
            class_weights[cls] = 0.0

    print("Class weights:")
    for name, cls in class_mapping.items():
        print(
            f"Class {name} ({cls}): "
            f"Weight {class_weights[cls]:.4f} "
            f"(Count: {class_counts[cls]})"
        )

    return class_weights

def apply_class_weights(class_labels: np.ndarray, class_weights: dict[int, float],) -> np.ndarray:
    weights = np.ones_like(class_labels, dtype=np.float32)
    
    for cls in class_weights.keys():
        if cls not in np.unique(class_labels):
            print(f"Warning: Class {cls} not found in class_labels. It will be ignored in weight application.")
        mask = class_labels == cls
        weights[mask] = class_weights[cls]

    return weights

def train_weights(y_train, reco_pt_train, class_labels, weightingMethod, debug):
    """
    Re-balancing the class weights and then flatten them based on truth pT
    """
    if weightingMethod not in ["none", "ptref", "onlyclass"]:
        raise ValueError(
            "Oops!  Given weightingMethod not defined in train_weights(). Use either none, ptref, or onlyclass."
        )
    num_samples = y_train.shape[0]

    sample_weights = np.ones(num_samples)

    # Define pT bins (without the high pT part we don't care about)
    pt_bins = np.array(
        [15, 17, 19, 22, 25, 30, 35, 40, 45, 50, 60, 76, 97, 122, 154, np.inf]
    )  # Use np.inf to cover all higher values

    if weightingMethod == "onlyclass":
        pt_bins = np.array([0.0, np.inf])  # Use np.inf to cover all higher values

    # Initialize counts per class per pT bin
    class_pt_counts = {}

    # Calculate counts per class per pT bin
    for _label, idx in class_labels.items():
        class_mask = y_train[:, idx] == 1
        print(f"DEBUG - Class {idx} {len(class_mask)} samples ({len(reco_pt_train)})")
        class_pt_counts[idx], _ = np.histogram(reco_pt_train[class_mask], bins=pt_bins)
        print(f"DEBUG - Class {idx} ({_label}) counts per pT bin: {class_pt_counts[idx]})")

    # Compute the maximum counts per pT bin over all classes
    max_counts_per_bin = np.zeros(len(pt_bins) - 1)
    min_counts_per_bin = np.zeros(len(pt_bins) - 1)
    for bin_idx in range(len(pt_bins) - 1):
        counts_in_bin = [class_pt_counts[idx][bin_idx] for idx in class_labels.values()]
        max_counts_per_bin[bin_idx] = max(counts_in_bin)
        min_counts_per_bin[bin_idx] = min(counts_in_bin)

    # Weight all to one base class (b = 0)
    counts_per_bin = class_pt_counts[0]

    if weightingMethod == "ptref":
        # Try minimum and flat
        counts_per_bin = [min(min_counts_per_bin) for __ in min_counts_per_bin]
        # Try maximum and flat
        # counts_per_bin = [max(max_counts_per_bin) for __ in max_counts_per_bin]

    # Compute weights per class per pT bin
    weights_per_class_pt_bin = {}
    for idx in class_labels.values():
        weights_per_class_pt_bin[idx] = np.zeros(len(pt_bins) - 1)
        for bin_idx in range(len(pt_bins) - 1):
            class_count = class_pt_counts[idx][bin_idx]
            if class_count == 0:
                weights_per_class_pt_bin[idx][bin_idx] = 0.0
            else:
                weights_per_class_pt_bin[idx][bin_idx] = counts_per_bin[bin_idx] / class_count

    # Multiply by some custom class weights
    # All same weight
    weights_per_class = {
        0: 1.0,  # b
        1: 1.0,  # charm
        2: 1.0,  # light
        3: 1.0,  # gluon
        4: 1.0,  # taup
        5: 1.0,  # taum
        6: 1.0,  # muon
        7: 1.0,  # electron
        8: 1.0,  # unmatched
    }
    for idx in class_labels.values():
        weights_per_class_pt_bin[idx] = weights_per_class_pt_bin[idx] * weights_per_class[idx]
        print(f"DEBUG - Weights for class {idx}: {weights_per_class_pt_bin[idx]})")

    # Assign weights to samples
    for idx in class_labels.values():
        class_mask = y_train[:, idx] == 1
        print(f"DEBUG - Class {idx} has {np.sum(class_mask)} samples ({np.sum(class_mask)/num_samples*100:.2f}%)")
        class_truth_pt = reco_pt_train[class_mask]
        sample_indices = np.where(class_mask)[0]
        # Subtract 1 to get 0-based index
        bin_indices = np.digitize(class_truth_pt, pt_bins) - 1
        # Handle right edge
        bin_indices[bin_indices == len(pt_bins) - 1] = len(pt_bins) - 2
        sample_weights[sample_indices] = weights_per_class_pt_bin[idx][bin_indices]

        # Print weighted jets as closure test in debug mode
        if debug and weightingMethod != "none":
            print("DEBUG - Checking jets weighted by sample_weights as a function of pT:")
            print(np.histogram(class_truth_pt, bins=pt_bins, weights=sample_weights[sample_indices]))

    # Normalize sample weights
    sample_weights = sample_weights / np.mean(sample_weights)

    if weightingMethod == "none":
        return None
    return sample_weights