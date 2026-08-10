import numpy as np

def generate_balanced_sample_indices(class_info: dict, class_array: np.ndarray, njets_per_class: int = 1000):
    """
    Generate balanced sample indices for each class in the labels array.

    Parameters:
    - class_info: Dictionary containing class information.
    - class_array: Array of event class labels.
    - njets_per_class: Number of jets to select per class.

    Returns:
    - sample_indices: Array of indices corresponding to the balanced sample.
    """

    # Select a balanced sample of jets from each process for the embedding analysis
    for cls, info in class_info.items():
        mask = class_array == info['class']
        n_available = mask.sum()
        n_to_sample = min(njets_per_class, n_available)
        if n_to_sample < njets_per_class:
            print(f"Warning: Not enough jets for process {cls}, using {n_to_sample} jets")

        rng = np.random.default_rng(42)
        sample_indices_proc = rng.choice(np.where(mask)[0], size=n_to_sample, replace=False,)

        if cls == list(class_info.keys())[0]:
            sample_indices = sample_indices_proc
        else:
            sample_indices = np.concatenate([sample_indices, sample_indices_proc])

    return sample_indices