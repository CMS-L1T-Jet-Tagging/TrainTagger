import os
from argparse import ArgumentParser

# Third parties
import numpy as np
import tensorflow as tf

# Import from other modules
from tagger.data.tools import load_data, to_ML
from tagger.model.common.utils import fromFolder, fromYaml
from tagger.plot.basic import basic
from tagger.train.utils import calculate_class_weights, apply_class_weights, train_weights, set_gpu
set_gpu(0)  # Set GPU to use, if available

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
    class_counts = np.bincount(class_labels_train,  minlength=num_classes,)
    total_count = class_counts.sum()

    class_weights = {}
    for idx in range(num_classes):
        if class_counts[idx] > 0:
            class_weights[idx] = (total_count / (num_classes * class_counts[idx]))
        else:
            class_weights[idx] = 0.0

    print("Class weights:")
    for name, index in class_mapping.items():
        print(
            f"Class {name} ({index}): "
            f"Weight {class_weights[index]:.4f} "
            f"(Count: {class_counts[index]})"
        )

    return class_weights

def apply_class_weights(class_labels: np.ndarray, class_weights: dict[int, float],) -> np.ndarray:
    weight_lookup = np.array([class_weights[i] for i in range(len(class_weights))], dtype=np.float32,)
    return weight_lookup[class_labels]

def train(model, out_dir, percent):

    # Load the data, class_labels and input variables name, not really using input variable names to be honest
    data_train, data_test, class_labels, input_vars, extra_vars = load_data('training_data/All200', percentage=percent)
    model.set_labels(input_vars, extra_vars, class_labels,)

    # Make into ML-like data for training
    X_train, y_train, pt_target_train, extras_train = to_ML(data_train, class_labels, extra_vars=extra_vars)
    X_test, y_test, pt_target_test, extras_test = to_ML(data_test, class_labels, extra_vars=extra_vars)

    print("Removing pt_rel from training features due to issue with BSM samples")
    X_train = np.delete(X_train, input_vars.index("pt_rel"), axis=2)
    X_test = np.delete(X_test, input_vars.index("pt_rel"), axis=2)

    save_test_data(out_dir, X_test, y_test, extras_test['target_pt_phys'], extras_test['jet_pt_phys'])
    del data_train, data_test, X_test, y_test, pt_target_test, extras_test  # Free up memory

    class_weights = calculate_class_weights(y_train, class_labels)
    sample_weight = apply_class_weights(y_train, class_weights)

    # Get input shape
    y_train_ohe = tf.keras.utils.to_categorical(y_train, num_classes=len(class_labels))
    input_shape = X_train.shape[1:]  # First dimension is batch size
    output_shape = y_train_ohe.shape[1:]

    model.build_model(input_shape, output_shape)
    # Train it with a pruned model
    num_samples = X_train.shape[0] * (1 - model.training_config['validation_split'])
    model.compile_model(num_samples)
    model.fit(X_train, y_train_ohe, pt_target_train, sample_weight)
    model.save()
    model.plot_loss()

    return


if __name__ == "__main__":

    parser = ArgumentParser()
    # Training argument
    parser.add_argument(
        '-o', '--output', default='output/baseline', help='Output model directory path, also save evaluation plots'
    )
    parser.add_argument('-p', '--percent', default=100, type=int, help='Percentage of how much processed data to train on')
    parser.add_argument(
        '-y', '--yaml_config', default='tagger/model/configs/baseline_larger.yaml', help='YAML config for model'
    )

    # Basic ploting
    parser.add_argument('--plot-basic', action='store_true', help='Plot all the basic performance if set')
    parser.add_argument(
        '-sig', '--signal-processes', default=[], nargs='*', help='Specify all signal process for individual plotting'
    )

    args = parser.parse_args()

    if args.plot_basic:
        # All the basic plots!
        model = fromFolder(args.output)
        results = basic(model, args.signal_processes)

    else:
        model = fromYaml(args.yaml_config, args.output)
        train(model, args.output, args.percent)
