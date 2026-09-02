import os
from argparse import ArgumentParser
import gc
import json

# Third parties
import numpy as np
import awkward as ak
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Import from other modules
from tagger.data.tools import load_data, to_ML_new, make_unique_event_ids
from tagger.model.anomaly_models import Autoencoder, MahalanobisModel
from tagger.model.common.utils import fromFolder
from tagger.train.utils import calculate_class_weights, apply_class_weights
from tagger.data.tools import get_puppicand_fields, get_valid_jets
from tagger.plot.basic_ad import make_plots

def find_matching_samples(input_dir, sample_names):
    """
    Find matching samples in the input directory based on the provided sample names.

    Args:
        input_dir (str): The directory to search for sample files.
        sample_names (list): List of sample names to match. Can be exact names or patterns (e.g., 'QCD_Pt*').

    Returns:
        list: List of matching sample file paths.
    """

    matching_samples = []
    for sample_name in sample_names:
        for sample in os.listdir(input_dir):
            if sample_name.endswith('*'):
                # If the sample name ends with '*', treat it as a pattern
                if sample.startswith(sample_name[:-1]):
                    matching_samples.append(sample)
            else:
                # Exact match
                if sample == sample_name:
                    matching_samples.append(sample)
            
    return matching_samples


def load_all_datasets(out_dir, input_dir='training_data', percent=100, test_ratio=0.2):

    training_samples = ['MinBias_PU200', 'QCD_Pt*', 'DY*', 'TT_PU200', 'WJetsToLNu_PU200']
    testing_samples = ['MinBias_PU200', 'QCD_PU200', 'TT_PU200', 'GluGluHH*', 'SVJ_250', 'SVJ_500', 'SUEP']
    training_samples = find_matching_samples(input_dir, training_samples)
    testing_samples = find_matching_samples(input_dir, testing_samples)
    samples = sorted(list(set(training_samples + testing_samples)))
    all_data_train = []
    all_data_test = []

    # Create a mapping of sample names to integer labels for later use
    sample_iter = list(enumerate(samples))
    sample_labels = {sample: idx for idx, sample in sample_iter}
    with open(os.path.join(out_dir, "sample_labels.json"), "w") as f:
        json.dump(sample_labels, f)

    for idx, sample in sample_iter:
        if sample in training_samples and sample in testing_samples:
            test_ratio = 0.2
        elif sample in training_samples:
            test_ratio = 0
        elif sample in testing_samples:
            test_ratio = 1
        else:
            raise ValueError(f"Sample {sample} is neither in training nor testing samples.")

        data_train, data_test, class_labels, input_vars, extra_vars = load_data(
            outdir = os.path.join(input_dir, sample),
            test_ratio = test_ratio,
            percentage=percent
        )

        # Remove invalid jets based on kinematic cuts (training only - want to evaluate on all jets in testing)
        valid_jets_train = get_valid_jets(data_train['jet_pt_phys'], data_train['jet_eta_phys'], data_train['jet_reject'])
        data_train = data_train[valid_jets_train]

        # Create unique event IDs and assign sample labels
        data_train['event_id'] = make_unique_event_ids(sample, data_train['event'])
        data_test['event_id'] = make_unique_event_ids(sample, data_test['event'])
        data_train['sample_label'] = np.full(len(data_train), sample_labels[sample], dtype=np.int16)
        data_test['sample_label'] = np.full(len(data_test), sample_labels[sample], dtype=np.int16)

        all_data_train.append(data_train)
        all_data_test.append(data_test)

    print(f"Concatenating data")
    all_data_train = ak.concatenate(all_data_train, axis=0)
    all_data_test = ak.concatenate(all_data_test, axis=0)

    return all_data_train, all_data_test, class_labels, input_vars, extra_vars

def preprocess_data(data, class_labels, input_vars, extra_vars):

    # Make into ML-like data
    print("Preprocessing data to ML format")
    x, y, pt_target, extras = to_ML_new(data, class_labels, extra_vars=extra_vars)

    # Remove pt_rel from training features due to issue with BSM samples
    x = np.delete(x, input_vars.index("pt_rel"), axis=2)

    return x, y, pt_target, extras

def train_autoencoder(z_train, z_val, w_train, w_val, out_dir, layers=[16, 8, 4, 8, 16], epochs=250, batch_size=2048):

    # Build the autoencoder
    ae_model = Autoencoder(input_dim=z_train.shape[1], model_layers=layers)

    # Scale the embeddings using the autoencoder's scaler
    ae_model.fit_scaler(z_train)
    z_train_scaled = ae_model.scaler(z_train)
    z_val_scaled = ae_model.scaler(z_val)

    # Prepare validation data for the autoencoder training
    validation_data = (
        z_val_scaled,
        z_val_scaled,
        w_val,
    )

    # Train the autoencoder
    print("=========== Autoencoder model ===========")
    ae_model(tf.zeros((1, ae_model.input_dim)))
    ae_model.model.summary()

    ae_history = ae_model.fit(
        x=z_train_scaled,
        sample_weight=w_train,
        validation_data=validation_data,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=2,
    )

    print("Autoencoder training complete. Saving model and history")
    ae_model.save(os.path.join(out_dir, 'model', 'autoencoder.keras'))
    with open(os.path.join(out_dir, "autoencoder_history.json"), "w") as f:
        json.dump({"autoencoder_history": ae_history.history}, f)

    return ae_model, ae_history

def train_mahalanobis(z_train, w_train, out_dir):

    print("Fitting Mahalanobis model")
    mh_model = MahalanobisModel(input_dim=z_train.shape[1])
    mh_model.fit_reference(z_train)
    mh_model.save(os.path.join(out_dir, 'model', 'mahalanobis_model.keras'))

    return mh_model


def evaluate_models(z_test, ae_model, mh_model, data_test, extra_vars, out_dir):

    # Scale the test embeddings using the autoencoder's scaler
    z_test_scaled = ae_model.scaler(z_test)

    # Get reconstruction scores from the autoencoder
    reco_scores = ae_model.score(z_test_scaled, batch_size=20_000, verbose=2)

    # Get Mahalanobis scores
    mahalanobis_scores = mh_model.score(z_test, batch_size=20_000, verbose=2)

    # Save the test data for later evaluation
    data_dir = os.path.join(out_dir, "testing_data_ad")
    os.makedirs(data_dir, exist_ok=True)
    np.save(os.path.join(data_dir, "reco_score.npy"), reco_scores)
    np.save(os.path.join(data_dir, "mahalanobis_score.npy"), mahalanobis_scores)

    # Save other relevant information for analysis
    np.save(os.path.join(data_dir, "event_id.npy"), np.asarray(data_test['event_id'], dtype=np.int64))
    np.save(os.path.join(data_dir, "event_class.npy"), np.asarray(data_test['sample_label'], dtype=np.int16))
    np.save(os.path.join(data_dir, "jet_class.npy"), np.asarray(data_test['class_label'], dtype=np.int8))
    
    # Save data for the test set
    jet_features_test = np.stack([np.asarray(data_test[field]) for field in extra_vars], axis=1)
    np.save(os.path.join(data_dir, "jet_features.npy"), jet_features_test)

    # Save model outputs
    np.save(os.path.join(data_dir, "embedding.npy"), z_test)

    # Save metrics
    metrics = {
        'reco_score_mean': np.mean(reco_scores),
        'reco_score_std': np.std(reco_scores),
        'reco_score_median': np.median(reco_scores),
        'reco_score_p90': np.percentile(reco_scores, 90),
        'reco_score_p99': np.percentile(reco_scores, 99),
        'reco_score_p999': np.percentile(reco_scores, 99.9),
        'mahalanobis_score_mean': np.mean(mahalanobis_scores),
        'mahalanobis_score_std': np.std(mahalanobis_scores),
        'mahalanobis_score_median': np.median(mahalanobis_scores),
        'mahalanobis_score_p90': np.percentile(mahalanobis_scores, 90),
        'mahalanobis_score_p99': np.percentile(mahalanobis_scores, 99),
        'mahalanobis_score_p999': np.percentile(mahalanobis_scores, 99.9),
    }
    for metric_name, metric_value in metrics.items():
        metrics[metric_name] = float(metric_value)  # Convert to float for JSON serialization
        
    with open(os.path.join(out_dir, "metrics_ad.json"), "w") as f:
        json.dump(metrics, f)

    return reco_scores, mahalanobis_scores

def main(model, out_dir, percent):

    # ------------------------------------------
    # Load and preprocess the training and testing datasets
    # ------------------------------------------
    data_train, data_test, class_labels, input_vars, extra_vars = load_all_datasets(out_dir, percent=percent)
    X_train, y_train, pt_target_train, extras_train = preprocess_data(data_train, class_labels, input_vars, extra_vars)
    X_test, y_test, pt_target_test, extras_test = preprocess_data(data_test, class_labels, input_vars, extra_vars)

    # ------------------------------------------
    # Calculate class weights and apply them to the training data
    # ------------------------------------------
    class_labels = {**{'unmatched': -1}, **class_labels}  # Add unmatched class for weighting
    class_weights = calculate_class_weights(y_train, class_labels)
    w_train = apply_class_weights(y_train, class_weights)
    w_train = np.ones_like(w_train)  # Override weights to be uniform for autoencoder training

    # ------------------------------------------
    # Get embeddings from the trained model
    # ------------------------------------------
    z_train = model.embedding_predict(X_train, batch_size=20_000, verbose=2)
    z_test = model.embedding_predict(X_test, batch_size=20_000, verbose=2)

    # ------------------------------------------
    # Train Autoencoder and Mahalanobis model on the embeddings
    # ------------------------------------------
    z_train, z_val, w_train, w_val = train_test_split(z_train, w_train, test_size=0.1, random_state=42)

    ae_model, ae_history = train_autoencoder(z_train, z_val, w_train, w_val, out_dir)
    mh_model = train_mahalanobis(z_train, w_train, out_dir)
    del z_train, z_val; gc.collect()

    # ------------------------------------------
    # Save the test data and model outputs for later evaluation
    # ------------------------------------------
    reco_scores, mahalanobis_scores = evaluate_models(z_test, ae_model, mh_model, data_test, extra_vars, out_dir)

    print("Done.")


if __name__ == "__main__":

    parser = ArgumentParser()
    # Training argument
    parser.add_argument('-o', '--output', default='output/baseline', help='Output model directory path, also save evaluation plots')
    parser.add_argument('-p', '--percent', default=100, type=int, help='Percentage of how much processed data to train on')
    args = parser.parse_args()

    model = fromFolder(args.output)
    main(model, args.output, args.percent)
