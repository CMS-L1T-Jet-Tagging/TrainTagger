from argparse import ArgumentParser
import os, shutil, json

# Import from other modules
from tagger.data.tools import make_data, load_data, to_ML
from tagger.plot.basic import loss_history, basic
import models

# Third parties
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import mlflow
from datetime import datetime

num_threads = 24
os.environ["OMP_NUM_THREADS"] = str(num_threads)
os.environ["TF_NUM_INTRAOP_THREADS"] = str(num_threads)
os.environ["TF_NUM_INTEROP_THREADS"] = str(num_threads)

tf.config.threading.set_inter_op_parallelism_threads(
    num_threads
)
tf.config.threading.set_intra_op_parallelism_threads(
    num_threads
)

# GLOBAL PARAMETERS TO BE DEFINED WHEN TRAINING
tf.keras.utils.set_random_seed(420) # not a special number 
BATCH_SIZE = 1024
EPOCHS = 200
VALIDATION_SPLIT = 0.2 # 20% of training set will be used for validation set. 
LOSS_WEIGHTS = [1., 1.]
WEIGHT_METHOD = "onlyclass"
DEBUG = True

# Sparsity parameters
I_SPARSITY = 0.0 # Initial sparsity
F_SPARSITY = 0.1 # Final sparsity

def prune_model(model, num_samples):
    """
    Pruning settings for the model. Return the pruned model
    """

    print("Begin pruning the model...")

    # Calculate the ending step for pruning
    end_step = np.ceil(num_samples / BATCH_SIZE).astype(np.int32) * EPOCHS

    # Define the pruned model
    pruning_params = {'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=I_SPARSITY, final_sparsity=F_SPARSITY, begin_step=0, end_step=end_step)}
    pruned_model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)

    pruned_model.compile(optimizer='adam',
                            loss={'prune_low_magnitude_jet_id_output': 'categorical_crossentropy', 'prune_low_magnitude_pT_output': tf.keras.losses.Huber()},
                            metrics = {'prune_low_magnitude_jet_id_output': 'categorical_accuracy', 'prune_low_magnitude_pT_output': ['mae', 'mean_squared_error']},
                            weighted_metrics = {'prune_low_magnitude_jet_id_output': 'categorical_accuracy', 'prune_low_magnitude_pT_output': ['mae', 'mean_squared_error']},
                            loss_weights = LOSS_WEIGHTS)

    print(pruned_model.summary())

    return pruned_model

def save_test_data(out_dir, X_test, y_test, truth_pt_test, reco_pt_test, class_labels):

    os.makedirs(os.path.join(out_dir,'testing_data'), exist_ok=True)

    np.save(os.path.join(out_dir, "testing_data/X_test.npy"), X_test)
    np.save(os.path.join(out_dir, "testing_data/y_test.npy"), y_test)
    np.save(os.path.join(out_dir, "testing_data/truth_pt_test.npy"), truth_pt_test)
    np.save(os.path.join(out_dir, "testing_data/reco_pt_test.npy"), reco_pt_test)
    with open(os.path.join(out_dir, "class_label.json"), "w") as f: json.dump(class_labels, f, indent=4) #Dump output variables

    print(f"Test data saved to {out_dir}")

def train_weights(y_train, reco_pt_train, class_labels, weightingMethod = WEIGHT_METHOD):
    """
    Re-balancing the class weights and then flatten them based on truth pT
    """
    if weightingMethod not in ["none", "ptref", "onlyclass"]:
        raise ValueError("Oops!  Given weightingMethod not defined in train_weights(). Use either none, ptref, or onlyclass.")
    num_samples = y_train.shape[0]
    num_classes = y_train.shape[1]

    sample_weights = np.ones(num_samples)

    # Define pT bins (without the high pT part we don't care about)
    pt_bins = np.array([
        15, 17, 19, 22, 25, 30, 35, 40, 45, 50,
        60, 76, 97, 122, 154, np.inf  # Use np.inf to cover all higher values
    ])

    if weightingMethod == "onlyclass":
        pt_bins = np.array([
            0., np.inf  # Use np.inf to cover all higher values
        ])
    
    # Initialize counts per class per pT bin
    class_pt_counts = {}

    # Calculate counts per class per pT bin
    for label, idx in class_labels.items():
        class_mask = y_train[:, idx] == 1
        class_pt_counts[idx], _ = np.histogram(reco_pt_train[class_mask], bins=pt_bins)
    
    # Compute the maximum counts per pT bin over all classes
    max_counts_per_bin = np.zeros(len(pt_bins)-1)
    min_counts_per_bin = np.zeros(len(pt_bins)-1)
    for bin_idx in range(len(pt_bins)-1):
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
        weights_per_class_pt_bin[idx] = np.zeros(len(pt_bins)-1)
        for bin_idx in range(len(pt_bins)-1):
            class_count = class_pt_counts[idx][bin_idx]
            if class_count == 0:
                weights_per_class_pt_bin[idx][bin_idx] = 0.
            else:
                weights_per_class_pt_bin[idx][bin_idx] = counts_per_bin[bin_idx] / class_count

    # Multiply by some custom class weights
    # All same weight
    weights_per_class = {
        0: 1, # b
        1: 1, # charm
        2: 1., # light
        3: 1., # gluon
        4: 1., # taup
        5: 1., # taum
        6: 1., # muon
        7: 1. # electron
    }
    for idx in class_labels.values():
        weights_per_class_pt_bin[idx] = weights_per_class_pt_bin[idx] * weights_per_class[idx]

    # Assign weights to samples
    for idx in class_labels.values():
        class_mask = y_train[:, idx] == 1
        class_truth_pt = reco_pt_train[class_mask]
        sample_indices = np.where(class_mask)[0]
        bin_indices = np.digitize(class_truth_pt, pt_bins) - 1  # Subtract 1 to get 0-based index
        bin_indices[bin_indices == len(pt_bins)-1] = len(pt_bins)-2  # Handle right edge
        sample_weights[sample_indices] = weights_per_class_pt_bin[idx][bin_indices]
    
        # Print weighted jets as closure test in debug mode
        if DEBUG and weightingMethod != "none":
            print ("DEBUG - Checking jets weighted by sample_weights as a function of pT:")
            print (np.histogram(class_truth_pt, bins = pt_bins, weights = sample_weights[sample_indices]))

    # Normalize sample weights
    sample_weights = sample_weights / np.mean(sample_weights)


    if weightingMethod == "none":
        return None
    else:
        return sample_weights

def train(out_dir, percent, model_name, new_epochs = None):

    # Remove output dir if exists
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
        print(f"Re-created existing directory: {out_dir}.")

    # Create dir to save results
    os.makedirs(out_dir)

    # Load the data, class_labels and input variables name, not really using input variable names to be honest
    data_train, data_test, class_labels, input_vars, extra_vars = load_data("training_data/", percentage=percent)
    
    # Save input variables and extra variables metadata
    with open(os.path.join(out_dir, "input_vars.json"), "w") as f: json.dump(input_vars, f, indent=4) #Dump input variables
    with open(os.path.join(out_dir, "extra_vars.json"), "w") as f: json.dump(extra_vars, f, indent=4) #Dump output variables

    # store model metadata for later analysis
    model_metadata = {
        "model_name": model_name,
        "epochs": EPOCHS,
        "batchsize": BATCH_SIZE,
        "valsplit": VALIDATION_SPLIT,
        "loss_weights": LOSS_WEIGHTS,
        "percent": percent,
        "weighting": WEIGHT_METHOD,
    }
    with open(os.path.join(out_dir, "model_metadata.json"), "w") as f: json.dump(model_metadata, f, indent=4) #Dump model variables

    # Make into ML-like data for training
    X_train, y_train, pt_target_train, truth_pt_train, reco_pt_train = to_ML(data_train, class_labels)

    # Save X_test, y_test, and truth_pt_test for plotting later
    X_test, y_test, _, truth_pt_test, reco_pt_test = to_ML(data_test, class_labels)
    save_test_data(out_dir, X_test, y_test, truth_pt_test, reco_pt_test, class_labels)

    # Calculate the sample weights for training
    sample_weight = train_weights(y_train, reco_pt_train, class_labels)
    if DEBUG:
        print ("DEBUG - Checking sample_weight:")
        print (sample_weight)

    # Get input shape
    input_shape = X_train.shape[1:] # First dimension is batch size
    output_shape = y_train.shape[1:]

    # Dynamically get the model
    try:
        model_func = getattr(models, model_name)
        model = model_func(input_shape, output_shape)  # Assuming the model function doesn't require additional arguments
    except AttributeError:
        raise ValueError(f"Model '{model_name}' is not defined in the 'models' module.")

    # Train it with a pruned model
    num_samples = X_train.shape[0] * (1 - VALIDATION_SPLIT)
    pruned_model = prune_model(model, num_samples)

    # Now fit to the data
    callbacks = [tfmot.sparsity.keras.UpdatePruningStep(),
                 EarlyStopping(monitor='val_loss', patience=10),
                 ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_lr=1e-5)]

    history = pruned_model.fit({'model_input': X_train},
                            {'prune_low_magnitude_jet_id_output': y_train, 'prune_low_magnitude_pT_output': pt_target_train},
                            sample_weight=sample_weight,
                            epochs=EPOCHS,
                            batch_size=BATCH_SIZE,
                            verbose=2,
                            validation_split=VALIDATION_SPLIT,
                            callbacks = [callbacks],
                            shuffle=True)
    
    # Export the model
    model_export = tfmot.sparsity.keras.strip_pruning(pruned_model)

    export_path = os.path.join(out_dir, "model/saved_model.h5")
    model_export.save(export_path)
    print(f"Model saved to {export_path}")

    # Produce some basic plots with the training for diagnostics
    plot_path = os.path.join(out_dir, "plots/training")
    os.makedirs(plot_path, exist_ok=True)

    # Plot history
    loss_history(plot_path, history)

    return

if __name__ == "__main__":

    parser = ArgumentParser()

    # Making input arguments
    parser.add_argument('--make-data', action='store_true', help='Prepare the data if set.')
    parser.add_argument('-i','--input', default='/eos/cms/store/cmst3/group/l1tr/sewuchte/l1teg/fp_jettuples_090125/All200.root' , help = 'Path to input training data')
    parser.add_argument('-r','--ratio', default=1, type=float, help = 'Ratio (0-1) of the input data root file to process')
    parser.add_argument('-s','--step', default='100MB' , help = 'The maximum memory size to process input root file')
    parser.add_argument('-e','--extras', default='extra_fields', help= 'Which extra fields to add to output tuples, in pfcand_fields.yml')

    # Training argument
    parser.add_argument('-o','--output', default='output/baseline', help = 'Output model directory path, also save evaluation plots')
    parser.add_argument('-p','--percent', default=100, type=int, help = 'Percentage of how much processed data to train on')
    parser.add_argument('-m','--model', default='baseline', help = 'Model object name to train on')
    parser.add_argument('-n','--name', default='baseline', help = 'Model experiment name')
    parser.add_argument('-t','--tree', default='outnano/Jets', help = 'Tree within the ntuple containing the jets')

    # Basic ploting
    parser.add_argument('--plot-basic', action='store_true', help='Plot all the basic performance if set')
    parser.add_argument('-sig', '--signal-processes', default=[], nargs='*', help='Specify all signal process for individual plotting')

    args = parser.parse_args()

    mlflow.set_experiment(os.getenv('CI_COMMIT_REF_NAME'))

    #create plot folder
    plot_path = os.path.join(args.output, "plots/training")
    os.makedirs(plot_path, exist_ok=True)

    #Either make data or start the training
    if args.make_data:
        make_data(infile=args.input, step_size=args.step, extras=args.extras, ratio=args.ratio, tree=args.tree) #Write to training_data/, can be specified using outdir, but keeping it simple here for now
        # Format all the signal processes used for plotting later
        for signal_process in args.signal_processes:
            signal_input = os.path.join(os.path.dirname(args.input), f"{signal_process}.root")
            signal_output = os.path.join("signal_process_data", signal_process)
            if not os.path.exists(signal_output):
                make_data(infile=signal_input, outdir=signal_output, step_size=args.step, extras=args.extras,ratio=args.ratio, tree=args.tree)

    elif args.plot_basic:
        model_dir = args.output
        #All the basic plots!
        results = basic(model_dir, args.signal_processes)
        if os.path.isfile("mlflow_run_id.txt"):
            f = open("mlflow_run_id.txt", "r")
            run_id = (f.read())
            mlflow.get_experiment_by_name(os.getenv('CI_COMMIT_REF_NAME'))
            with mlflow.start_run(experiment_id=1,
                                run_name=args.name,
                                run_id=run_id # pass None to start a new run
                                ):
                for class_label in results.keys():
                    mlflow.log_metric(class_label + ' ROC AUC',results[class_label])

    else:
        with mlflow.start_run(run_name=args.name) as run:
            mlflow.set_tag('gitlab.CI_JOB_ID', os.getenv('CI_JOB_ID'))
            mlflow.keras.autolog()
            train(args.output, args.percent, model_name=args.model)
            run_id = run.info.run_id
        sourceFile = open('mlflow_run_id.txt', 'w')
        print(run_id, end="", file = sourceFile)
        sourceFile.close()



