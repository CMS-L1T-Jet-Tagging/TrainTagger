from argparse import ArgumentParser
import os, shutil, json, yaml

# Import from other modules
from tagger.data.tools import make_data, load_data, to_ML
from tagger.plot.basic import loss_history, basic
from tagger.model.modelLoader import fromYaml, fromFolder


# Third parties
import numpy as np
import mlflow
from datetime import datetime

# num_threads = 24
# os.environ["OMP_NUM_THREADS"] = str(num_threads)
# os.environ["TF_NUM_INTRAOP_THREADS"] = str(num_threads)
# os.environ["TF_NUM_INTEROP_THREADS"] = str(num_threads)

# tf.config.threading.set_inter_op_parallelism_threads(
#     num_threads
# )
# tf.config.threading.set_intra_op_parallelism_threads(
#     num_threads
# )

# # GLOBAL PARAMETERS TO BE DEFINED WHEN TRAINING
# tf.keras.utils.set_random_seed(420) # not a special number 
LOSS_WEIGHTS = [1., 1.]
WEIGHT_METHOD = "onlyclass"
DEBUG = True



def save_test_data(out_dir, X_test, y_test, truth_pt_test, reco_pt_test):

    os.makedirs(os.path.join(out_dir,'testing_data'), exist_ok=True)

    np.save(os.path.join(out_dir, "testing_data/X_test.npy"), X_test)
    np.save(os.path.join(out_dir, "testing_data/y_test.npy"), y_test)
    np.save(os.path.join(out_dir, "testing_data/truth_pt_test.npy"), truth_pt_test)
    np.save(os.path.join(out_dir, "testing_data/reco_pt_test.npy"), reco_pt_test)

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

def train(model,out_dir,yaml_path,percent):

    # Load the data, class_labels and input variables name, not really using input variable names to be honest
    data_train, data_test, class_labels, input_vars, extra_vars = load_data("training_data/", percentage=percent)
    model.set_labels(input_vars,extra_vars,class_labels,)
    
    # Make into ML-like data for training
    X_train, y_train, pt_target_train, truth_pt_train, reco_pt_train = to_ML(data_train, class_labels)

    # Save X_test, y_test, and truth_pt_test for plotting later
    X_test, y_test, _, truth_pt_test, reco_pt_test = to_ML(data_test, class_labels)
    save_test_data(out_dir, X_test, y_test, truth_pt_test, reco_pt_test)

    # Calculate the sample weights for training
    sample_weight = train_weights(y_train, reco_pt_train, class_labels,weightingMethod = model.training_config['weight_method'] )
    if model.run_config['debug']:
        print ("DEBUG - Checking sample_weight:")
        print (sample_weight)

    # Get input shape
    input_shape = X_train.shape[1:] #First dimension is batch size
    output_shape = y_train.shape[1:]

    model.build_model(input_shape,output_shape)
    #Train it with a pruned model
    num_samples = X_train.shape[0] * (1 - model.training_config['validation_split'])
    model.compile_model(num_samples)
    history = model.fit(X_train,y_train,pt_target_train,sample_weight)

    model.save()

    model.plot_loss()

    return

if __name__ == "__main__":

    parser = ArgumentParser()
    # Training argument
    parser.add_argument('-o','--output', default='output/baseline', help = 'Output model directory path, also save evaluation plots')
    parser.add_argument('-p','--percent', default=100, type=int, help = 'Percentage of how much processed data to train on')
    parser.add_argument('-y','--yaml_config', default='tagger/model/configs/baseline_larger.yaml', help = 'YAML config for model')

    # Basic ploting
    parser.add_argument('--plot-basic', action='store_true', help='Plot all the basic performance if set')
    parser.add_argument('-sig', '--signal-processes', default=[], nargs='*', help='Specify all signal process for individual plotting')

    args = parser.parse_args()

    #mlflow.set_experiment(os.getenv('CI_COMMIT_REF_NAME'))

    

    if args.plot_basic:
        #All the basic plots!
        model = fromFolder(args.output)
        results = basic(model, args.signal_processes)
        # if os.path.isfile("mlflow_run_id.txt"):
        #     f = open("mlflow_run_id.txt", "r")
        #     run_id = (f.read())
        #     mlflow.get_experiment_by_name(os.getenv('CI_COMMIT_REF_NAME'))
        #     with mlflow.start_run(experiment_id=1,
        #                         run_name=args.name,
        #                         run_id=run_id # pass None to start a new run
        #                         ):
        #         for class_label in results.keys():
        #             mlflow.log_metric(class_label + ' ROC AUC',results[class_label])

    else:
        model = fromYaml(args.yaml_config,args.output)
        train(model, args.output, args.yaml_config, args.percent)
        # with mlflow.start_run(run_name=args.name) as run:
        #     mlflow.set_tag('gitlab.CI_JOB_ID', os.getenv('CI_JOB_ID'))
        #     mlflow.keras.autolog()
        #     train(args.output, args.yaml_config, args.percent)
        #     run_id = run.info.run_id
        # sourceFile = open('mlflow_run_id.txt', 'w')
        # print(run_id, end="", file = sourceFile)
        # sourceFile.close()



