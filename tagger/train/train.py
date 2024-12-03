from argparse import ArgumentParser
import os, shutil, json

#Import from other modules
from tagger.data.tools import make_data, load_data, to_ML
from tagger.plot.basic import loss_history, basic_ROC, pt_correction_hist, rms
from tagger.models.baseline import Baseline
from tagger.models.inet import IntNet

#Third parties
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot

VALIDATION_SPLIT = 0.1 # 10% of training set will be used for validation set. 

def train(out_dir, percent, model_name):

    #Remove output dir if exists
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
        print(f"Removed existing directory: {out_dir}")
    else:
        os.makedirs(out_dir, exist_ok=True)

    #Load the data, class_labels and input variables name
    #not really using input variable names to be honest
    data_train, data_test, class_labels, input_vars, _ = load_data("training_data/", percentage=percent)

    #Make into ML-like data for training
    X_train, y_train, pt_target_train, truth_pt_train = to_ML(data_train, class_labels)

    #Get input shape
    input_shape = X_train.shape[1:] #First dimension is batch size
    output_shape = y_train.shape[1:]

    #Dynamically get the model
    if model_name == 'baseline':
        model_class = Baseline(input_shape, output_shape)  # Assuming the model function doesn't require additional arguments
    if model_name == 'IntNet':
        model_class = IntNet(input_shape, output_shape)
    
    #try:
    #except AttributeError:
    #    raise ValueError(f"Model '{model_name}' is not defined in the 'models' module.")

    #Train it with a pruned model
    num_samples = X_train.shape[0] * (1 - VALIDATION_SPLIT)
    model_class.compile_model(num_samples)
    print(model_class)

    #Now fit to the data

    history = model_class.fit(X_train,y_train,pt_target_train)
    
    model_class.save_model(out_dir)

    #Produce some basic plots with the training for diagnostics
    plot_path = os.path.join(out_dir, "plots/training")
    os.makedirs(plot_path, exist_ok=True)

    #Plot history
    loss_history(plot_path, history)

    #Save X_test, y_test, and truth_pt_test for plotting later
    X_test, y_test, pt_target_test, truth_pt_test = to_ML(data_test, class_labels)
    
    data_path = os.path.join(out_dir, "testing_data")
    os.makedirs(data_path, exist_ok=True)

    np.save(os.path.join(data_path, "X_test.npy"), X_test)
    np.save(os.path.join(data_path, "y_test.npy"), y_test)
    np.save(os.path.join(data_path, "pt_target_test.npy"), pt_target_test)
    with open(os.path.join(out_dir, "class_label.json"), "w") as f: json.dump(class_labels, f, indent=4) #Dump output variables

    print(f"Test data saved to {out_dir}")

    model_class.basic_ROC(out_dir)
    model_class.basic_residual(out_dir)
    model_class.basic_histo(out_dir)

    return

if __name__ == "__main__":

    parser = ArgumentParser()

    #Making input arguments
    parser.add_argument('--make-data', action='store_true', help='Prepare the data if set.')
    parser.add_argument('-i','--input', default='/eos/cms/store/cmst3/group/l1tr/sewuchte/l1teg/fp_ntuples_v131Xv9/baselineTRK_4param_021024/All200.root' , help = 'Path to input training data')
    parser.add_argument('-s','--step', default='100MB' , help = 'The maximum memory size to process input root file')
    parser.add_argument('-e','--extras', default='extra_fields', help= 'Which extra fields to add to output tuples, defined in pfcand_fields.yml')

    #Training argument
    parser.add_argument('-o','--output', default='output/baseline', help = 'Output model directory path, also save evaluation plots')
    parser.add_argument('-p','--percent', default=100, type=int, help = 'Percentage of how much processed data to train on')
    parser.add_argument('-m','--model', default='baseline', help = 'Model object name to train on')

    #Basic ploting
    parser.add_argument('--plot-basic', action='store_true', help='Plot all the basic performance if set')

    args = parser.parse_args()

    #Either make data or start the training
    if args.make_data:
        make_data(infile=args.input, step_size=args.step, extras=args.extras) #Write to training_data/, can be specified using outdir, but keeping it simple here for now
    elif args.plot_basic:
        model_dir = args.output
        
        #All the basic plots!
        basic_ROC(model_dir)
    else:
        train(args.output, args.percent, model_name=args.model)
        
