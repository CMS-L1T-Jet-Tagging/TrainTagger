from argparse import ArgumentParser
import os, shutil

#Import from other modules
from tagger.data.tools import make_data, load_data, to_ML
from tagger.plot.basic import loss_history
import models

#Third parties
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot

# GLOBAL PARAMETERS TO BE DEFINED WHEN TRAINING
BATCH_SIZE = 1024
EPOCHS = 100
VALIDATION_SPLIT = 0.1 # 10% of training set will be used for validation set. 

# Sparsity parameters
I_SPARSITY = 0.0 #Initial sparsity
F_SPARSITY = 0.6 #Final sparsity

# Loss function parameters
GAMMA = 0.3 #Loss weight for classification, 1-GAMMA for pt regression

def prune_model(model, num_samples):
    """
    Pruning settings for the model. Return the pruned model
    """

    print("Begin pruning the model...")

    #Calculate the ending step for pruning
    end_step = np.ceil(num_samples / BATCH_SIZE).astype(np.int32) * EPOCHS

    #Define the pruned model
    pruning_params = {'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=I_SPARSITY, final_sparsity=F_SPARSITY, begin_step=0, end_step=end_step)}
    pruned_model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)

    pruned_model.compile(optimizer='adam',
                            loss={'prune_low_magnitude_jet_id_output': 'binary_crossentropy', 'prune_low_magnitude_pT_output': 'mean_squared_error'},
                            loss_weights={'prune_low_magnitude_jet_id_output': GAMMA,  'prune_low_magnitude_pT_output': 1 - GAMMA}, metrics=['accuracy'])

    print(pruned_model.summary())

    return pruned_model

def train(out_dir, percent, model_name):

    #Remove output dir if exists
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
        print(f"Removed existing directory: {out_dir}")

    #Load the data, class_labels and input variables name
    #not really using input variable names to be honest
    data, class_labels, input_vars = load_data("training_data/", percentage=percent)

    #Make into ML-like data for training
    X_train, y_train, pt_target_train = to_ML(data, class_labels)

    #Get input shape
    input_shape = X_train.shape[1:] #First dimension is batch size
    output_shape = y_train.shape[1:]

    #Dynamically get the model
    try:
        model_func = getattr(models, model_name)
        model = model_func(input_shape, output_shape)  # Assuming the model function doesn't require additional arguments
    except AttributeError:
        raise ValueError(f"Model '{model_name}' is not defined in the 'models' module.")

    #Train it with a pruned model
    num_samples = X_train.shape[0] * (1 - VALIDATION_SPLIT)
    pruned_model = prune_model(model, num_samples)

    #Now fit to the data
    callbacks = [tfmot.sparsity.keras.UpdatePruningStep(),
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', verbose=2, patience=5)]

    history = pruned_model.fit({'model_input': X_train},
                            {'prune_low_magnitude_jet_id_output': y_train, 'prune_low_magnitude_pT_output': pt_target_train},
                            epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2, validation_split=VALIDATION_SPLIT, callbacks = [callbacks])
    
    #Export the model
    model_export = tfmot.sparsity.keras.strip_pruning(pruned_model)

    export_path = os.path.join(out_dir, "saved_model.h5")
    model_export.save(export_path)
    print(f"Model saved to {export_path}")

    #Produce some basic plots with the training for diagnostics
    plot_path = os.path.join(out_dir, "plots")
    os.makedirs(plot_path, exist_ok=True)

    #Plot
    loss_history(plot_path, history)

    return

if __name__ == "__main__":

    parser = ArgumentParser()

    #Making input arguments
    parser.add_argument('--make-data', action='store_true', help='Prepare the data if set.')
    parser.add_argument('-i','--input', default='/eos/cms/store/cmst3/group/l1tr/sewuchte/l1teg/fp_ntuples_v131Xv9/baselineTRK_4param_021024/All200.root' , help = 'Path to input training data')
    parser.add_argument('-s','--step', default='100MB' , help = 'The maximum memory size to process input root file')

    #Training argument
    parser.add_argument('-o','--output', default='output/baseline', help = 'Output model directory path, also save evaluation plots')
    parser.add_argument('-p','--percent', default=100, type=int, help = 'Percentage of how much processed data to train on')
    parser.add_argument('-m','--model', default='baseline', help = 'Model object name to train on')
    
    args = parser.parse_args()

    #Either make data or start the training
    if args.make_data:
        make_data(infile=args.input, step_size=args.step) #Write to training_data/, can be specified using outdir, but keeping it simple here for now
    else:
        train(args.output, args.percent, model_name=args.model)
