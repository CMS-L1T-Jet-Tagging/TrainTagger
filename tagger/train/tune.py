import os
import tempfile
from argparse import ArgumentParser

# Third parties
import numpy as np

# Import from other modules
from tagger.data.tools import load_data, to_ML
from tagger.model.common import fromFolder, fromYaml,fromDict
from tagger.plot.basic import basic

from tagger.train.train import train_weights

from functools import partial

from ray import tune
from ray.tune import Checkpoint
from ray.tune.schedulers import ASHAScheduler

from sklearn.metrics import auc, roc_curve

from ray.tune import Result


import tensorflow as tf

def train_jet_model(config, data_train, data_test, class_labels, input_vars, extra_vars):
    print(config['embedding_learning_rate'])
        
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        # tf.config.set_logical_device_configuration(
        #         gpu,
        #         [tf.config.LogicalDeviceConfiguration(memory_limit=1024)]  # MB
            #)
    config_dict = {'model': 'TransformerEmbeddingModel',
                'run_config' : {'verbose' : 2,
                                'debug' : True,
                                'num_threads' : 4},
                'model_config' : {'name' : 'transformer',
                                    'emb_layers' : [8,8],
                                    'transformer_layers' : [[2,4,2,8],[2,4,2,8]] ,#num_heads, mha_hidden_dim, num_dense_layers, dim_dense_layers
                                    'classification_layers' : [32,16],
                                    'regression_layers' : [10],
                                    'projection_dims' : config['projection_dims'],
                                    'kernel_initializer' : 'lecun_uniform',
                                    'projection_blocks' : config['projection_blocks']
                                    },
                'training_config' :{'weight_method': "onlyclass",
                                        'embedding_epochs' : 40,
                                        'finetuning_epochs' : 200,
                                        'batch_size' : config['batch_size'],
                                        'learning_rate' : config['learning_rate'],
                                        'embedding_lr' : config['embedding_learning_rate'],
                                        'masking_probability' : config['masking_probability'],
                                        "loss_temperature" : config['loss_temperature'],
                                        'validation_split' : 0.1,
                                        'loss_weights' : config['loss_weights'],
                                        'EarlyStopping_patience' : 20,
                                        'ReduceLROnPlateau_factor' : 0.5,
                                        'ReduceLROnPlateau_patience' : 10,
                                        'ReduceLROnPlateau_min_lr' : 0.00001},
                }

    model = fromDict(config_dict,'output/autoencoder')
    model.set_labels(
        input_vars,
        extra_vars,
        class_labels,
    )
        # Make into ML-like data for training
    X_train, y_train, pt_target_train, truth_pt_train, reco_pt_train = to_ML(data_train, class_labels)

    X_test, y_test, _, truth_pt_test, reco_pt_test = to_ML(data_test, class_labels)

        # Calculate the sample weights for training
    sample_weight = train_weights(
            y_train,
            reco_pt_train,
            class_labels,
            weightingMethod='onlyclass',
            debug=False)
    
    # Get input shape
    input_shape = X_train.shape[1:]  # First dimension is batch size
    output_shape = y_train.shape[1:]
        
    model.build_model(input_shape, output_shape)
    # Train it with a pruned model
    num_samples = X_train.shape[0] * (1 - model.training_config['validation_split'])
    model.compile_model(num_samples)
    model.fit(X_train, y_train, pt_target_train, sample_weight)

    model_outputs = model.predict(X_test)
    # Get classification outputs
    y_pred = model_outputs[0]
    pt_ratio = model_outputs[1]
    
    aucs = 0
    
    for i, class_label in enumerate(class_labels):

        # Get true labels and predicted probabilities for the current class
        # Extract the one-hot column for the current class
        y_true = y_test[:, i]
        y_score = y_pred[:, i]  # Predicted probabilities for the current class

        # Compute FPR, TPR, and AUC
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        aucs += roc_auc
        
    roc_score = roc_auc / len(class_labels)
    
    with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
        checkpoint = None
        if (i + 1) % 5 == 0:
            # This saves the model to the trial directory
            model.save()
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)

    tune.report({"roc_score": roc_score}, checkpoint=checkpoint)

if __name__ == "__main__":
    data_train, data_test, class_labels, input_vars, extra_vars = load_data("training_data/", percentage=1)
    
    train_fn = tune.with_parameters(
        train_jet_model,
        data_train=data_train,
        data_test=data_test,
        class_labels=class_labels,
        input_vars=input_vars,
        extra_vars=extra_vars
    )
    
    
    tuner = tune.Tuner(
        tune.with_resources(train_fn,{"cpu": 4,"gpu":0.1,"memory":10e9}),
        tune_config=tune.TuneConfig(
            scheduler=ASHAScheduler(metric="roc_score", mode="max"),
            num_samples=1000,
            max_concurrent_trials=10,
        ),
        run_config=tune.RunConfig(
            name="exp",
            stop={"roc_score": 0.8},
        ),
        param_space={
            "embedding_learning_rate": tune.uniform(0.001, 0.1),
            "learning_rate" :  tune.uniform(0.0001, 0.1),
            "projection_dims" : tune.choice([4, 8, 10, 16, 20]),
            "projection_blocks" : tune.choice([1, 2, 3, 4, 5]),
            "masking_probability" : tune.uniform(0.0, 0.8),
            "batch_size" : tune.choice([256,512,1024,2048,4096]),
            "loss_temperature" : tune.uniform(0.001,1.0),
            "loss_weights" : tune.choice([[1,1],[1,2],[2,1],[4,1],[1,4]])
        },
    )
    results = tuner.fit()
    

    # Get the result with the maximum test set `mean_accuracy`
    best_result = results.get_best_result(metric="roc_score", mode="max")
    print(best_result.config)
    # Get the result with the minimum `mean_accuracy`
    worst_performing_result=  results.get_best_result(
        metric="roc_score", mode="min"
    )
    print(worst_performing_result.config)
    
    result_df = best_result.metrics_dataframe
    
    print(result_df.head())

