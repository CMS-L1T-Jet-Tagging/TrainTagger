import numpy as np
import sys
import os
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
import threading

import matplotlib.pyplot as plt

import ydf

from argparse import ArgumentParser

from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler

# Import from other modules
from tagger.data.tools import load_data, to_ML
from tagger.model.common import fromFolder, fromYaml
from tagger.plot.basic import basic

from tagger.train.train import save_test_data, train_weights

def train_model_wrapper(config):  
    
    os.system('cp -r /builds/ml_l1/TrainTagger/traindata .')
    os.system('cp -r /builds/ml_l1/TrainTagger/testdata .')
    
    #os.system('cp -r /root/TrainTagger/traindata .')
    #os.system('cp -r /root/TrainTagger/testdata .')
     
    with open("traindata/train_features", "rb") as fp:   # Unpickling
        train_features = pickle.load(fp)
    with open("testdata/test_features", "rb") as fp:   # Unpickling
        test_features = pickle.load(fp)
    with open("traindata/train_labels", "rb") as fp:   # Unpickling
        train_labels = pickle.load(fp)
    with open("testdata/test_labels", "rb") as fp:   # Unpickling
        test_labels = pickle.load(fp)
    with open("traindata/train_weights", "rb") as fp:   # Unpickling
        train_weights = pickle.load(fp)
 
    train_weight = train_weights[config['sample_weight']]
    random_indices = np.random.choice(int(len(train_labels)), int(0.1*len(train_labels)))
    
    sub_X = []
    sub_y = []
    sub_weight = []
    
    for X_i in random_indices:
        sub_X.append(train_features[X_i])
        sub_y.append(train_labels[X_i])
        sub_weight.append(train_weight[X_i])
    
    train_dataset = {"label": np.array(sub_y,dtype=int), "feature": sub_X, "weights":sub_weight}
    
    if config['split_axis'] == 'AXIS_ALIGNED':

        learner = ydf.GradientBoostedTreesLearner(  weights= "weights",
                                                    max_depth = config['max_depth'],
                                                    use_hessian_gain=config['use_hessian_gain'],
                                                    num_candidate_attributes_ratio=config['num_candidate_attributes_ratio'],
                                                    min_examples=config['min_examples'],
                                                    shrinkage=config['shrinkage'],
                                                    label="label",
                                                    num_trees = config['num_trees'],
                                                    num_threads = 4,
                                                    discretize_numerical_columns=True,
        )
    
    elif config['split_axis'] == 'SPARSE_OBLIQUE':
        learner = ydf.GradientBoostedTreesLearner(  weights= "weights",
                                                    max_depth = config['max_depth'],
                                                    num_trees = config['num_trees'],
                                                    use_hessian_gain=config['use_hessian_gain'],
                                                    num_candidate_attributes_ratio=config['num_candidate_attributes_ratio'],
                                                    min_examples=config['min_examples'],
                                                    shrinkage=config['shrinkage'],
                                                    split_axis=config['split_axis'],
                                                    sparse_oblique_num_projections_exponent=config['sparse_oblique_num_projections_exponent'],
                                                    label="label",
                                                    num_threads = 4,
                                                    discretize_numerical_columns=True,
        )
    model = learner.train(train_dataset, verbose=0)
    
    random_test_indices = np.random.choice(int(len(test_labels)), int(0.1*len(test_labels)))
    
    sub_X_test = []
    sub_y_test = []
    
    for test_i in random_test_indices:
        sub_X_test.append(test_features[test_i])
        sub_y_test.append(test_labels[test_i])
        
    test_dataset  = {"label": np.array(sub_y_test,dtype=int), "feature": sub_X_test}
    #print(model.evaluate(test_dataset))
    return {'loss': model.evaluate(test_dataset).loss}


def tune_bdt():
    sched = AsyncHyperBandScheduler(
        time_attr="training_iteration", max_t=4000, grace_period=20
    )

    tuner = tune.Tuner(
        tune.with_resources(train_model_wrapper, resources={"cpu":24 , "gpu": 0.0}),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=sched,
            num_samples=100,
        ),
        run_config=tune.RunConfig(
            name="exp",
            stop={"training_iteration": 5000},
        ),
        param_space={
            "threads": 4,
            "max_depth": tune.randint(3, 6),
            "num_trees": tune.randint(100, 300),
            "use_hessian_gain":tune.choice([True,False]),
            "num_candidate_attributes_ratio":tune.uniform(0.1, 1.0),
            "min_examples":tune.randint(10,20),
            "shrinkage":tune.loguniform(0.0001, 0.1, base=10),
            "split_axis":tune.choice(['SPARSE_OBLIQUE','AXIS_ALIGNED']),
            "sparse_oblique_num_projections_exponent":tune.uniform(1.0, 2.0),
            "sample_weight" : tune.choice(["none", "ptref", "onlyclass"])
        },
    )
    results = tuner.fit(verbose=1)
    return results

if __name__ == "__main__":

    parser = ArgumentParser()
    # Training argument
    parser.add_argument(
        '-o', '--output', default='output/baseline', help='Output model directory path, also save evaluation plots'
    )
    parser.add_argument('-p', '--percent', default=50, type=int, help='Percentage of how much processed data to train on')
    parser.add_argument(
        '-y', '--yaml_config', default='tagger/model/configs/baseline_larger.yaml', help='YAML config for model'
    )
    parser.add_argument(
        '-sig', '--signal-processes', default=[], nargs='*', help='Specify all signal process for individual plotting'
    )

    args = parser.parse_args()
    
    model = fromYaml(args.yaml_config, args.output)
    # Load the data, class_labels and input variables name, not really using input variable names to be honest
    data_train, data_test, class_labels, input_vars, extra_vars = load_data("training_data/", percentage=args.percent)
    model.set_labels(
        input_vars,
        extra_vars,
        class_labels,
    )

    # Make into ML-like data for training
    X_train, y_train, pt_target_train, truth_pt_train, reco_pt_train = to_ML(data_train, class_labels)

    # Save X_test, y_test, and truth_pt_test for plotting later
    X_test, y_test, _, truth_pt_test, reco_pt_test = to_ML(data_test, class_labels)
    save_test_data(args.output, X_test, y_test, truth_pt_test, reco_pt_test)

    sample_weight_choices = ["none", "ptref", "onlyclass"]
    sample_weights = {}
    # Calculate the sample weights for training
    for weight_choice in sample_weight_choices:
        sample_weight = train_weights(
            y_train,
            reco_pt_train,
            class_labels,
            weightingMethod=weight_choice,
            debug=model.run_config['debug'],
        )
        
        sample_weights[weight_choice] = sample_weight
        
    if model.run_config['debug']:
        print("DEBUG - Checking sample_weight:")
        print(sample_weight)

    # Get input shape
    input_shape = X_train.shape[1:]  # First dimension is batch size
    output_shape = y_train.shape[1:]

    model.build_model(input_shape, output_shape)
    # Train it with a pruned model
    num_samples = X_train.shape[0] * (1 - model.training_config['validation_split'])
    model.compile_model(num_samples)
    
    X_train_array = []
    y_train_array = []
        
    for ibatch,batch in enumerate(X_train):
        vectors_list = []
        y_list = []
        if ibatch % 250000 == 0:
            print(ibatch , " out of ", len(X_train) )
        for icandidate,candidate in enumerate(X_train[ibatch]):
            if np.abs(np.sum(candidate)) > 0:
                vectors_list.append([candidate])
                #print(np.sum(candidate),candidate,y_train[icandidate])
        vectors = np.array(np.concatenate(vectors_list, axis=0)) 
        X_train_array.append(vectors)
        index = np.where(y_train[ibatch] == 1)
        y_train_array.append(index[0][0])
        
    X_test_array = []
    y_test_array = []
        
    for ibatch,batch in enumerate(X_test):
        vectors_list = []
        y_list = []
        if ibatch % 250000 == 0:
            print(ibatch , " out of ", len(X_test) )
        for icandidate,candidate in enumerate(X_test[ibatch]):
            if np.abs(np.sum(candidate)) > 0:
                vectors_list.append([candidate])
                #print(np.sum(candidate),candidate,y_test[icandidate])
        vectors = np.array(np.concatenate(vectors_list, axis=0)) 
        X_test_array.append(vectors)
        index = np.where(y_test[ibatch] == 1)
        y_test_array.append(index[0][0])
    
    os.makedirs('traindata', exist_ok=True)
    os.makedirs('testdata', exist_ok=True)
    with open("traindata/train_features", "wb") as fp:   #Pickling
        pickle.dump(X_train_array, fp)
    with open("testdata/test_features", "wb") as fp:   #Pickling
        pickle.dump(X_test_array, fp)
    with open("traindata/train_labels", "wb") as fp:   #Pickling
        pickle.dump(y_train_array, fp)
    with open("testdata/test_labels", "wb") as fp:   #Pickling
        pickle.dump(y_test_array, fp)
    with open("traindata/train_weights", "wb") as fp:   #Pickling
        pickle.dump(sample_weights, fp)


    results = tune_bdt()
    print(f"Best hyperparameters found were: {results.get_best_result().config} | loss: {results.get_best_result().metrics['loss']}")
        
    if results.get_best_result().config['split_axis'] == 'AXIS_ALIGNED':
            learner = ydf.GradientBoostedTreesLearner(  max_depth = results.get_best_result().config['max_depth'],
                                                      weights = 'weights',
                                                    use_hessian_gain=results.get_best_result().config['use_hessian_gain'],
                                                    num_candidate_attributes_ratio=results.get_best_result().config['num_candidate_attributes_ratio'],
                                                    min_examples=results.get_best_result().config['min_examples'],
                                                    shrinkage=results.get_best_result().config['shrinkage'],
                                                    label="label",
                                                    num_trees=results.get_best_result().config['num_trees'],
                                                    num_threads = os.cpu_count(),
                                                    discretize_numerical_columns=True,
        )
    
    elif results.get_best_result().config['split_axis'] == 'SPARSE_OBLIQUE':
        learner = ydf.GradientBoostedTreesLearner(  max_depth = results.get_best_result().config['max_depth'],
                                                    weights = 'weights',
                                                    use_hessian_gain=results.get_best_result().config['use_hessian_gain'],
                                                    num_candidate_attributes_ratio=results.get_best_result().config['num_candidate_attributes_ratio'],
                                                    min_examples=results.get_best_result().config['min_examples'],
                                                    shrinkage=results.get_best_result().config['shrinkage'],
                                                    split_axis=results.get_best_result().config['split_axis'],
                                                    sparse_oblique_num_projections_exponent=results.get_best_result().config['sparse_oblique_num_projections_exponent'],
                                                    label="label",
                                                    num_trees=results.get_best_result().config['num_trees'],
                                                    num_threads = os.cpu_count(),
                                                    discretize_numerical_columns=True,
        )
    
    train_dataset = {"label": np.array(y_train_array), "feature": X_train_array, "weights" : sample_weight}
    test_dataset  = {"label": np.array(y_test_array), "feature": X_test_array}
    model = learner.train(train_dataset, verbose=2)
    print(model.describe())
    print(model.evaluate(test_dataset))
    print(model.analyze(test_dataset, sampling=0.1))
    model.save("model_tuned")
    results = basic(model, args.signal_processes)

