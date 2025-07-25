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
from tagger.data.tools import load_data, to_ML, calculate_scale, fit_scale
from tagger.model.common import fromFolder, fromYaml
from tagger.plot.basic import basic

from tagger.train.train import save_test_data, train_weights

def train_model_wrapper(config):  
    
    os.system('cp -r /builds/ml_l1/TrainTagger/traindata .')
    os.system('cp -r /builds/ml_l1/TrainTagger/testdata .')
    
    # os.system('cp -r /root/TrainTagger/traindata .')
    # os.system('cp -r /root/TrainTagger/testdata .')
     
    with open("traindata/train_features", "rb") as fp:   # Unpickling
        train_dict = pickle.load(fp)
    with open("testdata/test_features", "rb") as fp:   # Unpickling
        test_dict = pickle.load(fp)
    with open("traindata/train_labels", "rb") as fp:   # Unpickling
        train_labels = pickle.load(fp)
    with open("testdata/test_labels", "rb") as fp:   # Unpickling
        test_labels = pickle.load(fp)
    with open("traindata/train_weights", "rb") as fp:   # Unpickling
        train_weights = pickle.load(fp)
    with open("testdata/test_weights", "rb") as fp:   # Unpickling
        test_weights = pickle.load(fp)
 
    train_weight = train_weights[config['sample_weight']]
    random_indices = np.random.choice(int(len(train_labels)), int(0.1*len(train_labels)))
    
    sub_X = {key : [] for key in train_dict}
    sub_y = []
    sub_weight = []
    
    for X_i in random_indices:
        for key in sub_X:
            sub_X[key].append(train_features[key][X_i])
        sub_X.append(train_dict[X_i])
        sub_y.append(train_labels[X_i])
        sub_weight.append(train_weight[X_i])
        
    sub_X["label"] = np.array(sub_y,dtype=int)
    sub_X["weights"] = sub_weight

    learner = ydf.GradientBoostedTreesLearner(  weights= "weights",
                                                max_depth = config['max_depth'],
                                                use_hessian_gain=config['use_hessian_gain'],
                                                num_candidate_attributes_ratio=config['num_candidate_attributes_ratio'],
                                                min_examples=config['min_examples'],
                                                shrinkage=config['shrinkage'],
                                                label="label",
                                                num_trees = 100,
                                                num_threads = 4,
                                                discretize_numerical_columns=True,
        )
    tuning_model = learner.train(sub_X, verbose=0)
    
    test_weight = test_weights[config['sample_weight']]
    random_test_indices = np.random.choice(int(len(test_labels)), int(0.1*len(test_labels)))
    
    sub_X_test =  {key : [] for key in test_dict}
    sub_y_test = []
    sub_weight_test = []
    
    for test_i in random_test_indices:
        sub_X_test.append(test_features[test_i])
        sub_y_test.append(test_labels[test_i])
        sub_weight_test.append(test_weight[test_i])
        
    sub_X_test["label"] = np.array(sub_y_test,dtype=int)
    sub_X_test["weights"] = sub_weight_test
    #print(model.evaluate(test_dataset))
    return {'loss': tuning_model.evaluate(sub_X_test).loss}


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
            num_samples=10,
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
            "shrinkage":tune.loguniform(0.01, 0.5, base=10),
            "sample_weight" : tune.choice(["none", "ptref", "onlyclass"])
        },
    )
    results = tuner.fit()
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
    sample_weights_train = {}
    # Calculate the sample weights for training
    for weight_choice in sample_weight_choices:
        sample_weight_train = train_weights(
            y_train,
            reco_pt_train,
            class_labels,
            weightingMethod=weight_choice,
            debug=model.run_config['debug'],
        )
        
        sample_weights_train[weight_choice] = sample_weight_train
        
    if model.run_config['debug']:
        print("DEBUG - Checking sample_weight:")
        print(sample_weight_train)
        
    sample_weights_test = {}
    # Calculate the sample weights for training
    for weight_choice in sample_weight_choices:
        sample_weight_test = train_weights(
            y_test,
            reco_pt_test,
            class_labels,
            weightingMethod=weight_choice,
            debug=model.run_config['debug'],
        )
        
        sample_weights_test[weight_choice] = sample_weight_test
        
    if model.run_config['debug']:
        print("DEBUG - Checking sample_weight:")
        print(sample_weight_test)

    # Get input shape
    input_shape = X_train.shape[1:]  # First dimension is batch size
    output_shape = y_train.shape[1:]

    model.build_model(input_shape, output_shape)
    # Train it with a pruned model
    num_samples = X_train.shape[0] * (1 - model.training_config['validation_split'])
    model.compile_model(num_samples)
    
    X_train_dict = {'pt':[],'pt_rel':[],'pt_log':[],
                        'delta':[],'pid':[],'z0':[],'dxy':[],
                        'puppiweight':[],'quality':[],
                        'avg_pt':[],'avg_pt_rel':[],'avg_pt_log':[],
                        'avg_deta':[],'avg_dphi':[],'avg_z0':[],'avg_dxy':[],
                        'avg_puppiweight':[],'avg_quality':[],
                        'std_pt':[],'std_pt_rel':[],'std_pt_log':[],
                        'std_deta':[],'std_dphi':[],'std_z0':[],'std_dxy':[],
                        'std_puppiweight':[],'std_quality':[]}
    y_train_array = []
        
    for ibatch,batch in enumerate(X_train):
            if ibatch % 250000 == 0:
                print(ibatch , " out of ", len(X_train) )
            '''
              0 pt
              1 pt_rel
              2 pt_log
              3 deta
              4 dphi
              5 mass
              6 isPhoton
              7 isElectronPlus
              8 isElectronMinus
              9 isMuonPlus
              10 isMuonMinus
              11 isNeutralHadron
              12 isChargedHadronPlus
              13 isChargedHadronMinus
              14 z0
              15 dxy
              16 isfilled
              17 puppiweight
              18 emid
              19 quality
            '''            
            X_train_dict['pt'].append(np.array([[batch[j,0]] for j in range(len(batch))]))
            X_train_dict['avg_pt'].append(np.mean([[batch[j,0]] for j in range(len(batch))]))
            X_train_dict['std_pt'].append(np.std([[batch[j,0]] for j in range(len(batch))]))
            X_train_dict['pt_rel'].append(np.array([[batch[j,1]] for j in range(len(batch))]))
            X_train_dict['avg_pt_rel'].append(np.mean([[batch[j,1]] for j in range(len(batch))]))
            X_train_dict['std_pt_rel'].append(np.std([[batch[j,1]] for j in range(len(batch))]))
            X_train_dict['pt_log'].append(np.array([[batch[j,2]] for j in range(len(batch))]))
            X_train_dict['avg_pt_log'].append(np.mean([[batch[j,2]] for j in range(len(batch))]))
            X_train_dict['std_pt_log'].append(np.std([[batch[j,2]] for j in range(len(batch))]))
            X_train_dict['delta'].append(np.array([[batch[j,3],batch[j,4]] for j in range(len(batch))]))
            X_train_dict['avg_deta'].append(np.mean([[batch[j,3],batch[j,4]] for j in range(len(batch))]))
            X_train_dict['std_deta'].append(np.std([[batch[j,3],batch[j,4]] for j in range(len(batch))]))
            X_train_dict['avg_dphi'].append(np.mean([[batch[j,3],batch[j,4]] for j in range(len(batch))]))
            X_train_dict['std_dphi'].append(np.std([[batch[j,3],batch[j,4]] for j in range(len(batch))]))
            X_train_dict['pid'].append(np.array([[batch[j,6],batch[j,7],batch[j,8],batch[j,9],batch[j,10],batch[j,11],batch[j,12],batch[j,13],batch[j,18]] for j in range(len(batch))]))
            X_train_dict['z0'].append(np.array([[batch[j,14]] for j in range(len(batch))]))
            X_train_dict['avg_z0'].append(np.mean([[batch[j,14]] for j in range(len(batch))]))
            X_train_dict['std_z0'].append(np.std([[batch[j,14]] for j in range(len(batch))]))
            X_train_dict['dxy'].append(np.array([[batch[j,15]] for j in range(len(batch))]))
            X_train_dict['avg_dxy'].append(np.mean([[batch[j,15]] for j in range(len(batch))]))
            X_train_dict['std_dxy'].append(np.std([[batch[j,15]] for j in range(len(batch))]))
            X_train_dict['puppiweight'].append(np.array([[batch[j,17]] for j in range(len(batch))]))
            X_train_dict['avg_puppiweight'].append(np.mean([[batch[j,17]] for j in range(len(batch))]))
            X_train_dict['std_puppiweight'].append(np.std([[batch[j,17]] for j in range(len(batch))]))
            X_train_dict['quality'].append(np.array([[batch[j,19]] for j in range(len(batch))]))
            X_train_dict['avg_quality'].append(np.mean([[batch[j,19]] for j in range(len(batch))]))
            X_train_dict['std_quality'].append(np.std([[batch[j,19]] for j in range(len(batch))]))

            index = np.where(y_train[ibatch] == 1)
            y_train_array.append(index[0][0])
            
    X_test_dict = {'pt':[],'pt_rel':[],'pt_log':[],
                        'delta':[],'pid':[],'z0':[],'dxy':[],
                        'puppiweight':[],'quality':[],
                        'avg_pt':[],'avg_pt_rel':[],'avg_pt_log':[],
                        'avg_deta':[],'avg_dphi':[],'avg_z0':[],'avg_dxy':[],
                        'avg_puppiweight':[],'avg_quality':[],
                        'std_pt':[],'std_pt_rel':[],'std_pt_log':[],
                        'std_deta':[],'std_dphi':[],'std_z0':[],'std_dxy':[],
                        'std_puppiweight':[],'std_quality':[]}
    y_test_array = []
        
    for ibatch,batch in enumerate(X_test):
            if ibatch % 250000 == 0:
                print(ibatch , " out of ", len(X_test) )
                        X_test_dict['pt'].append(np.array([[batch[j,0]] for j in range(len(batch))]))
            X_test_dict['avg_pt'].append(np.mean([[batch[j,0]] for j in range(len(batch))]))
            X_test_dict['std_pt'].append(np.std([[batch[j,0]] for j in range(len(batch))]))
            X_test_dict['pt_rel'].append(np.array([[batch[j,1]] for j in range(len(batch))]))
            X_test_dict['avg_pt_rel'].append(np.mean([[batch[j,1]] for j in range(len(batch))]))
            X_test_dict['std_pt_rel'].append(np.std([[batch[j,1]] for j in range(len(batch))]))
            X_test_dict['pt_log'].append(np.array([[batch[j,2]] for j in range(len(batch))]))
            X_test_dict['avg_pt_log'].append(np.mean([[batch[j,2]] for j in range(len(batch))]))
            X_test_dict['std_pt_log'].append(np.std([[batch[j,2]] for j in range(len(batch))]))
            X_test_dict['delta'].append(np.array([[batch[j,3],batch[j,4]] for j in range(len(batch))]))
            X_test_dict['avg_deta'].append(np.mean([[batch[j,3],batch[j,4]] for j in range(len(batch))]))
            X_test_dict['std_deta'].append(np.std([[batch[j,3],batch[j,4]] for j in range(len(batch))]))
            X_test_dict['avg_dphi'].append(np.mean([[batch[j,3],batch[j,4]] for j in range(len(batch))]))
            X_test_dict['std_dphi'].append(np.std([[batch[j,3],batch[j,4]] for j in range(len(batch))]))
            X_test_dict['pid'].append(np.array([[batch[j,6],batch[j,7],batch[j,8],batch[j,9],batch[j,10],batch[j,11],batch[j,12],batch[j,13],batch[j,18]] for j in range(len(batch))]))
            X_test_dict['z0'].append(np.array([[batch[j,14]] for j in range(len(batch))]))
            X_test_dict['avg_z0'].append(np.mean([[batch[j,14]] for j in range(len(batch))]))
            X_test_dict['std_z0'].append(np.std([[batch[j,14]] for j in range(len(batch))]))
            X_test_dict['dxy'].append(np.array([[batch[j,15]] for j in range(len(batch))]))
            X_test_dict['avg_dxy'].append(np.mean([[batch[j,15]] for j in range(len(batch))]))
            X_test_dict['std_dxy'].append(np.std([[batch[j,15]] for j in range(len(batch))]))
            X_test_dict['puppiweight'].append(np.array([[batch[j,17]] for j in range(len(batch))]))
            X_test_dict['avg_puppiweight'].append(np.mean([[batch[j,17]] for j in range(len(batch))]))
            X_test_dict['std_puppiweight'].append(np.std([[batch[j,17]] for j in range(len(batch))]))
            X_test_dict['quality'].append(np.array([[batch[j,19]] for j in range(len(batch))]))
            X_test_dict['avg_quality'].append(np.mean([[batch[j,19]] for j in range(len(batch))]))
            X_test_dict['std_quality'].append(np.std([[batch[j,19]] for j in range(len(batch))]))
            y_test_array.append(0)
            
        
    X_test_dict["label"] =  np.array(y_test_array,dtype=int) 
    
    os.makedirs('traindata', exist_ok=True)
    os.makedirs('testdata', exist_ok=True)
    with open("traindata/train_features", "wb") as fp:   #Pickling
        pickle.dump(X_train_dict, fp)
    with open("testdata/test_features", "wb") as fp:   #Pickling
        pickle.dump(X_test_dict, fp)
    with open("traindata/train_labels", "wb") as fp:   #Pickling
        pickle.dump(y_train_array, fp)
    with open("testdata/test_labels", "wb") as fp:   #Pickling
        pickle.dump(y_test_array, fp)
    with open("traindata/train_weights", "wb") as fp:   #Pickling
        pickle.dump(sample_weights_train, fp)
    with open("testdata/test_weights", "wb") as fp:   #Pickling
        pickle.dump(sample_weights_test, fp)


    results = tune_bdt()
    print(f"Best hyperparameters found were: {results.get_best_result().config} | loss: {results.get_best_result().metrics['loss']}")
        
    learner = ydf.GradientBoostedTreesLearner(  max_depth = results.get_best_result().config['max_depth'],
                                                weights = 'weights',
                                                use_hessian_gain=results.get_best_result().config['use_hessian_gain'],
                                                num_candidate_attributes_ratio=results.get_best_result().config['num_candidate_attributes_ratio'],
                                                min_examples=results.get_best_result().config['min_examples'],
                                                shrinkage=results.get_best_result().config['shrinkage'],
                                                label="label",
                                                num_trees=500,
                                                num_threads = os.cpu_count(),
                                                discretize_numerical_columns=True,
        )

    
    X_train_dict["label"] = np.array(y_train_array,dtype=int)
    X_train_dict["weights"] = sample_weight_train
    
    X_test_dict["label"] = np.array(y_test_array,dtype=int)
    X_test_dict["weights"] = sample_weight_test
    
    model.jet_model = learner.train(X_test_dict, verbose=2)
    print(model.jet_model.describe())
    print(model.jet_model.evaluate(X_test_dict))
    print(model.jet_model.analyze(X_test_dict, sampling=0.1))
    model.save("model_tuned")
    results = basic(model, args.signal_processes)

