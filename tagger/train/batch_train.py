import os
from argparse import ArgumentParser

# Third parties
import numpy as np

# Import from other modules
from tagger.data.tools import load_data, to_ML
from tagger.data.make_c2v_data import *
from tagger.model.common import fromFolder, fromYaml
from tagger.plot.basic import basic
from tagger.data.config import N_PARTICLES

import json


def train(yaml_config, training_dir, out_dir, percent):
    # Load the data, class_labels and input variables name, not really using input variable names to be honest
    X_train, y_train = sample_and_merge_chunks(
        outdir=training_dir,
        sample_fraction=0.01*float(percent),  
        random_seed=42,
        n_particles = N_PARTICLES,
        file_type='train'
    )
        
    ebops_values =  [50000,100000,500000,750000,1000000,2500000,5000000]
    for ebops in ebops_values:
    
        model_dir = out_dir + "_" + str(ebops)
    
        model = fromYaml(args.yaml_config, model_dir)
        model.set_labels(
            [ "pt", "pt_rel", "pt_log", "deta", "dphi", "mass", "isPhoton", "isElectronPlus", "isElectronMinus", "isMuonPlus", "isMuonMinus", "isNeutralHadron",
                    "isChargedHadronPlus", "isChargedHadronMinus", "z0", "dxy", "isfilled", "puppiweight" ],
            [ "jet_pt_phys","jet_genmatch_pt" ],
            { "b": 0,"charm": 1, "light": 2, "gluon": 3 }
        )


        # Get input shape
        input_shape = X_train.shape[1:]  # First dimension is batch size
        output_shape = y_train.shape[1:]
        
        dxy_noise = np.random.normal(loc=0.0, scale=0.0, size=(X_train.shape[0], X_train.shape[1], 1))
        X_train[:,:,15:16] *= 0#dxy_noise
        features = [0,1,2,3,4,14,15,16,17]
        X_train = X_train[:,:,features]
        input_shape = X_train.shape[1:]  # First dimension is batch size
        print(input_shape)

        num_training_samples = X_train.shape[0]
        print(f"With {num_training_samples} jets")
        model.build_model(input_shape, output_shape)
        # Train it with a pruned model
        
        print("Training with ",input_shape," jets")
        model.compile_model(num_training_samples,ebops)
        model.fit(X_train,y_train)

        model.save()

        model.plot_loss()

    return


if __name__ == "__main__":

    parser = ArgumentParser()
    # Training argument
    parser.add_argument(
        '-o', '--output', default='output/baseline', help='Output model directory path, also save evaluation plots'
    )
    parser.add_argument(
        '-y', '--yaml_config', default='tagger/model/configs/baseline_larger.yaml', help='YAML config for model'
    )
    
    parser.add_argument(
        '-i', '--input', default='/eos/project/c/cms-l1t-jet-tagger/Collid2V/processed_dataset_64_candidates/'
    )
    
    parser.add_argument(
        '-p', '--percent', default=10
    )

    # Basic ploting
    parser.add_argument('--plot-basic', action='store_true', help='Plot all the basic performance if set')


    args = parser.parse_args()

    if args.plot_basic:
        
        X_test, y_test = sample_and_merge_chunks(
            outdir=args.input,
            sample_fraction=1,  
            random_seed=42,
            n_particles = N_PARTICLES,
            file_type='test'
            )
        
        ebops_values =  [50000,100000,500000,750000,1000000,2500000,5000000]
        for ebops in ebops_values:
            # All the basic plots!
            model_dir = args.output + "_" + str(ebops)
            model = fromFolder(model_dir)
            # Get input shape
            input_shape = X_test.shape[1:]  # First dimension is batch size
            output_shape = y_test.shape[1:]
            
            dxy_noise = np.random.normal(loc=0.0, scale=0.0, size=(X_test.shape[0], X_test.shape[1], 1))
            X_test[:,:,15:16] *= 0#dxy_noise
            features = [0,1,2,3,4,14,15,16,17]
            X_test = X_test[:,:,features]
            results = basic(model, X_test,y_test)

    else:
        train(args.yaml_config, args.input, args.output,args.percent,args.ebops )