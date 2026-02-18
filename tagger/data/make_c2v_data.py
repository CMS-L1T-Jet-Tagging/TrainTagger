import os
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
# Import from other modules
from tagger.data.tools import make_data
from c2v_utils import process_single_file
import numpy as np
from tqdm import tqdm
import shutil
import json

from tagger.plot.basic import plot_input_vars

if __name__ == "__main__":

    input_directory = '/eos/project/f/foundational-model-dataset/samples/production_final/'
    outdir = 'training_data/'  
    if os.path.exists(outdir):
        confirm = input(f"The directory '{outdir}' already exists. Do you want to delete it and continue? [y/n]: ")
        if confirm.lower() == 'y':
            shutil.rmtree(outdir)
            print(f"Deleted existing directory: {outdir}")
        else:
            print("Exiting without making changes.")

    # Create output training dataset
    os.makedirs(outdir+'plots/', exist_ok=True)
    print("Output directory:", outdir)
    
    
    
    max_files_per_process = 10
    
    test_fraction = 0.1
    
    #process_folders = {'QCD_HT50tobb': 'QCD_HT50tobb-NEVENT10000-RS25'}
    
    # process_folders = {'HH_bbtautau': 'HH_bbtautau-NEVENT10000-RS22'}
    
    process_folders = {'HH_4b' :'HH_4b-NEVENT10000-RS20',
                       'HH_bbgammagamma': 'HH_bbgammagamma-NEVENT10000-RS21',
                       'HH_bbtautau': 'HH_bbtautau-NEVENT10000-RS22',
                       'QCD_HT50toInf': 'QCD_HT50toInf-NEVENT10000-RS26',
                       'tt0123j_5f_ckm_LO_MLM_hadronic':'tt0123j_5f_ckm_LO_MLM_hadronic-NEVENT10000-RS28',
                       'ggHgluglu':'ggHgluglu-NEVENT10000-RS16',
                       'ggHtautau':'ggHtautau-NEVENT10000-RS17',
                       'ggHbb':'ggHbb-NEVENT10000-RS13',
                       'ggHcc':'ggHcc-NEVENT10000-RS14',
                       'VBFHbb':'VBFHbb-NEVENT10000-RS35',
                       'VBFHcc':'VBFHcc-NEVENT10000-RS36',
                       'VBFHgammagamma':'VBFHgammagamma-NEVENT10000-RS37',
                       'VBFHgluglu':'VBFHgluglu-NEVENT10000-RS38',
                       'VBFHtautau':'VBFHtautau-NEVENT10000-RS39',
                       'WJetsToLNu_13TeV-madgraphMLM-pythia8':'WJetsToLNu_13TeV-madgraphMLM-pythia8-NEVENT10000-RS44'}
    
    full_feature_array = []
    full_pt_target_array = []
    full_class_label_array = []
    full_reco_pt_array = []

    for iprocess, process in tqdm(enumerate(process_folders.keys())):
        for ifile in tqdm(range(max_files_per_process)):
            input_file = input_directory + process + '/' + process_folders[process] + str(ifile+1).zfill(6) + '.parquet'

            X_train, y_train, pt_target_train, reco_pt_train = process_single_file(input_file)
            
            full_feature_array.append(X_train)
            full_pt_target_array.append(pt_target_train)
            full_class_label_array.append(y_train)
            full_reco_pt_array.append(reco_pt_train)

            
    new_feature_array = np.concatenate( full_feature_array)
    pt_target_array = np.concatenate(full_pt_target_array)
    class_label_array = np.concatenate(full_class_label_array)
    reco_pt_array = np.concatenate(full_reco_pt_array)   
    
    print("Total number of jets: ",len(new_feature_array))
    
    p = np.random.permutation(len(new_feature_array))
    
    p_train, p_test = train_test_split(p, test_size=test_fraction, random_state=42)
    
    
    input_vars = ["pt", "pt_rel", "pt_log", "deta", "dphi", "mass",
                  "isPhoton", "isElectronPlus", "isElectronMinus", "isMuonPlus",
                  "isMuonMinus", "isNeutralHadron", "isChargedHadronPlus", "isChargedHadronMinus",
                  "z0", "dxy", "isfilled", "puppiweight", "emid", "quality"]
    
    extra_vars = ["jet_pt_phys", "jet_genmatch_pt"]
    
    class_labels = {"b": 0, "charm": 1, "light": 2, "gluon": 3,
                    "taup": 4, "taum": 5, "muon": 6, "electron": 7 }
    
    print(class_label_array)
  
    num_classes = len(class_labels.keys())
    class_label_array = np.eye(num_classes)[class_label_array]
     
    dataset_metadata_file = os.path.join(outdir, 'metadata.json')

    metadata = {
        "outputs": class_labels,
        "inputs": input_vars,
        "extras": extra_vars,
    }

    with open(dataset_metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)

    np.save(outdir+'X_train.npy', new_feature_array[p_train])
    np.save(outdir+'pt_target_train.npy', pt_target_array[p_train])
    np.save(outdir+'y_train.npy', class_label_array[p_train])
    np.save(outdir+'reco_pt_train.npy', reco_pt_array[p_train])
    
    np.save(outdir+'X_test.npy', new_feature_array[p_test])
    np.save(outdir+'pt_target_test.npy', pt_target_array[p_test])
    np.save(outdir+'y_test.npy', class_label_array[p_test])
    np.save(outdir+'reco_pt_test.npy', reco_pt_array[p_test])
    
    plot_input_vars(new_feature_array[p_train], class_label_array[p_train], input_vars, class_labels, outdir+'plots')
    
    
    
            