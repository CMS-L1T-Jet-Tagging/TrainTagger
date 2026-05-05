import os
from argparse import ArgumentParser

# Import from other modules
from tagger.data.tools import make_data, _split_flavor,_make_nn_inputs,_get_puppicand_fields,to_ML
from tagger.data.config import EXTRA_FIELDS, FILTER_PATTERN, INPUT_TAG, N_PARTICLES

from argparse import ArgumentParser

import uproot
from tqdm import tqdm
import numpy as np

from tagger.model.common import fromFolder

parser = ArgumentParser()

parser.add_argument(
        '-i', '--input', default='output/trf', help='input model for jet embedding'
    )

parser.add_argument(
        '-o', '--output', default='event_training_data_trf/', help='output folder for datafiles'
    )

files = {
    'TT_PU200':{'path':'TT_PU200.root ','fraction':20, 'label':1},
     'HH_4b':{'path':'GluGluHHTo4B_PU200.root','fraction':100, 'label':2},
     'minbias':{'path':'MinBias_PU200.root ','fraction':100, 'label':0},
        #   'SMJ_cascadeA':{'path':'SMJ_cascadeA.root ','fraction':100, 'label':3},
        #   'SMJ_cascadeC':{'path':'SMJ_cascadeC.root ','fraction':100, 'label':4},
        #   'SVJ_250':{'path':'SVJ_250.root','fraction':100, 'label':5},
        #   'SVJ_500':{'path':'SVJ_500.root ','fraction':100, 'label':6},
        #   'SUEP':{'path':'SUEP.root ','fraction':100, 'label':7},
         }

args = parser.parse_args()

max_number_jets = 12
max_number_candidates = 16

base_dir = '/eos/cms/store/cmst3/group/l1tr/sewuchte/l1teg/fp_jettuples_191125_151X/'

jet_class_model = args.input
output_dir = args.output

model = fromFolder(jet_class_model)

for root_file in files.keys():

    outdir = output_dir + root_file
    infile = base_dir+files[root_file]['path']

    os.makedirs(outdir, exist_ok=True)
    print("Output directory:", outdir)

    # Loop through the entries
    num_entries = uproot.open(infile)["outnano/Jets"].num_entries
    num_entries_done = 0
    chunk = 0
    
    total_entries = int(0.01*files[root_file]['fraction']*num_entries)
    
    jet_class_labels = []
    jet_class_predictions = []
    jet_pt_predictions = []
    jet_embeddings = []
    
    candidate_features = []
    jet_features = []
    
    bonus_jet_info = []
    bonus_event_info = []
    
    event_label = []
        

    for data in uproot.iterate(infile, filter_name=FILTER_PATTERN, how="zip", step_size="1GB", max_workers=24):
        #pbar.set_description(f'Processing chunk {chunk}')

        # Define jet kinematic cuts
        jet_cut = (data['jet_pt_phys'] > 15) & (np.abs(data['jet_eta_phys']) < 2.4) & (data['jet_reject'] == 0)
        data = data[jet_cut]
        
        
        data_split, class_labels = _split_flavor(data)
        
        _make_nn_inputs(data_split, INPUT_TAG, N_PARTICLES)
        extra_features = _get_puppicand_fields(EXTRA_FIELDS)

        # Save them to a root file
        save_fields = ['nn_inputs', 'class_label', 'target_pt', 'target_pt_phys'] + extra_features

        # Filter the data_split to only include save_fields
        filtered_data = {field: data_split[field] for field in save_fields}
        
        num_entries_done += len(data_split)  # count after cuts
        
        X, y, pt_target, truth_pt, reco_pt, jet_features_from_ml,event = to_ML(filtered_data,class_labels )
        
        jet_class_predict, jet_pt_predict = model.predict(X)
        jet_embeddings_predict = model.embedding_predict(X)
        

        current_jet = 0
        for ievent in np.unique(filtered_data['event']):
            event_label.append(files[root_file]['label'])
            
            jet_indices = np.where(filtered_data['event'] == ievent)
            
            event_jet_class_labels = []
            event_jet_class_predictions = []
            event_jet_pt_predictions = []
            event_jet_embeddings = []
            event_candidate_features = []
            event_jet_features = []
            event_bonus_jet_info = []
            #1544064
            num_jets = 0
            event_ht = 0
            
            for ijet in range(max_number_jets):
                if ijet < len(filtered_data['jet_pt_phys'][jet_indices]):

                    jet_feature_vector = np.zeros(3)
                    event_ht += filtered_data['jet_pt_phys'][jet_indices][ijet]
                    jet_feature_vector[0] = filtered_data['jet_pt_phys'][jet_indices][ijet]
                    jet_feature_vector[1] = filtered_data['jet_eta_phys'][jet_indices][ijet]
                    jet_feature_vector[2] = filtered_data['jet_phi_phys'][jet_indices][ijet]
                    
                    event_bonus_jet_info.append(np.array([filtered_data['jet_pt'][jet_indices][ijet],filtered_data['jet_genmatch_pt'][jet_indices][ijet]]))                    
                    
                    event_jet_class_labels.append(y[current_jet])
                    
                    event_jet_class_predictions.append(jet_class_predict[current_jet])
                    event_jet_pt_predictions.append(jet_pt_predict[current_jet])
                    
                    event_jet_embeddings.append(jet_embeddings_predict[current_jet])
                    
                    event_jet_features.append(jet_feature_vector)
                    
                    event_candidate_features.append(X[current_jet])
                        
                    current_jet += 1
                    num_jets += 1
                    
                else:
                                        
                    event_jet_class_labels.append(np.zeros(8))
                    event_jet_class_predictions.append(np.zeros(8))
                    event_jet_pt_predictions.append(0)
                    
                    event_bonus_jet_info.append(np.array([0,0]))
                    
                    event_jet_embeddings.append(np.zeros(jet_embeddings_predict.shape[1]))
                    
                    event_candidate_features.append(np.zeros((16,20)))
                    event_jet_features.append(np.zeros(3))
                                

            jet_class_labels.append(event_jet_class_labels)
            jet_class_predictions.append(event_jet_class_predictions)
            jet_pt_predictions.append(event_jet_pt_predictions)
            jet_embeddings.append(event_jet_embeddings)
            
            candidate_features.append(event_candidate_features)
            jet_features.append(event_jet_features)
            
            bonus_jet_info.append(event_bonus_jet_info)
            
            bonus_event_info.append([num_jets,event_ht])
                    
        
        # if num_entries_done > total_entries:
        #     break
        
    
    jet_class_labels_vector = np.stack(jet_class_labels)
    jet_class_predictions_vector = np.stack(jet_class_predictions)
    jet_pt_predictions_vector = np.stack(jet_pt_predictions)
    jet_embeddings_vector = np.stack(jet_embeddings)
        
    candidate_features_vector = np.stack(candidate_features)
    jet_features_vector = np.stack(jet_features)
    
    event_label_vector = np.array(event_label)
    bonus_event_info_vector = np.array(bonus_event_info)
    
    bonus_jet_info_vector = np.stack(bonus_jet_info)
    
    print(jet_class_labels_vector.shape)
    print(jet_class_predictions_vector.shape)
    print(jet_pt_predictions_vector.shape)
    print(jet_embeddings_vector.shape)
    print(candidate_features_vector.shape)
    print(jet_features_vector.shape)
    print(event_label_vector.shape)
    print(bonus_jet_info_vector.shape)
    print(bonus_event_info_vector.shape)
    
    
    np.save(outdir+'/jet_class_labels.npy',jet_class_labels_vector)
    np.save(outdir+'/jet_class_predictions.npy',jet_class_predictions_vector)
    np.save(outdir+'/jet_pt_predictions.npy',jet_pt_predictions_vector)
    np.save(outdir+'/jet_embeddings.npy',jet_embeddings_vector)
    np.save(outdir+'/candidate_features.npy',candidate_features_vector)
    np.save(outdir+'/jet_features.npy',jet_features_vector)
    np.save(outdir+'/event_label.npy',event_label_vector)
    np.save(outdir+'/bonus_jet_info.npy',bonus_jet_info_vector)
    np.save(outdir+'/bonus_event_info.npy',bonus_event_info_vector)
    
    


