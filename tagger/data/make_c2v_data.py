import os
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
# Import from other modules
from tagger.data.tools import make_data
from c2v_utils import process_single_file


if __name__ == "__main__":

    input_directory = '/eos/project/f/foundational-model-dataset/samples/production_final/'
    outdir = 'training_data'  
    if os.path.exists(outdir):
        confirm = input(f"The directory '{outdir}' already exists. Do you want to delete it and continue? [y/n]: ")
        if confirm.lower() == 'y':
            shutil.rmtree(outdir)
            print(f"Deleted existing directory: {outdir}")
        else:
            print("Exiting without making changes.")
            return

    # Create output training dataset
    os.makedirs(outdir, exist_ok=True)
    print("Output directory:", outdir)
    
    max_files_per_process = 10
    
    train_test_split = 0.1
    
    process_folders = ['QCD_HT50tobb']
    
    full_feature_array = []
    full_pt_target_array = []
    full_class_label_array = []
    full_reco_pt_array = []
    
    num_entries_done = 0
    chunk = 0
    
    for process in process_folders:
        for ifile in range(max_files_per_process):
            input_file = input_directory + process + '/' + process + '-NEVENT10000-RS2500000' + str(ifile) + '.parquet'

            X_train, y_train, pt_target_train, reco_pt_train = process_single_file(input_file)
            
            full_feature_array.append(X_train)
            full_pt_target_array.append(pt_target_train)
            full_class_label_array.append(y_train)
            full_reco_pt_array.append(reco_pt_train)
            
            num_entries_done += len(full_feature_array) 
            
            
            if chunk == 0:
                _save_dataset_metadata(outdir, class_labels, tag, extras)
            
            chunk += 1
            
    new_feature_array = np.stack( full_feature_array)
    pt_target_array = np.array(full_pt_target_array)
    class_label_array = np.array(full_class_label_array)
    reco_pt_array = np.array(full_reco_pt_array)   
    
    
    p = np.random.permutation(len(new_feature_array))
    
    p_train, p_test = train_test_split(p, test_size=train_test_split, random_state=42)

    
    np.save(new_feature_array[p_train], outdir+'X_train.npy')
    np.save(pt_target_array[p_train], outdir+'truth_pt_train.npy')
    np.save(class_label_array[p_train], outdir+'y_train.npy')
    np.save(reco_pt_array[p_train], outdir+'reco_pt_train.npy')
    
    np.save(new_feature_array[p_test], outdir+'X_test.npy')
    np.save(pt_target_array[p_test], outdir+'truth_pt_test.npy')
    np.save(class_label_array[p_test], outdir+'y_test.npy')
    np.save(reco_pt_array[p_test], outdir+'reco_pt_test.npy')
    
    
    
            