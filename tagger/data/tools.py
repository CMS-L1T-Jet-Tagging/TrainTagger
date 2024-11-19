#Python
import os, gc, json, glob, shutil

#Third party
import numpy as np
import awkward as ak
import tensorflow as tf
import uproot, yaml

#Dataset configuration
from .config import FILTER_PATTERN, N_PARTICLES, INPUT_TAG

gc.set_threshold(0)

#>>>>>>>>>>>>>>>>>>>PRIVATE FUNCTIONS<<<<<<<<<<<<<<<<<<<<<<
def _add_response_vars(data):
    data['jet_ptUncorr_div_ptGen'] = ak.nan_to_num(data['jet_pt_phys']/data['jet_genmatch_pt'], copy=True, nan=0.0, posinf=0., neginf=0.)
    data['jet_ptCorr_div_ptGen'] = ak.nan_to_num(data['jet_pt_corr']/data['jet_genmatch_pt'], copy=True, nan=0.0, posinf=0., neginf=0.)
    data['jet_ptRaw_div_ptGen'] = ak.nan_to_num(data['jet_pt_raw']/data['jet_genmatch_pt'], copy=True, nan=0.0, posinf=0., neginf=0.)

def _split_flavor(data):
    """
    Splits data by particle flavor and applies conditions for each category. Also creates the pT target.

    Parameters:
        data (awkward array): The input data to split.

    Returns:
        dict: A dictionary containing the split data by label.
    """

    genmatch_pt_base = data['jet_genmatch_pt'] > 0

    # Define conditions for each label
    conditions = {
        "b": (
            genmatch_pt_base &
            (data['jet_muflav'] == 0) &
            (data['jet_tauflav'] == 0) &
            (data['jet_elflav'] == 0) &
            (data['jet_genmatch_hflav'] == 5)
        ),
        "charm": ( #Charm
            genmatch_pt_base &
            (data['jet_muflav'] == 0) &
            (data['jet_tauflav'] == 0) &
            (data['jet_elflav'] == 0) &
            (data['jet_genmatch_hflav'] == 4)
        ),
        "light": (
            genmatch_pt_base &
            (data['jet_muflav'] == 0) &
            (data['jet_tauflav'] == 0) &
            (data['jet_elflav'] == 0) &
            (data['jet_genmatch_hflav'] == 0) &
            ((abs(data['jet_genmatch_pflav']) == 0) | (abs(data['jet_genmatch_pflav']) == 1) | (abs(data['jet_genmatch_pflav']) == 2) | (abs(data['jet_genmatch_pflav']) == 3))
        ),
        "gluon": ( #Gluon
            genmatch_pt_base &
            (data['jet_muflav'] == 0) &
            (data['jet_tauflav'] == 0) &
            (data['jet_elflav'] == 0) &
            (data['jet_genmatch_hflav'] == 0) &
            (data['jet_genmatch_pflav'] == 21)
        ),
        "taup": (
            genmatch_pt_base &
            (data['jet_muflav'] == 0) &
            (data['jet_tauflav'] == 1) &
            (data['jet_taucharge'] > 0) &
            (data['jet_elflav'] == 0)
        ),
        "taum": (
            genmatch_pt_base &
            (data['jet_muflav'] == 0) &
            (data['jet_tauflav'] == 1) &
            (data['jet_taucharge'] < 0) &
            (data['jet_elflav'] == 0)
        ),
        "muon": (
            genmatch_pt_base &
            (data['jet_muflav'] == 1) &
            (data['jet_tauflav'] == 0) &
            (data['jet_elflav'] == 0)
        ),
        "electron": (
            genmatch_pt_base &
            (data['jet_muflav'] == 0) &
            (data['jet_tauflav'] == 0) &
            (data['jet_elflav'] == 1)
        ),
    }

    # Automatically generate class labels based on the order of keys in conditions
    class_labels = {label: idx for idx, label in enumerate(conditions)}

    # Initialize the new array in data for numeric labels with default -1 for unmatched entries
    data['class_label'] = ak.full_like(data['jet_genmatch_pt'], -1)

    # Assign numeric values based on conditions using awkward's where function
    for label, condition in conditions.items():
        data['class_label'] = ak.where(condition, class_labels[label], data['class_label'])

    #Set pt regression target
    hadrons = (conditions["b"] | conditions["charm"] | conditions["light"] | conditions["gluon"])
    leptons = (conditions["taup"] | conditions["taum"] | conditions["muon"] | conditions["electron"])

    hadron_pt_ratio = ak.nan_to_num(data["jet_genmatch_pt"]/data["jet_pt_phys"], nan=0, posinf=0, neginf=0)
    lepton_pt_ratio = ak.nan_to_num((data["jet_genmatch_lep_vis_pt"]/data["jet_pt_phys"]),nan=0,posinf=0,neginf=0)

    hadron_pt = ak.nan_to_num(data["jet_genmatch_pt"],nan=0,posinf=0,neginf=0)
    lepton_pt = ak.nan_to_num((data["jet_genmatch_lep_vis_pt"]),nan=0,posinf=0,neginf=0)

    data['target_pt'] = np.clip(hadrons * hadron_pt_ratio + leptons * lepton_pt_ratio, 0.3, 2)
    data['target_pt_phys'] = hadrons * hadron_pt + leptons*lepton_pt

    # Apply pt_cut
    jet_ptmin_gen = (data['target_pt_phys'] > 5.)
    for key in conditions: conditions[key] = conditions[key] & jet_ptmin_gen

    # Sanity check for data consistency
    split_data_sum = sum(sum(conditions[label]) for label, condition in conditions.items())
    if split_data_sum != len(data[jet_ptmin_gen]):
        raise ValueError(f"Data splitting error: Total entries ({split_data_sum}) do not match the filtered data length ({len(data[jet_ptmin_gen])}).")

    return data[jet_ptmin_gen], class_labels

def _get_pfcand_fields(tag):
    
    # Get the directory of the current file (tools.py)
    current_dir = os.path.dirname(__file__)

    # Construct the path to pfcand_fields.yml relative to tools.py
    pfcand_fields_path = os.path.join(current_dir, "pfcand_fields.yml")

    # Load the YAML file as a dictionary
    with open(pfcand_fields_path, "r") as file: pfcand_fields = yaml.safe_load(file)

    return pfcand_fields[tag]

def _pad_fill(array, target):
    '''
    pad an array to target length and then fill it with 0s
    '''
    return ak.fill_none(ak.pad_none(array, target, axis=1, clip=True), 0)

def _make_nn_inputs(data_split, tag, n_parts):
    
    features = _get_pfcand_fields(tag)

    #Concatenate all the inputs
    inputs_list = []

    #Vertically stacked them to create input sets
    #https://awkward-array.org/doc/main/user-guide/how-to-restructure-concatenate.html
    #Also pad and fill them with 0 to the number of constituents we are using (nconstit)
    for i in range(len(features)):

        field = features[i]
        field_array = data_split["jet_pfcand"][field]

        padded_filled_array = _pad_fill(field_array, n_parts)
        inputs_list.append(padded_filled_array[:, np.newaxis])

    inputs = ak.concatenate(inputs_list, axis=1)

    data_split['nn_inputs'] = inputs

    return

def _save_chunk_metadata(metadata_file, chunk, entries, outfile):

    chunk_info = {
        "chunk": chunk,
        "entries": entries,
        "file": outfile
    }

    # Load existing metadata or start a new list
    if os.path.exists(metadata_file):
        with open(metadata_file, "r") as f:
            content = f.read()
            metadata = json.loads(content) if content.strip() else []
    else:
        metadata = []

    metadata.append(chunk_info)

    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)

    return

def _save_dataset_metadata(outdir, class_labels, tag):

    dataset_metadata_file = os.path.join(outdir, 'variables.json')

    metadata = {"outputs": class_labels,
                "inputs": _get_pfcand_fields(tag)}

    with open(dataset_metadata_file, "w") as f: json.dump(metadata, f, indent=4)

    return

def _process_chunk(data_split, tag, n_parts, chunk, outdir):
    """
    Process chunk of data_split to save/parse it for training datasets
    """

    #Create the NN inputs
    _make_nn_inputs(data_split, tag, n_parts)

    #Save them to a root file
    save_fields=['nn_inputs', 'class_label', 'target_pt', 'target_pt_phys']

    # Filter the data_split to only include save_fields
    filtered_data = {field: data_split[field] for field in save_fields}

    #Save chunk to files
    outfile = os.path.join(outdir, f'data_chunk_{chunk}.root')
    with uproot.recreate(outfile) as f:
        f["data"] = filtered_data
        print(f"Saved chunk {chunk} to {outfile}")

    # Log metadata
    metadata_file = os.path.join(outdir, "metadata.json")
    _save_chunk_metadata(metadata_file, chunk, len(data_split), outfile) #Chunk, Entries, Outfile

    #Delete the variables to save memory
    gc.collect()

    return

# >>>>>>FUNCTIONS THAT SHOULD BE USED EXTERNALLY!<<<<<<<
def extract_array(tree, field, entry_stop):
    """
    Extracts an array from the tree with a limit on the number of entries.
    """
    return tree[field].array(entry_stop=entry_stop)

def to_ML(data, class_labels):
    """
    Take in the data from make_data (loaded by load_data) and make them ready for training.
    """

    X = np.asarray(data['nn_inputs'])
    y = tf.keras.utils.to_categorical(np.asarray(data['class_label']), num_classes=len(class_labels))
    pt_target = np.asarray(data['target_pt'])
    truth_pt = np.asarray(data['target_pt_phys'])

    return X, y, pt_target, truth_pt

def load_data(outdir, percentage, test_ratio=0.2, fields=None):
    """
    Load a specified percentage of the dataset using uproot.concatenate.

    Parameters:
        outdir (str): The output directory containing the data chunks.
        percentage (float): The percentage of TOTAL data to load (0-100).
        test_ratio (float): how much of the total data would be used for testing (0-1)
        fields (list, optional): Specific fields to load. If None, load all fields.

    Returns:
        awkward.Array: Concatenated data arrays from selected chunks.
    """
    import json
    import numpy as np

    print("Loading data from: ", outdir)

    # Load metadata to determine chunks to load
    metadata_file = os.path.join(outdir, "metadata.json")
    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    total_chunks = len(metadata)
    chunks_to_load = int(np.ceil((percentage / 100) * total_chunks))

    # Collect the file paths for the chunks to load
    chunk_files = [metadata[i]["file"] for i in range(chunks_to_load)]

    # Use uproot.concatenate to load and combine data from multiple files
    data = uproot.concatenate(chunk_files, filter_name=fields, library="ak")

    # Shuffle the data indices
    total_data_len = len(data)
    indices = np.arange(total_data_len)
    np.random.shuffle(indices)

    # Split indices based on test_ratio
    split_index = int((1 - test_ratio) * total_data_len)
    train_indices, test_indices = indices[:split_index], indices[split_index:]

    # Split the data into training and testing sets
    train_data = data[train_indices]
    test_data = data[test_indices]

    #Load corresponding metadata for classlabels/input variables
    data_metadata_file  = os.path.join(outdir, "variables.json")
    with open(data_metadata_file, "r") as f:
        variables = json.load(f)
        class_labels = variables['outputs']
        input_vars = variables['inputs']
    
    return train_data, test_data, class_labels, input_vars

def make_data(infile='/eos/cms/store/cmst3/group/l1tr/sewuchte/l1teg/fp_ntuples_v131Xv9/baselineTRK_4param_021024/All200.root', 
              outdir='training_data/',
              tag=INPUT_TAG,
              n_parts=N_PARTICLES,
              step_size='100MB'):
    """
    Process the data set in chunks from the input ntuples file.

    Parameters:
        infile (str): The input file path.
        outdir (str): The output directory.
        tag (str): Input tags to use from pfcands, defined in pfcand_fields.yml.
        n_parts (int): Number of constituent particles to use for tagging.
        step_size (str): Step size for uproot iteration.
    """

    #Check if output dir already exists, remove if so
    if os.path.exists(outdir):
        confirm = input(f"The directory '{outdir}' already exists. Do you want to delete it and continue? [y/n]: ")
        if confirm.lower() == 'y':
            shutil.rmtree(outdir)
            print(f"Deleted existing directory: {outdir}")
        else:
            print("Exiting without making changes.")
            return

    #Create output training dataset
    os.makedirs(outdir, exist_ok=True)
    print("Output directory:", outdir)

    #Loop through the entries
    num_entries = uproot.open(infile)["jetntuple/Jets"].num_entries
    num_entries_done = 0
    chunk = 0

    for data in uproot.iterate(infile, filter_name=FILTER_PATTERN, how="zip",step_size=step_size):
        num_entries_done += len(data)
        print(f"Processing {num_entries_done}/{num_entries} entries | {np.round(num_entries_done / num_entries * 100, 1)}%")
        
        #Define jet kinematic cuts
        jet_cut = (data['jet_pt_phys'] > 15) & (np.abs(data['jet_eta_phys']) < 2.4) & (data['jet_reject'] == 0)
        data = data[jet_cut]

        #Add additional response variables
        # _add_response_vars(data)

        #Split data into all the training classes
        data_split, class_labels = _split_flavor(data)

        #If first chunk then save metadata of the dataset
        if chunk == 0: _save_dataset_metadata(outdir, class_labels, tag)

        #Process and save training data for a given feature set
        _process_chunk(data_split, tag=tag, n_parts=n_parts, chunk=chunk, outdir=outdir)

        #Uncomment when testing
        chunk += 1
        if chunk == 2: break