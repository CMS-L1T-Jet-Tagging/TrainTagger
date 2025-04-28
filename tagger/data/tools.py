# Python
import os, gc, json, glob, shutil

# Third party
import numpy as np
import awkward as ak
import tensorflow as tf
import uproot, yaml

# Dataset configuration
from .config import FILTER_PATTERN, N_PARTICLES, INPUT_TAG, EXTRA_FIELDS

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
        "b": ( # Bottom
            genmatch_pt_base &
            (data['jet_muflav'] == 0) &
            (data['jet_tauflav'] == 0) &
            (data['jet_elflav'] == 0) &
            (data['jet_genmatch_hflav'] == 5)
        ),
        "charm": ( # Charm
            genmatch_pt_base &
            (data['jet_muflav'] == 0) &
            (data['jet_tauflav'] == 0) &
            (data['jet_elflav'] == 0) &
            (data['jet_genmatch_hflav'] == 4)
        ),
        "light": ( # uds
            genmatch_pt_base &
            (data['jet_muflav'] == 0) &
            (data['jet_tauflav'] == 0) &
            (data['jet_elflav'] == 0) &
            (data['jet_genmatch_hflav'] == 0) &
            ((abs(data['jet_genmatch_pflav']) == 0) | (abs(data['jet_genmatch_pflav']) == 1) | (abs(data['jet_genmatch_pflav']) == 2) | (abs(data['jet_genmatch_pflav']) == 3))
        ),
        "gluon": ( # Gluon
            genmatch_pt_base &
            (data['jet_muflav'] == 0) &
            (data['jet_tauflav'] == 0) &
            (data['jet_elflav'] == 0) &
            (data['jet_genmatch_hflav'] == 0) &
            (data['jet_genmatch_pflav'] == 21)
        ),
        "taup": ( # Tau +
            genmatch_pt_base &
            (data['jet_muflav'] == 0) &
            (data['jet_tauflav'] == 1) &
            (data['jet_taucharge'] > 0) &
            (data['jet_elflav'] == 0)
        ),
        "taum": ( # Tau -
            genmatch_pt_base &
            (data['jet_muflav'] == 0) &
            (data['jet_tauflav'] == 1) &
            (data['jet_taucharge'] < 0) &
            (data['jet_elflav'] == 0)
        ),
        "muon": ( # muon
            genmatch_pt_base &
            (data['jet_muflav'] == 1) &
            (data['jet_tauflav'] == 0) &
            (data['jet_elflav'] == 0)
        ),
        "electron": ( # electron
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
    for field in features:
        field_array = data_split["jet_pfcand"][field]

        padded_filled_array = _pad_fill(field_array, n_parts)
        inputs_list.append(padded_filled_array[:,:,np.newaxis])

    #batch_size, n_particles, n_features
    inputs = ak.concatenate(inputs_list, axis=2)
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

def _save_dataset_metadata(outdir, class_labels, tag, extras):

    dataset_metadata_file = os.path.join(outdir, 'variables.json')

    metadata = {"outputs": class_labels,
                "inputs": _get_pfcand_fields(tag),
                "extras": _get_pfcand_fields(extras),}

    with open(dataset_metadata_file, "w") as f: json.dump(metadata, f, indent=4)

    return

def _process_chunk(data_split, tag, extras, n_parts, chunk, outdir):
    """
    Process chunk of data_split to save/parse it for training datasets
    """

    #Create the NN inputs
    _make_nn_inputs(data_split, tag, n_parts)
    extra_features = _get_pfcand_fields(extras)

    #Save them to a root file
    save_fields=['nn_inputs', 'class_label', 'target_pt', 'target_pt_phys'] + extra_features

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

    del data_split, filtered_data, outfile
    #Delete the variables to save memory
    gc.collect()

    return

# >>>>>>FUNCTIONS THAT SHOULD BE USED EXTERNALLY!<<<<<<<
def extract_array(tree, field, entry_stop):
    """
    Extracts an array from the tree with a limit on the number of entries.
    """
    return tree[field].array(entry_stop=entry_stop)

def extract_nn_inputs(data, input_vars, n_parts=16 ,n_entries=None):
    """
    Extract nn inputs based on the input_vars list
    """

    #Concatenate all the inputs
    inputs_list = []

    for field in input_vars:

        field_array = extract_array(data, f"jet_pfcand_{field}", n_entries)

        padded_filled_array = _pad_fill(field_array, n_parts)
        inputs_list.append(padded_filled_array[:,:,np.newaxis])

    #batch_size, n_particles, n_features
    inputs = ak.concatenate(inputs_list, axis=2)

    return inputs

def group_id_values(event_id, *arrays, num_elements = 2):
    '''
    Group values according to event id.
    Filter out events that has less than num_elements
    '''

    # Use ak.argsort to sort based on event_id
    sorted_indices = ak.argsort(event_id)
    sorted_event_id = event_id[sorted_indices]

    # Find unique event_ids and counts manually
    unique_event_id, counts = np.unique(sorted_event_id, return_counts=True)
    
    # Use ak.unflatten to group the arrays by counts
    grouped_id = ak.unflatten(sorted_event_id, counts)
    grouped_arrays = [ak.unflatten(arr[sorted_indices], counts) for arr in arrays]

    #Filter out groups that don't have at least num_elements elements
    mask = ak.num(grouped_id) >= num_elements
    filtered_grouped_arrays = [arr[mask] for arr in grouped_arrays]

    return grouped_id[mask], filtered_grouped_arrays

def to_ML(data, class_labels):
    """
    Take in the data from make_data (loaded by load_data) and make them ready for training.
    """

    X = np.asarray(data['nn_inputs'])
    y = tf.keras.utils.to_categorical(np.asarray(data['class_label']), num_classes=len(class_labels))
    pt_target = np.asarray(data['target_pt'])
    truth_pt = np.asarray(data['target_pt_phys'])
    reco_pt = np.asarray(data['jet_pt_phys'])

    return X, y, pt_target, truth_pt, reco_pt

def load_data(outdir, percentage, test_ratio=0.1, fields=None):
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
    print("Loading percentage: ", percentage)
    print("With test ratio of: ", test_ratio)

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
        extra_vars = variables['extras']
    
    return train_data, test_data, class_labels, input_vars, extra_vars

def make_data(infile='/eos/cms/store/cmst3/group/l1tr/sewuchte/l1teg/fp_ntuples_v131Xv9/baselineTRK_4param_221124/All200.root', 
              outdir='training_data/',
              tag=INPUT_TAG,
              extras=EXTRA_FIELDS,
              n_parts=N_PARTICLES,
              ratio=1.0,
              step_size="100MB",
              tree="outnano/jets"):
    """
    Process the data set in chunks from the input ntuples file.

    Parameters:
        infile (str): The input file path.
        outdir (str): The output directory.
        tag (str): Input tags to use from pfcands, defined in pfcand_fields.yml.
        extras (str): Extra fields to store for plotting, defined in pfcand_fields.yml
        n_parts (int): Number of constituent particles to use for tagging.
        fraction (float) : fraction from (0-1) of data to process for training/testing
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
    num_entries = uproot.open(infile)[tree].num_entries
    num_entries_done = 0
    chunk = 0

    for data in uproot.iterate(infile, filter_name=FILTER_PATTERN, how="zip", step_size=step_size, max_workers=8):
        
        num_entries_done += len(data) # count before cuts

        #Define jet kinematic cuts
        jet_cut = (data['jet_pt_phys'] > 15) & (np.abs(data['jet_eta_phys']) < 2.4) & (data['jet_reject'] == 0)
        data = data[jet_cut]

        #Add additional response variables
        # _add_response_vars(data)
        #Split data into all the training classes
        data_split, class_labels = _split_flavor(data)

        #If first chunk then save metadata of the dataset
        if chunk == 0: _save_dataset_metadata(outdir, class_labels, tag, extras)

        #Process and save training data for a given feature set
        _process_chunk(data_split, tag=tag, extras=extras, n_parts=n_parts, chunk=chunk, outdir=outdir)

        #Number of chunk for indexing files
        chunk += 1
        print(f"Processed {num_entries_done}/{num_entries} entries | {np.round(num_entries_done / num_entries * 100, 1)}%")
        if num_entries_done / num_entries >= ratio: break
