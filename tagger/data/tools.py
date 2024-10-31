import os, gc
import numpy as np
import awkward as ak
import uproot
import yaml
import tensorflow as tf

#Dataset configuration
from .config import FILTER_PATTERN, N_PARTICLES, INPUT_TAG

gc.set_threshold(0)

def _add_response_vars(data):
    data['jet_ptUncorr_div_ptGen'] = ak.nan_to_num(data['jet_pt_phys']/data['jet_genmatch_pt'], copy=True, nan=0.0, posinf=0., neginf=0.)
    data['jet_ptCorr_div_ptGen'] = ak.nan_to_num(data['jet_pt_corr']/data['jet_genmatch_pt'], copy=True, nan=0.0, posinf=0., neginf=0.)
    data['jet_ptRaw_div_ptGen'] = ak.nan_to_num(data['jet_pt_raw']/data['jet_genmatch_pt'], copy=True, nan=0.0, posinf=0., neginf=0.)

def _split_flavor(data):
    """
    Splits data by particle flavor and applies conditions for each category.

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

    hadron_pt_norm = ak.nan_to_num(data["jet_genmatch_pt"]/data["jet_pt_phys"], nan=0, posinf=0, neginf=0)
    lepton_pt_norm = ak.nan_to_num((data["jet_genmatch_lep_vis_pt"]/data["jet_pt_phys"]),nan=0,posinf=0,neginf=0)

    hadron_pt = ak.nan_to_num(data["jet_genmatch_pt"],nan=0,posinf=0,neginf=0)
    lepton_pt = ak.nan_to_num((data["jet_genmatch_lep_vis_pt"]),nan=0,posinf=0,neginf=0)

    data['target_pt'] = np.clip(hadrons * hadron_pt_norm + leptons * lepton_pt_norm, 0.3, 2)
    data['target_pt_phys'] = hadrons * hadron_pt + leptons*lepton_pt

    # Apply pt_cut
    jet_ptmin_gen = (data['target_pt_phys'] > 5.)
    for key in conditions: conditions[key] = conditions[key] & jet_ptmin_gen

    # Sanity check for data consistency
    split_data_sum = sum(sum(conditions[label]) for label, condition in conditions.items())
    if split_data_sum != len(data[jet_ptmin_gen]):
        raise ValueError(f"Data splitting error: Total entries ({total_entries}) do not match the filtered data length ({len(data[jet_ptmin_gen])}).")

    return data[jet_ptmin_gen]

def _get_pfcand_fields(tag):
    
    # Get the directory of the current file (tools.py)
    current_dir = os.path.dirname(__file__)

    # Construct the path to pfcand_fields.yml relative to tools.py
    pfcand_fields_path = os.path.join(current_dir, "pfcand_fields.yml")

    # Load the YAML file as a dictionary
    with open(pfcand_fields_path, "r") as file: pfcand_fields = yaml.safe_load(file)

    return pfcand_fields


    """
    Extracts an array from the tree with a limit on the number of entries.
    """
    return tree[field].array(entry_stop=entry_stop)

def _create_data(data_split, tag, n_parts):
    """
    Create training/testing data for the model from data_split and pfcand_fields
    """

    features = _get_pfcand_fields(tag)

    #Create X, concatenate all the inputs
    inputs_list = []

    #Vertically stacked them to create input sets
    #https://awkward-array.org/doc/main/user-guide/how-to-restructure-concatenate.html
    #Also pad and fill them with 0 to the number of constituents we are using (n_parts)
    for i in range(len(features)):

        field = f"jet_pfcand_{features[i]}"
        field_array = data_split[field]

        padded_filled_array = pad_fill(field_array, n_parts)
        inputs_list.append(padded_filled_array[:, np.newaxis])

    X = ak.concatenate(inputs_list, axis=1)

    #Create y (the classes)
    y = tf.keras.utils.to_categorical(data_split['class_label'])

    #Create pt_target

    #Create 

    return X, y, 
    

def _process_chunk(data_split, tag, n_parts, chunk, outdir):
    """
    Process chunk of data_split to save/parse it for training datasets
    """

    #Create the classes, x, y
    # X, y, pt_target, classes, jet_features = _create_data(data_split, tag, n_parts)

    #Reshape the arrays for training


    #Save them to parquet

    #Delete the variables to save memory
    gc.collect()

    return

########### Functions that should/can be use externally!
def pad_fill(array, target):
    '''
    pad an array to target length and then fill it with 0s
    '''
    return ak.fill_none(ak.pad_none(array, target, axis=1, clip=True), 0)

def extract_array(tree, field, entry_stop):
    """
    Extracts an array from the tree with a limit on the number of entries.
    """
    return tree[field].array(entry_stop=entry_stop)

def make_data(infile='/eos/cms/store/cmst3/group/l1tr/sewuchte/l1teg/fp_ntuples_v131Xv9/baselineTRK_4param_021024/All200.root', 
              outdir='../training_data/',
              tag=INPUT_TAG,
              n_parts=N_PARTICLES):
    """
    Process the data set in chunks from the input ntuples file.

    Parameters:
        infile (str): The input file path.
        outdir (str): The output directory.
        tags (str): input tags to use from pfcands, defined in pfcand_fields.yml
        n_parts (int): Number of constituent particles to use for tagging.
    """

    #Create output training dataset
    os.makedirs(outdir, exist_ok=True)
    print("Output directory:", outdir)

    #Loop through the entries
    num_entries = uproot.open(infile)["jetntuple/Jets"].num_entries
    num_entries_done = 0
    chunk = 0

    for data in uproot.iterate(infile, filter_name=FILTER_PATTERN, how="zip",step_size=5000):
        num_entries_done += len(data)
        print(f"Processing {num_entries_done}/{num_entries} entries | {np.round(num_entries_done / num_entries * 100, 1)}%")
        
        #Define jet kinematic cuts
        jet_cut = (data['jet_pt_phys'] > 15) & (np.abs(data['jet_eta_phys']) < 2.4) & (data['jet_reject'] == 0)
        data = data[jet_cut]

        #Add additional response variables
        _add_response_vars(data)

        #Split data into all the training classes
        data_split = _split_flavor(data)

        #Process and save training data for a given feature set
        _process_chunk(data_split, tag=tag, n_parts=n_parts, chunk=chunk, outdir=outdir)

        chunk += 1
        if chunk == 3: break