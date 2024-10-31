import os, gc
import numpy as np
import awkward as ak
import uproot
import yaml

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

    # Get unique values
    unique_values = np.unique(data['jet_genmatch_pflav'])

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

    # Split data based on conditions
    split_data = {label: data[condition] for label, condition in conditions.items()}

    # Sanity check for data consistency
    total_entries = sum(len(split_data[label]) for label in split_data)
    if total_entries != len(data[jet_ptmin_gen]):
        raise ValueError(f"Data splitting error: Total entries ({total_entries}) do not match the filtered data length ({len(data[jet_ptmin_gen])}).")

    return split_data

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

    # Get the directory of the current file (tools.py)
    current_dir = os.path.dirname(__file__)

    # Construct the path to pfcand_fields.yml relative to tools.py
    pfcand_fields_path = os.path.join(current_dir, "pfcand_fields.yml")

    # Load the YAML file as a dictionary
    with open(pfcand_fields_path, "r") as file: pfcand_fields = yaml.safe_load(file)

    #Create output training dataset
    os.makedirs(outdir, exist_ok=True)
    print("Output directory:", outdir)

    #Loop through the entries
    num_entries = uproot.open(infile)["jetntuple/Jets"].num_entries
    num_entries_done = 0
    chunk = 0

    for data in uproot.iterate(infile, filter_name=FILTER_PATTERN, how="zip",step_size=5000):
        num_entries_done += len(data)
        print(f"Processed {num_entries_done}/{num_entries} entries | {np.round(num_entries_done / num_entries * 100, 1)}%")
        
        #Define jet kinematic cuts
        jet_cut = (data['jet_pt_phys'] > 15) & (np.abs(data['jet_eta_phys']) < 2.4) & (data['jet_reject'] == 0)
        data = data[jet_cut]

        #Add additional response variables
        _add_response_vars(data)

        #Split data into all the training classes
        data_split = _split_flavor(data)


        chunk += 1
        if chunk == 3: break