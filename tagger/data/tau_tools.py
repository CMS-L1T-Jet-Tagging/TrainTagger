import os, gc, json, glob, shutil

# Third party
import numpy as np
import awkward as ak
import uproot, yaml

from sklearn.utils import shuffle
from .tools import _save_chunk_metadata

gc.set_threshold(0)

tau_inputs = [
    "pt",
    "deta",
    "dphi", 
    "isPhoton",
    "isElectronPlus",
    "isMuonPlus",
    "isNeutralHadron",
    "isChargedHadronPlus"
]

def _save_tau_dataset_metadata(outdir, class_labels):

    dataset_metadata_file = os.path.join(outdir, 'variables.json')

    metadata = {"outputs": class_labels,
                "inputs": tau_inputs,
                "extras": []}

    with open(dataset_metadata_file, "w") as f: json.dump(metadata, f, indent=4)

    return


def _process_tau_chunk(filtered_data, chunk, outdir):
    """
    Process chunk of data_split to save/parse it for training datasets
    """

    #Save chunk to files
    outfile = os.path.join(outdir, f'data_chunk_{chunk}.root')
    with uproot.recreate(outfile) as f:
        f["data"] = filtered_data
        print(f"Saved chunk {chunk} to {outfile}")

    # Log metadata
    metadata_file = os.path.join(outdir, "metadata.json")
    nevents = len(filtered_data['class_label'])

    _save_chunk_metadata(metadata_file, chunk, nevents, outfile) #Chunk, Entries, Outfile

    #Delete the variables to save memory
    gc.collect()

    return

def _process_tau(data):

    dR_match = 0.2
    gen_pt_cut = 5.0

    n_parts = 10
    n_feats = 8

    class_labels = {
            "light": 0,
            "taus" : 1,
        }

    out = {}
    # Initialize the new array in data for numeric labels with default -1 for unmatched entries
    #data['class_label'] = ak.full_like(data['gendr1'], 0)
    out['class_label'] = np.zeros(len(data['gendr1']), dtype=np.int)
    out['jet_pt_phys'] = np.asarray(data['pt'])

    #tau_match = (np.abs(data['gendr1']) < dR_match)  & (data['genpt1'] > gen_pt_cut)
    tau_match = (np.abs(np.asarray(data['gendr1'])) < dR_match)  & (np.asarray(data['genpt1']) > gen_pt_cut)

    # Assign class label
    out['class_label'][tau_match] = 1

    #Set pt regression target
    gen_pt = np.nan_to_num(np.asarray(data["genpt1"]),nan=0,posinf=0,neginf=0)
    tau_pt_ratio = np.nan_to_num(gen_pt/np.asarray(data["pt"]), nan=0, posinf=0, neginf=0)
    tau_pt_ratio = np.clip(tau_pt_ratio, 0.3, 2)

    out['target_pt'] = np.ones(len(out['class_label']))
    out['target_pt_phys'] = np.asarray(ak.copy(data['pt']))

    out['target_pt'][tau_match] = tau_pt_ratio[tau_match] 
    out['target_pt_phys'][tau_match] = gen_pt[tau_match]

    out['nn_inputs'] = np.asarray(data['m_inputs']).reshape(-1, n_parts, n_feats)

    #shuffle order of training inputs
    out['class_label'], out['target_pt'], out['target_pt_phys'], out['nn_inputs'] = shuffle(
        out['class_label'], out['target_pt'], out['target_pt_phys'], out['nn_inputs'], random_state = 42)

    # Sanity check for data consistency
    # TODO 

    return out, class_labels

def make_tau_data(infile='/eos/cms/store/cmst3/group/l1tr/sewuchte/l1teg/fp_ntuples_v131Xv9/baselineTRK_4param_221124/All.root', 
              outdir='training_data/',
              n_parts=10,
              ratio=1.0,
              step_size="100MB",
              tree="ntuplePupSingle/tree",
              **kwargs):
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

    pt_cut = 15
    eta_cut = 2.4

    for data in uproot.iterate(infile+":"+tree, how="zip", step_size=step_size, max_workers=8):
        jet_cut = (data['pt'] > pt_cut) & (np.abs(data['eta']) < eta_cut)
        data = data[jet_cut]

        #Add additional response variables
        # _add_response_vars(data)
        #Split data into all the training classes
        data_split, class_labels = _process_tau(data)

        #If first chunk then save metadata of the dataset
        if chunk == 0: _save_tau_dataset_metadata(outdir, class_labels)

        #Process and save training data for a given feature set
        _process_tau_chunk(data_split, chunk=chunk, outdir=outdir)

        #Number of chunk for indexing files
        chunk += 1
        num_entries_done += len(data)
        print(f"Processed {num_entries_done}/{num_entries} entries | {np.round(num_entries_done / num_entries * 100, 1)}%")
        if num_entries_done / num_entries >= ratio: break
