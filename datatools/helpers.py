import numpy as np
import awkward as ak
from datatools.createDataset import dict_fields

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

    #Filter out groups that don't have at least 2 elements
    mask = ak.num(grouped_id) >= num_elements
    filtered_grouped_arrays = [arr[mask] for arr in grouped_arrays]

    return grouped_id[mask], filtered_grouped_arrays

def extract_nn_inputs(minbias, input_fields_tag='ext7', nconstit=16, n_entries=None):
    """
    Extract NN inputs based on input_fields_tag
    """

    #The complete input sets are defined in utils/createDataset.py
    features = dict_fields[input_fields_tag]

    print("Using the following features: ", features)

    #Concatenate all the inputs
    inputs_list = []

    #Vertically stacked them to create input sets
    #https://awkward-array.org/doc/main/user-guide/how-to-restructure-concatenate.html
    #Also pad and fill them with 0 to the number of constituents we are using (nconstit)
    for i in range(len(features)):

        if features[i] != "dxySqrt":
            field = f"jet_pfcand_{features[i]}"
            field_array = extract_array(minbias, field, n_entries)
        else:
            field = f"jet_pfcand_dxy_custom"
            dxy_custom = extract_array(minbias, field, n_entries)
            field_array = np.nan_to_num(np.sqrt(dxy_custom), nan=0.0, posinf=0., neginf=0.)

        padded_filled_array = pad_fill(field_array, nconstit)
        inputs_list.append(padded_filled_array[:, np.newaxis])

    inputs = ak.concatenate(inputs_list, axis=1)

    return inputs