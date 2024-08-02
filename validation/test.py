import numpy as np
import awkward as ak

def group_id_values(event_id, *arrays):
    '''
    Group multiple arrays according to event id.
    '''
    # Use ak.argsort to sort based on event_id
    sorted_indices = ak.argsort(event_id)
    sorted_event_id = event_id[sorted_indices]

    # Find unique event_ids and their counts
    unique_event_id, counts = np.unique(sorted_event_id, return_counts=True)

    # Group each array by the sorted indices and counts
    grouped_arrays = [ak.unflatten(arr[sorted_indices], counts) for arr in arrays]

    return unique_event_id, grouped_arrays

# Test function
def test_group_id_values():
    # Dummy data
    event_id = np.array([1, 2, 1, 3, 2, 1])
    values1 = np.array([10, 20, 30, 40, 50, 60])
    values2 = np.array([100, 200, 300, 400, 500, 600])

    # Expected results
    expected_unique_event_id = np.array([1, 2, 3])
    expected_grouped_values1 = ak.Array([[10, 30, 60], [20, 50], [40]])
    expected_grouped_values2 = ak.Array([[100, 300, 600], [200, 500], [400]])

    # Run the function
    unique_event_id, grouped_arrays = group_id_values(event_id, values1, values2)

    grouped_values1, grouped_values2 = grouped_arrays

    # Check if the results are as expected
    assert np.array_equal(unique_event_id, expected_unique_event_id), "Unique event IDs do not match!"
    assert ak.to_list(grouped_values1) == ak.to_list(expected_grouped_values1), "Grouped values1 do not match!"
    assert ak.to_list(grouped_values2) == ak.to_list(expected_grouped_values2), "Grouped values2 do not match!"

    print("All tests passed!")

# Run the test
test_group_id_values()
