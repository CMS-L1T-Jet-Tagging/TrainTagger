import numpy as np
import awkward as ak

# Sample data
event_id = np.array([1, 2, 1, 2, 3])
tau_pt = np.array([10.5, 20.1, 11.0, 21.2, 30.3])

# Step 1: Convert to Awkward Arrays
event_id_ak = ak.Array(event_id)
tau_pt_ak = ak.Array(tau_pt)

# Step 2: Use ak.argsort to sort based on event_id
sorted_indices = ak.argsort(event_id_ak)
sorted_event_id = event_id_ak[sorted_indices]
sorted_tau_pt = tau_pt_ak[sorted_indices]

# Step 3: Find unique event_ids and counts manually
unique_event_id = np.unique(sorted_event_id)
counts = [np.sum(sorted_event_id == eid) for eid in unique_event_id]

# Step 4: Use ak.unflatten to group the tau_pt by counts
grouped_event_id = ak.unflatten(sorted_event_id, counts)
grouped_tau_pt = ak.unflatten(sorted_tau_pt, counts)

print(grouped_event_id)
print(grouped_tau_pt)