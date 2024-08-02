import awkward as ak
import numpy as np

def pad_and_fill(array, target):
    '''
    pad an array to target length and then fill it with 0s
    '''
    return ak.fill_none(ak.pad_none(array, target, axis=1), 0)

pt = pad_and_fill(ak.Array([[1], [1, 2], [1, 2, 3]]), 16)
eta = pad_and_fill(ak.Array([[4], [4, 5], [4, 5, 6]]), 16)


a = np.asarray(ak.concatenate([pt[:, np.newaxis], eta[:, np.newaxis]], axis=1))

print(a[0])
print(np.asarray(a).shape)



