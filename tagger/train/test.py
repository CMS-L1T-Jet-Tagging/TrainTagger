from tagger.data.tools import make_data, load_data
import numpy as np

#Create training data set from infile (see README on how to create it)
make_data()


# test = load_data('training_data', 100)

# print(test)
# print(test.fields)
# print(np.asarray(test['target_pt']).shape)
# print(test['target_pt'])