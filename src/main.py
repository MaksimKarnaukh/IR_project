# use the python code below to extract ground truth dictionary

import pickle

# read binary data
with open('../gt', 'rb') as binary_file:
    binary_data = binary_file.read()

# Process binary_data
# Unpickle (deserialize) the binary data
deserialized_data = pickle.loads(binary_data)

#example

deserialized_data["Assassin's Creed IV: Black Flag"]