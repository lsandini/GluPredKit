import os

model_file = '/home/lorenzo/GluPredKit/data/trained_models/lstm__my_config_5__60.pkl'
if os.path.exists(model_file):
    print("Model file exists.")
else:
    print("Model file does not exist.")

import pickle

with open(model_file, 'rb') as f:
    model_instance = pickle.load(f)

