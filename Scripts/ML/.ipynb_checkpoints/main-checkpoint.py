import numpy as np
import xarray as xr
import os
import joblib

from tools.l2o_modified import open_all_datasets_post
from tools.dataloading import extract_features

# change input and output directory
INPUT_DIR = os.path('C:\Users\danie\Documents\Local Gits\CapstoneAI_TU_2023-2024\scipts\Datasets\small_dataset')
OUTPUT_DIR = 'Results\small_dataset_predictions'

# load data
experimentdata = ExperimentData.from_file(INPUT_DIR)
dsds = open_all_datasets_post(experimentdata)
X = extract_features(dsds)

# load model
model = joblib.load('models\best_model_untuned')

# make predictions and save to csv
IDs = dsds.itemID.values
predictions = model.predict(dsds)
result = pd.DataFrame({ID: pred for ID, pred in zip(IDs, predictions)})
result.to_csv(OUTPUT_DIR)

result