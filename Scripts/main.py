import sys
import os

# directory of this file
dirname = os.path.abspath('')
sys.path.append(os.path.join(dirname, 'ML/tools/'))

import pandas as pd
import joblib

from f3dasm import ExperimentData

from l2o_modified import open_all_datasets_post
from DataLoading import extract_features

# change input and output directory
INPUT_DIR = 'Input/YOUR_DATASET'
OUTPUT_PATH = 'Results/Results.csv'

# load data
experimentdata = ExperimentData.from_file(os.path.join(dirname, INPUT_DIR))
dsds = open_all_datasets_post(experimentdata)
X = extract_features(dsds)

print('Features extracted')

# load model
MODEL_PATH = 'ML/models/best_model_untuned'
model = joblib.load(os.path.join(dirname, MODEL_PATH))

# make predictions and save to csv
IDs = dsds.itemID.values
predictions = model.predict(X)

print('Predictions made')

result = pd.DataFrame(zip(IDs, predictions), columns=['itemID', 'prediction'])
result.to_csv(os.path.join(dirname, OUTPUT_PATH), index=False)

print('Results saved')
