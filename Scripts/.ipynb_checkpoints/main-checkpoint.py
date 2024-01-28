import torch
from DatasetExtraction import extract_features, extract_dataset
import joblib
from UsefulFunctions import YourStrategy
import pandas as pd
from UsefulFunctions import Net


input_dir = 'small_dataset'
output_dir = 'Results/'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
scaler = joblib.load('ready_files/scaler.save')
model = torch.load('ready_files/DL_best_model.pt', map_location=device)

strategy = YourStrategy(model, scaler, device)

input_dataset = extract_dataset(input_dir, False)

IDs = input_dataset.itemID.values
predictions = strategy.predict(input_dataset)
result = {ID: pred for ID, pred in zip(IDs, predictions)}
# result = pd.DataFrame(zip(IDs, predictions))
# result = pd.DataFrame(predictions)


print(result)