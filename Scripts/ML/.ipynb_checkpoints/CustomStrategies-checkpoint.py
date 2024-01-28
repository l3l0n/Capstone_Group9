from l2o import CustomStrategy
from DataLoading import extract_features
import numpy as np
import xarray as xr
import dask as dsk
from typing import Any, Callable
from sklearn.base import BaseEstimator, TransformerMixin

# ============================ CUSTOM STRATEGIES ============================

class SingleLabelStrategy(CustomStrategy):    
    def __init__(self, model: BaseEstimator, scaler: TransformerMixin):
        self.model: BaseEstimator = model
        if model.__class__.__name__ == 'GridSearch':
            self.name : str = model.model.__class__.__name__
        else:
            self.name : str = model.__class__.__name__
        
        self.scaler: TransformerMixin = scaler

    def predict(self, features: xr.Dataset) -> np.ndarray:
        X = extract_features(features)
        Xt = self.scaler.transform(X)
        return self.model.predict(Xt)

class MultiLabelStrategy(CustomStrategy):
    def __init__(self, model: BaseEstimator, scaler: TransformerMixin):
        self.model: BaseEstimator = model
        if model.__class__.__name__ == 'GridSearch':
            if model.model.__class__.__name__ == 'MultiOutputClassifier':
                self.name: str = model.model.estimator.__name__
            else:
                self.name: str = model.model.__class__.__name__
        elif model.__class__.__name__ == 'MultiOutputClassifier':
            self.name: str = model.estimator.__class__.__name__
        else:
            self.name: str = model.__class__.__name__
        
        self.scaler: TransformerMixin = scaler
    
    def predict(self, features: xr.Dataset) -> np.ndarray:
        X = extract_features(features)
        Xt = self.scaler.transform(X)
        predictions = self.model.predict(Xt)
        return features.optimizer.values[np.argmax(predictions, axis=1)]

class RegressionStrategy(CustomStrategy):
    def __init__(self, model: BaseEstimator, scaler: TransformerMixin):
        self.model: BaseEstimator = model
        if model.__class__.__name__ == 'GridSearch':
            if model.model.__class__.__name__ == 'MultiOutputRegressor':
                self.name: str = model.model.estimator.__name__
            else:
                self.name: str = model.model.__class__.__name__
        elif model.__class__.__name__ == 'MultiOutputRegressor':
            self.name: str = model.estimator.__class__.__name__
        else:
            self.name: str = model.__class__.__name__
        
        self.scaler: TransformerMixin = scaler
    
    def predict(self, features: xr.Dataset) -> np.ndarray:
        X = extract_features(features)
        Xt = self.scaler.transform(X)
        predictions = self.model.predict(Xt)
        return features.optimizer.values[np.argmin(predictions, axis=1)].reshape(-1,)
