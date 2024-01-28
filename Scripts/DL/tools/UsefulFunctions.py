#First import these functions. Partially for typehints 
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import xarray as xr
import numpy as np
import pandas as pd
import torch.nn as nn
from sklearn.model_selection import train_test_split
from collections.abc import Iterable
import l2o
from DatasetExtraction import extract_features


def preprocess_X(X: Iterable[pd.DataFrame], device: str) -> tuple[torch.Tensor] | torch.Tensor:
    """h are categorized as infinite. Add 1e-7 so
        that there aren't any log10(0)'s
        
        Args:
            X: tensor to preprocess
            device: gpu or cpu depending on device
    """
    res = []
    for x in X:
        x = torch.Tensor(np.array(x)).to(device)
        x = torch.nan_to_num(x)
        x[:, :30] = torch.log(F.relu(x[:, :30]) + 1e-7)
        res.append(x)
    if len(res) == 1:
        return res[0]
    return tuple(res)

def encode_y(Ys: Iterable[list], classes: tuple[str], device: str) -> tuple[torch.Tensor]:
    """
         The ys are still listed as ("cmaes", "adam", etc) to fix this use the classes iterable. Index the y in 
         the classes so that the output becomes [0, 1, etc]. Then make of that list a torch.Tensor and push it
         to the device.
         
         Args:
             Ys: Iterable of the names of optimizers
             classes: Iterable with the names of the classes
             device: gpu or cpu depending on device
    """
    res = []
    for Y in Ys:
        Y = [classes.index(y) for y in Y]
        Y_enc: torch.Tensor = torch.Tensor(np.array(Y)).long().to(device)
        res.append(Y_enc)

    return tuple(res)

class TorchStandardScaler:
    """
        Class for scaling the torch.Tensors. First use the fit function for mean and standard deviation.
        Then transform (new) data so that it is scaled.
        
        Args:
            x: tensor to scale
    """
    def fit(self, x: torch.Tensor) -> None:
        self.mean = x.mean(0, keepdim=True)
        self.std = x.std(0, unbiased=False, keepdim=True)
    def transform(self, x: torch.Tensor()) -> torch.Tensor():
        x -= self.mean
        x /= (self.std + 1e-7)
        return x

class EarlyStopper:
    """
        Class to find out if early stopping needs to be applied. Initialize the values for which you want it to
        check. With function early stop, check if the validation is improving or not. If not, then early stop.
        
        Args:
            patience: How many epochs to check for before early stopping
            min_delta: How much the validation needs to have been improved
            validation_acc: float for accuracy
    """
    def __init__(self, patience: int = 1, min_delta: float = 0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.max_validation_acc = -float('inf')

    def early_stop(self, validation_acc:float) -> bool:
        if validation_acc > self.max_validation_acc:
            self.max_validation_acc = validation_acc
            self.counter = 0
        elif validation_acc < (self.max_validation_acc - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def split(X: pd.DataFrame, yID: Iterable[list], dataset: xr.Dataset, val_size: float = 0.1, random_state: int = 42) -> tuple[tuple[pd.DataFrame], tuple[list], tuple[xr.Dataset]]:
    """
        Function to split the X and Y values using sklearn. Then make a tuple of them for use. 
        Datasets get categorized based on the ID's from the ys.
        
        Args:
            X: Dataframe from which the train, validation test data will be pulled
            yID: The ID's containing the names and ID's needed for selecting the correct dataset
            dataset: xarray dataset used for getting all input variables
            val_size: size of the validation set
            random_state: random state integer
    """
    
    X_train, X_val, yID_train, yID_val = tuple(train_test_split(X, yID, test_size=val_size, random_state=random_state))

    Xs: tuple = tuple([X_train, X_val])

    y_train, IDs_train = tuple([list(i) for i in list(zip(*yID_train))])
    y_val, IDs_val = tuple([list(i) for i in list(zip(*yID_val))])

    ys: tuple = tuple([y_train, y_val])

    dataset_train = dataset.sel(itemID=IDs_train)
    dataset_val = dataset.sel(itemID=IDs_val)

    datasets: tuple[xr.Dataset] = tuple([dataset_train, dataset_val])

    return Xs, ys, datasets

def scale(Xs: Iterable[torch.Tensor], scaler: TorchStandardScaler) -> tuple[torch.Tensor] | torch.Tensor:
    """
        Function for scaling the X values using a scaler. 
        
        Arg:
            Xs: Iterable which will be scaled
            scaler: function (or program) with which Xs will be transformed 
    """
    res = []
    for X in Xs:
        X[:, :32] = scaler.transform(X[:, :32])
        res.append(X)
    if len(res) == 1:
        return res[0]
    return tuple(res)

def train(train_loader: DataLoader, net: nn.Module,
          optimizer: torch.optim, criterion: nn.modules.loss) -> None:
    """
    Trains network for one epoch in batches.

    Args:
        train_loader: Data loader for training set.
        net: Neural network model.
        optimizer: Optimizer.
        criterion: Loss function.
    """

    net.train()

    # iterate through batches
    for data in train_loader:
        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


def test(loader: DataLoader, net: nn.Module, criterion: nn.modules.loss) -> tuple[float, float]:
    """
    Evaluates network in batches.

    Args:
        loader: Data loader for validation set.
        net: Neural network model.
        criterion: Loss function.
    """

    net.eval()

    avg_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        # iterate through batches
        for data in loader:
            # get the inputs
            inputs, labels = data

            # forward pass
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            # keep track of loss and accuracy
            avg_loss += float(loss)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return avg_loss / len(loader), 100 * correct / total


def true_predicted(net: nn.Module, loader: DataLoader) -> tuple[list, list]:
    """
        Function for getting the true labels and the predicted labels
        
        Args:
            net: The neural network
            loader: Dataloader for making predictions
    """
    true_list = []
    predict_list = []

    for data in loader:
        inputs, labels = data

        outputs = net(inputs)

        _, predicted = torch.max(outputs.data, 1)

        true_list.extend(np.array(labels.cpu()))
        predict_list.extend(np.array(predicted.cpu()))

    return true_list, predict_list

def subsets(dataset: l2o.PerformanceDataset) -> dict[str: l2o.PerformanceDataset]:
    """
        Function returning itemIDs of subsets of the dataset based on different criteria.

        Args:
            dataset: full dataset to take subsets from
    """
    res = dict()
    res['dim <= 10'] = dataset.itemID.where((dataset.dim <= 10).compute(), drop=True)
    res['dim > 10'] = dataset.itemID.where((dataset.dim > 10).compute(), drop=True)
    res['budget <= 1124'] = dataset.itemID.where((dataset.budget <= 1124).compute(), drop=True)
    res['budget > 1124'] = dataset.itemID.where((dataset.budget > 1124).compute(), drop=True)
    res['noisy'] = dataset.itemID.where((dataset.noise == 1).compute(), drop=True)
    res['not noisy'] = dataset.itemID.where((dataset.noise == 0).compute(), drop=True)
    res['convex'] = dataset.itemID.where((dataset.convex == 1).compute(), drop=True)
    res['not convex'] = dataset.itemID.where((dataset.convex == 0).compute(), drop=True)
    res['separable'] = dataset.itemID.where((dataset.separable == 1).compute(), drop=True)
    res['not separable'] = dataset.itemID.where((dataset.separable == 0).compute(), drop=True)
    res['multimodal'] = dataset.itemID.where((dataset.multimodal == 1).compute(), drop=True)
    res['not multimodal'] = dataset.itemID.where((dataset.multimodal == 0).compute(), drop=True)

    return res


# Net

class Net(nn.Module):
    """
        Neural network model

        Args:
            n_layers: No. of fully connected layers in the network.
            n_units: No. of units used in hidden layers.
            k: Dropout ratio of dropout layers.
    """
    def __init__(self, n_layers: int, n_units: int, k: float) -> None:
        super().__init__()
        self.n_layers = n_layers - 2
        self.fc1 = nn.Linear(36, n_units)

        layer = nn.Linear(n_units, n_units)
        for i in range(n_layers):
            name = 'fc' + str(i + 2)
            setattr(self, name, layer)

        self.fclast = nn.Linear(n_units, 5)
        self.dropout = nn.Dropout(k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))

        for i in range(self.n_layers):
            name = 'fc' + str(i + 2)
            layer = getattr(self, name)
            h = h + F.relu(layer(h))
            if i < self.n_layers - 1:
                h = self.dropout(h)

        return self.fclast(h)

    def predict(self, x: torch.Tensor) -> list[str]:
        classes = ('CMAES', 'PSO', 'Adam', 'LBFGSB', 'RandomSearch')
        self.eval()
        y = self(x)
        pred = torch.max(y, 1)[1]
        pred = [classes[i] for i in pred]

        return pred


class YourStrategy(l2o.CustomStrategy):
    """
        Optimization strategy with implemented own prediction model.

        Args:
            model: Model to be used for predictions
            scaler: Scaler used for preprocessing of inputs
            device: gpu or cpu depending on device.
            name: Name of the strategy.
            features: Dataset serving as input for predictions.
    """
    def __init__(self, model, scaler, device, name: str = "") -> None:
        self.model = model
        self.name = name
        self.scaler = scaler
        self.device = device

    def predict(self, features: l2o.PerformanceDataset) -> list[str]:
        X_predict = extract_features(features)
        X_predict = preprocess_X([X_predict], self.device)
        X_predict = scale([X_predict], self.scaler)

        if self.name == 'KNN':
            prediction = self.model.predict(X_predict[:, 30:].cpu())
        else:
            prediction = self.model.predict(X_predict)

        return prediction
