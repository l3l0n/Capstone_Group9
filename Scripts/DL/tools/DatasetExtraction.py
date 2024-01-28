from f3dasm import ExperimentData
import xarray as xr
import numpy as np
import l2o as l2o
import pandas as pd
from os.path import exists


def extract_dataset(name: str, save: bool = True) -> l2o.PerformanceDataset:
    """
        Extracts a dataset from a directory with experiment files.
        Saves the dataset in a file for later use.

        Args:
            name: Name of the dataset.
            save: Bool whether to save the extracted dataset in a separate file.
    """
    if name == 'small_dataset' or name == 'big_dataset':
        data = ExperimentData.from_file(f'../Datasets/{name}')
    else:
        data = ExperimentData.from_file(name)
    print('data extracted')

    dataset = l2o.open_all_datasets_post(data)
    if save:
        dataset.to_netcdf(f'ready_files/{name}.nc')
    print('dataset extracted')

    return dataset

def extract_features(data: l2o.PerformanceDataset, name: str = "") -> pd.DataFrame:
    """
        Extracts features from a dataset to be used for training and predicting.
        Optionally, saves features in a file for later use.

        Args:
            data: Dataset to extract features from
            name(optional): Name of the dataset.
    """

    res = {'dim': data.dim,
           'budget': data.budget,
           'noise': data.noise,
           'convex': data.convex,
           'separable': data.separable,
           'multimodal': data.multimodal}
    res = pd.DataFrame(res)

    samples = []
    for ID in data.itemID.values:
        r = int(str(ID)[-1])
        sam = np.reshape(data.samples_output.sel(itemID=ID, realization=r).values, -1)
        samples.append(sam)
    samples = pd.DataFrame(samples)

    res = pd.concat([samples, res], axis=1)
    res.columns = res.columns.astype(str)

    if name != "":
        res.to_csv(f'ready_files/X_{name}.csv', index=False)
        print('features extracted')

    return res

def best_optimizers_with_IDs(data: l2o.PerformanceDataset, name: str) -> list[tuple[str, int]]:
    """
        Extracts labels used for training and testing with item IDs of their experiments.

        Args:
            data: Dataset to extract labels and IDs from.
            name: Name of the dataset.
    """

    labels = data.coords['optimizer'].values[np.argmin(data['ranking'].to_numpy(), axis=1)].reshape(-1, )
    IDs = data['itemID'].to_numpy()
    res = list(zip(labels, IDs))

    pd.DataFrame(res).to_csv(f'ready_files/yID_{name}.csv', index=False)
    print('labels extracted')

    return res

def extract(name: str) -> tuple[pd.DataFrame, list[tuple[str, int]], l2o.PerformanceDataset]:
    """
        Loads datasets, features and labels from files if they are available.
        Otherwise, extracts them, using functions defined above.

        Args:
            name: Name of the dataset.
    """

    if exists(f'ready_files/{name}.nc'):
        dataset = xr.load_dataset(f'ready_files/{name}.nc')
        print('dataset loaded')
    else:
        dataset = extract_dataset(name)

    if exists(f'ready_files/X_{name}.csv'):
        X = pd.read_csv(f'ready_files/X_{name}.csv', index_col=False)
        print('features loaded')
    else:
        X = extract_features(dataset, name)

    if exists(f'ready_files/yID_{name}.csv'):
        yID = pd.read_csv(f'ready_files/yID_{name}.csv', index_col=False)
        yID = [tuple(i) for i in yID.values.tolist()]
        print('labels loaded')
    else:
        yID = best_optimizers_with_IDs(dataset, name)

    return X, yID, dataset
