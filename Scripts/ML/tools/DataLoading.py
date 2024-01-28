import numpy as np
import xarray as xr
import scipy.stats as st
import dask as dsk

# ============================ SPLITTING ============================

def train_test_split(
    xarr: xr.Dataset, 
    test_size: float = 0.2, 
    stratified: bool = False, 
    random_state: int = None
) -> tuple[xr.Dataset, xr.Dataset]:
    rs = np.random.RandomState(seed=random_state)
    
    N = len(xarr.itemID)
    N_test = int(test_size * N)

    if stratified:
        labels = extract_labels(xarr)

        test_index = np.array([], dtype=np.int16)
        for label in np.unique(labels):
            class_proba = np.sum(labels == label) / N
            class_index = np.arange(N)[np.nonzero(labels == label)]
            test_index = np.append(test_index, rs.choice(class_index, (round(N_test * class_proba),), replace=False))
    else:
        test_index = rs.choice(np.arange(N), (N_test,), replace=False)

    train_index = np.delete(np.arange(N), test_index)

    xarr_train, xarr_test = xarr.isel(itemID=train_index), xarr.isel(itemID=test_index)

    return xarr_train, xarr_test

# ============================ DATA EXTRACTION ============================

def extract_labels(xarr: xr.Dataset) -> np.ndarray:
    return xarr.coords['optimizer'].values[np.argmin(xarr.ranking.values, axis=1)].reshape(-1,)

def extract_rankings(xarr: xr.Dataset) -> np.ndarray:
    return xarr.ranking.values.reshape((-1, 5))

def extract_multilabels(xarr: xr.Dataset, f: float) -> np.ndarray:
    rankings = extract_rankings(xarr)
    return np.less_equal(rankings, np.repeat(np.min(rankings, axis=1) * f, 5).reshape((-1, 5))) * 1

def extract_features(xarr: xr.Dataset) -> np.ndarray:
    features = np.array([xarr.dim, xarr.budget, xarr.noise, xarr.convex, xarr.separable, xarr.multimodal]).T
    so = xarr.samples_output.values.reshape((-1, 300))
    so_log = np.log(so - np.min(so) + 1e-30)
    
    features_so = np.array([np.mean(so_log, axis=1), np.std(so_log, axis=1), 
                            np.min(so_log, axis=1), np.max(so_log, axis=1),
                            st.skew(so, axis=1), st.kurtosis(so, axis=1)]).T
    
    return np.concatenate([features, features_so], axis=1)