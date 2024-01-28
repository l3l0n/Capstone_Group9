import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import _scorer
from xarray import Dataset
from typing import Tuple

# ============================ SCORERS ============================

def multilabel_scorer_func(ym_true: np.ndarray, ym_pred: np.ndarray) -> float:    
    union = (np.sum(ym_true * ym_pred, axis=1) > 0)
    return np.less_equal(np.sum(ym_pred, axis=1), union).sum() / len(ym_true)

multilabel_scorer: _scorer._PredictScorer = make_scorer(multilabel_scorer_func)

def regression_scorer_func(r_true: np.ndarray, r_pred: np.ndarray) -> float:
    return np.sum(np.argmin(r_pred, axis=1) == np.argmin(r_true, axis=1)) / len(r_true)

regression_scorer: _scorer._PredictScorer = make_scorer(regression_scorer_func)

# ============================ EVALUATION ============================

def integrate_perf_profile(arr: np.ndarray) -> float:
    return np.trapz(arr, dx=4/len(arr))

def compare_perf_profiles(xr_f: Dataset) -> pd.DataFrame:
    df = {'strategy': [], 'f_one': [], 'area': [], 'f_one_score': [], 'area_score' : [], 'score': []}

    best_pp = xr_f.sel(strategy='best_perf_profile').values.reshape((-1,))
    best_pp_area = integrate_perf_profile(best_pp)
    
    for strategy in xr_f.strategy:
        pp = xr_f.sel(strategy=strategy).values.reshape((-1,))
        pp_area = integrate_perf_profile(pp)
        
        df['strategy'].append(strategy.values)
        df['f_one'].append(pp[0])
        df['area'].append(pp_area)
        df['f_one_score'].append(pp[0]/best_pp[0])
        df['area_score'].append(pp_area/best_pp_area)
        df['score'].append(np.sqrt(pp[0] / best_pp[0] * pp_area / best_pp_area))

    return pd.DataFrame(df).sort_values('score', ascending=False)

def plot_perf_profiles(xr_f: Dataset, title: str = None) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(10, 5))
    for strategy in xr_f.strategy:
        f_one = xr_f.sel(f=1.0, strategy=strategy).values[0]
        ax.plot(xr_f.f, xr_f.sel(strategy=strategy), label=f"{strategy.values} = {f_one}", zorder=-1)
        ax.scatter(1.0, xr_f.sel(f=1.0, strategy=strategy), marker="x", zorder=1)

    if title is not None:
        ax.set_title(title)
    ax.set_xlabel("Performance ratio (f)")
    ax.set_ylabel("Fraction of problems solved")
    ax.legend()
    return fig, ax

def gs_results(opt: GridSearchCV) -> pd.DataFrame:
    params, scores = np.array(opt.get_param_scores()).T
    return pd.concat((pd.DataFrame(list(params)), pd.DataFrame({'score': scores})), axis='columns')