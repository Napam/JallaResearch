from typing import Dict, Iterable, Optional
import numpy as np 
from numpy.typing import ArrayLike
import pandas as pd 
from sklearn.model_selection import ParameterGrid, ParameterSampler, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
import torch
from torch import nn 
import scipy.stats.distributions as dists


def fit_and_evaluate_models(
    models: Iterable, X: ArrayLike, candidate_kwargs: Optional[dict]=None, 
    static_kwargs: Optional[dict]=None
):
    if candidate_kwargs is None:
        candidate_kwargs = {}

    if static_kwargs is None:
        static_kwargs = {}

    hist = {'model':[], 'params':[], 'fit_time':[], 'train_mae':[], 'val_mae':[]}

    for model in models:
        model


if __name__ == '__main__':
    from matplotlib import pyplot as plt 
    N = 256
    X = np.random.randn(N)
    y = X + np.random.randn(N)*0.5 + 10

    model1 = nn.Sequential(
        nn.Linear(2, 10),
        nn.ReLU(),
        nn.Linear(10, 1)
    )
    
    model2 = nn.Sequential(
        nn.Linear(2, 10),
        nn.LeakyReLU(),
        nn.Linear(10, 1)
    )
    
    model3 = nn.Sequential(
        nn.Linear(2, 10),
        nn.Tanh(),
        nn.Linear(10, 1)
    )

    models = [model1, model2, model3]

    fit_and_evaluate_models(models, X)
    