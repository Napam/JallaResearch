import torch
from matplotlib import pyplot as plt
from numpy.typing import ArrayLike
import numpy as np

def get_lims(X: torch.Tensor, padding: float | ArrayLike = 0.25):
    if isinstance(padding, float):
        padding = np.array([padding] * 2)

    x_min, x_max = X[:,0].min(), X[:,0].max()
    y_min, y_max = X[:,1].min(), X[:,1].max()
    x_std, y_std = X[:,0].std(), X[:,1].std()
    return (x_min - x_std * padding[0], x_max + x_std * padding[0]), (y_min - y_std * padding[1], y_max + y_std * padding[1])


def plotHyperplane(xspace: ArrayLike, intercept: float, xslope: float, yslope: float, n: int = 3, ax: plt.Axes = None, **kwargs):
    ax = ax or plt.gca()
    xspace = np.array(xspace)
    a, b = - xslope / yslope, intercept / yslope
    f = lambda x: a * x + b
    plt.plot(xspace, f(xspace))
    xmin, xmax = xspace.min(), xspace.max()
    diff = xmax - xmin
    arrowxs = np.linspace(xmin + diff * 0.15, xmax - diff * 0.15, n)
    for arrowx in arrowxs:
        plt.arrow(arrowx, f(arrowx), xslope, yslope, **kwargs)
    return ax