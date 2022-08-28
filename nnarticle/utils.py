import torch
from matplotlib import pyplot as plt
from numbers import Number
from numpy.typing import ArrayLike
import numpy as np


def get_lims(X: torch.Tensor, padding: float | ArrayLike = 0.25):
    if isinstance(padding, Number):
        padding = np.array([padding, padding])

    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    x_std, y_std = X[:, 0].std(), X[:, 1].std()
    return (x_min - x_std * padding[0], x_max + x_std * padding[0]), (y_min - y_std * padding[1], y_max + y_std * padding[1])


def plot_hyperplane(
        xspace: ArrayLike,
        intercept: float,
        xslope: float,
        yslope: float,
        n: int = 3,
        ax: plt.Axes = None,
        c=None,
        alpha=None,
        plot_kwargs=None,
        quiver_kwargs=None
):

    plot_kwargs = plot_kwargs or {}
    quiver_kwargs = quiver_kwargs or {}
    if c is not None:
        plot_kwargs['c'] = c
        quiver_kwargs['color'] = c

    if alpha is not None:
        plot_kwargs['alpha'] = alpha
        quiver_kwargs['alpha'] = alpha

    ax = ax or plt.gca()
    xspace = np.array(xspace)

    a, b = - xslope / yslope, - intercept / yslope
    def f(x): return a * x + b
    plt.plot(xspace, f(xspace), **plot_kwargs)

    xmin, xmax = xspace.min(), xspace.max()
    diff = (xmax - xmin) / (n + 1)
    arrowxs = np.linspace(xmin + diff, xmax - diff, n)
    norm = np.linalg.norm([xslope, yslope])
    plt.quiver(arrowxs, f(arrowxs), xslope / norm, yslope / norm, **quiver_kwargs)
    print(xmin, xmax)
    print(arrowxs.min(), arrowxs.max())
    return ax


def normalize_data(X: ArrayLike):
    """Standardization"""
    X_mean = X.mean(0)
    X_std = X.std(0)
    return (X - X_mean) / X_std, X_mean, X_std


def unnormalize_plane(m: ArrayLike, s: ArrayLike, intercept: Number, xslope: Number, yslope: Number):
    intercept_ = intercept - (m[0] * xslope) / s[0] - (m[1] * yslope) / s[1]
    xslope_ = xslope / s[0]
    yslope_ = yslope / s[1]
    return intercept_, xslope_, yslope_


if __name__ == '__main__':
    xspace = np.linspace(-2, 2, 10)
    ax = plot_hyperplane(xspace, 0, 1, 1, n=10, c='r', quiver_kwargs={'units': 'dots'})
    ax = plot_hyperplane(xspace, -1, 1, -1, n=10, c='g', quiver_kwargs={'units': 'dots'})
    ax = plot_hyperplane(xspace, -1, -1, -1, n=10, c='b', quiver_kwargs={'units': 'dots'})
    ax = plot_hyperplane(xspace, 0, -1, 1, n=10, c='y', quiver_kwargs={'units': 'dots'})
    ax.set_aspect('equal')
    # ax.set_xlim((-10,10))
    # ax.set_ylim((-10,10))
    plt.show()
