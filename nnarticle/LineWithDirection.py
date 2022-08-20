from matplotlib import pyplot as plt
import numpy as np
from numpy.typing import ArrayLike


def plotHyperplane(xspace: ArrayLike, intercept: float, xslope: float, yslope: float, n: int = 3, ax: plt.Axes = None, **kwargs):
    ax = ax or plt.gca()
    xspace = np.array(xspace)
    a, b = - xslope / yslope, intercept / yslope
    f = lambda x: a * x + b
    plt.plot(xspace, f(xspace))
    xmin, xmax = xspace.min(), xspace.max()
    diff = xmax - xmin
    arrowxs = np.linspace(xmin + diff * 0.15, xmax - diff * 0.15, n)
    plt.quiver(arrowxs, f(arrowxs), xslope, yslope, units='dots')
    return ax


if __name__ == '__main__':
    xspace = np.linspace(-10, 10, 10)
    ax = plotHyperplane(xspace, 0, 1, 1, width=0.1)
    ax = plotHyperplane(xspace, -2, 1, -1, n=5, width=0.1)
    ax.set_aspect('equal')
    # ax.set_xlim((-10,10))
    # ax.set_ylim((-10,10))
    plt.show()