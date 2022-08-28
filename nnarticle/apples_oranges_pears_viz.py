import torch
import pandas as pd
from matplotlib import pyplot as plt

from utils import get_lims, plot_hyperplane, unnormalize_plane
plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'dejavuserif'})

df = pd.read_csv("datasets/apples_oranges_pears.csv")
X, y = df[["weight", "height"]].values, df["class"].map({"apple": 0, "orange": 1, "pear": 2}).values
x_lim, y_lim = get_lims(X, padding=[0.75, 1.5])


def visualize_data_set():
    plt.xlabel("Weight (g)")
    plt.ylabel("Diameter (cm)")
    plt.title("Comparing apples, oranges and pears")
    plt.scatter(*X[y == 0].T, label="Apple", marker="^", c="greenyellow", edgecolor="black")
    plt.scatter(*X[y == 1].T, label="Orange", marker="o", c="orange", edgecolor="black")
    plt.scatter(*X[y == 2].T, label="Pear", marker="s", c="forestgreen", edgecolor="black", s=20)
    plt.legend(loc="upper right")
    plt.xlim(*x_lim)
    plt.ylim(*y_lim)
    plt.gca().set_aspect('equal')
    plt.gcf().set_figheight(10)
    plt.gcf().set_figwidth(10)
    plt.savefig("figures/apples_oranges_pears.pdf")
    plt.clf()


def visualize_two_lines():
    plt.xlabel("Weight (g)")
    plt.ylabel("Diameter (cm)")
    plt.title("Decision boundaries for apples and pears")
    plt.scatter(*X[y == 0].T, label="Apple", marker="^", c="greenyellow", edgecolor="black")
    plt.scatter(*X[y == 1].T, label="Orange", marker="o", c="orange", edgecolor="black")
    plt.scatter(*X[y == 2].T, label="Pear", marker="s", c="forestgreen", edgecolor="black", s=20)

    intercept1 = -1.3598535060882568
    xslope1 = -2.9026194
    yslope1 = -0.982407

    intercept2 = -1.6127370595932007
    xslope2 = 3.1835275
    yslope2 = -1.2765434

    m = [141.8463, 6.2363]
    s = [10.5088, 1.7896]

    xspace = torch.linspace(x_lim[0], x_lim[1], 4)

    plot_kwargs = {}
    quiver_kwargs = {'units': 'dots', 'width': 1.75, 'headwidth': 4, 'scale': 0.075, 'scale_units': 'dots'}

    intercept_1, xslope_1, yslope_1 = unnormalize_plane(m, s, intercept1, xslope1, yslope1)
    plot_hyperplane(
        xspace,
        intercept_1,
        xslope_1,
        yslope_1,
        5,
        c='greenyellow',
        plot_kwargs={**plot_kwargs, 'label': 'Apple boundary'},
        quiver_kwargs=quiver_kwargs
    )

    intercept_2, xslope_2, yslope_2 = unnormalize_plane(m, s, intercept2, xslope2, yslope2)
    plot_hyperplane(
        xspace,
        intercept_2,
        xslope_2,
        yslope_2,
        5,
        c='forestgreen',
        plot_kwargs={**plot_kwargs, 'linestyle': '--', 'label': 'Pear boundary'},
        quiver_kwargs=quiver_kwargs
    )

    plt.legend(loc="upper right")
    plt.xlim(*x_lim)
    plt.ylim(*y_lim)
    plt.gca().set_aspect('equal')
    plt.gcf().set_figheight(10)
    plt.gcf().set_figwidth(10)
    plt.savefig("figures/apples_oranges_pears_two_lines.pdf")
    # plt.show()
    plt.clf()


# visualize_data_set()
visualize_two_lines()
