import torch
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from utils import get_lims, plot_hyperplane, unnormalize_plane, unnormalize_planes
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


def visualize_three_lines():
    plt.xlabel("Weight (g)")
    plt.ylabel("Diameter (cm)")
    plt.title("Decision boundaries for apples, oranges and pears")
    plt.scatter(*X[y == 0].T, label="Apple", marker="^", c="greenyellow", edgecolor="black")
    plt.scatter(*X[y == 1].T, label="Orange", marker="o", c="orange", edgecolor="black")
    plt.scatter(*X[y == 2].T, label="Pear", marker="s", c="forestgreen", edgecolor="black", s=20)

    intercept1 = -1.360250473022461
    xslope1 = -3.2235553
    yslope1 = -1.1162834

    intercept2 = -1.2119554281234741
    xslope2 = 0.63510317
    yslope2 = 2.3010178

    intercept3 = -1.4134321212768555
    xslope3 = 2.7407901
    yslope3 = -1.087009

    m = [141.8463, 6.2363]
    s = [10.5088, 1.7896]

    xspace = torch.linspace(x_lim[0], x_lim[1], 4)

    plot_kwargs = {'linewidth': 3}
    quiver_kwargs = {'units': 'dots', 'width': 1.75, 'headwidth': 4, 'scale': 0.075, 'scale_units': 'dots'}

    intercept_1, xslope_1, yslope_1 = unnormalize_plane(m, s, intercept1, xslope1, yslope1)
    plot_hyperplane(
        xspace,
        intercept_1,
        xslope_1,
        yslope_1,
        8,
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
        8,
        c='orange',
        plot_kwargs={**plot_kwargs, 'linestyle': '-.', 'label': 'Orange boundary'},
        quiver_kwargs=quiver_kwargs
    )

    intercept_3, xslope_3, yslope_3 = unnormalize_plane(m, s, intercept3, xslope3, yslope3)
    plot_hyperplane(
        xspace,
        intercept_3,
        xslope_3,
        yslope_3,
        8,
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
    plt.savefig("figures/apples_oranges_pears_three_lines.pdf")
    # plt.show()
    plt.clf()


def visualize_strengths():
    plt.xlabel("Weight (g)")
    plt.ylabel("Diameter (cm)")
    plt.title("Strengths")
    # plt.scatter(*X[y == 0].T, label="Apple", marker="^", c="greenyellow", edgecolor="black", alpha=0.1)
    # plt.scatter(*X[y == 1].T, label="Orange", marker="o", c="orange", edgecolor="black", alpha=0.1)
    # plt.scatter(*X[y == 2].T, label="Pear", marker="s", c="forestgreen", edgecolor="black", s=20, alpha=0.1)

    intercepts = np.array([
        -1.360250473022461,
        -1.2119554281234741,
        -1.4134321212768555
    ])

    slopes = np.array(
        [[-3.2235553, -1.1162834],
         [0.63510317, 2.3010178],
         [2.7407901, -1.087009]]
    )

    m = np.array([141.8463, 6.2363])
    s = np.array([10.5088, 1.7896])

    uintercepts, uslopes = unnormalize_planes(m, s, intercepts, slopes)

    point = np.array([[125, 6]])
    plt.scatter(*point.T, label="Unknown", marker="x", c="black", s=60)

    def forward(X, intercepts, slopes):
        z = intercepts + X @ slopes.T
        # z = np.log(z)
        z = (z + abs(z.min())).clip(0.01, np.inf)
        z = np.sqrt(z)
        print('LOG:\x1b[33mDEBUG\x1b[0m:', 'z:', z)
        return z / z.sum()

    strengths = forward(point, uintercepts, uslopes)[0]
    print('LOG:\x1b[33mDEBUG\x1b[0m:', 'strengths:', strengths)

    xspace = torch.linspace(x_lim[0], x_lim[1], 4)

    plot_kwargs = {}
    quiver_kwargs = {'units': 'dots', 'width': 1.75, 'headwidth': 4, 'scale': 0.075, 'scale_units': 'dots'}

    linestyles = [
        None,
        '-.',
        '--'
    ]

    labels = [
        'Apple boundary',
        'Orange boundary',
        'Pear boundary'
    ]

    colors = [
        'greenyellow',
        'orange',
        'forestgreen'
    ]

    for i, (linestyle, label, color) in enumerate(zip(linestyles, labels, colors)):
        plot_hyperplane(
            xspace,
            uintercepts[i],
            *uslopes[i],
            8,
            c=color,
            plot_kwargs={**plot_kwargs, 'linestyle': linestyle, 'linewidth': max(strengths[i] * 10, 0.1), 'label': label},
            quiver_kwargs={**quiver_kwargs, 'scale': max((1 - strengths[i]) * 0.25, 0.05)}
        )

    # plt.legend(loc="upper right")
    plt.xlim(*x_lim)
    plt.ylim(*y_lim)
    plt.gca().set_aspect('equal')
    plt.gcf().set_figheight(10)
    plt.gcf().set_figwidth(10)
    plt.savefig("figures/apples_oranges_pears_strengths.pdf")
    # plt.show()
    plt.clf()


# visualize_data_set()
# visualize_two_lines()
# visualize_three_lines()
visualize_strengths()
