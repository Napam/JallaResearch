from pprint import pprint
import torch
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import math

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
        -0.08808770030736923,
        -0.09143412113189697,
        -0.09384874999523163
    ])

    slopes = np.array(
        [[-0.19972077, -0.03343868],
         [-0.021978999, 0.14851315],
         [0.20376714, -0.11762319]]
    )

    m = np.array([141.8463, 6.2363])
    s = np.array([10.5088, 1.7896])

    uintercepts, uslopes = unnormalize_planes(m, s, intercepts, slopes)

    point = np.array([[140, 6]])
    plt.scatter(*point.T, label="Unknown", marker="x", c="black", s=60)

    def forward(X, intercepts, slopes):
        z = intercepts + X @ slopes.T
        print('LOG:\x1b[33mDEBUG\x1b[0m:', 'z:', z)
        z = z + abs(z.min())
        z = z ** 2 / z.sum()
        return z

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


def visualize_strengths_animated():
    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel("Weight (g)")
    ax.set_ylabel("Diameter (cm)")
    fig.suptitle("Strengths")

    intercepts = np.array([
        -0.08808770030736923,
        -0.09143412113189697,
        -0.09384874999523163
    ])

    slopes = np.array(
        [[-0.19972077, -0.03343868],
         [-0.021978999, 0.14851315],
         [0.20376714, -0.11762319]]
    )

    m = np.array([141.8463, 6.2363])
    s = np.array([10.5088, 1.7896])

    uintercepts, uslopes = unnormalize_planes(m, s, intercepts, slopes)

    point = np.array([[140, 6]], dtype=float)
    scatter = ax.scatter(*point.T, label="Unknown", marker="x", c="black", s=60, zorder=100)

    def forward(X, intercepts, slopes):
        z = intercepts + X @ slopes.T
        z = z + abs(z.min())
        z = z ** 2 / z.sum()
        return z

    strengths = forward(point, uintercepts, uslopes)[0]
    xspace = torch.linspace(x_lim[0], x_lim[1], 4)

    plot_kwargs = {}
    quiver_kwargs = {'units': 'dots', 'width': 1.75, 'headwidth': 4, 'scale': 0.075, 'scale_units': 'dots'}

    linestyles = [
        None,
        None,
        None
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

    plane_artists = {}

    for i, (linestyle, label, color) in enumerate(zip(linestyles, labels, colors)):
        _, artists = plot_hyperplane(
            xspace,
            uintercepts[i],
            *uslopes[i],
            8,
            c=color,
            plot_kwargs={**plot_kwargs, 'linestyle': linestyle, 'linewidth': max(strengths[i] * 10, 0.1), 'label': label},
            quiver_kwargs={**quiver_kwargs, 'scale': max((1 - strengths[i]) * 0.25, 0.05)},
            return_artists=True
        )
        plane_artists[label] = artists

    line0 = plane_artists[labels[0]]['line'][0]
    line1 = plane_artists[labels[1]]['line'][0]
    line2 = plane_artists[labels[2]]['line'][0]
    arrows0 = plane_artists[labels[0]]['arrows']
    arrows1 = plane_artists[labels[1]]['arrows']
    arrows2 = plane_artists[labels[2]]['arrows']

    n = 60
    point = np.array([[0, 0]], dtype=float)
    centerx = np.mean(x_lim)
    centery = np.mean(y_lim)

    # pprint(dir(arrows0))
    # exit()

    def step(i):
        point[0, 0] = centerx + 20 * math.cos(i * 0.004)
        point[0, 1] = centery + 5 * math.sin(i * 0.004)
        z = forward(point, uintercepts, uslopes)[0]
        scatter.set_offsets(point)
        line0.set_linewidth(max(z[0] * 20, 1))
        line1.set_linewidth(max(z[1] * 20, 1))
        line2.set_linewidth(max(z[2] * 20, 1))
        arrows0.scale = max((1 - z[0]) * 0.1, 0.05)
        arrows1.scale = max((1 - z[1]) * 0.1, 0.05)
        arrows2.scale = max((1 - z[2]) * 0.1, 0.05)

        return (scatter, line0, line1, line2, arrows0, arrows1, arrows2)

    # plt.legend(loc="upper right")
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.set_aspect('equal')
    fig.set_figheight(4)
    fig.set_figwidth(10)
    # plt.savefig("figures/apples_oranges_pears_strengths.pdf")
    anim = FuncAnimation(fig, step, blit=True, interval=0)
    plt.show()
    plt.clf()


# visualize_data_set()
# visualize_two_lines()
# visualize_three_lines()
# visualize_strengths()
visualize_strengths_animated()
