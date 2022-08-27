import torch
import pandas as pd
from matplotlib import pyplot as plt

from utils import get_lims, plot_hyperplane, unnormalize_plane
plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'dejavuserif'})

df = pd.read_csv("datasets/apples_and_oranges.csv")
X, y = df[["weight", "height"]].values, df["class"].map({"apple": 0, "orange": 1}).values
x_lim, y_lim = get_lims(X, padding=1.25)


def visualize_data_set():
    plt.xlabel("Weight (g)")
    plt.ylabel("Diameter (cm)")
    plt.title("Comparing apples and oranges")
    plt.scatter(*X[y == 0].T, label="Apple", marker="^", c="greenyellow", edgecolor="black")
    plt.scatter(*X[y == 1].T, label="Orange", marker="o", c="orange", edgecolor="black")
    plt.legend(loc="upper right")
    plt.xlim(*x_lim)
    plt.ylim(*y_lim)
    plt.gca().set_aspect('equal')
    plt.gcf().set_figheight(10)
    plt.gcf().set_figwidth(10)
    plt.savefig("figures/applesnoranges.pdf")
    plt.clf()


def visualize_data_set_with_unknown_point():
    plt.xlabel("Weight (g)")
    plt.ylabel("Diameter (cm)")
    plt.title("Comparing apples and oranges with an unknown")
    plt.scatter(*X[y == 0].T, label="Apple", marker="^", c="greenyellow", edgecolor="black")
    plt.scatter(*X[y == 1].T, label="Orange", marker="o", c="orange", edgecolor="black")
    plt.scatter([130], [5.5], label="Unknown", marker="x", c="black")
    plt.legend(loc="upper right")
    plt.xlim(*x_lim)
    plt.ylim(*y_lim)
    plt.gca().set_aspect('equal')
    plt.gcf().set_figheight(10)
    plt.gcf().set_figwidth(10)
    plt.savefig("figures/applesnoranges_unknown_point.pdf")
    plt.clf()


def visualize_data_set_with_unknown_point_and_line():
    plt.xlabel("Weight (g)")
    plt.ylabel("Diameter (cm)")
    plt.title("Comparing apples and oranges with an unknown and a decision boundary")
    plt.scatter(*X[y == 0].T, label="Apple", marker="^", c="greenyellow", edgecolor="black")
    plt.scatter(*X[y == 1].T, label="Orange", marker="o", c="orange", edgecolor="black")
    plt.scatter([130], [5.5], label="Unknown", marker="x", c="black")

    intercept = 0.06541142612695694
    xslope = 1.0689597
    yslope = 1.5919806
    s = [6.5687, 1.5144]
    m = [135.7327, 6.8887]

    xspace = torch.linspace(x_lim[0], x_lim[1], 4)
    intercept_, xslope_, yslope_ = unnormalize_plane(m, s, intercept, xslope, yslope)
    plot_hyperplane(xspace, intercept_, xslope_, yslope_, 3, c='k', alpha=0.75, quiver_kwargs={
                    'units': 'dots', 'width': 1.75, 'scale': 0.075, 'scale_units': 'dots'})
    plt.legend(loc="upper right")
    plt.xlim(*x_lim)
    plt.ylim(*y_lim)
    plt.gca().set_aspect('equal')
    plt.gcf().set_figheight(10)
    plt.gcf().set_figwidth(10)
    plt.savefig("figures/applesnoranges_unknown_point_with_line.pdf")
    plt.clf()


visualize_data_set()
visualize_data_set_with_unknown_point()
visualize_data_set_with_unknown_point_and_line()
