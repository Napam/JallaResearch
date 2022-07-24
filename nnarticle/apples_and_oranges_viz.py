import torch
import pandas as pd
from matplotlib import pyplot as plt

from utils import get_lims
plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset':'dejavuserif'})

df = pd.read_csv("datasets/apples_and_oranges.csv")
X, y = df[["weight", "height"]].values, df["class"].map({"apple": 0, "orange": 1}).values
x_lim, y_lim = get_lims(X)

def visualize_data_set():
    plt.xlabel("Weight (g)")
    plt.ylabel("Diameter (cm)")
    plt.title("Comparing apples and oranges")
    plt.scatter(*X[y == 0].T, label="Orange", marker="o", c="orange", edgecolor="black")
    plt.scatter(*X[y == 1].T, label="Apple", marker="^", c="greenyellow", edgecolor="black")
    plt.legend(loc="upper right")
    plt.xlim(*x_lim)
    plt.ylim(*y_lim)
    plt.savefig("figures/applesnoranges.pdf")
    plt.clf()

def visualize_data_set_with_unknown_point():
    plt.xlabel("Weight (g)")
    plt.ylabel("Diameter (cm)")
    plt.title("Comparing apples and oranges with an unknown")
    plt.scatter(*X[y == 0].T, label="Orange", marker="o", c="orange", edgecolor="black")
    plt.scatter(*X[y == 1].T, label="Apple", marker="^", c="greenyellow", edgecolor="black")
    plt.scatter([130], [5.5], label="Unknown", marker="x", c="black")
    plt.legend(loc="upper right")
    plt.xlim(*x_lim)
    plt.ylim(*y_lim)
    plt.savefig("figures/applesnoranges_unknown_point.pdf")
    plt.clf()

def visualize_data_set_with_unknown_point_and_line():
    plt.xlabel("Weight (g)")
    plt.ylabel("Diameter (cm)")
    plt.title("Comparing apples and oranges with\nan unknown and a decision boundary")
    plt.scatter(*X[y == 0].T, label="Orange", marker="o", c="orange", edgecolor="black")
    plt.scatter(*X[y == 1].T, label="Apple", marker="^", c="greenyellow", edgecolor="black")
    plt.scatter([130], [5.5], label="Unknown", marker="x", c="black")
    xrange = torch.linspace(x_lim[0] * 0.5, x_lim[1] * 1.5, 4)
    plt.plot(xrange, -0.2084 * xrange + 35.2242, c='k', label='Decision boundary')
    plt.legend(loc="upper right")
    plt.xlim(*x_lim)
    plt.ylim(*y_lim)
    plt.savefig("figures/applesnoranges_unknown_point_with_line.pdf")
    plt.clf()

# visualize_data_set()
# visualize_data_set_with_unknown_point()
visualize_data_set_with_unknown_point_and_line()
