import torch
import pandas as pd
from matplotlib import pyplot as plt

from utils import get_lims
plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset':'dejavuserif'})

df = pd.read_csv("datasets/apples_oranges_pears.csv")
X, y = df[["weight", "height"]].values, df["class"].map({"apple": 0, "orange": 1, "pear": 2}).values
x_lim, y_lim = get_lims(X)

def visualize_data_set():
    plt.xlabel("Weight (g)")
    plt.ylabel("Diameter (cm)")
    plt.title("Comparing apples, oranges and pears")
    plt.scatter(*X[y == 0].T, label="Orange", marker="o", c="orange", edgecolor="black")
    plt.scatter(*X[y == 1].T, label="Apple", marker="^", c="greenyellow", edgecolor="black")
    plt.scatter(*X[y == 2].T, label="Pear", marker="s", c="forestgreen", edgecolor="black", s=20)
    plt.legend(loc="upper right")
    plt.xlim(*x_lim)
    plt.ylim(*y_lim)
    plt.savefig("figures/apples_oranges_pears.pdf")
    plt.clf()

visualize_data_set()
