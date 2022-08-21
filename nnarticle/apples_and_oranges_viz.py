import torch
import pandas as pd
from matplotlib import pyplot as plt

from utils import get_lims, plot_hyperplane
plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset':'dejavuserif'})

df = pd.read_csv("datasets/apples_and_oranges.csv")
X, y = df[["weight", "height"]].values, df["class"].map({"apple": 0, "orange": 1}).values
x_lim, y_lim = get_lims(X, padding = 2.0)

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
    intercept = -0.009632732719182968
    xslope = -0.93068683
    yslope = -1.2795106
    s_1, s_2 = 7.5490, 1.5144
    m_1, m_2 = 135.9769, 6.8887



    xspace = torch.linspace(x_lim[0] * 0.5, x_lim[1] * 1.5, 4) 
    intercept_ = - intercept + (m_1 * xslope) / s_1 + (m_2 * yslope) / s_2
    xslope_ = xslope / s_1
    yslope_ = yslope / s_2
    plot_hyperplane(xspace, intercept_, xslope_, yslope_, 10)
    plt.legend(loc="upper right")
    plt.xlim(*x_lim)
    plt.ylim(*y_lim)
    # plt.savefig("figures/applesnoranges_unknown_point_with_line.pdf")
    plt.gca().set_aspect('equal')
    plt.show()
    plt.clf()

# visualize_data_set()
# visualize_data_set_with_unknown_point()
visualize_data_set_with_unknown_point_and_line()
