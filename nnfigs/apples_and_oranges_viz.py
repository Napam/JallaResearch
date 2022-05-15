import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from matplotlib import pyplot as plt
plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset':'dejavuserif'})

df = pd.read_csv("datasets/apples_and_oranges.csv")
X, y = df[["weight", "height"]].values, df["class"].map({"apple": 0, "orange": 1}).values
plt.xlabel("Weight (g)")
plt.ylabel("Diameter (cm)")
plt.title("Comparing apples and oranges")
plt.scatter(*X[y == 0].T, label="Orange", marker="o", c="orange", edgecolor="black")
plt.scatter(*X[y == 1].T, label="Apple", marker="^", c="greenyellow", edgecolor="black")
plt.legend()
plt.savefig("figures/applesnoranges.pdf")
