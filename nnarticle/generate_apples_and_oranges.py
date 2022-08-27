import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from matplotlib import pyplot as plt
plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'dejavuserif'})
rng = np.random.default_rng(2)


def make_blobs(sizes: ArrayLike, means: ArrayLike, stds: ArrayLike):
    X = np.concatenate([
        rng.standard_normal((size, 2)) * std + mean for size, mean, std in zip(sizes, means, stds)
    ])
    y = np.arange(len(sizes)).repeat(sizes)

    shufflerState = np.random.default_rng(1)
    shufflerState.shuffle(X)
    shufflerState = np.random.default_rng(1)
    shufflerState.shuffle(y)
    return X, y


avg_orange_diameter = 8  # cm
avg_orange_weight = 140  # gram
avg_apple_diameter = 6  # cm
avg_apple_weight = 130  # gram

centers = [[avg_orange_weight, avg_orange_diameter], [avg_apple_weight, avg_apple_diameter]]

X, y = make_blobs([30, 30], centers, [[4.5, 1]] * len(centers))

df = pd.DataFrame({
    "weight": X[:, 0],
    "height": X[:, 1],
    "class": np.vectorize({0: "orange", 1: "apple"}.get)(y)
})
print(df)
df.to_csv("datasets/apples_and_oranges.csv", index=False)
