# https://stackoverflow.com/questions/70326790/sort-array-with-repeated-values/70327394#70327394
import numpy as np


def groupsort(a: np.ndarray):
    uniques, counts = np.unique(a, return_counts=True)
    min_count = np.min(counts)
    counts -= min_count
    n_easy = min_count * len(uniques)

    # Pre allocate array
    values = np.empty(n_easy + counts.sum(), dtype=a.dtype)

    # Set easy values
    temp = values[:n_easy].reshape(min_count, len(uniques))
    temp[:] = uniques

    # Set "hard" values
    i = n_easy
    while np.any(mask := counts > 0): # Python 3.8 trick
        masksum = mask.sum()
        values[i : i + masksum] = uniques[mask]
        counts -= mask
        i += masksum
    return values


a = np.array(list(range(4)) * 2 + [1, 1, 1, 1, 1, 0, 0, 0, 2, 2])
np.random.shuffle(a)
print(a)
print(groupsort(a))
