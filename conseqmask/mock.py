import numpy as np
import numba as nb
from numpy.lib.stride_tricks import as_strided


A = np.array([1, 1, 0, 1, 0, 0, 0, 0, 0, 0])
N = 3

@nb.njit(fastmath=True)
def strideloop(A: np.ndarray, N: int):
    B = np.zeros_like(A, np.bool8)
    windows = as_strided(A, (len(A) - N + 1, N), (A.itemsize, A.itemsize))
    print(windows)
    for i, window in enumerate(windows):
        B[i] = window.sum() == 0
    return B

print(strideloop(A, N))