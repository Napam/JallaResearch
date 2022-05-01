# https://stackoverflow.com/questions/68122821/customized-search-for-consequtive-values-in-numpy-array
import numpy as np
import numba
from numpy.lib.stride_tricks import as_strided
from matplotlib import pyplot as plt


A = np.array([1, 1, 0, 1, 0, 0, 0, 0, 0, 0])
N = 3


def baseline(A: np.ndarray):
    '''
    Your solution
    '''
    B = np.zeros(len(A), dtype=np.bool8)
    for i in range(len(A)):
        if (i + N <= len(A)) and (sum(A[i:i + N]) == 0):
            B[i] = 1
    return B


@numba.njit(fastmath=True)
def branchless_jumper(A: np.ndarray, N: int = 1000):
    '''
    You may need to run this once first to see performance gains
    as this has to be JIT compiled by Numba
    '''
    A = A != 0
    B = np.zeros(len(A), dtype=np.bool_)
    limit = len(A) - N + 1
    i = 0
    while i < limit:
        s = A[i:i + N].sum()
        B[i] = s == 0
        i += s or 1
    return B

@numba.njit(fastmath=True)
def numba_linear(A, N=1000):
    B = np.zeros(len(A), dtype=np.bool_)
    c = 0
    for i in range(len(A)):
        c = c+1 if A[i] == 0 else 0
        if c >= N: B[i - N + 1] = 1
    return B


def correlational(A: np.ndarray):
    '''
    Use a correlation operation
    '''
    B = np.zeros_like(A, dtype=np.bool8)
    B[(np.correlate(A, np.full(N, 1), 'valid') == 0).nonzero()[0]] = 1
    return B


def stridetricks(A: np.ndarray):
    '''
    Same idea as using correlation, but with stride tricks
    Maybe it will be faster? Probably not.
    '''
    u = np.array(A.itemsize)
    B = np.zeros_like(A, dtype=np.bool8)
    B[(~as_strided(A, (len(A) - N + 1, N), u * (1, 1)).sum(1, bool)).nonzero()[0]] = 1
    return B

@numba.njit(fastmath=True)
def strideloop(A: np.ndarray, N: int = 1000):
    B = np.zeros_like(A, np.bool8)
    windows = as_strided(A, (len(A) - N + 1, N), (A.itemsize, A.itemsize))
    for i, window in enumerate(windows):
        B[i] = window.sum() == 0
    return B

@numba.njit(fastmath=True)
def all_equal_idx_back(arr, start, stop, value=0):
    for i in range(stop - 1, start - 1, -1):
        if arr[i] != value:
            return i - start
    return -1


# A = np.array([1, 1, 0, 1, 0, 0, 0, 0, 0, 0])

@numba.njit(fastmath=True)
def find_blocks_while_nb(arr, size=1000, value=0):
    n = len(arr)
    result = np.zeros(n, dtype=np.bool_)
    i = j = 0
    while i < n - size + 1:
        if j == -1 and arr[i + size - 1] == value:
            result[i] = True
            i += 1
        else:
            j = all_equal_idx_back(arr, i, i + size, value)
            if j == -1:
                result[i] = True
                i += 1
            else:
                i += j + 1
    return result

print(numba_linear(A, N).astype(np.int8))
print(branchless_jumper(A, N).astype(np.int8))
print(find_blocks_while_nb(A, N).astype(np.int8))
# exit()
N = 1000

import perfplot
out = perfplot.plot(
    setup=lambda n: np.random.randint(0, 2, n),
    n_range=np.arange(100_000, 400_000 + 1, 25_000),
    kernels=[
        branchless_jumper,
        numba_linear,
        find_blocks_while_nb
    ],
    xlabel='np.arange(x)',
    logx=False,
    logy=False
)
print(out)
plt.show()
# out.show()
