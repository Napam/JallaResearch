import numpy as np 
from numpy.lib.stride_tricks import as_strided
from typing import Tuple


def get_sliding_window_2d(x: np.ndarray, width: int, height: int, rowstride: int, colstride: int):
    """
    x: np.array
    width: width of window
    height: height of window
    rowstride: horizontal window step size
    colstride: vertical window step size 
    """
    imgRows, imgCols = x.shape
    u = np.array(x.itemsize)
    return as_strided(x,
        shape=((imgRows-width)//rowstride+1, (imgCols-height)//colstride+1, width, height), 
        strides=u*(imgCols*rowstride, colstride, imgCols, 1)
    )

a = np.arange(4*4).reshape(4,4)+1
for windows in get_sliding_window_2d(a, 2, 2, 2, 2):
    for window in windows:
        print(window, end="\n\n")