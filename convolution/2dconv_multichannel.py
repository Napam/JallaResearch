import numpy as np 
from numpy.lib.stride_tricks import as_strided
import time

def get_sliding_window(x: np.ndarray, k: np.ndarray, rowstride: int, colstride: int):
    imgChannels, imgRows, imgCols = x.shape
    _, kernelRows, kernelCols = k.shape
    u = np.array(x.itemsize)
    
    return as_strided(x,
        shape=((imgRows-kernelRows)//rowstride+1, (imgCols-kernelCols)//colstride+1, imgChannels, kernelRows, kernelCols), 
        strides=u*(imgCols*rowstride, colstride, imgRows*imgCols, imgCols, 1)
    )


def conv2d(x: np.ndarray, k: np.ndarray, rowstride: int, colstride: int):
    """
    Performs 2d convolution on images with arbitrary number of channels where you can
    specify the strides as well. 

    x: np.ndarray, image array of shape (C x N x M), where C is number of channels
    k: np.ndarray, convolution kernel of shape (C x P x Q), where C is number of channels
    rowstride: int, "vertical" step size
    colstride: int, "horizontal" step size
    """
    sliding_window_view = get_sliding_window(x, k, rowstride, colstride)

    return np.tensordot(sliding_window_view, k, axes=3)


x = np.array([
    [[1,1,1,1],
     [1,1,1,1],
     [2,2,2,2],
     [2,2,2,2]], 

    [[1,1,2,2],
     [1,1,2,2],
     [4,4,8,8],
     [4,4,8,8]]
])


k = np.array([
    [[1,1],  
     [1,1]],

    [[1,1],  
     [1,1]]
]) / 8

print(conv2d(x,k,1,1))
print(conv2d(x,k,2,2))

def conv2d_asciiviz(x: np.ndarray, k: np.ndarray, rowstride: int, colstride: int):
    x = x.copy().astype(object)
    sliding_window_view = get_sliding_window(x, k, rowstride, colstride)
    highlighter = np.vectorize(lambda x: f"\x1b[33m{x}\x1b[0m")
    r = np.full(sliding_window_view.shape[:2], np.nan)
    with np.printoptions(nanstr="", formatter={"all":lambda x: str(x)}):
        for i, row in enumerate(sliding_window_view):
            for j, window in enumerate(row):
                temp = window.copy()
                r[i,j] = np.tensordot(window, k, axes=3)
                window[...] = highlighter(window)
                print(f"\x1b[JChannels:\n{x}\n\nResult:\n{str(r)}\x1b[{x.shape[0]*x.shape[1]+len(r)+4}A")
                window[...] = temp
                time.sleep(0.69)
    print(f"\x1b[{x.shape[0]*x.shape[1]+len(r)+4}B")
    return r

print("Output:\n",conv2d(x,k,1,1))