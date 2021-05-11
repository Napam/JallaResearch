import numpy as np 
from numpy.lib.stride_tricks import sliding_window_view

a = np.array([
    [1,2,3,4],
    [1,2,3,4],
    [1,2,3,4],
    [1,2,3,4],
])

print(sliding_window_view(a, (2,2)).shape)