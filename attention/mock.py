import numpy as np 
from numpy.lib.stride_tricks import as_strided, sliding_window_view

x = np.stack([
    np.array([
        [1,1,1,1],
        [1,1,1,1],
        [1,1,1,1],
        [1,1,1,1],
    ]),
    np.array([
        [2,2,2,2],
        [2,2,2,2],
        [2,2,2,2],
        [2,2,2,2],
    ]),
    np.array([
        [3,3,3,3],
        [3,3,3,3],
        [3,3,3,3],
        [3,3,3,3],
    ])
])

def get_sliding_window(x, window_size, strides):
    as_strided(x)

get_sliding_window(x, window_size=(2,2), strides=(2,2))

