# https://stackoverflow.com/questions/67557724/coalescing-rows-from-boolean-mask?noredirect=1#comment119461081_67557724
import numpy as np 

rows = np.r_['1,2,0', :6, :6]
mask = np.tile([1, 1, 0, 0, 1, 1], (2,1)).T.astype(bool)

def maskforwardfill(a: np.ndarray, mask: np.ndarray):
    mask = mask.copy()
    mask[np.diff(mask,prepend=[0]) == 1] = False # set leading True to False
    return a[~mask] # index out wanted rows

def maskforwardfill2(a: np.ndarray, mask: np.ndarray):
    mask = mask.copy()
    mask[1:] = mask[1:] & mask[:-1] # Set leading Trues to Falses
    mask[0] = False
    return a[~mask] # index out wanted rows

# Reduce mask's dimension since I assume that you only do complete rows
print(maskforwardfill2(rows, mask.any(1)))
#[[0 0]
# [2 2]
# [3 3]
# [4 4]]

