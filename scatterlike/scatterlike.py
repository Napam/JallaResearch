import numpy as np

# Just extend the problem a little because why not
A = np.array([[[ 1,  2,  3],
               [ 4,  5,  6],
               [ 7,  8,  9]],

              [[10, 11, 12],
               [13, 14, 15],
               [16, 17, 18]],
            
              [[19, 20, 21],
               [22, 23, 24],
               [25, 26, 27]],
])

I = np.array([[0, 2],
              [2, 1],
              [1, 1]])


B = np.stack([A,A])
C = np.array([
    [[0,2],
     [2,1],
     [1,1]],
    
    [[0,0],
     [1,1],
     [2,2]],
])

def scatterlike(A: np.ndarray, I: np.ndarray, target: float=0):
    A_ = A.reshape(-1, *A.shape[-I.shape[-1]:])
    A_[(range(len(A_)),*I.reshape(-1,I.shape[-1]).T.tolist())] = target
    return A_.reshape(A.shape)


from numpy.typing import ArrayLike
from typing import Union
def scatterlike_general(A: np.ndarray, I: np.ndarray, target: Union[ArrayLike,float] = 0):
    A_ = A.reshape(-1, *A.shape[-I.shape[-1]:])
    A_[(range(len(A_)),*I.reshape(-1,I.shape[-1]).T.tolist())] = target
    return A_.reshape(A.shape)

