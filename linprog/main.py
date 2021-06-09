from matrixtools import *

get_dict_from_basis(
    A=[
        [ 1, 2, 3],
        [-7, 5,-1],
    ],
    b=[
        [ 1],
        [-3],
    ],
    c=[1,2,4],
    basis=[1,3]
)

ranging(
    DICp=[
        [ 6,-1,-1,-2],
        [ 3,-1,-1,-1],
        [ 4, 2,-2, 3]
    ],
    c=[1,2,1,0,0],
    dc=[0,1,0,0,0],
    basic=[2,5],
    nonbasic=[1,3,4],
)


A = np.array([
    [1,2],
    [3,4],
])

print(np.linalg.inv(A))