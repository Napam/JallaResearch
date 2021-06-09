import numpy as np 
from typing import Callable, Sequence
from numbers import Number, Integral
from fractions import Fraction

def a_mu(mu: float):
    return (1 - 2*mu + np.sqrt(1+4*mu**2))/2

if __name__ == '__main__':
    mu = 0.0001
    a = a_mu(mu)
    
    A = np.array([
        [1,0],
        [0,1]
    ])

    b = np.array([
        [1],
        [1]
    ])

    x_ = np.array([
        [  a],
        [1/2]
    ])

    w_ = np.array([
        [1-a],
        [1/2]
    ])

    y_ = np.array([
        [mu/(1-a)],
        [2*mu]
    ])

    z_ = np.array([
        [mu/a],
        [2*mu]
    ])

    print(A@x_+w_)
    print(A.T@y_-z_)
    print(x_*z_)
    print(y_*w_)

