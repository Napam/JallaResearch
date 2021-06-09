import numpy as np
from typing import Callable, Sequence
from numbers import Number
from fractions import Fraction
from numpy.core.fromnumeric import ravel
from sympy.solvers.inequalities import solve_poly_inequality
from sympy.polys import Poly 
import sympy as sp
from itertools import combinations
from pprint import pprint

UTF = {
    0: u'\u2080',
    1: u'\u2081',
    2: u'\u2082',
    3: u'\u2083',
    4: u'\u2084',
    5: u'\u2085',
    6: u'\u2086',
    7: u'\u2087',
    8: u'\u2088',
    9: u'\u2089',
    10: '₁₀', 
    11: '₁₁', 
    12: '₁₂', 
    'cal B': '𝓑', 
    'cal N': '𝓝',
    '_B': 'ᵦ',
    '_N': 'ₙ',
    'inv': '⁻¹',
    'T': 'ᵀ',
    'zeta': 'ζ',
    'geq': '≥',
    'Delta': 'Δ',
    'rarrow':'⇒',
} 

def get_formatter(spacing: int) -> Callable:
    def formatter(x):
        a = str(Fraction(x).limit_denominator())
        return f'{a:>{spacing}}'
    return formatter


def print_lp_dict(DICp: Sequence[Sequence[Number]], basic: np.ndarray, nonbasic: np.ndarray):
    with np.printoptions(formatter={'float':get_formatter(4)}):    
        strbasic = [f'x{UTF[i]}' for i in basic+1]
        strnonbasic = [f'x{UTF[i]}' for i in nonbasic+1]
    
        objstring = f'{UTF["zeta"]}  = {DICp[0]}'

        print(f'x{UTF["_N"]} = ', ' , '.join(strnonbasic))
        print(objstring)
        print(f'{"":_^{len(objstring)}}')
        for basicvar, row in zip(strbasic, DICp[1:]):
            print(f'{basicvar} = {row}')


def get_dict_from_basis(
    A: Sequence[Sequence[Number]], 
    b: Sequence[Number], 
    c: Sequence[Number], 
    basis: Sequence[int]
) -> None:
    A = np.array(A)
    b = np.array(b)
    c = np.array(c)
    basis = np.array(basis)

    m, n = A.shape

    A = np.c_[A, np.eye(m, dtype=int)]
    c = np.concatenate([c, np.zeros(m)])
    
    basic = basis - 1
    allvars = np.arange(m+n)
    nonbasic = np.setdiff1d(allvars, basic)
    c_basic = c[basic]
    c_nonbasic = c[nonbasic]

    print('Variables:', *['x'+UTF[v] for v in allvars+1])
    print(UTF['cal B']+' =', basic+1)
    print(UTF['cal N']+' =', nonbasic+1)
    print('c'+UTF['_B'], c_basic)
    print('c'+UTF['_N'], c_nonbasic)

    B = A[:,basic]
    N = A[:,nonbasic]

    with np.printoptions(formatter={'float':get_formatter(4)}):    
        print('\nB =')
        print(B)

        print('\nN =')
        print(N)

        print('\nb =')
        print(b)

        Binv = np.linalg.inv(B)

        print('\nB'+UTF['inv']+'N =')
        print(Binv@N)

    with np.printoptions(formatter={'float':get_formatter(4)}):    
        Binvb = Binv@b
        print('\nB'+UTF['inv']+'b =')
        print(Binvb)

        cBb = c_basic@Binv@b
        print('\nc'+UTF['T']+UTF['_B']+'B'+UTF['inv']+'b =', cBb)
    
    with np.printoptions(formatter={'float':get_formatter(4)}):    
        BinvN = Binv@N
        zn = (BinvN.T)@c_basic - c_nonbasic
        print(f'\n(B{UTF["inv"]}N){UTF["T"]}c{UTF["_B"]}-c{UTF["_N"]} =', zn)
        
    print('\nDictionary:')
    DICp = np.zeros((m+1, n+m-1))
    DICp[0] = [cBb, *(-zn)]
    DICp[1:,0] = Binvb.ravel()
    DICp[1:,1:] = -BinvN
    
    print_lp_dict(DICp, basic, nonbasic)


def ranging(
    DICp: Sequence[Sequence[Number]],
    c: Sequence[Number],
    dc: Sequence[Number], 
    basic: Sequence[int],
    nonbasic: Sequence[int],
) -> None:
    DICp = np.array(DICp)
    c = np.array(c)
    dc = np.array(dc)
    basic = np.array(basic)
    nonbasic = np.array(nonbasic)

    zn = -DICp[0,1:].reshape(-1,1)
    
    basic = np.array(basic)-1
    nonbasic = np.array(nonbasic)-1
    
    dc_basic = dc[basic].reshape(-1,1)
    dc_nonbasic = dc[nonbasic].reshape(-1,1)
    
    BinvN = -DICp[1:,1:]

    BinvNTcb = BinvN.T@dc_basic
    dzn = BinvNTcb - dc_nonbasic

    print('Got dicitonary:')
    print_lp_dict(DICp, basic, nonbasic)

    print(f'\n-B{UTF["inv"]}N =')
    print(-BinvN)
    print()

    print(f'{UTF["Delta"]}c{UTF["T"]} = {dc.ravel()}')
    print(f'{UTF["Delta"]}c{UTF["_B"]}{UTF["T"]} = {dc_basic.ravel()}')
    print(f'{UTF["Delta"]}c{UTF["_N"]}{UTF["T"]} = {dc_nonbasic.ravel()}')


    print(f'\n(B{UTF["inv"]}N){UTF["T"]} =')
    print(BinvN.T)
    print()

    print(f'{UTF["Delta"]}z{UTF["_N"]}= '
          f'(B{UTF["inv"]}N){UTF["T"]}{UTF["Delta"]}c{UTF["_B"]} - c{UTF["_N"]} =')
    print(f'{BinvNTcb.ravel()}{UTF["T"]} - {dc_nonbasic.ravel()}{UTF["T"]} = {dzn.ravel()}{UTF["T"]}')

    print(f'\nz*{UTF["_N"]} + t{UTF["Delta"]}z{UTF["_N"]} {UTF["geq"]} 0 {UTF["rarrow"]} '
          f'{zn.ravel()}{UTF["T"]} + t{dzn.ravel()}{UTF["T"]} {UTF["geq"]} 0')    
    print(UTF['rarrow'])
    for i, j in zip(zn.ravel(), dzn.ravel()):
        string = ''
        string += str(i)
        if j >= 0:
            string += ' +'
        elif j < 0:
            string += ' -'
        string += f' {abs(j)}t {UTF["geq"]} 0'
        print(string)

    t = sp.Symbol('t', real=True)
    polys = [(Poly(i + j*t), '>=') for i, j in zip(zn.ravel(), dzn.ravel()) if j != 0]
    result = sp.Intersection(*[s for p in polys for s in solve_poly_inequality(*p)])

    # l for lower, u for upper    
    l_bound, u_bound = result.left, result.right

    l_c = c + dc*l_bound
    u_c = c + dc*u_bound
    # Fix nans 
    mask = l_c == sp.nan
    l_c[mask] = c[mask]    
    
    mask = u_c == sp.nan
    u_c[mask] = c[mask]    

    print()
    print(f'Respective to {UTF["Delta"]}c = {dc}')
    print(f'We see that we can vary c in:')
    print(l_c)
    print('to')
    print(u_c)

def submatrix(A: Sequence[Sequence[Number]], b: Sequence[Sequence[Number]]) -> None:
    A = np.array(A)
    b = np.array(b)
    
    m, n = A.shape

    print(f'A = \n{A}')
    print(f'b = \n{b}\n')

    indices = np.arange(n)
    pairs = np.array(list(combinations(indices,2)))
    submatrices = np.zeros((len(pairs), m, m))
    for i, pair in enumerate(pairs):
        mat = A[:,pair]
        submatrices[i] = (mat)

    for pair, mat in zip(pairs, submatrices):
        print(f'Basic vars: {pair} {UTF["rarrow"]}')
        print('B=')
        print(mat)
        det = np.linalg.det(mat)
        print(f'det = {det:.2f}')
        if np.linalg.det(mat) != 0:
            print('Is nonsingular')
        else:
            print('Is SINGULAR')
        print()

    inverses = np.zeros_like(submatrices)
    vertices = np.zeros((len(submatrices), m))
    for i, mat in enumerate(submatrices):
        Binv = np.linalg.inv(mat)
        inverses[i] = Binv
        vertices[i] = (Binv@b).ravel()

    feasible_mask = (vertices < 0).sum(axis=1) == 0

    print('\n\nBasis feasible')
    for isbasis, pair, mat in zip(feasible_mask, pairs, submatrices):
        if isbasis:
            print(pair)
            print(mat)

if __name__ == '__main__':
    submatrix(
        A=[
            [-1, 4, 1, 0],
            [ 2, 6, 0, 1]
        ],
        b=[
            [-1],
            [ 6],
        ]
    )

    # get_dict_from_basis(
    #     A=[
    #         [-1,-1, 0, 0, 0, 0],
    #         [ 1, 0,-1, 0,-1, 0],
    #         [ 0, 1, 1,-1, 0, 0],
    #         [ 0, 0, 0, 1, 0, 1],
    #         [ 0, 0, 0, 0, 1,-1],
    #     ],
    #     b=[
    #         [ 2],
    #         [ 2],
    #         [-1],
    #         [-4],
    #         [ 1],
    #     ],
    #     c=[4,2,1,1,1,3],
    #     basis=[1,3,4,6]
    # )

    # Compulsory 2
    # get_dict_from_basis(
    #     A=[
    #         [ 1, 2, 3, 4, 5],
    #         [ 0, 5,-3,-2,-1],
    #     ],
    #     b=[
    #         [ 2],
    #         [ 3],
    #     ],
    #     c=[1,2,4,8,16],
    #     basis=[1,5]
    # )

    # H15 Ex2
    # get_dict_from_basis(
    #     A=[
    #         [ 1, 2, 3],
    #         [-7, 5,-1],
    #     ],
    #     b=[
    #         [ 1],
    #         [-3],
    #     ],
    #     c=[1,2,4],
    #     basis=[1,3]
    # )

    # ranging(
    #     DICp=[
    #         [ 6,-1,-1,-2],
    #         [ 3,-1,-1,-1],
    #         [ 4, 2,-2, 3]
    #     ],
    #     c=[1,2,1,0,0],
    #     dc=[0,1,0,0,0],
    #     basic=[2,5],
    #     nonbasic=[1,3,4],
    # )

    # ranging(
    #     DICp=[
    #         [ 3,-1,-0,-1],
    #         [ 3,-1,-1,-1],
    #         [10,-2,-4, 1]
    #     ],
    #     c=[1,0,1,0,0],
    #     dc=[0,1,0,0,0],
    #     basic=[1,5],
    #     nonbasic=[2,3,4],
    # )