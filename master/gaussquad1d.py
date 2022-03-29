from math import ceil
import numpy as np
from sympy import nroots
from jacobi import jacobi

def gaussquad1d(pgauss):
    
    n = ceil((pgauss+1)/2)
    nth_leg_poly = jacobi(n, 0, 0)  # Computes the nth legendre polynomial

    n_roots = nth_leg_poly.roots()  # Points in the gauss quadrature scheme.

    A = np.zeros((n, n))

    for i in np.arange(n):
        A[i,:] = jacobi(i, 0, 0)(n_roots)

    rhs = np.zeros_like(A[:,0])
    rhs[0] = 2
    weights = np.linalg.solve(A, rhs)

    x = (n_roots+1)/2   # Transformation from (-1, 1) -> (0, 1)
    w = weights/2       # Need to account for transformation from (-1, 1) -> (0, 1)

    return x, w

if __name__ == '__main__':
    import sys
    gaussquad1d(int(sys.argv[1]))