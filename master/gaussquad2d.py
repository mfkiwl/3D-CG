from math import ceil
import numpy as np
from jacobi import jacobi
from scipy.io import loadmat

def gaussquad1d(pgauss):

    n = ceil((pgauss+1)/2)
    nth_leg_poly = jacobi(n, 0, 0)  # Computes the nth legendre polynomial

    n_roots = nth_leg_poly.roots()  # Points in the gauss quadrature scheme.

    A = np.zeros((n, n))

    for i in np.arange(n):
        A[i, :] = jacobi(i, 0, 0)(n_roots)

    rhs = np.zeros_like(A[:, 0])
    rhs[0] = 2
    weights = np.linalg.solve(A, rhs)

    x = (n_roots+1)/2   # Transformation from (-1, 1) -> (0, 1)
    w = weights/2    # Need to account for transformation from (-1, 1) -> (0, 1)

    return x, w


def gaussquad2d(pgauss):

    if pgauss <= 16:
        import os
        file = loadmat('../../master/gaussquad_import/gptsweights.mat')
        x = file['gpts2'][pgauss][0]
        w = np.squeeze(file['gweights2'][pgauss][1])

        return x, w
    else:

        # Get points and weights for integrating a 1D function exactly using gaussquad1d
        pts, weights = gaussquad1d(pgauss)
        weights = weights[:,None]   # Adding second dimension for later
        pts = 2*(pts) - 1     # Map from (0, 1) -> (-1, 1)
        weights *= 2        # Multiply by 2 because we stretched the domain by 2x

        # Map them to a square
        x, y = np.meshgrid(pts, pts, indexing='ij')     # Equivalent to matlab ngrid

        # List of gauss points on the master quad
        eta = np.concatenate((np.reshape(x, (-1, 1), order='F'),
                            np.reshape(y, (-1, 1), order='F')), axis=1)    # Specifying Fortran ordering to prevent having to sort at the end to get it to match the matlab ordering
        xi = np.copy(eta)
        xi[:, 0] = (1+xi[:, 0])*(1-xi[:, 1])/2 - 1  # See slide 25 in 16.930 Interpolation slides for forward transform from square to triangle. We now have a list of the gauss points as transformed to the master triangle, vertices at (-1, -1), (1, -1), (-1, 1)
        xi = (xi+1)/2   # Shifting and scaling to the master element: vertices at (0, 0), (1, 0), (0, 1)

        # Adjust weights due to the transformation
        w = weights*weights.T
        w = w.reshape((-1, 1))
        w *= (1-xi[:,1][:, None]) * (1/4)      # Multiply by (1-y) to scale weight by "jacobian" by the same scaling of a differential volumen element in the transformation from square -> triangle. In this case, the gauss pts are compressed as y changes.
        # Divided by 8 to reflect the 4x shrinkage of the triangle area and then 2x from square->triangle

        return xi, w

def gaussquad3d(pgauss):

    if pgauss <= 15:
        import os
        file = loadmat('../../master/gaussquad_import/gptsweights.mat')
        x = file['gpts3'][pgauss][0]
        w = np.squeeze(file['gweights3'][pgauss][1])/6      # Explain this magic number
    else:
        raise NotImplementedError('GQ degree > 16 not supported!')

    return x, w


if __name__ == '__main__':
    import sys
    x, w = gaussquad2d(int(sys.argv[1]))

    print(x)
    print()
    print(w)

    # gaussquad1d(int(sys.argv[1]))
