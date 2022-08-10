import numpy as np
from jacobi import jacobi
from master.pascalindex import pascalindex2d
from numpy.polynomial import Polynomial as poly

def koornwinder2d(pts_eval, porder):
    """
        KOORNWINDER2D Vandermonde matrix for Koornwinder polynomials in
                    the master triangle[0, 0]-[1, 0]-[0, 1]
        [F, FX, FY] = KOORNWINDER(X, P)

            X:         Coordinates of the points wherethe polynomials
                        are to be evaluated(npoints, dim)
            PORDER:    Maximum order of the polynomials consider. That
        is all polynomials of complete degree up to p,
                        npoly = (PORDER+1)*(PORDER+2)/2
            F:         Vandermonde matrix(npoints, npoly)
            FX:        Vandermonde matrix for the derivative of the Koornwinder
                        polynomials w.r.t. x(npoints, npoly)
            FY:        Vandermonde matrix for the derivative of the Koornwinder
                        polynomials w.r.t. y(npoints, npoly)
    """

    if pts_eval.shape[1] > 2:
        print('The input list of x coordinates must be 2 dimensional')
        raise ValueError


    poly_indices = pascalindex2d(porder)
    npoly = poly_indices.shape[0]

    # Empty vandermonde matrices for the shape functions and their derivatives
    V = np.zeros((pts_eval.shape[0], npoly))
    Vx = np.zeros_like(V)
    Vy = np.zeros_like(V)

    # Points come in as coordinates on the master triangle with vertices (0, 0), (1, 0), and (0, 1). They must be transformed first to the standard triangle centered at the origin, and then to the standard square sitting on the origin between (-1, 1)x(-1, 1)
    # Perform a transformation from the master element (triangle) to a larger master triangle centered at the origin with side legnth 2 - the element on the right of diagram on pg 25 of Interpolation 16.930 notes
    xi = 2*pts_eval-1   # Operates on both x and y - stretching and shifting triangle so that midpoint of hypotenuse is on the origin - in "standard" configuration
    ind_1 = np.argwhere(xi == 1)[0]
    xi[ind_1] = 1-1e-8

    # Perform coordinate transformation from triangle to square - taken from slide 25 of 16.930 notes. Note that eta2 = xi2 in the transformation.
    eta = xi
    eta[:, 0] = 2*(1+xi[:, 0])/(1-xi[:, 1]) - 1
    eta_deriv = eta     # Saving points at which to evaluate the derivative before resettting the corner point to 1
    eta[ind_1] = 1      # Switching this back and forth to allow both the polynomial evaluation and the derivative evaluation to work out

    jac = np.zeros_like(xi)
    jac[:, 0] = 2/(1-eta[:, 1])
    jac[:, 1] = 2*(1+eta[:,0])/(1-eta[:, 1])**2

    for idx, (m, n) in enumerate(poly_indices):
        Px = jacobi(m, 0, 0)
        Py = jacobi(n, 2*m+1, 0)*(poly([1, -1])/2)**m
        dPx = Px.deriv(1)
        dPy = Py.deriv(1)

        Px_val = Px(eta[:, 0])
        Py_val = Py(eta[:, 1])
        dPx_val = dPx(eta_deriv[:, 0])
        dPy_val = dPy(eta_deriv[:, 1])

        norm = (2*(2*m+1)*(m+n+1))**0.5        # Normalization factor such that the function integrates to 1

        V[:, idx] = Px_val * Py_val*norm
        Vx[:, idx] = dPx_val*Py_val*jac[:,0]*norm
        Vy[:, idx] = norm*(dPx_val*Py_val*jac[:, 1] + Px_val*dPy)

    Vx *= 2
    Vy *= 2

    return V, Vx. Vy
