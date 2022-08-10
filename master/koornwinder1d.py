import numpy as np
from jacobi import jacobi

def koornwinder1d(pts_eval, p):
    """
    X: Coordinates of the points wherethe polynomials are to be evaluated (NPOINTS)
    P: Maximum order of the shape functions. That is all polynomials of degree up to P, NPOLY=P+1

    Returns:
    F: Vandermonde matrix (NPOINTS,NPOLY)
    FX: Vandermonde matrix for the derivative of the Koornwinder polynomials w.r.t. x (NPOINTS,NPOLY)
    """

    if pts_eval.shape[1] > 1:
        print('The input list of x coordinates must be 1 dimensional')
        raise ValueError

    # Perform a transformation from the master element (triangle) to a larger master triangle centered at the origin with side legnth 2 - the element on the right of diagram on pg 25 of Interpolation 16.930 notes
    pts_eval = 2*pts_eval-1

    npoly = p+1
    V = np.zeros((pts_eval.shape[0], npoly))    # Empty vandermonde matrix for the shape functions
    Vx = np.zeros_like(V)    # Empty vandermonde matrix for the shape function derivatives

    for n in np.arange(p):
        poly = jacobi(n, 0, 0) * (2*n+1)**0.5       # Normalization factor such that the function integrates to 1
        dpoly = poly.deriv(1)

        V[:, n] = poly(pts_eval)
        Vx[:, n] = dpoly(pts_eval)

    Vx *= 2

    return V, Vx
