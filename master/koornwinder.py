import numpy as np
from jacobi import jacobi
from pascalindex import pascalindex2d, pascalindex3d
from numpy.polynomial import Polynomial as poly

""" It is important to note that the koornwinder polynomials are not the shape functions (nodal basis functions). They are merely a 2D equivalent of the legendre orthogonal basis functions in 1D.
    In 1D: legendre polynomials formed the columns in the vandermonde matrix, so the columns were orthogonal. The nodal basis functions were created by solving VL=I, since the shape functions sampled at the nodes yields the identity matrix.
    The equivalent to skipping the legendre polynomials and using a monomial basis would be like skipping the koornwinder polynomials and forming the shape functions using a basis of {1, x1, x2, x1x2, x1^2, x2^2, ...}.
"""

def koornwinder(pts_eval, p):
    """
    Driver for the koornwinder shape function generation
    "Tensor products of Jacobi polynomials"
    """

    dim = pts_eval.shape[1]
    if dim == 1:
        return koornwinder1d(pts_eval, p)
    elif dim == 2:
        return koornwinder2d(pts_eval, p)
    elif dim == 3:
        raise NotImplementedError('Dimension 3 not implemented yet')
    else:
        raise ValueError('Dim must be 1, 2, or 3')

def koornwinder1d(pts_eval, porder):
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

    npoly = porder+1
    # Empty vandermonde matrix for the shape functions
    V = np.zeros((pts_eval.shape[0], npoly))
    # Empty vandermonde matrix for the shape function derivatives
    Vx = np.zeros_like(V)

    for n in np.arange(npoly):
        # Normalization factor such that the function integrates to 1
        poly = jacobi(n, 0, 0) * (2*n+1)**0.5
        dpoly = poly.deriv(1)

        V[:, n] = np.squeeze(poly(pts_eval))
        Vx[:, n] = np.squeeze(dpoly(pts_eval)[:, None])

    Vx *= 2

    return V, Vx

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

    if pts_eval.shape[1] != 2:
        raise ValueError('The input list of x coordinates must be 2 dimensional')

    poly_indices = pascalindex2d(porder)
    npoly = poly_indices.shape[0]   # Equivalent to (k+1)*(k+2)/2

    # Empty vandermonde matrices for the shape functions and their derivatives
    V = np.zeros((pts_eval.shape[0], npoly))
    Vx = np.zeros_like(V)
    Vy = np.zeros_like(V)

    # Points come in as coordinates on the master triangle with vertices (0, 0), (1, 0), and (0, 1). They must be transformed first to the standard triangle centered at the origin, and then to the standard square sitting on the origin between (-1, 1)x(-1, 1)
    # Perform a transformation from the master element (triangle) to a larger master triangle centered at the origin with side legnth 2 - the element on the right of diagram on pg 25 of Interpolation 16.930 notes
    xi = 2*pts_eval-1   # Operates on both x and y - stretching and shifting triangle so that midpoint of hypotenuse is on the origin - in "standard" configuration
    
    ind_1 = np.argwhere(xi[:, 1] > 0.99999999)  # Or, we could have hard set the singularity while performing the transformation (see koornwinder3d)
    # id_1 if the corner point exists: [[9]], shape = (1, 1)
    # id_1 if the corner point does not exist: [], shape = (0, 1)
    if ind_1.shape[0]:   # There will not always be a point at the corner, for example if koornwinder is being used to evaluate at the GQpts
        xi[ind_1, 1] = 0.99999999

    # Perform coordinate transformation from triangle to square - taken from slide 25 of 16.930 notes.
    eta = np.copy(xi)
    eta[:, 0] = 2*(1+xi[:, 0])/(1-xi[:, 1]) - 1
    #eta2 = xi2 in the transformation

    # Saving points at which to evaluate the derivative before resettting the corner point to 1
    eta_displaced = np.copy(eta)
    # Moving the point back to the corner in the actual eta vector
    eta[ind_1] = 1      # FIX THIS!

    jac = np.zeros_like(xi)
    jac[:, 0] = 2/(1-xi[:, 1])
    jac[:, 1] = 2*(1+xi[:, 0])/(1-xi[:, 1])**2

    for idx, (m, n) in enumerate(poly_indices):
        # if idx != 4:
        #     continue
        Px = jacobi(m, 0, 0)
        Py = jacobi(n, 2*m+1, 0)*(poly([1, -1])/2)**m
        dPx = Px.deriv(1)
        dPy = Py.deriv(1)

        # IMPORTANT: The points used to calculate the shape function values include the corner (singularity) because the polynomials are defined there...
        Px_val = Px(eta[:, 0])
        Py_val = Py(eta[:, 1])
        
        # However, the jacobian is singular at the corner, so we can't evaluate that point directly in our integral. Instead, we look at the point that is slightly displaced from the corner and evaluate the derivative of the function at that displaced point.
        # Note that since we have split up the function into muliple parts to differentiate via the product rule, we still need to evaluate both the derivative and re-evaluate the shape function at the displaced point. The only difference between eta and eta_displaced is that the eta coord of the corner is shifted down slightly.
        Px_val_disp = Px(eta_displaced[:, 0])
        Py_val_disp = Py(eta_displaced[:, 1])
        dPx_val = dPx(eta_displaced[:, 0])
        dPy_val = dPy(eta_displaced[:, 1])

        # Normalization factor such that the function integrates to 1
        norm = (2*(2*m+1)*(m+n+1))**0.5

        V[:, idx] = Px_val * Py_val*norm
        Vx[:, idx] = dPx_val*Py_val_disp*jac[:, 0]*norm
        Vy[:, idx] = norm*(dPx_val*Py_val_disp*jac[:, 1] + Px_val_disp*dPy_val)

    Vx *= 2
    Vy *= 2

    return V, Vx, Vy

def koornwinder3d(pts_eval, porder):
    """
        KOORNWINDER3D Vandermonde matrix for Koornwinder polynomials in
                    the master tetrahedron [0, 0]-[1, 0]-[0, 1]-[1, 1]
        [F, FX, FY, FZ] = KOORNWINDER(X, P)

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
            FZ:        Vandermonde matrix for the derivative of the Koornwinder
                        polynomials w.r.t. z(npoints, npoly)
    """

    if pts_eval.shape[1] != 3:
        print('The input list of x coordinates must be 3 dimensional')
        raise ValueError

    poly_indices = pascalindex3d(porder)
    npoly = poly_indices.shape[0]   # Equivalent to (k+1)*(k+2)*(k+3)/6
    numpts = pts_eval.shape[0]

    # Empty vandermonde matrices for the shape functions and their derivatives
    V = np.zeros((numpts, npoly))
    Vx = np.zeros_like(V)
    Vy = np.zeros_like(V)
    Vz = np.zeros_like(V)

    # Points come in as coordinates on the master tetrahedron with vertices (0, 0), (1, 0), (0, 1), and (1, 1). They must be transformed first to the standard triangle centered at the origin, and then to the standard cube sitting on the origin between (-1, 1)^3
    # Perform a transformation from the master element (triangle) to a larger master triangle centered at the origin with side legnth 2 - the element on the right of diagram on pg 25 of Interpolation 16.930 notes
    xi = 2*pts_eval-1   # Operates on both x and y - stretching and shifting triangle so that midpoint of hypotenuse is on the origin - in "standard" configuration


    # # FIX
    # ind_1 = np.argwhere(xi[:, 1] > 0.99999999)
    # # id_1 if the corner point exists: [[9]], shape = (1, 1)
    # # id_1 if the corner point does not exist: [], shape = (0, 1)
    # if ind_1.shape[0]:   # There will not always be a point at the corner, for example if koornwinder is being used to evaluate at the GQpts
    #     xi[ind_1, 1] = 0.99999999

    # # FIX
    # # Perform coordinate transformation from triangle to square - taken from slide 25 of 16.930 notes. Note that eta2 = xi2 in the transformation.
    # eta = np.copy(xi)
    # eta[:, 0] = -2*(1+xi[:,0])/(xi[:,1]+xi[:,2]) - 1
    # eta[:, 1] = 2*(1+xi[:, 1])/(1-xi[:, 2]) - 1
    # # eta3 is equal to xi3 in the transformation

    # # Saving points at which to evaluate the derivative before resettting the corner point to 1
    # eta_displaced = np.copy(eta)
    # # Moving the point back to the corner in the actual eta vector
    # eta[ind_1] = 1

    eta = np.zeros_like(xi)
    xi_displaced = np.copy(xi)
    eta_displaced = np.zeros_like(xi)

    eps = 1e-8
    for i, pt in enumerate(xi):
        if abs((pt[1] + pt[2])) < eps:
            eta[i,0] = -1
            xi_displaced[i,2] = -pt[1] - eps
        else:
            # Normal eta1 transform
            eta[i, 0] = -2*(1+pt[0])/(pt[1]+pt[2]) - 1
        if pt[2] > 1-eps:
            eta[i, 1] = -1
            xi_displaced[i,2] = 1-eps
        else:
            # Normal eta 2 transform
            eta[i, 1] = 2*(1+pt[1])/(1-pt[2]) - 1

        eta[i:,2] = pt[2]


    eta_displaced[:, 0] = -2*(1+xi_displaced[:,0])/(xi_displaced[:,1]+xi_displaced[:, 2]) - 1
    eta_displaced[:, 1] = 2*(1+xi_displaced[:, 1])/(1-xi_displaced[:, 2]) - 1
    eta_displaced[:, 2] = xi_displaced[:,2]

    jac = np.zeros((numpts, 3,3))
    jac[:, 0, 0] = -2/(xi_displaced[:, 1] + xi_displaced[:, 2])
    jac[:, 1, 0] = 2*(1+xi_displaced[:, 0])/(xi_displaced[:, 1] + xi_displaced[:, 2])**2
    jac[:, 2, 0] = 2*(1+xi_displaced[:, 0])/(xi_displaced[:, 1] + xi_displaced[:, 2])**2  # same as above
    jac[:, 0, 1] = 0
    jac[:, 1, 1] = 2/(1-xi_displaced[:, 2])
    jac[:, 2, 1] = 2*(1+xi_displaced[:, 1])/(1-xi_displaced[:, 2])**2
    jac[:, 0, 2] = 0
    jac[:, 1, 2] = 0
    jac[:, 2, 2] = 1

    for idx, (m, n, l) in enumerate(poly_indices):
        Px = jacobi(m, 0, 0)
        Py = jacobi(n, 2*m+1, 0)*(poly([1, -1])/2)**m
        Pz = jacobi(l, 2*m+2*n+2, 0)*(poly([1, -1])/2)**(m+n)
        dPx = Px.deriv(1)
        dPy = Py.deriv(1)
        dPz = Pz.deriv(1)

        # IMPORTANT: The points used to calculate the shape function values include the corner (singularity) because the polynomials are defined there...
        Px_val = Px(eta[:, 0])
        Py_val = Py(eta[:, 1])
        Pz_val = Pz(eta[:, 2])

        # Normalization factor such that the function integrates to 1
        norm = (2*(2*m+1) *2*(m+n+1) * (m+n+l+2))**0.5
        # print(norm)
        # Why not add in a factor of 2^(4i+2j+6)?

        # However, the jacobian is singular at the corner, so we can't evaluate that point directly in our integral. Instead, we look at the point that is slightly displaced from the corner and evaluate the derivative of the function at that displaced point.
        # Note that since we have split up the function into muliple parts to differentiate via the product rule, we still need to evaluate both the derivative and re-evaluate the shape function at the displaced point. The only difference between eta and eta_displaced is that the eta coord of the corner is shifted down slightly.
        Px_val_disp = Px(eta_displaced[:, 0])
        Py_val_disp = Py(eta_displaced[:, 1])
        Pz_val_disp = Pz(eta_displaced[:, 2])

        dPx_val = dPx(eta_displaced[:, 0])
        dPy_val = dPy(eta_displaced[:, 1])
        dPz_val = dPz(eta_displaced[:, 2])

        df_deta1 = dPx_val*Py_val_disp*Pz_val_disp*norm
        df_deta2 = Px_val_disp*dPy_val*Pz_val_disp*norm
        df_deta3 = Px_val_disp*Py_val_disp*dPz_val*norm

        for pt in np.arange(numpts):
            [df_dxi1, df_dxi2, df_dxi3] = jac[pt,:,:] @ np.array([df_deta1[pt], df_deta2[pt], df_deta3[pt]])

            Vx[pt, idx] = df_dxi1
            Vy[pt, idx] = df_dxi2
            Vz[pt, idx] = df_dxi3

        V[:, idx] = norm*Px_val*Py_val*Pz_val

    Vx *= 2
    Vy *= 2
    Vz *= 2

    return V, Vx, Vy, Vz

if __name__ == '__main__':
    np.set_printoptions(suppress=True, linewidth=np.inf)
    # np.set_printoptions(linewidth=np.inf, precision=10)
    import sys
    sys.path.insert(0, '../mesh')
    sys.path.insert(0, '../util')
    from mkmshlocal import mkmshlocal
    from import_util import load_mat

    pl2d, _ = mkmshlocal(3)
    pl2d = pl2d[:,1:]
    loc1d_idx = np.squeeze(np.argwhere(pl2d[:, -1] == 0))
    pl1d = pl2d[loc1d_idx, :-1]

    # Testing 1D
    # V, Vx = koornwinder(pl1d[:,1:], 3)
    # print(V)
    # print(Vx)

    # Testing 2D
    # V, Vx, Vy = koornwinder(pl2d, 3)
    # print(V)
    # print()
    # print(Vx)
    # print()
    # print(Vy)

    # V_mat = load_mat('V')
    # Vx_mat = load_mat('Vx')
    # Vy_mat = load_mat('Vy')

    # print(np.allclose(V, V_mat))
    # print(np.allclose(Vx, Vx_mat))
    # print(np.allclose(Vy, Vy_mat))

    # Testing 3D