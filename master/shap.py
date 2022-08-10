import numpy as np
from koornwinder import *

def shape1d(porder, plocal, pts):
    """
    Call koornwinder 1d to get the Vandermonde matrix of the 1D shape functions and their derivatives
    Get the weights of the koornwinder functions by solving VL=I -> we are using a nodal basis, so the value of the nodal basis functions at the nodes will be either 1 or 0
    With the weights for the basis functions known, we need to evaluate the polynomials at the points given to us in pts.
    This can be done by evaluating the koornwinder polynomials AGAIN at the specified points and then weighting the new vandermonde matix with the basis weights computed in the previous step. Do this for both the functions and their derivatives.
    """    

    """
    First, we need to generate the 1D shape functions on the domain. The
    shape functions will form a polynomial basis for the porder-dimensional
    space. We will take the plocal nodal points(plocal=porder+1) and
    construct lagrange polynomials that satisfy the dirac delta function
    delta_ij.

    To do this, we take in a set of orthogonal polynomials in the form of a
    Vandermonde matrix comprised of the Koorwinder(legendre polynomials)
    sampled at the nodal basis points.
    """
    plocal = plocal[:, None]    # Broadcast as a column vector
    pts = pts[:, None]

    V1d_nodal, _ = koornwinder1d(plocal, porder)

    """
    For each of the porder+1 shape functions, we need to find the proper
    weights so that the shape function can be generated from a linear
    combination of the orthogonal basis functions. Since the lagrange shape
    functions sampled at the nodes = delta_ij, the shape functions together
    form the identity matrix. Therefore, we have V*L = I, where L is the
    matrix of weights of the orthogonal basis polynomials. Then, L = inv(V).

    """

    L = np.linalg.inv(V1d_nodal)

    """
    With the weights in hand, our lagrange polynomials are defined, not just
    at the nodal points but everywhere. To interpolate the functions at
    points other than the nodal points(i.e. GQ points), simply regnerate the
    Vandermode matrix and re-multiply by the previously-computed L matrix.

    """

    V1d_sampled, V1d_sampled_der = koornwinder1d(pts, porder)
    # Shape functions run down the columns

    shape_fn_sampled = V1d_sampled @ L
    shape_fn_sampled_der = V1d_sampled_der @ L
    # Because the nGQ pts is greater than nplocal, these arrays will be "tall" - more samples at the GQ pts than there are shape functions.

    # Basis function array stores the value of each of the shape functions and their derivatives at each requested (GQ) point.

    # Reshape the "tall" array into a (num_pts, 1, nplocal) array. Then concatenate both arrays from the function evaluation and the derivative.
    shape_fn_reshaped = np.ravel(shape_fn_sampled).reshape((pts.shape[0], -1, 1))
    shape_fn_der_reshaped = np.ravel(shape_fn_sampled_der).reshape((pts.shape[0], -1, 1))
    shap = np.concatenate((shape_fn_reshaped, shape_fn_der_reshaped), axis=2)

    return shap


def shape2d(porder, plocal, pts):

    V2d_nodal, _, _ = koornwinder2d(plocal, porder)

    L = np.linalg.inv(V2d_nodal)

    V2, V2x, V2y = koornwinder2d(pts, porder)

    shape_fn_sampled = V2 @ L
    shape_fn_sampled_dx = V2x @ L
    shape_fn_sampled_dy = V2y @ L

    shape_fn_reshaped = np.ravel(shape_fn_sampled).reshape((pts.shape[0], -1, 1))
    shape_fn_dx_reshaped = np.ravel(shape_fn_sampled_dx).reshape((pts.shape[0], -1, 1))
    shape_fn_dy_reshaped = np.ravel(shape_fn_sampled_dy).reshape((pts.shape[0], -1, 1))

    shap = np.concatenate((shape_fn_reshaped, shape_fn_dx_reshaped, shape_fn_dy_reshaped), axis=2)

    return shap


def shape3d(porder, plocal, pts):
    V3d_nodal, _, _, _ = koornwinder3d(plocal, porder)  # These are the orthogonal basis functions
    np.set_printoptions(suppress=True, linewidth=np.inf, precision=2)
    L = np.linalg.inv(V3d_nodal)

    V, Vx, Vy, Vz = koornwinder3d(pts, porder)

    shape_fn_sampled = V @ L  # These are the nodal basis functions
    shape_fn_sampled_dx = Vx @ L
    shape_fn_sampled_dy = Vy @ L
    shape_fn_sampled_dz = Vz @ L

    shape_fn_reshaped = np.ravel(shape_fn_sampled).reshape((pts.shape[0], -1, 1))
    shape_fn_dx_reshaped = np.ravel(shape_fn_sampled_dx).reshape((pts.shape[0], -1, 1))
    shape_fn_dy_reshaped = np.ravel(shape_fn_sampled_dy).reshape((pts.shape[0], -1, 1))
    shape_fn_dz_reshaped = np.ravel(shape_fn_sampled_dz).reshape((pts.shape[0], -1, 1))

    shap = np.concatenate((shape_fn_reshaped, shape_fn_dx_reshaped, shape_fn_dy_reshaped, shape_fn_dz_reshaped), axis=2)

    return shap



if __name__ == '__main__':
    import sys
    sys.path.insert(0, '../mesh')
    from mkmesh_sqmesh_gmsh import mkmesh_square
    from gaussquad1d import gaussquad1d
    
    porder = 3
    mesh = mkmesh_square(porder)
    master = mkmaster(mesh, 2*porder)

    
    mkmaster(mesh)

    porder = 3
    pgauss = 4*porder   
    master['gp1d'] = gaussquad1d(pgauss)

    sh1d = shape1d(master['porder'], master['ploc1d'], master['gp1d'])

