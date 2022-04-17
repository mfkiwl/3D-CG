import numpy as np
from functools import partial
import multiprocessing as mp
from functools import partial
from pathos.multiprocessing import ProcessingPool as Pool
from scipy.sparse import lil_matrix, save_npz, load_npz
import logging
import time
import solvers

import sys
sys.path.insert(0, '../master')
sys.path.insert(0, '../util')
import logging
import math_helper_fcns
import helper

logger = logging.getLogger(__name__)

def elem_grad_calc(dgnodes, master, sol, ndim, idx):
    """
    Originally the sample points were taken at the DG nodes, but this was unreliable due to errors in the gradient on O(1e-9), likely due to the way the koornwinder polynomials are constructed.
    This happens because the gradients of the koornwinder polynomials don't exist on the boundaries
    
    """

    if idx % 10000 == 0:
        logging.info(str(idx) + '/'+str(dgnodes.shape[0]))
    ho_pts = dgnodes[idx,:,:]
    sol_vals = sol[:,idx][:,None]

    n_gqpts = master['gptsvol'].shape[0]
    nplocal = master['shapvol'].shape[1]

    # The following matrices are size (nplocal, npgauss), essentially restructuring the master.shap dataset
    PHI = master['shapvol'][:, :, 0]
    DPHI_DXI = master['shapvol'][:, :, 1]       # These are "tall" matrices, (ngaussquad x nplocal), due to np C ordering
    DPHI_DETA = master['shapvol'][:, :, 2]
    DPHI_DGAMMA = master['shapvol'][:, :, 3]

    # du/dxi at GQ pts
    DU_DXI = DPHI_DXI@sol_vals
    DU_DETA = DPHI_DETA@sol_vals
    DU_DGAMMA = DPHI_DGAMMA@sol_vals

    GRAD_XI = DPHI_DXI@ho_pts   # (DX_DXI, DY_DXI, DZ_DXI)
    GRAD_ETA = DPHI_DETA@ho_pts  # (DX_DETA, DY_DETA, DZ_DETA)
    GRAD_GAMMA = DPHI_DGAMMA@ho_pts    # (DX_DGAMMA, DY_DGAMMA, DZ_DGAMMA)

    J = np.zeros((n_gqpts, 3, 3))
    J[:, 0, :] = GRAD_XI
    J[:, 1, :] = GRAD_ETA
    J[:, 2, :] = GRAD_GAMMA

    # holds du/dx vals, size (nplocal, 1)
    DU_Dx = np.zeros_like(DU_DXI)
    # holds du/dy vals, size (nplocal, 1)
    DU_Dy = np.zeros_like(DU_DETA)
    # holds du/dZ vals, size (nplocal, 1)
    DU_Dz = np.zeros_like(DU_DGAMMA)

    # GQ weights in matrix form
    W = np.diag(master['gwvol'])

    J_inv, JAC_DET = math_helper_fcns.inv(J)

    JAC_DET = np.diag(np.squeeze(JAC_DET))

    for i in np.arange(n_gqpts):
        # Make this vectorized for efficiency
        tmp = J_inv[i,:,:]@np.array([DU_DXI[i], DU_DETA[i], DU_DGAMMA[i]])
        DU_Dx[i] = tmp[0]
        DU_Dy[i] = tmp[1]
        DU_Dz[i] = tmp[2]

    grad_gq = np.concatenate((DU_Dx, DU_Dy, DU_Dz), axis=1)
    PINV =np.linalg.pinv(PHI) 
    grad_nodal = PINV@grad_gq

    grad_a = PHI.T@W@JAC_DET@PHI    # Transpose is due to C ordering during slicing array, see above

    grad_e = np.zeros((nplocal, ndim))

    for i in np.arange(3):
        grad_e[:,i] = PHI.T@W@JAC_DET@PHI@grad_nodal[:,i]

    # Returns (nplocal x (nplocal + ndim)) array, with the three RHS vectors present
    return np.concatenate((grad_a, grad_e), axis=1)

def calc_gradient(mesh, master, sol, ndim, solver, solver_tol):

    # Reshape into DG high order data structure
    sol_dg = helper.reshape_field(mesh, sol, 'to_array', 'scalars')

    nplocal = mesh['plocal'].shape[0]
    nelem = mesh['t'].shape[0]

    ae = np.zeros((nelem, nplocal, nplocal))
    fe  = np.zeros((nplocal, nelem))

    logger.info('Populating elemental matrices...')
    pool = Pool(mp.cpu_count())
    result = pool.map(partial(elem_grad_calc, mesh['dgnodes'], master, sol_dg, ndim), np.arange(nelem))
    ae_fe = np.asarray(result)
    # ae_fe = np.asarray(list(map(partial(elem_grad_calc, mesh['dgnodes'], master, sol, ndim), np.arange(nelem))))
    ae = ae_fe[:, :, :-ndim]    # There are now ndim RHS vectors in the arrays
    fe = ae_fe[:, :, -ndim:]    # Now a full 3D array

    nnodes = mesh['pcg'].shape[0]
    A= lil_matrix((nnodes, nnodes))
    F_x = np.zeros((nnodes, 1))
    F_y = np.zeros((nnodes, 1))
    F_z = np.zeros((nnodes, 1))

    logger.info('Loading matrix...')
    start = time.perf_counter()
    for i, elem in enumerate(mesh['tcg']):
        if i %10000 == 0:
            logger.info(str(i)+'/'+str(ae.shape[0]))
        A[elem[:, None], elem] += ae[i, :, :]
        F_x[elem, 0] += fe[i, :, 0]
        F_y[elem, 0] += fe[i, :, 1]
        F_z[elem, 0] += fe[i, :, 2]

    logger.info('Loading matrix took '+ str(time.perf_counter()-start)+' s')

    logger.info('Solving with '+ solver)

    du_dx_pcg = solvers.solve(A, F_x, solver_tol, solver)
    du_dy_pcg = solvers.solve(A, F_y, solver_tol, solver)
    du_dz_pcg = solvers.solve(A, F_z, solver_tol, solver)

    grad = np.concatenate((du_dx_pcg, du_dy_pcg, du_dz_pcg), axis=1)
    mag = np.linalg.norm(grad, ord=2, axis=1)[:,None]

    return grad, mag