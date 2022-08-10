import numpy as np
import gradcalc2
from functools import partial
import multiprocessing as mp
from functools import partial
from pathos.multiprocessing import ProcessingPool as Pool

import sys
sys.path.insert(0, '../master')
sys.path.insert(0, '../util')
import shap
import logging
import math_helper_fcns

logger = logging.getLogger(__name__)

def elem_grad_calc(dgnodes, master, sol, idx):
    """
    Originally the sample points were taken at the DG nodes, but this was unreliable due to errors in the gradient on O(1e-9), likely due to the way the koornwinder polynomials are constructed.
    This happens because the gradients of the koornwinder polynomials don't exist on the boundaries
    
    """

    if idx % 10000 == 0:
        logging.info(str(idx) + '/'+str(dgnodes.shape[0]))
    ho_pts = dgnodes[idx,:,:]
    sol_vals = sol[:,idx][:,None]

    n_gqpts = master['gptsvol'].shape[0]

    # The following matrices are size (nplocal, npgauss), essentially restructuring the master.shap dataset
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

    J_inv,__ = math_helper_fcns.inv(J)

    for i in np.arange(n_gqpts):
        # Make this vectorized for efficiency
        tmp = J_inv[i,:,:]@np.array([DU_DXI[i], DU_DETA[i], DU_DGAMMA[i]])
        DU_Dx[i] = tmp[0]
        DU_Dy[i] = tmp[1]
        DU_Dz[i] = tmp[2]

    grad_gq = np.concatenate((DU_Dx, DU_Dy, DU_Dz), axis=1)

    grad_dg = master['phi_inv']@grad_gq
    return grad_dg

def calc_derivatives(mesh, master, sol, ndim):

    nelem = mesh['t'].shape[0]
    pool = Pool(mp.cpu_count())
    result = pool.map(partial(elem_grad_calc, mesh['dgnodes'], master, sol), np.arange(nelem))
    # result = np.asarray(list(map(partial(elem_grad_calc, mesh['dgnodes'], master, sol), np.arange(nelem))))
    grad = np.hstack(result)
    # mag = np.linalg.norm(grad, ord=2, axis=2)[:,:,None]

    return grad