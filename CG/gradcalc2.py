import numpy as np
import sys
sys.path.insert(0, '../master')
sys.path.insert(0, '../util')
from shap import shape3d
import logging
from math_helper_fcns import inv


logger = logging.getLogger(__name__)

def grad_calc(dgnodes, master, sol, idx):
    if idx % 10000 == 0:
        logging.info(str(idx) + '/'+str(dgnodes.shape[0]))
    ho_pts = dgnodes[idx,:,:]
    sol_pts = sol[:,idx]

    # First, prepare data structures to build PHI, DX, and DY, DETJ, and other matrices:
    npts = master['plocvol'].shape[0]  # Changing this to account for the DG nodes, not the GQ pts

    # The following matrices are size (nplocal, npgauss), essentially restructuring the master.shap dataset
    # Note: the tranposes are undoing each other here. This is simply to match the notation in the 930 handouts.
    DPHI_DXI = master['shapvol_nodal'][:, :, 1].T
    DPHI_DETA = master['shapvol_nodal'][:, :, 2].T
    DPHI_DGAMMA = master['shapvol_nodal'][:, :, 3].T

    # These lines are the equivalent of x_GQ, y_GQ, etc for elemmat_cg. They are the weights of the desired function (x, y, z, sol, etc) multiplied by the values of the basis functions.
    DU_DXI = DPHI_DXI.T@sol_pts[:,None]     # DU_DXI
    DU_DETA = DPHI_DETA.T@sol_pts[:, None]    # DU_DETA
    DU_DGAMMA = DPHI_DGAMMA.T@sol_pts[:, None]    # DU_DGAMMA

    J = np.zeros((npts, 3, 3))
    J[:, 0, 0] = DPHI_DXI.T@ho_pts[:, 0]     # DX_DXI
    J[:, 0, 1] = DPHI_DXI.T@ho_pts[:, 1]     # DY_DXI
    J[:, 0, 2] = DPHI_DXI.T@ho_pts[:, 2]     # DZ_DXI
    J[:, 1, 0] = DPHI_DETA.T@ho_pts[:, 0]    # DX_DETA
    J[:, 1, 1] = DPHI_DETA.T@ho_pts[:, 1]    # DY_DETA
    J[:, 1, 2] = DPHI_DETA.T@ho_pts[:, 2]    # DZ_DETA
    J[:, 2, 0] = DPHI_DGAMMA.T@ho_pts[:, 0]    # DX_DGAMMA
    J[:, 2, 1] = DPHI_DGAMMA.T@ho_pts[:, 1]    # DY_DGAMMA
    J[:, 2, 2] = DPHI_DGAMMA.T@ho_pts[:, 2]    # DZ_DGAMMA

    # holds du/dx vals, size (nplocal, 1)
    DU_Dx = np.zeros_like(DU_DXI)
    # holds du/dy vals, size (nplocal, 1)
    DU_Dy = np.zeros_like(DU_DETA)
    # holds du/dZ vals, size (nplocal, 1)
    DU_Dz = np.zeros_like(DU_DGAMMA)

    J_inv,__ = inv(J)

    for i in np.arange(npts):
        # Make this vectorized for efficiency
        tmp = J_inv[i,:,:]@np.array([DU_DXI[i], DU_DETA[i], DU_DGAMMA[i]])
        DU_Dx[i] = tmp[0]
        DU_Dy[i] = tmp[1]
        DU_Dz[i] = tmp[2]

    return np.concatenate((DU_Dx, DU_Dy, DU_Dz), axis=1)
