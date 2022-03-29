import numpy as np
import sys
sys.path.insert(0, '../master')
from shap import shape3d
import logging

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


    J00 = DPHI_DXI.T@ho_pts[:, 0]     # DX_DXI
    J01 = DPHI_DXI.T@ho_pts[:, 1]     # DY_DXI
    J02 = DPHI_DXI.T@ho_pts[:, 2]     # DZ_DXI
    J10 = DPHI_DETA.T@ho_pts[:, 0]    # DX_DETA
    J11 = DPHI_DETA.T@ho_pts[:, 1]    # DY_DETA
    J12 = DPHI_DETA.T@ho_pts[:, 2]    # DZ_DETA
    J20 = DPHI_DGAMMA.T@ho_pts[:, 0]    # DX_DGAMMA
    J21 = DPHI_DGAMMA.T@ho_pts[:, 1]    # DY_DGAMMA
    J22 = DPHI_DGAMMA.T@ho_pts[:, 2]    # DZ_DGAMMA


    J = np.zeros((npts, 3, 3))
    J[:, 0, 0] = J00
    J[:, 0, 1] = J01
    J[:, 0, 2] = J02
    J[:, 1, 0] = J10
    J[:, 1, 1] = J11
    J[:, 1, 2] = J12
    J[:, 2, 0] = J20
    J[:, 2, 1] = J21
    J[:, 2, 2] = J22

    # holds du/dx vals, size (nplocal, 1)
    DU_Dx = np.zeros_like(DU_DXI)
    # holds du/dy vals, size (nplocal, 1)
    DU_Dy = np.zeros_like(DU_DETA)
    # holds du/dZ vals, size (nplocal, 1)
    DU_Dz = np.zeros_like(DU_DGAMMA)


    for i in np.arange(npts):
        # Make this vectorized for efficiency
        J_inv = np.linalg.inv(J[i, :, :])

        tmp = J_inv@np.array([DU_DXI[i], DU_DETA[i], DU_DGAMMA[i]])
        DU_Dx[i] = tmp[0]
        DU_Dy[i] = tmp[1]
        DU_Dz[i] = tmp[2]

    return np.concatenate((DU_Dx, DU_Dy, DU_Dz), axis=1)
