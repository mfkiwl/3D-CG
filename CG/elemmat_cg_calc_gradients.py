import numpy as np
import sys
sys.path.insert(0, '../util')
sys.path.insert(0, '../master')
from shap import shape3d
from import_util import load_mat

def calc_grad(dgnodes, sol, master, ndim, idx):

    if idx % 1000 == 0:
        print(idx)
    dg_pts = dgnodes[idx,:,:]
    sol_pts = sol[:,idx]
    master['shapvol_nodal'] = shape3d(master['porder'], master['plocvol'], master['plocvol'])

    if ndim == 3:
        npts = master['plocvol'].shape[0]  # Changing this to account for the DG nodes, not the GQ pts

        # The following matrices are size (nplocal, npgauss), essentially SAMPLING the master.shap dataset
        # PHI = master['shapvol_nodal'][:, :, 0].T        # This should be the identity matrix
        DPHI_DXI = master['shapvol_nodal'][:, :, 1].T     
        DPHI_DETA = master['shapvol_nodal'][:, :, 2].T
        DPHI_DGAMMA = master['shapvol_nodal'][:, :, 3].T

        # If we had done PHI here we could have interpolated the numerical solution but instead we chose to interpolate the derivatives
        DU_DXI = DPHI_DXI.T@sol_pts[:, None]
        DU_DETA = DPHI_DETA.T@sol_pts[:, None]
        DU_DGAMMA = DPHI_DGAMMA.T@sol_pts[:, None]
        # x_GQ = PHI.T@dg_pts[:, 0][:, None]
        # y_GQ = PHI.T@dg_pts[:, 1][:, None]
        # z_GQ = PHI.T@dg_pts[:, 2][:, None]

        J00 = DPHI_DXI.T@dg_pts[:, 0]     # DX_DXI
        J01 = DPHI_DXI.T@dg_pts[:, 1]     # DY_DXI
        J02 = DPHI_DXI.T@dg_pts[:, 2]     # DZ_DXI
        J10 = DPHI_DETA.T@dg_pts[:, 0]    # DX_DETA
        J11 = DPHI_DETA.T@dg_pts[:, 1]    # DY_DETA
        J12 = DPHI_DETA.T@dg_pts[:, 2]    # DZ_DETA
        J20 = DPHI_DGAMMA.T@dg_pts[:, 0]    # DX_DGAMMA
        J21 = DPHI_DGAMMA.T@dg_pts[:, 1]    # DY_DGAMMA
        J22 = DPHI_DGAMMA.T@dg_pts[:, 2]    # DZ_DGAMMA

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

    # return np.concatenate((np.ones_like(DU_Dz), 2*np.ones_like(DU_Dz), 3*np.ones_like(DU_Dz)), axis=1)
    return np.concatenate((DU_Dx, DU_Dy, DU_Dz), axis=1)