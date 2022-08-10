import numpy as np
import logging
logger = logging.getLogger(__name__)
# Finding the sim root directory
import sys
from pathlib import Path
cwd = Path.cwd()
for dirname in tuple(cwd.parents):
    if dirname.name == '3D-CG':
        sim_root_dir = dirname
        continue

sys.path.append(str(sim_root_dir.joinpath('util')))
from math_helper_fcns import inv

def face_surface_integral(ho_pts, master, field, ndim):
    if ndim == 2:
        raise NotImplementedError

    elif ndim == 3:
        # First, prepare data structures to build PHI, DX, and DY, DETJ, and other matrices:
        n_gqpts = master['gptsface'].shape[0]

        # The following matrices are size (nplocal, npgauss), essentially restructuring the master.shap dataset
        PHI = master['shapface'][:, :, 0].T
        DPHI_DXI = master['shapface'][:, :, 1].T
        DPHI_DETA = master['shapface'][:, :, 2].T

        # GQ weights in matrix form
        W = np.diag(master['gwface'])

        # Why transpose again? And check the dimensions of g
        G_GQ = PHI.T@field    # For now, assume the neumann condition is uniform across the face. But be prepared to change that in the future.

        J = np.zeros((n_gqpts, 2, 3))
        J[:, 0, 0] = DPHI_DXI.T@ho_pts[:, 0]     # DX_DXI
        J[:, 0, 1] = DPHI_DXI.T@ho_pts[:, 1]     # DY_DXI
        J[:, 0, 2] = DPHI_DXI.T@ho_pts[:, 2]     # DZ_DXI
        J[:, 1, 0] = DPHI_DETA.T@ho_pts[:, 0]    # DX_DETA
        J[:, 1, 1] = DPHI_DETA.T@ho_pts[:, 1]    # DY_DETA
        J[:, 1, 2] = DPHI_DETA.T@ho_pts[:, 2]    # DZ_DETA

        # Determinants of Jacobians stored as a matrix, diagonal)
        JAC_DET = np.zeros((n_gqpts))

        for i in np.arange(n_gqpts):
            JAC_DET[i] = np.linalg.norm(np.cross(J[i, 0, :], J[i, 1, :])) # Magic number 1/3 to correct scaling issue seen - not quite sure why here - but maybe the (1/2) and (1/3) should be combined to make (1/6) under the mapping from a cube to a tet volume? Not quite sure here.

        JAC_DET = np.diag(JAC_DET)

        # This is basically the same as in the volume integral case, except that the jacobian determinants represent the transformation from a square in the x-y plane to an arbitrarily oriented square in R^3
        dF = PHI@W@JAC_DET@G_GQ     # This is the contribution to the total integral from this particular volume element

    # return np.sum(dF)   # Returning as a scalar
    return dF   # Returning as a scalar

def elem_volume_integral(ho_pts, master, field, ndim):
    if ndim == 2:
        raise NotImplementedError

    elif ndim == 3:
        # First, prepare data structures to build PHI, DX, and DY, DETJ, and other matrices:
        n_gqpts = master['gptsvol'].shape[0]

        # The following matrices are size (nplocal, npgauss), essentially restructuring the master.shap dataset
        PHI = master['shapvol'][:, :, 0].T
        DPHI_DXI = master['shapvol'][:, :, 1].T
        DPHI_DETA = master['shapvol'][:, :, 2].T
        DPHI_DGAMMA = master['shapvol'][:, :, 3].T

        # GQ weights in matrix form
        W = np.diag(master['gwvol'])

        # Why transpose again? And check the dimensions of g
        G_GQ = PHI.T@field    # For now, assume the neumann condition is uniform across the face. But be prepared to change that in the future.

        J = np.zeros((n_gqpts, 3, 3))
        J[:, 0, 0] = DPHI_DXI.T@ho_pts[:, 0]     # DX_DXI
        J[:, 0, 1] = DPHI_DXI.T@ho_pts[:, 1]     # DY_DXI
        J[:, 0, 2] = DPHI_DXI.T@ho_pts[:, 2]     # DZ_DXI
        J[:, 1, 0] = DPHI_DETA.T@ho_pts[:, 0]    # DX_DETA
        J[:, 1, 1] = DPHI_DETA.T@ho_pts[:, 1]    # DY_DETA
        J[:, 1, 2] = DPHI_DETA.T@ho_pts[:, 2]    # DZ_DETA
        J[:, 2, 0] = DPHI_DGAMMA.T@ho_pts[:, 0]    # DX_DGAMMA
        J[:, 2, 1] = DPHI_DGAMMA.T@ho_pts[:, 1]    # DY_DGAMMA
        J[:, 2, 2] = DPHI_DGAMMA.T@ho_pts[:, 2]    # DZ_DGAMMA

        # Determinants of Jacobians stored as a matrix, diagonal)
        # __, JAC_DET = inv(J)

        JAC_DET = np.zeros((n_gqpts, 1));           # Determinants of Jacobians stored as a matrix, diagonal)

        for i in np.arange(n_gqpts):
            JAC_DET[i] = np.linalg.det(J[i, :, :])

        JAC_DET = np.diag(np.squeeze(JAC_DET))

        # JAC_DET = np.diag(JAC_DET)

        # This is basically the same as in the volume integral case, except that the jacobian determinants represent the transformation from a square in the x-y plane to an arbitrarily oriented square in R^3
        dF = PHI@W@JAC_DET@G_GQ     # This is the contribution to the total integral from this particular volume element

    return np.sum(dF)   # Returning as a scalar