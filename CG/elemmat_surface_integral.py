import numpy as np
import sys
sys.path.insert(0, '../util')

def elemmat_surface_integral(ho_pts, master, g, ndim, idx=0):
    # if idx % 1000 == 0:
    #     print(idx)
    # ho_pts = dgnodes[idx,:,:]
    if ndim == 2:
        # # First, prepare data structures to build PHI, DX, and DY, DETJ, and other matrices:
        # n_gqpts = master['gptsvol'].shape[0]

        # # The following matrices are size (nplocal, npgauss), essentially restructuring the master.shap dataset
        # PHI = master['shapvol'][:, :, 0].T
        # DPHI_DXI = master['shapvol'][:, :, 1].T
        # DPHI_DETA = master['shapvol'][:, :, 2].T

        # x_GQ = PHI.T@ho_pts[:,0][:, None]
        # y_GQ = PHI.T@ho_pts[:, 1][:, None]

        # W = np.diag(master['gwvol']);                              # GQ weights in matrix form
        # K = param['kappa'] * np.diag(np.ones((n_gqpts,)));        # difusivity in matrix form kappa*I
        # S = param['s'] * np.diag(np.ones((n_gqpts,)));            # reaction rate s in matrix form s*I
        # C1 = param['c'][0] * np.diag(np.ones((n_gqpts,)));        # x-convective vel in matrix form c1*I
        # C2 = param['c'][1] * np.diag(np.ones((n_gqpts,)));        # y-convective vel in matrix form c2*I
        # P_GQ = np.concatenate((x_GQ, y_GQ), axis=1)
        # G_GQ = forcing(P_GQ)

        # J00 = DPHI_DXI.T@ho_pts[:,0]     # DX_DXI
        # J01 = DPHI_DXI.T@ho_pts[:,1]     # DY_DXI
        # J10 = DPHI_DETA.T@ho_pts[:,0]    # DX_DETA
        # J11 = DPHI_DETA.T@ho_pts[:,1]    # DY_DETA

        # J = np.zeros((n_gqpts, 2, 2))
        # J[:, 0, 0] = J00
        # J[:, 0, 1] = J01
        # J[:, 1, 0] = J10
        # J[:, 1, 1] = J11

        # JAC_DET = np.zeros((n_gqpts, 1));           # Determinants of Jacobians stored as a matrix, diagonal)
        # DPHI_Dx = np.zeros_like(PHI)        # holds dphi/dx vals, size (nplocal, ngpts)
        # DPHI_Dy = np.zeros_like(PHI)        # holds dphi/dy vals, size (nplocal, ngpts)

        # for i in np.arange(n_gqpts):
        #     JAC_DET[i] = np.linalg.det(J[i, :, :])
        #     J_inv = np.linalg.inv(J[i, :, :])            # Make this vectorized for efficiency

        #     tmp = J_inv@np.concatenate((DPHI_DXI[:, i][:,None], DPHI_DETA[:, i][:,None]), axis=1).T       # 1D indexed vectors need to be promoted to higher dimensional array (2) before concatenation
        #     DPHI_Dx[:, i] = tmp[0,:]
        #     DPHI_Dy[:, i] = tmp[1, :]

        # JAC_DET = np.diag(np.squeeze(JAC_DET))

        # """
        # DPHI_Dx is nplocal x ngpts (long)
        # DPHI_Dy is nplocal x ngpts (long)
        # K is ngpts x ngpts (square)
        # W is ngpts x ngpts (square)
        # JAC_DET is ngpts x ngpts (square)

        # laplacian needs to be a nplocal x nplocal (square) matrix
        # """

        # # Noted: input to np.diag has to be a 1D array

        # laplacian = DPHI_Dx@K@W@JAC_DET@DPHI_Dx.T + DPHI_Dy@K@W@JAC_DET@DPHI_Dy.T
        # conv = -(DPHI_Dx@C1@W@JAC_DET@PHI.T + DPHI_Dy@C2@W@JAC_DET@PHI.T)
        # source = PHI@S@W@JAC_DET@PHI.T

        # Ae = laplacian + conv + source

        # Fe = PHI@W@JAC_DET@G_GQ
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

        ###### THIS NEEDS TO BE FIXED SO THAT IT INTERPOLATES G AT THE GQ PTS!!!
        G_GQ = np.ones((n_gqpts, 1))*g    # For now, assume the neumann condition is uniform across the face. But be prepared to change that in the future.

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
            JAC_DET[i] = (1/2)*np.linalg.norm(np.cross(J[i, 0, :], J[i, 1, :]))*(1/3) # Magic number 1/3 to correct scaling issue seen - not quite sure why here - but maybe the (1/2) and (1/3) should be combined to make (1/6) under the mapping from a cube to a tet volume? Not quite sure here.

        JAC_DET = np.diag(JAC_DET)

        # This is basically the same as in the volume integral case, except that the jacobian determinants represent the transformation from a square in the x-y plane to an arbitrarily oriented square in R^3
        dF = PHI@W@JAC_DET@G_GQ     # This is the contribution to the total integral from this particular volume element

    return np.squeeze(dF)   # Returning as a 1D array