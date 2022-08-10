import numpy as np
import sys
sys.path.insert(0, '../util')
import logging
import math_helper_fcns

logger = logging.getLogger(__name__)

def elemmat_cg(dgnodes, master, forcing, param, ndim, idx):
    if idx % 10000 == 0:
        logging.info(str(idx) + '/'+str(dgnodes.shape[0]))
    ho_pts = dgnodes[idx,:,:]
    if ndim == 2:
        # First, prepare data structures to build PHI, DX, and DY, DETJ, and other matrices:
        n_gqpts = master['gptsvol'].shape[0]

        # The following matrices are size (nplocal, npgauss), essentially restructuring the master.shap dataset
        PHI = master['shapvol'][:, :, 0].T
        DPHI_DXI = master['shapvol'][:, :, 1].T
        DPHI_DETA = master['shapvol'][:, :, 2].T

        x_GQ = PHI.T@ho_pts[:,0][:, None]
        y_GQ = PHI.T@ho_pts[:, 1][:, None]

        W = np.diag(master['gwvol']);                              # GQ weights in matrix form
        K = param['kappa'] * np.diag(np.ones((n_gqpts,)));        # difusivity in matrix form kappa*I
        S = param['s'] * np.diag(np.ones((n_gqpts,)));            # reaction rate s in matrix form s*I
        C1 = param['c'][0] * np.diag(np.ones((n_gqpts,)));        # x-convective vel in matrix form c1*I
        C2 = param['c'][1] * np.diag(np.ones((n_gqpts,)));        # y-convective vel in matrix form c2*I
        P_GQ = np.concatenate((x_GQ, y_GQ), axis=1)
        F_GQ = forcing(P_GQ)

        J = np.zeros((n_gqpts, 2, 2))
        J[:, 0, 0] = DPHI_DXI.T@ho_pts[:,0]     # DX_DXI
        J[:, 0, 1] = DPHI_DXI.T@ho_pts[:,1]     # DY_DXI
        J[:, 1, 0] = DPHI_DETA.T@ho_pts[:,0]    # DX_DETA
        J[:, 1, 1] = DPHI_DETA.T@ho_pts[:,1]    # DY_DETA

        DPHI_Dx = np.zeros_like(PHI)        # holds dphi/dx vals, size (nplocal, ngpts)
        DPHI_Dy = np.zeros_like(PHI)        # holds dphi/dy vals, size (nplocal, ngpts)

        J_inv, JAC_DET = math_helper_fcns.inv(J)

        for i in np.arange(n_gqpts):
            tmp = J_inv[i,:,:]@np.concatenate((DPHI_DXI[:, i][:,None], DPHI_DETA[:, i][:,None]), axis=1).T       # 1D indexed vectors need to be promoted to higher dimensional array (2) before concatenation
            DPHI_Dx[:, i] = tmp[0,:]
            DPHI_Dy[:, i] = tmp[1, :]

        JAC_DET = np.diag(np.squeeze(JAC_DET))

        """
        DPHI_Dx is nplocal x ngpts (long)
        DPHI_Dy is nplocal x ngpts (long)
        K is ngpts x ngpts (square)
        W is ngpts x ngpts (square)
        JAC_DET is ngpts x ngpts (square)

        laplacian needs to be a nplocal x nplocal (square) matrix
        """

        # Noted: input to np.diag has to be a 1D array

        laplacian = DPHI_Dx@K@W@JAC_DET@DPHI_Dx.T + DPHI_Dy@K@W@JAC_DET@DPHI_Dy.T
        conv = -(DPHI_Dx@C1@W@JAC_DET@PHI.T + DPHI_Dy@C2@W@JAC_DET@PHI.T)
        source = PHI@S@W@JAC_DET@PHI.T

        Ae = laplacian + conv + source

        Fe = PHI@W@JAC_DET@F_GQ

    elif ndim == 3:
        # First, prepare data structures to build PHI, DX, and DY, DETJ, and other matrices:
        n_gqpts = master['gptsvol'].shape[0]

        # The following matrices are size (nplocal, npgauss), essentially restructuring the master.shap dataset
        PHI = master['shapvol'][:, :, 0].T
        DPHI_DXI = master['shapvol'][:, :, 1].T
        DPHI_DETA = master['shapvol'][:, :, 2].T
        DPHI_DGAMMA = master['shapvol'][:, :, 3].T

        x_GQ = PHI.T@ho_pts[:, 0][:, None]
        y_GQ = PHI.T@ho_pts[:, 1][:, None]
        z_GQ = PHI.T@ho_pts[:, 2][:, None]

        # GQ weights in matrix form
        W = np.diag(master['gwvol'])
        # difusivity in matrix form kappa*I
        K = param['kappa'] * np.diag(np.ones((n_gqpts,)))

        P_GQ = np.concatenate((x_GQ, y_GQ, z_GQ), axis=1)
        F_GQ = forcing(P_GQ)

        GRAD_XI = DPHI_DXI.T@ho_pts   # (DX_DXI, DY_DXI, DZ_DXI)
        GRAD_ETA = DPHI_DETA.T@ho_pts  # (DX_DETA, DY_DETA, DZ_DETA)
        GRAD_GAMMA = DPHI_DGAMMA.T@ho_pts    # (DX_DGAMMA, DY_DGAMMA, DZ_DGAMMA)

        J = np.zeros((n_gqpts, 3, 3))
        J[:, 0, :] = GRAD_XI
        J[:, 1, :] = GRAD_ETA
        J[:, 2, :] = GRAD_GAMMA

        # holds dphi/dx vals, size (nplocal, ngpts)
        DPHI_Dx = np.zeros_like(PHI)
        # holds dphi/dy vals, size (nplocal, ngpts)
        DPHI_Dy = np.zeros_like(PHI)
        # holds dphi/dZ vals, size (nplocal, ngpts)
        DPHI_Dz = np.zeros_like(PHI)

        J_inv, JAC_DET = math_helper_fcns.inv(J)

        for i in np.arange(n_gqpts):
            tmp = J_inv[i,:,:]@np.concatenate((DPHI_DXI[:, i][:, None],DPHI_DETA[:, i][:, None],DPHI_DGAMMA[:, i][:, None]), axis=1).T

            DPHI_Dx[:, i] = tmp[0, :]
            DPHI_Dy[:, i] = tmp[1, :]
            DPHI_Dz[:, i] = tmp[2, :]

        JAC_DET = np.diag(np.squeeze(JAC_DET))

        """
        DPHI_Dx is nplocal x ngpts (long)
        DPHI_Dy is nplocal x ngpts (long)
        DPHI_Dz is nplocal x ngpts (long)
        K is ngpts x ngpts (square)
        W is ngpts x ngpts (square)
        JAC_DET is ngpts x ngpts (square)

        laplacian needs to be a nplocal x nplocal (square) matrix
        """

        # Noted: input to np.diag has to be a 1D array
        laplacian = DPHI_Dx@K@W@JAC_DET@DPHI_Dx.T + DPHI_Dy@K@W@JAC_DET@DPHI_Dy.T + DPHI_Dz@K@W@JAC_DET@DPHI_Dz.T

        if not (param['c'][0] and param['c'][1] and param['c'][2]) and not param['s']:
            Ae = laplacian
        else:
            # reaction rate s in matrix form s*I
            S = param['s'] * np.diag(np.ones((n_gqpts,)))
            # x-convective vel in matrix form c1*I
            C1 = param['c'][0] * np.diag(np.ones((n_gqpts,)))
            # y-convective vel in matrix form c2*I
            C2 = param['c'][1] * np.diag(np.ones((n_gqpts,)))
            # z-convective vel in matrix form c3*I
            C3 = param['c'][2] * np.diag(np.ones((n_gqpts,)))

            conv = -(DPHI_Dx@C1@W@JAC_DET@PHI.T + DPHI_Dy@C2@W@JAC_DET@PHI.T + DPHI_Dz@C3@W@JAC_DET@PHI.T)
            source = PHI@S@W@JAC_DET@PHI.T

            Ae = laplacian + conv + source

        Fe = PHI@W@JAC_DET@F_GQ

    return np.concatenate((Ae, Fe), axis=1)