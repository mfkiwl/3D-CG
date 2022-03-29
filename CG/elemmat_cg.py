import numpy as np
import sys
sys.path.insert(0, '../util')
import logging

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

        # dgpts = load_mat('dg')
        # PHI_mat = load_mat('PHI')
        # DPHI_DXI_mat = load_mat('DPHI_DXI')
        # DPHI_DETA_mat = load_mat('DPHI_DETA')
        # X_GQ_mat = load_mat('x_GQ')
        # Y_GQ_mat = load_mat('y_GQ')
        # W_mat = load_mat('W_mat')
        # K_mat = load_mat('K')
        # S_mat = load_mat('S')
        # C1_mat = load_mat('C1')
        # C2_mat = load_mat('C2')
        # P_GQ_mat = load_mat('P_GQ')
        # F_GQ_mat = load_mat('F_GQ')

        # # fe_mat = load_mat('fe')
        # # ae_mat = load_mat('ae').transpose((2, 0, 1))

        # # np.allclose testing
        # rtol = 1e-13
        # atol = 4e-15

        # diff = np.abs(DPHI_DXI-DPHI_DXI_mat)
        # tol = np.abs(DPHI_DXI_mat*rtol) + atol
        # bool_arry = diff < tol
        # print(tol[-1,-4])
        # print(diff[-1,-4])
        # print(bool_arry[-1,-4])

        # print(bool_arry)
        # print()

        # print(np.isclose(DPHI_DXI, DPHI_DXI_mat, rtol, atol))

        # print('DGPTS:', np.allclose(ho_pts, dgpts, rtol, atol))
        # print('PHI:', np.allclose(PHI, PHI_mat, rtol, atol))
        # print('DPHI_DXI:', np.allclose(DPHI_DXI, DPHI_DXI_mat, rtol, atol))
        # print('DPHI_DETA:', np.allclose(DPHI_DETA, DPHI_DETA_mat, rtol, atol))
        # print('X_GQ:', np.allclose(x_GQ, X_GQ_mat, rtol, atol))
        # print('Y_GQ:', np.allclose(y_GQ, Y_GQ_mat, rtol, atol))
        # print('W:', np.allclose(W, W_mat, rtol, atol))
        # print('K:', np.allclose(K, K_mat, rtol, atol))
        # print('S:', np.allclose(S, S_mat, rtol, atol))
        # print('C1:', np.allclose(C1, C1_mat, rtol, atol))
        # print('C2:', np.allclose(C2, C2_mat, rtol, atol))
        # print('P_GQ:', np.allclose(P_GQ, P_GQ_mat, rtol, atol))
        # print('F_GQ:', np.allclose(F_GQ, F_GQ_mat, rtol, atol))

        # # print('fe:', np.allclose(fe, fe_mat))
        # # print('ae:', np.allclose(ae, ae_mat))

        # # diff = tmp_mat-ae
        # # for i, page in enumerate(diff):
        # #     print(i)
        # #     print(page)
        # #     print()

        # # print(np.allclose(tmp_mat, ae))
        # exit()


        J00 = DPHI_DXI.T@ho_pts[:,0]     # DX_DXI
        J01 = DPHI_DXI.T@ho_pts[:,1]     # DY_DXI
        J10 = DPHI_DETA.T@ho_pts[:,0]    # DX_DETA
        J11 = DPHI_DETA.T@ho_pts[:,1]    # DY_DETA

        J = np.zeros((n_gqpts, 2, 2))
        J[:, 0, 0] = J00
        J[:, 0, 1] = J01
        J[:, 1, 0] = J10
        J[:, 1, 1] = J11

        JAC_DET = np.zeros((n_gqpts, 1));           # Determinants of Jacobians stored as a matrix, diagonal)
        DPHI_Dx = np.zeros_like(PHI)        # holds dphi/dx vals, size (nplocal, ngpts)
        DPHI_Dy = np.zeros_like(PHI)        # holds dphi/dy vals, size (nplocal, ngpts)

        for i in np.arange(n_gqpts):
            JAC_DET[i] = np.linalg.det(J[i, :, :])
            J_inv = np.linalg.inv(J[i, :, :])            # Make this vectorized for efficiency

            tmp = J_inv@np.concatenate((DPHI_DXI[:, i][:,None], DPHI_DETA[:, i][:,None]), axis=1).T       # 1D indexed vectors need to be promoted to higher dimensional array (2) before concatenation
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

        J00 = DPHI_DXI.T@ho_pts[:, 0]     # DX_DXI
        J01 = DPHI_DXI.T@ho_pts[:, 1]     # DY_DXI
        J02 = DPHI_DXI.T@ho_pts[:, 2]     # DZ_DXI
        J10 = DPHI_DETA.T@ho_pts[:, 0]    # DX_DETA
        J11 = DPHI_DETA.T@ho_pts[:, 1]    # DY_DETA
        J12 = DPHI_DETA.T@ho_pts[:, 2]    # DZ_DETA
        J20 = DPHI_DGAMMA.T@ho_pts[:, 0]    # DX_DGAMMA
        J21 = DPHI_DGAMMA.T@ho_pts[:, 1]    # DY_DGAMMA
        J22 = DPHI_DGAMMA.T@ho_pts[:, 2]    # DZ_DGAMMA


        J = np.zeros((n_gqpts, 3, 3))
        J[:, 0, 0] = J00
        J[:, 0, 1] = J01
        J[:, 0, 2] = J02
        J[:, 1, 0] = J10
        J[:, 1, 1] = J11
        J[:, 1, 2] = J12
        J[:, 2, 0] = J20
        J[:, 2, 1] = J21
        J[:, 2, 2] = J22

        # Determinants of Jacobians stored as a matrix, diagonal)
        JAC_DET = np.zeros((n_gqpts, 1))
        # holds dphi/dx vals, size (nplocal, ngpts)
        DPHI_Dx = np.zeros_like(PHI)
        # holds dphi/dy vals, size (nplocal, ngpts)
        DPHI_Dy = np.zeros_like(PHI)
        # holds dphi/dZ vals, size (nplocal, ngpts)
        DPHI_Dz = np.zeros_like(PHI)

        for i in np.arange(n_gqpts):
            JAC_DET[i] = np.linalg.det(J[i, :, :])
            # Make this vectorized for efficiency
            J_inv = np.linalg.inv(J[i, :, :])

            # 1D indexed vectors need to be promoted to higher dimensional array (2) before concatenation
            tmp = J_inv@np.concatenate((DPHI_DXI[:, i][:, None],DPHI_DETA[:, i][:, None],DPHI_DGAMMA[:, i][:, None]), axis=1).T
            DPHI_Dx[:, i] = tmp[0, :]
            DPHI_Dy[:, i] = tmp[1, :]
            DPHI_Dz[:, i] = tmp[2, :]   # This could be a serious mistake.  

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