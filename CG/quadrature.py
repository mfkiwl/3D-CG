from inspect import CO_ASYNC_GENERATOR
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
import multiprocessing as mp
from functools import partial
import shap

def elem_surface_integral(ho_pts, master, field, ndim, returnType='scalar'):
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
            JAC_DET[i] = np.linalg.norm(np.cross(J[i, 0, :], J[i, 1, :]))

        JAC_DET = np.diag(JAC_DET)

        # This is basically the same as in the volume integral case, except that the jacobian determinants represent the transformation from a square in the x-y plane to an arbitrarily oriented square in R^3
        dF = PHI@W@JAC_DET@G_GQ     # This is the contribution to the total integral from this particular volume element

    if returnType == 'scalar':
        return np.sum(dF)   # Returning as a scalar
    elif returnType == 'vector':
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
        __, JAC_DET = inv(J)

        JAC_DET = np.diag(JAC_DET)

        # This is basically the same as in the volume integral case, except that the jacobian determinants represent the transformation from a square in the x-y plane to an arbitrarily oriented square in R^3
        dF = PHI@W@JAC_DET@G_GQ     # This is the contribution to the total integral from this particular volume element

    return np.sum(dF)   # Returning as a scalar

def surface_integral_serial(mesh, master, scalar_field, faces, nnodes_per_face):
   
    #NOTE: Eventually delete the nnodes_per_face param when the sims are re-run with the mesh supporting that field
    # Loop over each element like in the cgsolve BC assignment and perform the numerical integration 
    integral_qty = 0
    
    for face in faces:
        # print(face)
        facenum = face[0]      # t2f uses 1-indexing for the faces
        bdry_elem = face[nnodes_per_face+1] # Adding 1 because we put the global face index in the first column and moved the rest over

        loc_face_idx = np.where(mesh['t2f'][bdry_elem, :] == facenum)[0][0]
        # Pull the local nodes on that face from permface - we don't want to include all the nodes in the element
        loc_face_nodes = master['perm'][:, loc_face_idx]

        # Use the perm indices to grab the face nodes from tcg
        face_nodes = mesh['tcg'][bdry_elem][loc_face_nodes]

        field_vals = scalar_field[face_nodes]

        # Also try pts_on_face = mesh['dgnodes'][bdry_elem, loc_face_nodes, :] instead of mesh['pcg'][:,face_nodes], but they should be the same
        integral_qty += elem_surface_integral(mesh['pcg'][face_nodes,:], master, field_vals, mesh['ndim'], 'scalar')

    return integral_qty

def surface_integral_parallel(mesh, master, scalar_field, faces, nnodes_per_face):
   
    #NOTE: Eventually delete the nnodes_per_face param when the sims are re-run with the mesh supporting that field
    # Loop over each element like in the cgsolve BC assignment and perform the numerical integration 
    
    with mp.Pool(mp.cpu_count()) as pool:
        result = pool.map(partial(get_surface_dQ, mesh, master, scalar_field, nnodes_per_face, faces), np.arange(faces.shape[0]))
    return sum(result)

def get_surface_dQ(mesh, master, scalar_field, nnodes_per_face, faces, face_idx):
    face = faces[face_idx]
    facenum = face[0]      # t2f uses 1-indexing for the faces
    bdry_elem = face[nnodes_per_face+1] # Adding 1 because we put the global face index in the first column and moved the rest over

    # Collect nodes of faces on boundary ONLY
    # Find the index that the face is in in t2f
    loc_face_idx = np.where(mesh['t2f'][bdry_elem, :] == facenum)[0][0]
    # Pull the local nodes on that face from permface - we don't want to include all the nodes in the element
    loc_face_nodes = master['perm'][:, loc_face_idx]

    # Use the perm indices to grab the face nodes from tcg
    face_nodes = mesh['tcg'][bdry_elem][loc_face_nodes]

    field_vals = scalar_field[face_nodes]

    # Also try pts_on_face = mesh['dgnodes'][bdry_elem, loc_face_nodes, :] instead of mesh['pcg'][:,face_nodes], but they should be the same
    dQ = elem_surface_integral(mesh['pcg'][face_nodes,:], master, field_vals, mesh['ndim'], 'scalar')
    
    return dQ

def volume_integral(mesh, master, scalar_field, elements):
    """
    elements is a list of all the elements that are in the domain of integration

    """

    integral_qty = 0

    for element in elements:
        dgpts = mesh['dgnodes'][element,:,:]
        field_vals = scalar_field[mesh['tcg'][element]]

        integral_qty += elem_volume_integral(dgpts, master, field_vals, mesh['ndim'])

    return integral_qty

def get_elem_face_normals(dgnodes, master, ndim, face_idx):

    """
    dgnodes are the high order nodes on the face on which to calculate the normals

    loc_pts can be either the local nodes or GQ pts
    """

    if ndim == 2:
        raise NotImplementedError('2D not implemented yet')

    elif ndim == 3:
        # First, prepare data structures to build PHI, DX, and DY, DETJ, and other matrices:
        face_nodes_loc_idx = master['perm'][:, face_idx]

        # local_normal_pts could be any local points, but most likely the nodal or GQ pts. If they are at the GQ pts, this is the same as master['shapv'], but this can't be assumed so we need to remake it.

        # Refer to masternodes for the way the faces are indexed - this has to match up with the way the faces are indexed in mesh.f
        if face_idx == 0:   
            GRAD_XI = (master['shapvol_nodal'][:, :, 1]@dgnodes)[face_nodes_loc_idx,:]
            GRAD_ETA = (master['shapvol_nodal'][:, :, 2]@dgnodes)[face_nodes_loc_idx,:]
            GRAD_GAMMA = (master['shapvol_nodal'][:, :, 3]@dgnodes)[face_nodes_loc_idx,:]
            n = (GRAD_XI + GRAD_ETA + GRAD_GAMMA) / np.linalg.norm(GRAD_XI + GRAD_ETA + GRAD_GAMMA, axis=1)[:,None]

        elif face_idx == 1:
            GRAD_XI = (master['shapvol_nodal'][:, :, 1]@dgnodes)[face_nodes_loc_idx,:]
            n = -GRAD_XI/np.linalg.norm(GRAD_XI, axis=1)[:,None]

        elif face_idx == 2:
            GRAD_ETA = (master['shapvol_nodal'][:, :, 2]@dgnodes)[face_nodes_loc_idx,:]
            n = -GRAD_ETA/np.linalg.norm(GRAD_ETA, axis=1)[:,None]

        elif face_idx == 3:
            GRAD_GAMMA = (master['shapvol_nodal'][:, :, 3]@dgnodes)[face_nodes_loc_idx,:]
            n = -GRAD_GAMMA/np.linalg.norm(GRAD_GAMMA, axis=1)[:,None]

    # Returns a nloc_ptsx3 array of the normal vectors on the given face
    return n