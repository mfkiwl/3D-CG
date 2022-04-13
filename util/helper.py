import numpy as np
# Finding the sim root directory
import sys
from pathlib import Path
cwd = Path.cwd()
for dirname in tuple(cwd.parents):
    if dirname.name == '3D-CG':
        sim_root_dir = dirname
        continue

sys.path.append(str(sim_root_dir.joinpath('master')))

import shap
import masternodes

def reshape_field(mesh, data, case, type, porder=None):
    if porder == 1:
        tcg = mesh['t']
        nplocal = mesh['t'].shape[1]
    else:
        tcg = mesh['tcg']
        nplocal = mesh['plocal'].shape[0]

    if type == 'scalars':
        if case == 'to_array':
            # Reshape solution from column vector into high order array
            num_scalar_fields = data.shape[1]
            data_reshaped = np.zeros((nplocal, mesh['t'].shape[0]*num_scalar_fields))
            for scalar_idx in np.arange(num_scalar_fields):
                for ielem, __ in enumerate(mesh['t']):
                    data_reshaped[:,scalar_idx*num_scalar_fields+ielem] = data[tcg[ielem,:], scalar_idx]

        elif case == 'to_column':
            # Reshape back into a column vector from high order array
            numel = mesh['t'].shape[0]
            num_scalar_fields = int(data.shape[1]/numel)

            data_reshaped = np.zeros((mesh['pcg'].shape[0], num_scalar_fields))
            for scalar_idx in np.arange(num_scalar_fields):
                for ielem, __ in enumerate(mesh['t']):
                    data_reshaped[tcg[ielem,:],scalar_idx] = data[:,scalar_idx*num_scalar_fields+ielem]

    elif type == 'vectors':
        if case == 'to_array':
            raise NotImplementedError('Not implemented')

        elif case == 'to_column':
            ndim = mesh['ndim']
            data_reshaped = np.zeros((data.shape[0], mesh['pcg'].shape[0], mesh['ndim']))
            for vec_idx in np.arange(data.shape[0]):
                for dim in np.arange(mesh['ndim']):
                    data_reshaped[vec_idx,:,dim] = np.squeeze(reshape_field(mesh, data[vec_idx,:,:][:,dim::ndim], 'to_column', 'scalars'))

    return data_reshaped

def interpolate_high_order(porder_lo, porder_hi, ndim, lo_scalars=None, lo_vectors=None):
    """
    Interpolates a low order scalar or vector field onto a high order mesh

    lo_scalars has to be reshaped to the high order element configuration
    ho_mesh must be given like mesh[dgnodes]
    """

    # Get ho pts to evaluate at - this is plocal for the high order
    # # Low order shape functions sampled at the high order nodal pts - this will be a tall matrix when it is usually square (when setting up the shape functions)
    ploc_lo, __, _, _, _, _, _ = masternodes.masternodes(porder_lo, ndim)
    ploc_hi, __, _, _, _, _, _ = masternodes.masternodes(porder_hi, ndim)

    # Low order shape functions sampled at the high order nodal pts - this will be a tall matrix when it is usually square (when setting up the shape functions)
    if ndim == 3:
        shap_ho = shap.shape3d(porder_lo, ploc_lo, ploc_hi)[:,:,0]  # Only need the values of the basis functions
    elif ndim == 2:
        ploc_lo = ploc_lo[:,:-1]     # Indexing added because plocal in 2D returns points in 3D so we must chop off the z component
        ploc_hi = ploc_hi[:,:-1]     # Indexing added because plocal in 2D returns points in 3D so we must chop off the z component
        shap_ho = shap.shape2d(porder_lo, ploc_lo, ploc_hi)[:,:,0]  # Only need the values of the basis functions
    else:
        raise NotImplementedError('dim not in [2, 3] not implemented')

    # For each element: weight the lo values with the lo fcns sampled at the ho pts and combine
    if lo_scalars is not None:    
        hi_scalars = np.zeros((shap_ho.shape[0], lo_scalars.shape[1]))
        for i,__ in enumerate(lo_scalars.T):    # Iterates through rows
            hi_scalars[:,i] = np.squeeze(shap_ho@lo_scalars[:,i][:,None])
    else:
        hi_scalars = None
    if lo_vectors is not None:
        hi_vectors = np.zeros((lo_vectors.shape[0], shap_ho.shape[0], lo_vectors.shape[2]))
        for field_idx in np.arange(lo_vectors.shape[0]):
            for i,__ in enumerate(lo_vectors.T):    # Iterates through rows
                hi_vectors[field_idx,:,i] = np.squeeze(shap_ho@lo_vectors[field_idx,:,i][:,None])
    else:
        hi_vectors = None

    return hi_scalars, hi_vectors

