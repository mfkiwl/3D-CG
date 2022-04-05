import numpy as np
import sys

sys.path.append('../../master')
import shap
import master_nodes


"""
Sample low order basis functions at the high order points - this will be a singular mapping but that's fine
Weight each low order polynomial vector by the input values at the low order points and add - this will give you the vector of the low order scalar field interpolated onto the high order field


That's all you need to do if the points have already been generated. If they haven't, you need to build the high order mesh.
For example, for viz you need to make a mesh that is double the order and then interpoalte the solutino on that grid. 

Workflow: call mkdgnodes with 2x the order on the original mesh
With the points laid out, input the above to fill in the data at the high order points. This will work between 1->3, 3->6, etc
^ Since this functionality is specifically for viz, put this in the viz script folder.

"""

def interpolate_high_order(porder_lo, porder_hi, ndim, lo_scalars=None, lo_vectors=None):
    """
    Interpolates a low order scalar or vector field onto a high order mesh

    lo_scalars has to be reshaped to the high order element configuration
    ho_mesh must be given like mesh[dgnodes]
    """

    if ndim == 3:
        if lo_scalars is not None:

            # Get ho pts to evaluate at - this is plocal for the high order

            # ploc_hi, __, _, _, _, _, _ = master_nodes.master_nodes(porder_hi, ndim)

            # # Low order shape functions sampled at the high order nodal pts - this will be a tall matrix when it is usually square (when setting up the shape functions)
            ploc_lo, __, _, _, _, _, _ = master_nodes.master_nodes(porder_lo, ndim)
            ploc_hi, __, _, _, _, _, _ = master_nodes.master_nodes(porder_hi, ndim)

            # Low order shape functions sampled at the high order nodal pts - this will be a tall matrix when it is usually square (when setting up the shape functions)
            shap_ho = shap.shape3d(porder_lo, ploc_lo, ploc_hi)[:,:,0]  # Only need the values of the basis functions

            # For each element: weight the lo scalars with the lo fcns sampled at the ho pts and combine
            numel = lo_scalars.shape[1]
            hi_scalars = np.zeros((shap_ho.shape[0], numel))
            for i,__ in enumerate(lo_scalars.T):    # Iterates through rows
                hi_scalars[:,i] = np.squeeze(shap_ho@lo_scalars[:,i][:,None])

        else:
            hi_scalars = None


        if lo_vectors is not None:
            pass
        else:
            hi_vectors = None
        

        return hi_scalars, hi_vectors

