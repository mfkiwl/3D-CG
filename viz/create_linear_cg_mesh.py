import numpy as np
import sys
sys.path.append('../../mesh')

def create_linear_cg_mesh(mesh):
    """
        Take the high order connectivity matrix in dgnodes and convert to CG mesh of linear elements

        This is NOT the same as the "cgmesh" function. 

        Specifically, this function converts a high order "DG" mesh to a CG mesh with linear elements. This means that each high order element gets its own element, and the total number of elements will be (nt x ntlocal) instead of nt as with cgmesh.

    """

    ntlocal = mesh['tlocal'].shape[0]

    # Could be written as a double for loop, but vectorized for speed
    tcg_flat = mesh['tcg'].ravel()

    nplocal = mesh['plocal'].shape[0]
    nt = mesh['t'].shape[0]

    tiled = np.tile(mesh['tlocal'], (nt,1))
    fact=np.repeat(np.arange(nt), ntlocal)[:,None]*nplocal
    idx_arry = fact+tiled

    linear_cg_mesh=tcg_flat[idx_arry]

    mesh['t_linear'] = linear_cg_mesh

    return mesh
