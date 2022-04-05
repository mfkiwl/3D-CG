import numpy as np
import sys
sys.path.append('../../mesh')
import cgmesh
import time

"""
This is NOT the same as the "cgmesh" function. 

Specifically, this function converts a high order "DG" mesh to a CG mesh with linear elements. This means that each high order element gets its own element, and the total number of elements will be (nt x ntlocal) instead of nt as with cgmesh.

"""

def create_linear_cg_mesh(mesh):

    # Call cgmesh to get the global nodes. We need to do this only if the cgmesh is not passed in to the function. This will create both 'tcg' and 'pcg'.
    if 'pcg' not in mesh:
        mesh = cgmesh.cgmesh(mesh)

    start = time.perf_counter()
    # Use mesh['tlocal'] to get the local connectivity and map to the global connectivity
    linear_cg_mesh = np.zeros((mesh['t'].shape[0]*mesh['tlocal'].shape[0], mesh['t'].shape[1]))

    ntlocal = mesh['tlocal'].shape[0]


    # Try to get rid of these loops for speed
    for elnum, element in enumerate(mesh['tcg']):
        for iloc, loc_el in enumerate(mesh['tlocal']):
            linear_cg_mesh[ntlocal*elnum+iloc, :] = element[loc_el]


    mesh['linear_cg_mesh'] = linear_cg_mesh

    print('linear_cg_nonvectorized', time.perf_counter() -start)
    # If needed, add functionality here to convert a "field" - i.e. solution matrix at the dgnodes, into a vector of size (npcg,).

    return mesh

def create_linear_cg_mesh_vec(mesh):
    """
        Take the high order connectivity matrix in dgnodes and convert to CG mesh of linear elements

    """
    
    # Call cgmesh to get the global nodes. We need to do this only if the cgmesh is not passed in to the function. This will create both 'tcg' and 'pcg'.
    if 'pcg' not in mesh:
        mesh = cgmesh.cgmesh(mesh)

    # start = time.perf_counter()
    ntlocal = mesh['tlocal'].shape[0]

    # Double for loop vectorized for speed
    tcg_flat = mesh['tcg'].ravel()

    nplocal = mesh['plocal'].shape[0]
    nt = mesh['t'].shape[0]

    tiled = np.tile(mesh['tlocal'], (nt,1))
    fact=np.repeat(np.arange(nt), ntlocal)[:,None]*nplocal
    idx_arry = fact+tiled

    linear_cg_mesh=tcg_flat[idx_arry]

    mesh['linear_cg_mesh'] = linear_cg_mesh

    # If needed, add functionality here to convert a "field" - i.e. solution matrix at the dgnodes, into a vector of size (npcg,).

    # print('linear_cg_vectorized', time.perf_counter() -start)
    return mesh



    """
    Add high order mesh interpolation
    Vectorize
    Add support for fields

    Integratw with pviz!!

    """