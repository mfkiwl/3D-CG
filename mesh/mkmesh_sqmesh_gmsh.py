import shelve
import sys
import numpy as np
from mkt2f import mkt2f
from create_dg_nodes import create_dg_nodes
from setbndnbrs import setbndnbrs
sys.path.insert(0, '../util')
sys.path.insert(0, '../master')
from import_util import process_mesh, load_processed_mesh
from masternodes import masternodes
from assign_BC_flags import assign_BC_flags


def mkmesh_square(porder, ndim, meshfile):
    mesh = process_mesh(meshfile, ndim, 2, 1)
    gmsh_mapping = {0: {'node_ID': [15, 1, 2, 4], 'nnodes': [1, 2, 3, 4]}, 1: {'node_ID': [15, 1, 3, 5], 'nnodes': [1, 2, 4, 8]}}     # Maps
    mesh['gmsh_mapping'] = gmsh_mapping
    mesh['porder'] = porder
    mesh['ndim'] = ndim
    mesh['elemtype'] = 0    # 0 for simplex elements (triangles/tets), 1 for quads/hexes

    mesh['f'], mesh['t2f'] = mkt2f(mesh['t'], 2)
    
    mesh['plocal'], mesh['tlocal'], _, _, _, _, _ = masternodes(porder, 2)

    # set boundary numbers
    mesh = assign_BC_flags(mesh)

    # create dg nodes
    mesh['dgnodes'] = create_dg_nodes(mesh, 2)[:,:,:-1]     # Indexing added because plocal in 2D returns points in 3D so we must chop off the z component
    # Note that it's okay to leave the DG nodes as 3D (as opposed to stripping off the constant z-coordinate) because the z points are not accessed when forming the elemental matrices (see elemmat_cg.py)

    # with shelve.open('unstr_square_mesh_save') as shelf:
    #     shelf['mesh'] = mesh        # It is ok to save in a shelf instead of a .npy because the data structure is a dictionary

    return mesh

if __name__ == '__main__':
    porder = 3
    mkmesh_square(porder)