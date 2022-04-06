import shelve
import sys
import numpy as np
from mkf_parallel2 import mkt2f_new
from create_dg_nodes import create_dg_nodes
sys.path.insert(0, '../util')
sys.path.insert(0, '../master')
from import_util import process_mesh
from masternodes import masternodes
from assign_BC_flags import assign_BC_flags
import pickle
import os.path
import logging
import cgmesh

logger = logging.getLogger(__name__)

def mkmesh_cube(porder, ndim, meshfile, build_mesh, scale_factor=1.0, stepfile=None, body_surfs=None):

    mesh_save = meshfile + '_processed'

    if not build_mesh:
        # Reading mesh from disk
        logger.info('Mesh: Reading processed mesh from '+mesh_save)
        with open(mesh_save, 'rb') as file:
            mesh = pickle.load(file)
        return mesh
    
    else:
        logger.info('Mesh: processing mesh...')
        mesh = process_mesh(meshfile, ndim, 4, 2)
        gmsh_mapping = {0: {'node_ID': [15, 1, 2, 4], 'nnodes': [1, 2, 3, 4]}, 1: {'node_ID': [15, 1, 3, 5], 'nnodes': [1, 2, 4, 8]}}     # Maps
        mesh['gmsh_mapping'] = gmsh_mapping

        mesh['meshfile'] = meshfile+'.msh'

        mesh['body_surfs'] = body_surfs
        mesh['stepfile'] = stepfile
        mesh['scale_factor'] = scale_factor
        mesh['p'] *= scale_factor
        mesh['bbox_after_scale'] = {'x': [np.min(mesh['p'][:,0]), np.max(mesh['p'][:,0])], 'y': [np.min(mesh['p'][:,1]), np.max(mesh['p'][:,1])], 'z': [np.min(mesh['p'][:,2]), np.max(mesh['p'][:,2])]}

        mesh['porder'] = porder
        mesh['ndim'] = ndim
        mesh['elemtype'] = 0    # 0 for simplex elements (triangles/tets), 1 for quads/hexes

        logger.info('Mesh: make t2f, f')
        mesh['f'], mesh['t2f'] = mkt2f_new(mesh['t'], 3)

        logger.info('Mesh: master nodes')
        mesh['plocal'], mesh['tlocal'], _, _, _, _, _ = masternodes(porder, 3)

        # set boundary numbers
        logger.info('Mesh: assigning BC flags')
        mesh = assign_BC_flags(mesh)

        # create dg nodes
        logger.info('Mesh: creating high order nodes')
        mesh['dgnodes'] = create_dg_nodes(mesh, 3)

        logger.info('Converting high order mesh to CG...')
        mesh = cgmesh.cgmesh(mesh)
        
        # Saving mesh to disk
        with open(mesh_save, 'wb') as file:
            pickle.dump(mesh, file)

        return mesh

if __name__ == '__main__':
    porder = 3
    mkmesh_cube(porder, cg=True)