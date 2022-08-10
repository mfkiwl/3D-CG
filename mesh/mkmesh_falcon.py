import numpy as np
from scipy.io import loadmat
# Finding the sim root directory
import sys
from pathlib import Path

cwd = Path.cwd()
for dirname in tuple(cwd.parents):
    if dirname.name == '3D-CG':
        sim_root_dir = dirname
        continue

sys.path.append(str(sim_root_dir.joinpath('util')))
sys.path.append(str(sim_root_dir.joinpath('CG')))
sys.path.append(str(sim_root_dir.joinpath('mesh')))
sys.path.append(str(sim_root_dir.joinpath('master')))
import logging

import gmshwrite
import cgmesh
from import_util import process_mesh
from masternodes import masternodes
from assign_BC_flags import assign_BC_flags
from mkf_parallel2 import mkt2f_new
from create_dg_nodes import create_dg_nodes
import pickle

logger = logging.getLogger(__name__)

def import_falcon_mesh(porder, ndim, meshfile, build_mesh):
    mesh_save = meshfile + '_processed'

    if not build_mesh:
        # Reading mesh from disk
        logger.info('Reading processed mesh from '+mesh_save)
        with open(mesh_save, 'rb') as file:
            mesh = pickle.load(file)
        return mesh

    else:
        falcon = loadmat(meshfile+'.mat')
        mesh = {}
        mesh['p'] = falcon['msh']['p'][0,0].astype(np.float)
        mesh['t'] = falcon['msh']['t'][0,0].astype(np.int32)-1

        scale_factor = 1.25    # Falcon fuselage radius, m
        mesh['p'] *= scale_factor

        mesh['bbox_after_scale'] = {'x': [np.min(mesh['p'][:,0]), np.max(mesh['p'][:,0])], 'y': [np.min(mesh['p'][:,1]), np.max(mesh['p'][:,1])], 'z': [np.min(mesh['p'][:,2]), np.max(mesh['p'][:,2])]}

        mesh['meshfile'] = meshfile+'.msh'

        pg_list = {}
        pg_list[1] = {'idx': 1, 'nodes': np.where(np.isclose(mesh['p'][:,0], mesh['bbox_after_scale']['x'][0]))[0]}  # -X
        pg_list[2] = {'idx': 2, 'nodes': np.where(np.isclose(mesh['p'][:,0], mesh['bbox_after_scale']['x'][1]))[0]}  # +X
        pg_list[3] = {'idx': 3, 'nodes': np.where(np.isclose(mesh['p'][:,1], mesh['bbox_after_scale']['y'][0]))[0]}  # -Y
        pg_list[4] = {'idx': 4, 'nodes': np.where(np.isclose(mesh['p'][:,1], mesh['bbox_after_scale']['y'][1]))[0]}  # +Y
        pg_list[5] = {'idx': 5, 'nodes': np.where(np.isclose(mesh['p'][:,2], mesh['bbox_after_scale']['z'][0]))[0]}  # -Z
        pg_list[6] = {'idx': 6, 'nodes': np.where(np.isclose(mesh['p'][:,2], mesh['bbox_after_scale']['z'][1]))[0]}  # +Z

        tot_pts = np.arange(mesh['p'].shape[0])
        bdry_pts = np.concatenate((pg_list[1]['nodes'], pg_list[2]['nodes'], pg_list[3]['nodes'], pg_list[4]['nodes'], pg_list[5]['nodes'], pg_list[6]['nodes']))
        interior_pts = np.setdiff1d(tot_pts, bdry_pts)

        pg_list[7] = {'idx': 7, 'nodes': interior_pts}   # Assigning all points that aren't on the outer surface of the box to 'interior points'

        pg_dict = {}
        for pg in pg_list:
            pg_dict[pg] = {'idx': pg}
            pg_dict[pg]['nodes'] = {key.tobytes():None for key in pg_list[pg]['nodes'].astype(np.int32)}

        mesh['pg'] = pg_dict

        # Special treatment for the physical groups for this mesh because we can't import them from Gmsh
        mesh['porder'] = porder
        mesh['ndim'] = ndim
        mesh['elemtype'] = 0    # 0 for simplex elements (triangles/tets), 1 for quads/hexes

        logger.info('Mesh: make t2f, f')
        mesh['f'], mesh['t2f'] = mkt2f_new(mesh['t'], 3)

        logger.info('Mesh: master nodes')
        mesh['plocal'], mesh['tlocal'], _, _, _, _, _ = masternodes(porder, 3)

        gmsh_mapping = {0: {'node_ID': [15, 1, 2, 4], 'nnodes': [1, 2, 3, 4]}, 1: {'node_ID': [15, 1, 3, 5], 'nnodes': [1, 2, 4, 8]}}     # Maps
        mesh['gmsh_mapping'] = gmsh_mapping
        mesh['nnodes_per_face'] = mesh['gmsh_mapping'][mesh['elemtype']]['nnodes'][mesh['ndim']-1]

        # set boundary numbers
        logger.info('Mesh: assigning BC flags')
        mesh = assign_BC_flags(mesh)

        # create dg nodes
        logger.info('Mesh: creating high order nodes')
        mesh['dgnodes'] = create_dg_nodes(mesh, 3)

        logger.info('Converting high order mesh to CG...')
        mesh = cgmesh.cgmesh(mesh)

        logger.info('Writing gmsh .msh file...')
        gmshwrite.gmshwrite(mesh['p'], mesh['t'], meshfile, mesh['f'])

        # Saving mesh to disk
        logger.info('Saving mesh to '+mesh_save)
        with open(mesh_save, 'wb') as file:
            pickle.dump(mesh, file)

    return mesh

if __name__ == '__main__':
    porder=2
    import_falcon_mesh(porder, 3, './mesh/falconfine', True)