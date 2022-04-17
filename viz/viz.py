# Finding the sim root directory
import sys
from pathlib import Path
cwd = Path.cwd()
for dirname in tuple(cwd.parents):
    if dirname.name == '3D-CG':
        sim_root_dir = dirname
        continue

sys.path.append(str(sim_root_dir.joinpath('util')))
sys.path.append(str(sim_root_dir.joinpath('mesh')))
sys.path.append(str(sim_root_dir.joinpath('master')))
sys.path.append(str(sim_root_dir.joinpath('viz')))
sys.path.append(str(sim_root_dir.joinpath('postprocessing')))
import masternodes
import create_dg_nodes
import cgmesh
import helper
import create_linear_cg_mesh
import pickle
import numpy as np
import vtk
import pyvista as pv
import os
import logging
import gmshwrite
from mkf_parallel2 import mkt2f_new
from functools import partial
from pathos.multiprocessing import ProcessingPool as Pool
import multiprocessing as mp
import quadrature
import extract_surface

logger = logging.getLogger(__name__)

def generate_vtu(p, t, scalars, vectors, labels, viz_filename, call_pv):
    if t.shape[1] == 3:  # 2D - mesh of triangles
        ndim = 2
        theta = np.concatenate((3*np.ones((t.shape[0], 1)), t), axis=1)
        theta = np.hstack(theta).astype(int)

        # each cell is a VTK_TRIANGLE
        celltypes = np.empty(t.shape[0], dtype=np.uint8)
        celltypes[:] = vtk.VTK_TRIANGLE

    elif t.shape[1] == 4:   # 3D - mesh of tets
        ndim = 3
        theta = np.concatenate((4*np.ones((t.shape[0], 1)), t), axis=1)
        theta = np.hstack(theta).astype(int)

        # each cell is a VTK_TETRA
        celltypes = np.empty(t.shape[0], dtype=np.uint8)
        celltypes[:] = vtk.VTK_TETRA

    else:
        raise NotImplementedError('generate_vtu only supports triangles and tets')

    # Build mesh
    mesh = pv.UnstructuredGrid(theta, celltypes, p)

    if scalars is not None:
        for field_index, scalar_field in enumerate(scalars.T):  # Enumerate through rows
            mesh.point_data[labels['scalars'][field_index]] = scalar_field
    if vectors is not None:
        for ifield, field_index in enumerate(labels['vectors']):
            mesh.point_data[labels['vectors'][field_index]] = vectors[:,ndim*ifield:ndim*(ifield+1)]
    
    # Name assignment must be set after all the datasets are loaded because every field is loaded in as 'scalars' initially.
    if scalars is not None:
        mesh.set_active_scalars(labels['scalars'][0])
    if vectors is not None:
        mesh.set_active_vectors(labels['vectors'][0])

    fname = viz_filename+'.vtu'
    mesh.save(fname, binary=False)
    if call_pv:
        logger.info('Opening paraview!')
        os.system("paraview --data=" + fname + " &")
        logger.info('Wrote .vtu to ' + fname +', done...')
    else:
        logger.info('Wrote .vtu to ' + fname +', done...')

def visualize(mesh, visorder, labels, vis_filename, call_pv, scalars=None, vectors=None, type=None):
    if visorder > mesh['porder']:
        # Change to a single line
        ho_viz_mesh = {}
        ho_viz_mesh['ndim'] = mesh['ndim']
        ho_viz_mesh['p'] = mesh['p']
        ho_viz_mesh['t'] = mesh['t']
        ho_viz_mesh['porder'] = visorder    # This is the one change from the base mesh

        ho_viz_mesh['plocal'], ho_viz_mesh['tlocal'], __, __, __, __, __ = masternodes.masternodes(ho_viz_mesh['porder'], mesh['ndim'])
    
        ho_viz_mesh['dgnodes'] = create_dg_nodes.create_dg_nodes(ho_viz_mesh, mesh['ndim'])
        ho_viz_mesh = cgmesh.cgmesh(ho_viz_mesh, type=type)

        # Reshape solution from column vector into high order array
        if scalars is not None:
            scalars = helper.reshape_field(mesh, scalars, 'to_array', 'scalars')
        if vectors is not None:
            vectors = helper.reshape_field(mesh, vectors, 'to_array', 'scalars')
        # 'vectors' comes to us in the 'array' format already
    
        scalars, vectors = helper.interpolate_high_order(mesh['porder'], ho_viz_mesh['porder'], mesh['ndim'], lo_scalars=scalars, lo_vectors=vectors)

        # Reshape back into the column vector of high order
        if scalars is not None:
            scalars = helper.reshape_field(ho_viz_mesh, scalars, 'to_column', 'scalars')
        if vectors is not None:
            vectors = helper.reshape_field(ho_viz_mesh, vectors, 'to_column', 'scalars')

        viz_mesh = create_linear_cg_mesh.create_linear_cg_mesh(ho_viz_mesh)

    elif visorder == mesh['porder']:
        viz_mesh = create_linear_cg_mesh.create_linear_cg_mesh(mesh)

    else:
        raise ValueError('Vizorder must be greater than or equal to the mesh order')

    # if viz_mesh['ndim'] == 3:
    #     f,__ = mkt2f_new(viz_mesh['t_linear'], 3)
    #     gmshwrite.gmshwrite(viz_mesh['pcg'], viz_mesh['t_linear'], vis_filename+'_vizmesh_linear', f[f[:, -1] < 0, :], elemnumbering='individual')
    #     logger.info('Wrote linear mesh')
    # else:
    #     gmshwrite.gmshwrite(viz_mesh['pcg'], viz_mesh['t_linear'], vis_filename+'_vizmesh_linear')
    #     logger.info('Wrote linear mesh')

    logger.info('Wrote linear mesh out to ' + vis_filename+'_mesh_linear')

    generate_vtu(viz_mesh['pcg'], viz_mesh['t_linear'], scalars, vectors, labels, vis_filename, call_pv)

if __name__ == '__main__':
    with open('/media/homehd/saustin/lightning_research/3D-CG/tests/cube_sine_neumann/out/mesh', 'rb') as file:
        mesh = pickle.load(file)
    with open('/media/homehd/saustin/lightning_research/3D-CG/tests/cube_sine_neumann/out/master', 'rb') as file:
        master = pickle.load(file)
    with open('/media/homehd/saustin/lightning_research/3D-CG/tests/cube_sine_neumann/out/sol', 'rb') as file:
        uh = pickle.load(file)  # Note that for this application, uh is a column vector (not in the high order format)
        
    mesh = {}
    mesh['ndim'] = mesh['ndim']
    mesh['p'] = mesh['p']
    mesh['t'] = mesh['t']
    mesh['porder'] = mesh['porder']*2

    mesh['plocal'], mesh['tlocal'], __, __, __, __, __ = masternodes.masternodes(mesh['porder'], 3)
  
    mesh['dgnodes'] = create_dg_nodes.create_dg_nodes(mesh, 3)

    mesh = cgmesh.cgmesh(mesh)

    # Reshape solution from column vector into high order array
    sol_reshaped = np.zeros((mesh['plocal'].shape[0], mesh['t'].shape[0]))
    for ielem, elem in enumerate(mesh['dgnodes']):
        sol_reshaped[:,ielem] = uh[mesh['tcg'][ielem,:]]

    data, __ = interpolate_ho.interpolate_high_order(mesh, mesh, lo_scalars=sol_reshaped)

    # Reshape back into the column vector of high order
    ho_column_vec_scalar = np.zeros((mesh['pcg'].shape[0]))
    for ielem, elem in enumerate(mesh['dgnodes']):
        ho_column_vec_scalar[mesh['tcg'][ielem,:]] = data[:,ielem]

    visualize(mesh, ho_column_vec_scalar)

    # Running pyvista
    plot = pv.Plotter(window_size=[2000, 1500])
    plot.set_background('white')

    # Normal vector for clipping
    normal = (0, -1, 0)
    origin = (0, 264, 0)
    clipped_mesh = mesh.clip(normal, origin)

    plot.add_mesh(clipped_mesh, scalars='sol_data',show_edges=False,line_width=0.75,)
    plot.add_mesh(mesh, scalars='sol_data',show_edges=False,line_width=0.75,)

    plot.show(cpos='xy')