import sys
sys.path.insert(0, '../../util')
sys.path.insert(0, '../../mesh')
sys.path.insert(0, '../../master')
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

logger = logging.getLogger(__name__)

def generate_vtu(p, t, scalars, vectors, labels, viz_filename, call_pv):

    theta = np.concatenate((4*np.ones((t.shape[0], 1)), t), axis=1)
    theta = np.hstack(theta).astype(int)

    # each cell is a VTK_TETRA
    celltypes = np.empty(t.shape[0], dtype=np.uint8)
    celltypes[:] = vtk.VTK_TETRA

    # Build mesh
    mesh = pv.UnstructuredGrid(theta, celltypes, p)

    if scalars is not None:
        for field_index, scalar_field in enumerate(scalars.T):  # Enumerate through rows
            mesh.point_data[labels['scalars'][field_index]] = scalar_field
    if vectors is not None:
        for field_index in labels['vectors']:
            mesh.point_data[labels['vectors'][field_index]] = vectors[field_index,:,:]
    
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

def visualize(mesh, visorder, labels, vis_filename, call_pv, scalars=None, vectors=None):
    if visorder > mesh['porder']:
        # Change to a single line
        ho_viz_mesh = {}
        ho_viz_mesh['ndim'] = mesh['ndim']
        ho_viz_mesh['p'] = mesh['p']
        ho_viz_mesh['t'] = mesh['t']
        ho_viz_mesh['porder'] = visorder    # This is the one change from the base mesh

        ho_viz_mesh['plocal'], ho_viz_mesh['tlocal'], __, __, __, __, __ = masternodes.masternodes(ho_viz_mesh['porder'], mesh['ndim'])
    
        ho_viz_mesh['dgnodes'] = create_dg_nodes.create_dg_nodes(ho_viz_mesh, mesh['ndim'])

        ho_viz_mesh = cgmesh.cgmesh(ho_viz_mesh)

        # Reshape solution from column vector into high order array
        if scalars is not None:
            scalars = helper.reshape_field(mesh, scalars, 'to_array', 'scalars')
        # 'vectors' comes to us in the 'array' format already
            
        scalars, vectors = helper.interpolate_high_order(mesh['porder'], ho_viz_mesh['porder'], mesh['ndim'], lo_scalars=scalars, lo_vectors=vectors)

        # Reshape back into the column vector of high order
        scalars = helper.reshape_field(ho_viz_mesh, scalars, 'to_column', 'scalars')
        vectors = helper.reshape_field(ho_viz_mesh, vectors, 'to_column', 'vectors')

        viz_mesh = create_linear_cg_mesh.create_linear_cg_mesh(ho_viz_mesh)    # update
        
    elif visorder == mesh['porder']:
        viz_mesh = create_linear_cg_mesh.create_linear_cg_mesh(mesh)

        if vectors is not None:
            vectors = helper.reshape_field(mesh, vectors, 'to_column', 'vectors')
    else:
        raise ValueError('Vizorder must be greater than or equal to the mesh order')

    generate_vtu(viz_mesh['pcg'], viz_mesh['linear_cg_mesh'], scalars, vectors, labels, vis_filename, call_pv)


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