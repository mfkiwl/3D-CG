import sys
import create_linear_cg_mesh
sys.path.insert(0, '../../util')
sys.path.insert(0, '../../mesh')
sys.path.insert(0, '../../master')
import master_nodes
import create_dg_nodes
import cgmesh
import pickle
import numpy as np
import interpolate_ho
import vtk
import pyvista as pv
import os

def visualize(mesh, visorder, scalars, vectors, labels, call_pv):
    if visorder > mesh['porder']:
        ho_viz_mesh = {}
        ho_viz_mesh['ndim'] = mesh['ndim']
        ho_viz_mesh['p'] = mesh['p']
        ho_viz_mesh['t'] = mesh['t']
        ho_viz_mesh['porder'] = visorder    # This is the one change from the base mesh

        ho_viz_mesh['plocal'], ho_viz_mesh['tlocal'], __, __, __, __, __ = master_nodes.master_nodes(ho_viz_mesh['porder'], 3)
    
        ho_viz_mesh['dgnodes'] = create_dg_nodes.create_dg_nodes(ho_viz_mesh, 3)

        ho_viz_mesh = cgmesh.cgmesh(ho_viz_mesh)

        # Reshape solution from column vector into high order array
        sol_reshaped = np.zeros((mesh['plocal'].shape[0], mesh['t'].shape[0]))
        for ielem, elem in enumerate(mesh['dgnodes']):
            sol_reshaped[:,ielem] = uh[mesh['tcg'][ielem,:]]

        hi_scalars, __ = interpolate_ho.interpolate_high_order(mesh, ho_viz_mesh, lo_scalars=scalars_dg_reshaped, lo_vectors=vectors_dg_reshaped)

        # Reshape back into the column vector of high order
        ho_column_vec_scalar = np.zeros((ho_viz_mesh['pcg'].shape[0]))
        for ielem, elem in enumerate(ho_viz_mesh['dgnodes']):
            ho_column_vec_scalar[ho_viz_mesh['tcg'][ielem,:]] = hi_scalars[:,ielem]

        viz_mesh = create_linear_cg_mesh.create_linear_cg_mesh_vec(ho_viz_mesh)    # update
        
    elif visorder == mesh['porder']:
        viz_mesh = create_linear_cg_mesh.create_linear_cg_mesh_vec(mesh)    # update
    else:
        raise ValueError('Vizorder must be greater than or equal to the mesh order')

    viz.generate_vtu(viz_mesh['pcg'], viz_mesh['linear_cg_mesh'], scalars, vectors, labels, call_pv)      # Update

def generate_vtu(p, t, scalars, vectors, labels, viz_filename, call_pv):

    theta = np.concatenate((4*np.ones((t.shape[0], 1)), t), axis=1)
    theta = np.hstack(theta).astype(int)

    # each cell is a VTK_TETRA
    celltypes = np.empty(t.shape[0], dtype=np.uint8)
    celltypes[:] = vtk.VTK_TETRA

    # Build mesh
    mesh = pv.UnstructuredGrid(theta, celltypes, p)

    for field_index, scalar_field in enumerate(scalars.T):  # Enumerate through rows
        mesh.point_data[labels['scalars'][field_index]] = scalar_field
    for field_index, vector_field in enumerate(vectors.T):  # Enumerate through pages
        mesh.point_data[labels['vectors'][field_index]] = vector_field
    
    mesh.set_active_scalars('sol_data')
    mesh.set_active_vectors('vec')

    fname = viz_filename+'.vtu'
    mesh.save(fname, binary=False)
    if call_pv:
        print('Opening paraview!')
        os.system("paraview --data=" + fname + " &")
    else:
        print('Completed writing .vtu, closing...')

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

    mesh['plocal'], mesh['tlocal'], __, __, __, __, __ = master_nodes.master_nodes(mesh['porder'], 3)
  
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

    # Generate plot
    # plot = pv.Plotter(window_size=[2000, 1500])
    # plot.set_background('white')

    # Normal vector for clipping
    # normal = (0, -1, 0)
    # origin = (0, 264, 0)
    # clipped_mesh = mesh.clip(normal, origin)

    # plot.add_mesh(clipped_mesh, scalars='sol_data',show_edges=False,line_width=0.75,)
    # plot.add_mesh(mesh, scalars='sol_data',show_edges=False,line_width=0.75,)

    # plot.show(cpos='xy')