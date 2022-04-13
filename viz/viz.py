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
import gmshwrite
from mkf_parallel2 import mkt2f_new

logger = logging.getLogger(__name__)

def generate_vtu(p, t, scalars, vectors, labels, viz_filename, call_pv):
    if t.shape[1] == 3:  # 2D - mesh of triangles
        theta = np.concatenate((3*np.ones((t.shape[0], 1)), t), axis=1)
        theta = np.hstack(theta).astype(int)

        # each cell is a VTK_TRIANGLE
        celltypes = np.empty(t.shape[0], dtype=np.uint8)
        celltypes[:] = vtk.VTK_TRIANGLE

    elif t.shape[1] == 4:   # 3D - mesh of tets
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
        # 'vectors' comes to us in the 'array' format already
            
        scalars, vectors = helper.interpolate_high_order(mesh['porder'], ho_viz_mesh['porder'], mesh['ndim'], lo_scalars=scalars, lo_vectors=vectors)

        # Reshape back into the column vector of high order
        if scalars is not None:
            scalars = helper.reshape_field(ho_viz_mesh, scalars, 'to_column', 'scalars')
        if vectors is not None:
            vectors = helper.reshape_field(ho_viz_mesh, vectors, 'to_column', 'vectors')

        viz_mesh = create_linear_cg_mesh.create_linear_cg_mesh(ho_viz_mesh)    # update

    elif visorder == mesh['porder']:
        viz_mesh = create_linear_cg_mesh.create_linear_cg_mesh(mesh)

        if vectors is not None:
            vectors = helper.reshape_field(mesh, vectors, 'to_column', 'vectors')
    else:
        raise ValueError('Vizorder must be greater than or equal to the mesh order')

    if viz_mesh['ndim'] == 3:
        f,__ = mkt2f_new(viz_mesh['t_linear'], 3)
        gmshwrite.gmshwrite(viz_mesh['pcg'], viz_mesh['t_linear'], vis_filename+'_vizmesh_linear', f[f[:, -1] < 0, :], elemnumbering='individual')
        logger.info('Wrote linear mesh')
    else:
        gmshwrite.gmshwrite(viz_mesh['pcg'], viz_mesh['t_linear'], vis_filename+'_vizmesh_linear')
        logger.info('Wrote linear mesh')

    logger.info('Wrote linear mesh out to ' + vis_filename+'_mesh_linear')

    generate_vtu(viz_mesh['pcg'], viz_mesh['t_linear'], scalars, vectors, labels, vis_filename, call_pv)

def visualize_surface_field(mesh, master, face_groups, case, field, visorder, labels, vis_fname, call_pv):
    """
    face_groups can either be a list of individual faces to plot, or a list of physical groups to plot

    'case' can either be 'pg', meaning that the items in 'face_list' represent indices into the physical group, or 'face', meaning that
    each item in 'face_list' represents a single face index to visualize. The latter allows finer grain control over what is visualized.
    
    'field' is the full scalar field of size mesh['pcg'].shape[0]
    """

    f_with_index = np.concatenate((np.arange(mesh['f'].shape[0])[:,None]+1, mesh['f']), axis=1)

    # Extract those faces from mesh.f. Note that we cut off the physical group index as this is no longer necessary information
    if case == 'face':
        print('Did you make sure to 0-index face_groups?')
        faces = f_with_index[face_groups, :-1]   # Specific faces input to visualize    face_groups HAVE TO BE 0-INDEXED!!
    elif case == 'pg':
        faces = []
        for group in face_groups:
            face_indices = np.where(f_with_index[:,-1] == -group)[0]
            faces.append(f_with_index[face_indices, :-1])   # Specific faces input to visualize
        faces = np.asarray(faces).reshape((-1, f_with_index.shape[1]-1))
    
    # We now have a 2D connectivity matrix. Turn this into a HO CG mesh using a 2D version of cgmesh
    mesh_face, face_scalars = cgmesh.cgmesh(mesh, faces, master, case='surface_mesh', field=field)
    mesh_face['porder'] = mesh['porder']
    mesh_face['ndim'] = mesh['ndim'] - 1
    mesh_face['plocal'], mesh_face['tlocal'], _, _, __, _, _ = masternodes.masternodes(mesh_face['porder'], mesh_face['ndim'])

    visualize(mesh_face, visorder, labels, vis_fname, call_pv, face_scalars, None, type='surface_mesh') # Can only have scalars on a surface mesh

    return


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