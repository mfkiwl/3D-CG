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
import viz
import interpolate_ho

def visualize(mesh, sol):
    mesh = create_linear_cg_mesh.create_linear_cg_mesh_vec(mesh)
    viz.generate_vtu(mesh['pcg'], mesh['linear_cg_mesh'], sol)

if __name__ == '__main__':
    with open('/media/homehd/saustin/lightning_research/3D-CG/tests/cube_sine_neumann/out/mesh', 'rb') as file:
        mesh = pickle.load(file)
    with open('/media/homehd/saustin/lightning_research/3D-CG/tests/cube_sine_neumann/out/master', 'rb') as file:
        master = pickle.load(file)
    with open('/media/homehd/saustin/lightning_research/3D-CG/tests/cube_sine_neumann/out/sol', 'rb') as file:
        uh = pickle.load(file)  # Note that for this application, uh is a column vector (not in the high order format)

        # NOTE: in the future uh will need to be reshaped into a nplocal x numvisfields x numel when the derivatives are added

    # visscalars = ["temperature", 0]; # list of scalar fields for visualization
    # visvectors = ["temperature gradient", np.array([1, 2, 3]).astype(int)]; # list of vector fields for visualization

    ho_viz_mesh = {}
    ho_viz_mesh['ndim'] = mesh['ndim']
    ho_viz_mesh['p'] = mesh['p']
    ho_viz_mesh['t'] = mesh['t']
    ho_viz_mesh['porder'] = mesh['porder']*2

    ho_viz_mesh['plocal'], ho_viz_mesh['tlocal'], __, __, __, __, __ = master_nodes.master_nodes(ho_viz_mesh['porder'], 3)
  
    ho_viz_mesh['dgnodes'] = create_dg_nodes.create_dg_nodes(ho_viz_mesh, 3)

    ho_viz_mesh = cgmesh.cgmesh(ho_viz_mesh)

    # Reshape solution from column vector into high order array
    sol_reshaped = np.zeros((mesh['plocal'].shape[0], mesh['t'].shape[0]))
    for ielem, elem in enumerate(mesh['dgnodes']):
        sol_reshaped[:,ielem] = uh[mesh['tcg'][ielem,:]]

    hi_scalars, __ = interpolate_ho.interpolate_high_order(mesh['porder'], ho_viz_mesh['porder'], ho_viz_mesh['ndim'], lo_scalars=sol_reshaped)

    # print(hi_scalars)
    # exit()
    # Reshape back into the column vector of high order

    ho_column_vec_scalar = np.zeros((ho_viz_mesh['pcg'].shape[0]))
    for ielem, elem in enumerate(ho_viz_mesh['dgnodes']):
        ho_column_vec_scalar[ho_viz_mesh['tcg'][ielem,:]] = hi_scalars[:,ielem]

    visualize(ho_viz_mesh, ho_column_vec_scalar)