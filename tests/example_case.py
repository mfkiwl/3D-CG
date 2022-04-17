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
sys.path.append(str(sim_root_dir.joinpath('CG')))
sys.path.append(str(sim_root_dir.joinpath('postprocessing')))

import numpy as np
from import_util import load_mat
import viz
from cgmesh import cgmesh
import mkmesh_cube
import mkmaster
import cg_solve
import pickle
import calc_derivative
import os
import logging
import logging.config
import helper
import domain_helper_fcns
import cg_gradient
import extract_surface

### Example test case illustrating surface extraction/visualization and gradient calculation with both the new and old methods


def test_3d_cube_sine_neumann(porder, meshfile, solver):

    ########## INITIALIZE LOGGING ##########
    logging.config.fileConfig('../logging/loggingDEPRECATED.conf', disable_existing_loggers=False)
    logger = logging.getLogger('root')
    logger.info('*************************** INITIALIZING SIM ***************************')

    # Check whether the specified path exists or not
    if not os.path.exists('out/'):
        os.makedirs('out/')
        logger.info('out/ directory not present, created...')

    ########## TOP LEVEL SIM SETUP ##########

    outdir = 'out/'
    meshfile = './data/' + meshfile
    build_mesh = True
    vis_filename = 'cube_sol'
    ndim = 3
    call_pv = True
    # call_pv = False
    # viz_labels = {'scalars': {0: 'Solution'}, 'vectors': {0: 'Solution Gradient'}}
    surf_viz_labels = {'scalars': {0: 'Solution', 1: 'Gradient dot normal'}, 'vectors': {}}
    visorder = porder
    solver_tol=1e-10
    
    # NOTE: Bugs with orders 4, 5, and 6 here in various parts

    ########## BCs ##########
    # Dirichlet
    dbc = {
        4: -1,   # +X
        # 6: 0,   # -X
    }

    # Neumann
    nbc = {
        6: -4.71238898038,   # -X
        3: 0,   # -Y
        5: 0,   # +Y
        1: 0,   # -Z
        2: 0,   # +Z
    }

    ########## CREATE MESH ##########

    mesh = mkmesh_cube.mkmesh_cube(porder, ndim, meshfile, build_mesh)
    logger.info('Converting high order mesh to CG...')
    mesh = cgmesh(mesh)
    mesh['dbc'] = dbc
    mesh['nbc'] = nbc

    logger.info('Preparing master data structure...')
    master = mkmaster.mkmaster(mesh, ndim=3, pgauss=2*mesh['porder'])

    ########## PHYSICS PARAM ##########
    param = {'kappa': 1, 'c': np.array([0, 0, 0]), 's': 0}

    ########## SOLVE ##########

    sol = cg_solve.cg_solve(master, mesh, domain_helper_fcns.forcing_cube_mod_sine, param, ndim, outdir, buildAF=True, solver=solver, solver_tol=solver_tol)

    ########## SAVE DATA ##########

    # NOTE: in the future uh will need to be reshaped into a nplocal x numvisfields x numel when the derivatives are added
    with open(outdir+'mesh', 'wb') as file:
        pickle.dump(mesh, file)
    with open(outdir+'master', 'wb') as file:
        pickle.dump(master, file)
    with open(outdir+'sol', 'wb') as file:
        pickle.dump(sol, file)
    logger.info('Wrote solution to file...')

    sol = np.squeeze(sol)

    ########## ERROR CALCULATION ##########
    # sol = domain_helper_fcns.exact_cube_mod_sine(mesh['pcg'])
    exact = domain_helper_fcns.exact_cube_mod_sine(mesh['pcg'])
    # Reshape into DG high order data structure
    sol_reshaped = np.zeros((mesh['plocal'].shape[0], mesh['t'].shape[0]))
    exact_reshaped = np.zeros((mesh['plocal'].shape[0], mesh['t'].shape[0]))
    for i in np.arange(mesh['t'].shape[0]):          # Set dirichlet BC
        sol_reshaped[:, i] = sol[mesh['tcg'][i, :]]
        exact_reshaped[:, i] = exact[mesh['tcg'][i, :]]

    error = exact_reshaped.ravel()-sol_reshaped.ravel()
    norm_error = np.linalg.norm(error, np.inf)
    logger.info('L-inf error: '+str(norm_error))

    ########## CALC DERIVATIVES ##########

    logger.info('Calculating derivatives')

    # Reshape into DG high order data structure
    sol_reshaped = helper.reshape_field(mesh, sol[:,None], 'to_array', 'scalars')

    grad_exact = domain_helper_fcns.grad_sine_1d(mesh['pcg'], m=1.5, axis='x')

    # Old gradients must be reshaped from array -> column vector first
    grad_old = calc_derivative.calc_derivatives(mesh, master, sol_reshaped, ndim)
    grad_col_old = helper.reshape_field(mesh, grad_old[None,:,:], 'to_column', 'vectors')[0,:,:]

    # New gradient calc returns the column vector format already
    grad, grad_mag = cg_gradient.calc_gradient(mesh, master, sol[:,None], ndim, solver, solver_tol)
    
    grad_error = np.linalg.norm((grad-grad_exact).ravel(), np.inf)

    viz_grad = np.concatenate((grad, grad_exact, (grad-grad), grad_col_old, (grad_col_old-grad_exact)), axis=1)
    vis_scalars = np.concatenate((exact[:,None], sol[:,None], grad_mag), axis=1)

    viz_labels = {'scalars': {0: 'Computed Solution', 1: 'Exact Solution', 2: 'Gradient magnitude'}, 'vectors': {0: 'Computed Gradient CG', 1: 'Exact Gradient', 2: 'Gradient Error CG', 3: 'Computed Gradient old', 4: 'Gradient error old'}}

    result_out = np.concatenate((sol[:,None], grad), axis=1)

    with open(outdir+'solution' + vis_filename + '.npy', 'wb') as file:
        np.save(file, result_out)
    logger.info('Wrote solution to /out')

    ########## VISUALIZE SOLUTION ##########
    # logger.info('Running visualization for volume fields')
    # viz.visualize(mesh, visorder, viz_labels, 'vis_tet', call_pv, scalars=vis_scalars, vectors=viz_grad)

    # import gmshwrite
    # gmshwrite.gmshwrite(mesh['p'], mesh['t'], 'base_mesh_test', mesh['f'], elemnumbering='individual')

    grad[:,0] = 1
    grad[:,1] = 1
    grad[:,2] = 1

    logger.info('Visualizing extracted surface')
    surf_face_pg = np.array([1, 2, 3, 4, 5, 6])

    # Uncomment to visualize specific faces
    # viz.visualize_surface_scalars(mesh, master, np.array([50, 51]), 'face', sol[:,None], visorder, surf_viz_labels, vis_filename+'_surface_face', call_pv)
    
    mesh_face, face_scalars = extract_surface.extract_surfaces(mesh, master, surf_face_pg, 'pg', sol[:,None])
    __, face_field_dot_normal = extract_surface.extract_surfaces(mesh, master, surf_face_pg, 'pg', grad, return_normal_quantity=True)
    
    face_scalars = np.concatenate((face_scalars, face_field_dot_normal), axis=1)

    viz.visualize(mesh_face, visorder, surf_viz_labels, vis_filename+'_surface', call_pv, face_scalars, None, type='surface_mesh') # Can only have scalars on a surface mesh

    return norm_error, grad_error

if __name__ == '__main__':
    # print('porder', 2, 'cube24', 'direct')
    # print(test_3d_cube_sine_neumann(2, 'cube24', 'direct'))
    # print('porder', 2, 'cube100', 'direct')
    # print(test_3d_cube_sine_neumann(2, 'cube100', 'direct'))
    # print('porder', 2, 'cube4591', 'gmres')
    # print(test_3d_cube_sine_neumann(2, 'cube4591', 'gmres'))

    # print('porder', 3, 'cube24', 'direct')
    # print(test_3d_cube_sine_neumann(3, 'cube24', 'direct'))
    # print('porder', 3, 'cube24', 'cg')
    # print(test_3d_cube_sine_neumann(3, 'cube24', 'cg'))
    # print('porder', 3, 'cube100', 'gmres')
    # print(test_3d_cube_sine_neumann(3, 'cube100', 'gmres'))

    # print('porder', 3, 'cube100', 'direct')
    # print(test_3d_cube_sine_neumann(3, 'cube100', 'direct'))
    # print('porder', 3, 'cube100', 'cg')
    # print(test_3d_cube_sine_neumann(3, 'cube100', 'cg'))
    # print('porder', 3, 'cube100', 'gmres')
    # print(test_3d_cube_sine_neumann(3, 'cube100', 'gmres'))

    # print('porder', 3, 'cube4591', 'cg')
    # print(test_3d_cube_sine_neumann(3, 'cube4591', 'cg'))
    print('porder', 3, 'cube4591', 'gmres')
    print(test_3d_cube_sine_neumann(3, 'cube4591', 'gmres'))