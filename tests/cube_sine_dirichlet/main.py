import sys
sys.path.append('../../util')
sys.path.append('../../mesh')
sys.path.append('../../master')
sys.path.append('../../viz')
sys.path.append('../../CG')
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

def test_3d_cube_sine_dirichlet(porder, meshfile, solver):

    ########## INITIALIZE LOGGING ##########
    logging.config.fileConfig('../../logging/loggingDEPRECATED.conf', disable_existing_loggers=False)
    # root logger, no __name__ as in submodules further down the hierarchy - this is very important - cost me a lot of time when I passed __name__ to the main logger
    logger = logging.getLogger('root')
    logger.info('*************************** INITIALIZING SIM ***************************')

    # Check whether the specified path exists or not
    if not os.path.exists('out/'):
        os.makedirs('out/')
        logger.info('out/ directory not present, created...')

    ########## TOP LEVEL SIM SETUP ##########

    outdir = 'out/'
    meshfile = '../data/' + meshfile
    build_mesh = True
    vis_filename = 'cube_sol'
    ndim = 3
    # call_pv = True
    call_pv = False
    viz_labels = {'scalars': {0: 'Solution'}, 'vectors': {0: 'Solution Gradient'}}
    visorder = porder
    solver_tol=1e-10
    
    # NOTE: Bugs with orders 4, 5, and 6 here in various parts

    ########## BCs ##########
    # Dirichlet
    dbc = {
        4: -1,   # +X
        6: 0,   # -X
    }

    # Neumann
    nbc = {
        # 6: -4.71238898038,   # -X
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

    exact = domain_helper_fcns.exact_cube_mod_sine(mesh['pcg'])[:,None]
    grad_exact = domain_helper_fcns.grad_sine_1d(mesh['pcg'], m=1.5, axis='x')

    ########## SAVE DATA ##########

    # NOTE: in the future uh will need to be reshaped into a nplocal x numvisfields x numel when the derivatives are added
    with open(outdir+'mesh', 'wb') as file:
        pickle.dump(mesh, file)
    with open(outdir+'master', 'wb') as file:
        pickle.dump(master, file)
    with open(outdir+'sol', 'wb') as file:
        pickle.dump(sol, file)
    logger.info('Wrote solution to file...')

    ########## CALC DERIVATIVES ##########
    logger.info('Calculating derivatives')
    grad, __ = cg_gradient.calc_gradient(mesh, master, sol, ndim, solver, solver_tol)

    ########## ERROR CALCULATION ##########

    sol_error = exact-sol
    norm_grad_error = np.linalg.norm((grad-grad_exact).ravel(), np.inf)
    logger.info('L-inf error of gradient: '+str(norm_grad_error))

    norm_error = np.linalg.norm(sol_error, np.inf)
    logger.info('L-inf error of solution: '+str(norm_error))

    ########## VISUALIZE SOLUTION ##########
    logger.info('Running visualization for volume fields')

    viz_grad = np.concatenate((grad, grad_exact, (grad_exact-grad)), axis=1)
    vis_scalars = np.concatenate((exact, sol), axis=1)

    viz_labels = {'scalars': {0: 'Computed Solution', 1: 'Exact Solution'}, 'vectors': {0: 'Computed Gradient', 1: 'Exact Gradient', 2: 'Gradient Error CG'}}
    viz.visualize(mesh, visorder, viz_labels, 'vis_tet', call_pv, scalars=vis_scalars, vectors=viz_grad)

    ########## SAVE GRADIENT DATA ##########

    result_out = np.concatenate((sol, grad), axis=1)
    with open(vis_filename+'_and_grad' + '.npy', 'wb') as file:
        np.save(file, result_out)
    logger.info('Wrote solution to /out')

    return norm_error, norm_grad_error

if __name__ == '__main__':
    print('porder', 2, 'cube24', 'direct')
    print(test_3d_cube_sine_dirichlet(2, 'cube24', 'direct'))
    print('porder', 2, 'cube100', 'direct')
    print(test_3d_cube_sine_dirichlet(2, 'cube100', 'gmres'))
    print('porder', 2, 'cube4591', 'gmres')
    print(test_3d_cube_sine_dirichlet(2, 'cube4591', 'gmres'))

    print('porder', 3, 'cube24', 'direct')
    print(test_3d_cube_sine_dirichlet(3, 'cube24', 'direct'))
    print('porder', 3, 'cube24', 'cg')
    print(test_3d_cube_sine_dirichlet(3, 'cube24', 'cg'))
    print('porder', 3, 'cube100', 'gmres')
    print(test_3d_cube_sine_dirichlet(3, 'cube24', 'gmres'))

    print('porder', 3, 'cube100', 'direct')
    print(test_3d_cube_sine_dirichlet(3, 'cube100', 'direct'))
    print('porder', 3, 'cube100', 'cg')
    print(test_3d_cube_sine_dirichlet(3, 'cube100', 'cg'))
    print('porder', 3, 'cube100', 'gmres')
    print(test_3d_cube_sine_dirichlet(3, 'cube100', 'gmres'))

    print('porder', 3, 'cube4591', 'cg')
    print(test_3d_cube_sine_dirichlet(3, 'cube4591', 'cg'))
    print('porder', 3, 'cube4591', 'gmres')
    print(test_3d_cube_sine_dirichlet(3, 'cube4591', 'gmres'))