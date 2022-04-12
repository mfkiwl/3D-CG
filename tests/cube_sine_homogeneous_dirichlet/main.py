import sys
sys.path.append('../../util')
sys.path.append('../../mesh')
sys.path.append('../../master')
# sys.path.append('../../viz/old2')
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

def test_3d_cube_sine_homoegeneous_dirichlet(porder, meshfile, solver):

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
    vis_filename = outdir+'solution_' + vis_filename
    ndim = 3
    call_pv = False
    viz_labels = {'scalars': {0: 'Solution'}, 'vectors': {0: 'Solution Gradient'}}
    visorder = porder
    solver_tol=1e-10
    
    # NOTE: Bugs with orders 4, 5, and 6 here in various parts

    ########## BCs ##########
    # Dirichlet
    dbc = {
    1: 0,   # -X
    2: 0,   # +X
    3: 0,   # +Z
    4: 0,   # -Z
    5: 0,   # -Y
    6: 0,   # -Z
    }

    # Neumann
    nbc = {}

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

    sol = cg_solve.cg_solve(master, mesh, domain_helper_fcns.forcing_sine_cube, param, ndim, outdir, buildAF=True, solver=solver, solver_tol=solver_tol)

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
    exact = domain_helper_fcns.exact_sine_cube(mesh['pcg'])
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

    grad = calc_derivative.calc_derivatives(mesh, master, sol_reshaped, ndim)

    result_out = np.concatenate((sol_reshaped, grad), axis=1)

    with open(vis_filename + '.npy', 'wb') as file:
        np.save(file, result_out)
    logger.info('Wrote solution to /out')

    # ########## VISUALIZE SOLUTION ##########
    # Might have to tweak dimensions based on how the viz functions in the HPC version handle it
    viz.visualize(mesh, visorder, viz_labels, vis_filename, call_pv, scalars=sol[:,None], vectors=grad[None,:,:])

    # Old version of viz scripts
    # logger.info('Calculating derivatives')
    # grad = calc_derivative.calc_derivatives(mesh, master, sol_reshaped, ndim)
    # import viz_driver
    # viz_driver.viz_driver(mesh, sol_reshaped[:,None,:], 'cuong_viz', True)
    
    return norm_error