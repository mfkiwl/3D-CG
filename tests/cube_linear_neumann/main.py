import sys
sys.path.append('../../util')
sys.path.append('../../mesh')
sys.path.append('../../master')
sys.path.append('../../viz')
sys.path.append('../../CG')
import numpy as np
from import_util import load_mat
# import viz_driver
from cgmesh import cgmesh
import mkmesh_cube
import mkmaster
import cg_solve
import pickle
import calc_derivative
import os
import logging
import logging.config

def exact_linear(p, axis):
    if axis == 'x':
        return p[:,0]
    elif axis == 'y':
        return p[:,1]
    elif axis == 'z':
        return p[:, 2]

def forcing_zero(p):
    return np.zeros((p.shape[0],1))

def test_3d_cube_linear_neumann(porder, meshfile, solver):

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
    vis_filename = 'cube_sol'
    call_pv = False
    build_mesh = True
    buildAF = True
    vis_filename = outdir+vis_filename
    ndim = 3

    # NOTE: Bugs with orders 4, 5, and 6 here in various parts
    # viz, and assigning BCs in accessing loc_face_nodes

    ########## BCs ##########
    # Dirichlet
    dbc = {
        6: 0,   # -X
    }

    # Neumann
    nbc = {
        4: 1,   # +X
        3: 0,   # -Y
        5: 0,   # +Y
        1: 0,   # -Z
        2: 0,   # +Z
    }

    # Eventually: add support for the neumann condition being entered as a vector dot product
    # Long term: neumann robin BC

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

    sol = cg_solve.cg_solve(master, mesh, forcing_zero, param, ndim, outdir, buildAF=True, solver=solver)

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
    exact = exact_linear(mesh['pcg'], 'x')
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

    # logger.info('Calculating derivatives')
    # grad = calc_derivative.calc_derivatives(mesh, master, sol_reshaped, ndim)
    # result = np.concatenate((sol_reshaped[:,None,:], grad.transpose(1,2,0)), axis=1)

    # # ########## VISUALIZE SOLUTION ##########
    # viz_driver.viz_driver(mesh, master, result, vis_filename, call_pv)
    return norm_error