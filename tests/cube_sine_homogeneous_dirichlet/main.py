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


def exact_cube(p):
    m = 1
    n = 1
    l = 1

    x = p[:, 0]
    y = p[:, 1]
    z = p[:, 2]
    exact = np.sin(m*np.pi*x) * np.sin(n*np.pi*y) * np.sin(l*np.pi*z)
    return exact

def forcing_cube(p):
    # Note: doesn't take kappa into account, might add in later
    m = 1
    n = 1
    l = 1

    forcing_cube = (m**2+n**2+l**2)*np.pi**2*exact_cube(p)        # We can do this because of the particular sin functions chosen for the exact solution
    return forcing_cube[:, None]   # returns as column vector

def forcing_zero(p):
    return np.zeros((p.shape[0], 1))

def test_3d_cube_sine_homoegeneous_dirichlet(porder, meshfile, solver):
    ########## INITIALIZE LOGGING ##########
    logging.config.fileConfig('../../logging/logging.conf', disable_existing_loggers=False)
    # root logger, no __name__ as in submodules further down the hierarchy - this is very important - cost me a lot of time when I passed __name__ to the main logger
    logger = logging.getLogger('root')
    logger.info('*************************** INITIALIZING SIM ***************************')

    # Check whether the specified path exists or not
    if not os.path.exists('out/'):
        os.makedirs('out/')
        logger.info('out/ directory not present, created...')


    ########## TOP LEVEL SIM SETUP ##########
    outdir = 'out/'
    meshfile = '../data/'+meshfile
    vis_filename = 'cube_sol'
    call_pv = True
    build_mesh = True
    buildAF = True
    vis_filename = outdir+vis_filename
    porder = 3
    ndim = 3
    visorder = 6
    labels = {'scalars': {0: 'Temperature'}, 'vectors': {0: 'Gradient'}}

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

    # sol = cg_solve(master, mesh, forcing_cube, param, ndim, outdir, buildAF, use_preconditioning)
    sol, _ = cg_solve.cg_solve(master, mesh, forcing_cube, param, ndim, outdir, approx_sol=None, buildAF=buildAF, solver=solver)

    ########## SAVE DATA ##########

    # NOTE: in the future uh will need to be reshaped into a nplocal x numvisfields x numel when the derivatives are added
    with open(outdir+'mesh', 'wb') as file:
        pickle.dump(mesh, file)
    with open(outdir+'master', 'wb') as file:
        pickle.dump(master, file)
    with open(outdir+'sol', 'wb') as file:
        pickle.dump(sol, file)
    logger.info('Wrote solution to file...')


    ########## ERROR CALCULATION ##########
    exact = exact_cube(mesh['pcg'])
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
    grad = calc_derivative.calc_derivatives(mesh, master, sol_reshaped, ndim)[None,:,:]

    # ########## VISUALIZE SOLUTION ##########
    viz.visualize(mesh, visorder, labels, vis_filename, call_pv, scalars=sol[:,None], vectors=grad)

    return norm_error