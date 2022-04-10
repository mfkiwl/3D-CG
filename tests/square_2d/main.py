import sys
import numpy as np

sys.path.append('../../util')
sys.path.append('../../mesh')
sys.path.append('../../master')
sys.path.append('../../CG')

from mkmesh_sqmesh_gmsh import mkmesh_square
from import_util import load_mat
from cgmesh import cgmesh
from import_util import load_mat
import mkmaster
import cg_solve
import pickle
import logging
import logging.config
import os


def test_2d_square():
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
    # Fewer options in 2D
    case_name = 'square26'
    meshfile = '../data/' + case_name
    porder = 3
    ndim = 2

    outdir = 'out/'
    buildAF = True

    ########## BCs ##########
    # Dirichlet
    dbc = {
        1: 0,
        2: 0,
        3: 0,
        4: 0,
    }

    # Neumann
    nbc = {}


    mesh = mkmesh_square(porder, ndim, meshfile)
    mesh = cgmesh(mesh)
    mesh['dbc'] = dbc
    mesh['nbc'] = nbc

    master = mkmaster.mkmaster(mesh, 2, 2*mesh['porder'])

    param = {'kappa': 0.1, 'c': np.array([1, -2]), 's': 1}

    n = 3
    m = 2

    def exact_square(p):
        x = p[:, 0]
        y = p[:, 1]
        return np.sin(n*np.pi*x) * np.sin(m*np.pi*y)


    def forcing_square(p):
        x = p[:, 0]
        y = p[:, 1]
        forcing_square = param['kappa']*(n**2+m**2)*np.pi*np.pi*(np.sin(n*np.pi*x)*np.sin(m*np.pi*y)) + param['c'][0]*n*np.pi*(np.cos(
            n*np.pi*x)*np.sin(m*np.pi*y)) + param['c'][1]*m*np.pi*(np.sin(n*np.pi*x)*np.cos(m*np.pi*y)) + param['s']*(np.sin(n*np.pi*x)*np.sin(m*np.pi*y))
        return forcing_square[:, None]   # returns as column vector

    uh =  cg_solve.cg_solve(master, mesh, forcing_square, param, 2, outdir, buildAF=buildAF, solver='direct')

    uh = np.squeeze(uh)

    # Reshape into DG high order data structure
    uh_reshaped = np.zeros((mesh['plocal'].shape[0], mesh['t'].shape[0]))
    for i in np.arange(mesh['t'].shape[0]):          # Set dirichlet BC
        uh_reshaped[:, i] = uh[mesh['tcg'][i, :]]
        
    # Compare to matlab
    uh_mat = load_mat('uh2d', 'uh')
    # print('Python solution matches Matlab:', np.allclose(uh_reshaped, uh_mat, rtol=1e-13, atol=4e-15))
    return np.allclose(uh_reshaped, uh_mat, rtol=1e-13, atol=4e-15)
