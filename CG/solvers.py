import numpy as np
import scipy.sparse.linalg as splinalg
import pyamg
import logging
import time
from functools import partial

logger = logging.getLogger(__name__)
iter_count = 0
last_x = 0

def solve(A, F, tol, solver):
    if solver=='cg':
        sol = solve_cg(A, F, tol)
    elif solver=='gmres':
        sol = solve_gmres(A, F, tol)

    elif solver=='amg':
        sol = solve_amg(A, F, tol)

    elif solver == 'direct':
        sol = solve_direct(A, F, tol)
    
    return sol

def call_iter(A_csr, b, tol, x):
    global iter_count
    global last_x
    if iter_count %1 == 0:
        residual = A_csr@x[:,None] - b
        res_norm = np.linalg.norm(residual)
        stopping = tol*np.linalg.norm(b)
        error_factor = res_norm/stopping
        delta_x = np.linalg.norm(x-last_x)
        last_x = x
        logger.info('Iteration ' + str(iter_count) + ', current residual norm is {:.5E}, {:.5E} req\'d for stopping, ratio: {:.3f}, norm(Delta x)={:.5E}'.format(res_norm, stopping, error_factor, delta_x))
    iter_count += 1
    last_x = x

def solve_direct(A, F, tol=None):
    logger.info('Solving directly...')

    sol = np.linalg.solve(A.todense(), F)

    return sol

def solve_amg(A, F, solver_tol):
    global last_x
    last_x = 0
    logger.info('Solving with AMG...')
    A_csr = A.tocsr()
    ml = pyamg.ruge_stuben_solver(A_csr, max_levels=20)                    # construct the multigrid hierarchy
    start = time.perf_counter()
    sol = ml.solve(F, x0=None, tol=solver_tol, maxiter=None, callback=partial(call_iter, A_csr, F, solver_tol))
    logger.info('Solution time: '+ str(round(time.perf_counter()-start, 5)) +'s')

    return sol


def solve_cg(A, F, solver_tol):
    global last_x
    last_x = 0
    logger.info('Solving with conjugate gradient...')
    A_csr = A.tocsr()

    logger.info('Prior to solving: Residual norm is {:.5E}'.format(np.linalg.norm(np.linalg.norm(F))))

    ml = pyamg.ruge_stuben_solver(A_csr, max_levels=20)    # Multigrid preconditioner
    P = ml.aspreconditioner()

    start = time.perf_counter()
    res = splinalg.cg(A, F, M=P, x0=None, tol=solver_tol, callback=partial(call_iter, A_csr, F, solver_tol), atol=0)
    logger.info('Solution time: '+ str(round(time.perf_counter()-start, 5)) +'s')

    if not res[1]:    # Successful exit
        sol = res[0][:,None]
        logger.info('Successfully solved with CG...')
        logger.info('Solution time: '+ str(round(time.perf_counter()-start, 5)) +'s')
    else:
        logger.error('CG did not converge, reached ' +str(res[1]) + ' iterations')
        raise ValueError('CG did not converge, reached ' +str(res[1]) + ' iterations')

    return sol

def solve_gmres(A, F, solver_tol):
    global last_x
    last_x = 0
    logger.info('Solving with GMRES...')
    A_csr = A.tocsr()

    ml = pyamg.ruge_stuben_solver(A_csr, max_levels=20)    # Multigrid preconditioner
    P = ml.aspreconditioner()

    start = time.perf_counter()
    res = splinalg.gmres(A, F, M=P, x0=None, tol=solver_tol, callback=partial(call_iter, A_csr, F, solver_tol), atol=0, callback_type='x', restart=20)
    logger.info('Solution time: '+ str(round(time.perf_counter()-start, 5)) +'s')

    if not res[1]:    # Successful exit
        sol = res[0][:,None]
        logger.info('Successfully solved with GMRES...')
        logger.info('Solution time: '+ str(round(time.perf_counter()-start, 5)) +'s')
    else:
        logger.error('GMRES did not converge, reached ' +str(res[1]) + ' iterations')
        raise ValueError('GMRES did not converge, reached ' +str(res[1]) + ' iterations')
    return sol

