import numpy as np
import gradcalc2
from functools import partial
import multiprocessing as mp
from functools import partial
from pathos.multiprocessing import ProcessingPool as Pool

def calc_derivatives(mesh, master, sol, ndim):

    nelem = mesh['t'].shape[0]
    pool = Pool(mp.cpu_count())
    result = pool.map(partial(gradcalc2.grad_calc, mesh['dgnodes'], master, sol), np.arange(nelem))
    # result = np.asarray(list(map(partial(grad_calc, mesh['dgnodes'], master, sol), np.arange(nelem))))
    grad = np.asarray(result)
    # mag = np.linalg.norm(grad, ord=2, axis=2)[:,:,None]

    # return np.concatenate((grad, mag), axis=2)
    return grad