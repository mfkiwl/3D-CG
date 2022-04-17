import numpy as np
import sys
sys.path.append('../../util')
import gmsh
from gmsh import model as gm
import pickle
import helper
import logging
import multiprocessing as mp
from functools import partial
import time

logger = logging.getLogger(__name__)


def forcing_zero(p):
    return np.zeros((p.shape[0], 1))

def exact_linear(p, axis):
    if axis == 'x':
        return p[:,0]
    elif axis == 'y':
        return p[:,1]
    elif axis == 'z':
        return p[:, 2]

def exact_cube_mod_sine(p, m=1.5):
    x = p[:, 0]
    y = p[:, 1]
    z = p[:, 2]
    exact = np.sin(m*np.pi*x)# * np.sin(n*np.pi*y) * np.sin(l*np.pi*z)
    return exact

def forcing_cube_mod_sine(p, m=1.5):
    forcing_cube = (m**2)*np.pi**2*exact_cube_mod_sine(p, m)        # We can do this because of the particular sin functions chosen for the exact solution
    return forcing_cube[:, None]   # returns as column vector

def exact_sine_cube(p, m=1):

    x = p[:, 0]
    y = p[:, 1]
    z = p[:, 2]
    exact = np.sin(m*np.pi*x) * np.sin(m*np.pi*y) * np.sin(m*np.pi*z)
    return exact

def forcing_sine_cube(p, m=1):
    # Note: doesn't take kappa into account, might add in later

    forcing_cube = (m**2+m**2+m**2)*np.pi**2*exact_sine_cube(p)        # We can do this because of the particular sin functions chosen for the exact solution
    return forcing_cube[:, None]   # returns as column vector

def grad_linear_solution(p, axis):
    grad = np.zeros_like(p)
    if axis == 'x':
        grad[:,0] = 1
    if axis == 'y':
        grad[:,1] = 1
    if axis == 'z':
        grad[:,2] = 1

    return grad

def grad_sine_1d(p, m, axis):
    grad = np.zeros_like(p)
    if axis == 'x':
        grad[:,0] = m*np.pi*np.cos(m*np.pi*p[:,0])
    if axis == 'y':
        grad[:,1] = m*np.pi*np.cos(p[:,1])
    if axis == 'z':
        grad[:,2] = m*np.pi*np.cos(p[:,2])

    return grad

def grad_sine_3d(p, m):
    # Assumes the scaling factor for the sine function is constant in all axes
    grad = np.zeros_like(p)
    grad[:,0] = m*np.pi*np.cos(m*np.pi*p[:,0])*np.sin(m*np.pi*p[:,1])*np.sin(m*np.pi*p[:,2])
    grad[:,1] = m*np.pi*np.sin(m*np.pi*p[:,0])*np.cos(m*np.pi*p[:,1])*np.sin(m*np.pi*p[:,2])
    grad[:,2] = m*np.pi*np.sin(m*np.pi*p[:,0])*np.sin(m*np.pi*p[:,1])*np.cos(m*np.pi*p[:,2])

    return grad

# Deprecated functions for computing the approximate solution
def get_distances(entity_dim, pts, surf_idx):
        logger.info('Getting distances for surface ' + str(surf_idx))
        pts_on_surf = gm.getClosestPoint(entity_dim, surf_idx, pts.ravel())[0].reshape((pts.shape[0],3))
        return np.linalg.norm(pts_on_surf-pts, axis=1)[:,None]

def batch_distance_to_body_gmsh(stepfile, pts, surfaces, sf):

    gmsh.initialize()
    gm.add('model')
    gm.occ.importShapes(stepfile)

    tags = gm.occ.getEntities(-1)
    gm.occ.dilate(tags, 0, 0, 0, sf, sf, sf)
    gm.occ.synchronize()

    distance_arry = np.zeros((pts.shape[0], len(surfaces)))

    start = time.perf_counter()

    # Uncomment for parallel processing
    with mp.Pool(mp.cpu_count()) as pool:
        distance_arry = np.hstack(pool.map(partial(get_distances, 2, pts), surfaces))

    logger.info('Time to calculate distance to body for all points: ' + str(time.perf_counter() - start) + 's')

    min_distances = np.min(distance_arry, axis=1)[:,None]    # Return as a column vector 

    gmsh.finalize()

    return min_distances

def approx_sol_charge(mesh):
    surfaces = np.asarray(mesh['body_surfs']) # Fill this out

    logger.info('Calling gmsh batch distance function...')
    distances = batch_distance_to_body_gmsh(mesh['stepfile'], mesh['p'], surfaces, mesh['scale_factor'])
    approx_charge = 1/(1+distances)**0.8    # Coords are scaled relative to the fuselage diameter - origin needs to be at the center of the fuselage to prevent singularity on surface

    logger.info('Interpolating approximate solution to high order mesh...')
    approx_charge = helper.reshape_field(mesh, approx_charge, 'to_array', 'scalars', porder=1)
    high_order_interp, __ = helper.interpolate_high_order(1, mesh['porder'], mesh['ndim'], approx_charge)
    approx_charge = helper.reshape_field(mesh, high_order_interp, 'to_column', 'scalars')

    return approx_charge

def linear_fcn(x1, x2, u1, u2, x):
    m = (u2-u1)/(x2-x1)
    return m*(x-x1)+u1

def approx_sol_E_field(mesh, case, u1, u2):
    if (u1 is None) or (u2 is None):
        raise ValueError('u1 and u2 cannot be None')
    if case == 'Ex':
        x1 = mesh['bbox_after_scale']['x'][0]
        x2 = mesh['bbox_after_scale']['x'][1]
        p = mesh['pcg'][:, 0]
    elif case == 'Ey':
        x1 = mesh['bbox_after_scale']['y'][0]
        x2 = mesh['bbox_after_scale']['y'][1]
        p = mesh['pcg'][:, 1]
    elif case == 'Ez':
        x1 = mesh['bbox_after_scale']['z'][0]
        x2 = mesh['bbox_after_scale']['z'][1]
        p = mesh['pcg'][:, 2]

    return linear_fcn(x1, x2, u1, u2, p)[:,None]


if __name__ == '__main__':
    with open('./mesh/boeing_', 'rb') as file:
        mesh = pickle.load(file)

    gmsh.fltk.run()
