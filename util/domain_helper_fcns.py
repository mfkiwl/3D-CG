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

def forcing_zero(p):
    return np.zeros((p.shape[0], 1))

# def approx_sol_x(mesh):
#     x1 = mesh['bbox_after_scale']['x'][0]
#     x2 = mesh['bbox_after_scale']['x'][1]
#     u1 = None
#     u2 = None

#     x = mesh['pcg'][:, 0]
    
#     u_approx = linear_fcn(x1, x2, u1, u2, x)

#     return u_approx

# def approx_sol_y(mesh, p):
#     y1 = mesh['bbox_after_scale']['y'][0]
#     y2 = mesh['bbox_after_scale']['y'][1]
#     u1 = None
#     u2 = None

#     y = p[:, 0]
    
#     u_approx = linear_fcn(y1, y2, u1, u2, y)

#     return u_approx

# def approx_sol_z(mesh, p):
#     z1 = mesh['bbox_after_scale']['z'][0]
#     z2 = mesh['bbox_after_scale']['z'][1]
#     u1 = None
#     u2 = None

#     z = p[:, 0]
    
#     u_approx = linear_fcn(z1, z2, u1, u2, z)

#     return u_approx

if __name__ == '__main__':
    with open('./mesh/boeing_', 'rb') as file:
        mesh = pickle.load(file)

    gmsh.fltk.run()
