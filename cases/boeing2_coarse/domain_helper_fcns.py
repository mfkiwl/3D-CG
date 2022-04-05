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

# def setup_gmsh_model(mesh):
#     gmsh.initialize()
#     gm.add('model')
#     gm.occ.importShapes(mesh['meshfile'])
#     gm.occ.dilate((3, 1), 0, 0, 0, mesh['scale_factor'], mesh['scale_factor'], mesh['scale_factor'])

#     # bbox = gm.occ.getBoundingBox()
#     gm.occ.synchronize()

#     return gm

def get_distances(entity_dim, pts, surf_idx):
        print(surf_idx)
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

    # distance_arry = np.hstack(list(map(partial(get_distances, 2, pts), surfaces)))
    print(time.perf_counter() - start)

    min_distances = np.min(distance_arry, axis=1)[:,None]    # Return as a column vector 

    gmsh.finalize()

    return min_distances

def linear_fcn(x1, x2, u1, u2, x):
    m = (u2-u1)/(x2-x1)
    return m*(x-x1)+u1

def forcing_zero(mesh, p):
    return np.zeros((p.shape[0], 1))

def approx_sol_x(mesh, p):
    x1 = mesh['bbox_after_scale']['x'][0]
    x2 = mesh['bbox_after_scale']['x'][1]
    u1 = None
    u2 = None

    x = p[:, 0]
    
    u_approx = linear_fcn(x1, x2, u1, u2, x)

    return u_approx

def approx_sol_y(mesh, p):
    y1 = mesh['bbox_after_scale']['y'][0]
    y2 = mesh['bbox_after_scale']['y'][1]
    u1 = None
    u2 = None

    y = p[:, 0]
    
    u_approx = linear_fcn(y1, y2, u1, u2, y)

    return u_approx


def approx_sol_z(mesh, p):
    z1 = mesh['bbox_after_scale']['z'][0]
    z2 = mesh['bbox_after_scale']['z'][1]
    u1 = None
    u2 = None

    z = p[:, 0]
    
    u_approx = linear_fcn(z1, z2, u1, u2, z)

    return u_approx


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

# def approx_sol_charge(p):

#     # Assumes the plane is aligned with the x-direction

#     def is_in_wing(pfr, pbr, pfl, point):
#         is_in_wing_bool = True
#         # Check x:
#         if point[0] > pfr[0] or point[0] < pbr[0]:
#             is_in_wing_bool = False

#         # Check y:
#         if point[1] < pfr[1] or point[1] > pfl[1]:
#             is_in_wing_bool = False

#         # No need to check z because if it is above or below the wing in the x-y plane, is it guaranteed to lie outside the wing surface itself.
#         return is_in_wing_bool

#     approx_potential = np.zeros((p.shape[0]))

#     # Defining cylinder
#     rear_p1 = np.array([-50, 292, 403])
#     rear_p2 = np.array([-50, 241, 403])
#     front_p1 = np.array([400, 292, 403])    # Don't need second front point because all we need are the cylinder radius and center axis

#     # Top of wing
#     wing_front_right = np.array([217, 103, 405])
#     wing_back_right = np.array([166, 103, 405])
#     wing_front_left = np.array([217, 433, 405])
#     thickness = 5

#     center = (rear_p1+rear_p2)/2    # Center point of rear circular cross section
#     r_fuselage = (rear_p1[1]-rear_p2[1])/2

#     # Compute r
#     for i, pt in enumerate(p):
#         x = pt[0]
#         y = pt[1]
#         z = pt[2]

#         if x<rear_p1[0] or x > front_p1[0]:         # Check to see if the point is in front of or behind the fuselage in x
#             r = ((y-center[1])**2+(z-center[2])**2)**0.5 # r = sqrt((y-b)^2 + (z-c)^2), pythagorean theorem
#             if r<r_fuselage:        # Point is directly in front of or behind the plane's cross section, contained inside cross section in y and z
#                 if x < rear_p1[0]:
#                     dist = np.abs(rear_p1[0]-x)
#                 elif x > front_p1[0]:
#                     dist = np.abs(front_p1[0]-x)

#                 approx_potential[i] = 1/((dist+r_fuselage)/r_fuselage)**0.8   # Scale length is r_fuselage, and the distance from the surface starts at 1/fuselage

#         elif x>rear_p1[0] and x < front_p1[0]:        # Check to see if the point is next to the fuselage in y and z
#             r = ((y-center[1])**2+(z-center[2])**2)**0.5 # r = sqrt((y-b)^2 + (z-c)^2), pythagorean theorem
#             approx_potential_fuselage = 1/(r/r_fuselage)**0.8     # Will = 1 on surface of fuselage, and anything > 1 is due to the fact that the element surfaces aren't curved and some of them lie inside the fuselage. These points closest to the surface will get taken out by setting the dirichlet condition anyway.

#             if is_in_wing(wing_front_right, wing_back_right, wing_front_left, pt):
#                 if pt[2] > wing_front_right[2]:      # On top of wing
#                     dist = pt[2] - wing_front_right[2]
#                 else:      # On bottom of wing
#                     dist = wing_front_right[2]-thickness - pt[2]

#                 approx_potential_wing = 1/((dist+r_fuselage)/r_fuselage)**0.8   # Scale length is r_fuselage, and the distance from the surface starts at 1/fuselage
#                 approx_potential[i] = max([approx_potential_fuselage, approx_potential_wing])
#             else:
#                 approx_potential[i] = approx_potential_fuselage

#     return approx_potential


if __name__ == '__main__':
    with open('./mesh/boeing_', 'rb') as file:
        mesh = pickle.load(file)


    gmsh.fltk.run()
