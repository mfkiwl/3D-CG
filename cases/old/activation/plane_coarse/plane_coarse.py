import sys
import numpy as np

sys.path.insert(0, '../../util')
sys.path.insert(0, '../../mesh')
sys.path.insert(0, '../../master')
sys.path.insert(0, '../../viz')
sys.path.insert(0, '../../CG')
from import_util import load_mat
import viz_driver
from cgmesh import cgmesh
import mkmesh_cube
import mkmaster
import cg_solve
import pickle
import calc_derivative
import logging
import logging.config

########## INITIALIZE LOGGING ##########
logging.config.fileConfig('logging.conf', disable_existing_loggers=False)
# root logger, no __name__ as in submodules further down the hierarchy - this is very important - cost me a lot of time when I passed __name__ to the main logger
logger = logging.getLogger('root')


########## TOP LEVEL SIM SETUP ##########
case_name = '797_coarse'
meshfile = 'mesh/' + case_name
porder = 3
ndim = 3

outdir = 'out/'
vis_filename = 'dirichlet1'
call_pv = False
build_mesh = False
buildAF = False
use_preconditioning = True
vis_filename = outdir+vis_filename

########## BCs ##########
# # Dirichlet
# dbc = {
#     1: 0,
#     2: 0,
#     3: 0,
#     4: 0,
#     5: 0,
#     6: 0,
#     7: 0,
#     8: 0,
#     9: 0,
#     10: 0,
#     11: 0,
#     12: 0,
#     13: 0,
#     14: 0,
#     15: 0,
#     16: 0,
#     17: 0,
#     18: 0,
#     19: 0,
#     20: 0,
#     21: 0,
#     22: 0,
#     23: 0,
#     24: 0,
#     25: 0,
#     26: 0,
#     27: 0,
#     28: 0,
#     29: 0,
#     30: 0,
#     31: 0,
#     32: 0,
#     33: 0,
#     34: 0,
#     35: 0,
#     36: 0,
#     37: 0,
#     38: 0,
#     39: 0,
#     40: 0,
#     41: 0,
# }


# # Neumann
# nbc = {
#     42: -1,   # -X
#     47: 1,   # +X

#     43: 0,   # -Y
#     45: 0,   # +Y

#     46: 0,   # -Z
#     44: 0,   # +Z
# }



# Dirichlet
dbc = {
    1: 1,
    2: 1,
    3: 1,
    4: 1,
    5: 1,
    6: 1,
    7: 1,
    8: 1,
    9: 1,
    10: 1,
    11: 1,
    12: 1,
    13: 1,
    14: 1,
    15: 1,
    16: 1,
    17: 1,
    18: 1,
    19: 1,
    20: 1,
    21: 1,
    22: 1,
    23: 1,
    24: 1,
    25: 1,
    26: 1,
    27: 1,
    28: 1,
    29: 1,
    30: 1,
    31: 1,
    32: 1,
    33: 1,
    34: 1,
    35: 1,
    36: 1,
    37: 1,
    38: 1,
    39: 1,
    40: 1,
    41: 1,
    42: 0,
    43: 0,
    44: 0,
    45: 0,
    46: 0,
    47: 0,
}

# Neumann
nbc = {

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

def forcing_zero(p):
    return np.zeros((p.shape[0], 1))

def approx_sol_x(p):
    x = p[:, 0]
    u1 = -240
    u2 = 390
    x1 = -100
    x2 = 600
    m = (u2-u1)/(x2-x1)
    return m*(x-x1)+u1

def approx_sol_y(p):
    y = p[:, y]
    return 'not implemented'

def approx_sol_z(p):
    z = p[:, 2]
    return 'not implemented'

def approx_sol_charge(p):
    # MAKE SURE TO CHECK THE ORIENTATION OF THE AXES THAT ARE BEING USED!! THIS ASSUMES A SPECIFIC ORIENTATION.

    # Defining cylinder
    rear_p1 = np.array([-50, 292, 403])
    rear_p2 = np.array([-50, 241, 403])
    front_p1 = np.array([400, 292, 403])    # Don't need second front point because all we need are the cylinder radius and center axis

    # Top of wing
    wing_front_right = np.array([217, 103, 405])
    wing_back_right = np.array([166, 103, 405])
    wing_front_left = np.array([217, 433, 405])
    thickness = 5

    def is_in_wing(pfr, pbr, pfl, point):
        is_in_wing_bool = True
        # Check x:
        if point[0] > pfr[0] or point[0] < pbr[0]:
            is_in_wing_bool = False

        # Check y:
        if point[1] < pfr[1] or point[1] > pfl[1]:
            is_in_wing_bool = False

        # No need to check z because if it is above or below the wing in the x-y plane, is it guaranteed to lie outside the wing surface itself.
        return is_in_wing_bool

    approx_potential = np.zeros((p.shape[0]))

    center = (rear_p1+rear_p2)/2    # Center point of rear circular cross section
    r_fuselage = (rear_p1[1]-rear_p2[1])/2

    # Compute r
    for i, pt in enumerate(p):
        x = pt[0]
        y = pt[1]
        z = pt[2]

        if x<rear_p1[0] or x > front_p1[0]:         # Check to see if the point is in front of or behind the fuselage in x
            r = ((y-center[1])**2+(z-center[2])**2)**0.5 # r = sqrt((y-b)^2 + (z-c)^2), pythagorean theorem
            if r<r_fuselage:        # Point is directly in front of or behind the plane's cross section, contained inside cross section in y and z
                if x < rear_p1[0]:
                    dist = np.abs(rear_p1[0]-x)
                elif x > front_p1[0]:
                    dist = np.abs(front_p1[0]-x)

                approx_potential[i] = 1/((dist+r_fuselage)/r_fuselage)**0.8   # Scale length is r_fuselage, and the distance from the surface starts at 1/fuselage

        elif x>rear_p1[0] and x < front_p1[0]:        # Check to see if the point is next to the fuselage in y and z
            r = ((y-center[1])**2+(z-center[2])**2)**0.5 # r = sqrt((y-b)^2 + (z-c)^2), pythagorean theorem
            approx_potential_fuselage = 1/(r/r_fuselage)**0.8     # Will = 1 on surface of fuselage, and anything > 1 is due to the fact that the element surfaces aren't curved and some of them lie inside the fuselage. These points closest to the surface will get taken out by setting the dirichlet condition anyway.

            if is_in_wing(wing_front_right, wing_back_right, wing_front_left, pt):
                if pt[2] > wing_front_right[2]:      # On top of wing
                    dist = pt[2] - wing_front_right[2]
                else:      # On bottom of wing
                    dist = wing_front_right[2]-thickness - pt[2]

                approx_potential_wing = 1/((dist+r_fuselage)/r_fuselage)**0.8   # Scale length is r_fuselage, and the distance from the surface starts at 1/fuselage
                approx_potential[i] = max([approx_potential_fuselage, approx_potential_wing])
            else:
                approx_potential[i] = approx_potential_fuselage

    return approx_potential


# ########## CALCULATE APPROX SOLUTION ##########
logger.info('Computing approximate solution')
approx_x = np.zeros((mesh['pcg'].shape[0]))
for i in np.arange(mesh['tcg'].shape[0]):
    approx_x[mesh['tcg'][i, :]] = approx_sol_charge(mesh['pcg'][mesh['tcg'][i, :], :])

########## SOLVE ##########

sol, x0 = cg_solve.cg_solve(master, mesh, forcing_zero, param, ndim, outdir, approx_x, buildAF, use_preconditioning, case_name)

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

approx_reshaped = np.zeros((mesh['plocal'].shape[0], mesh['t'].shape[0]))
for i in np.arange(mesh['t'].shape[0]):
    if i%1000 == 0:
        logger.info(str(i)+'/'+str(mesh['t'].shape[0]))
    approx_reshaped[:, i] = x0[mesh['tcg'][i, :]]
approx_reshaped = approx_reshaped[:,None,:]

logger.info('Calculating derivatives')
# Reshape into DG high order data structure
sol_reshaped = np.zeros((mesh['plocal'].shape[0], mesh['t'].shape[0]))
for i in np.arange(mesh['t'].shape[0]):          # Set dirichlet BC
    sol_reshaped[:, i] = sol[mesh['tcg'][i, :], 0]

grad = calc_derivative.calc_derivatives(mesh, master, sol_reshaped, ndim)
result = np.concatenate((sol_reshaped[:,None,:], grad.transpose(1,2,0), approx_reshaped), axis=1)

########## VISUALIZE SOLUTION ##########
logger.info('Running visualization script...')
viz_driver.viz_driver(mesh, master, result, vis_filename, call_pv)