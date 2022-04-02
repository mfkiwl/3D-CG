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
import os

########## INITIALIZE LOGGING ##########
logging.config.fileConfig('logging.conf', disable_existing_loggers=False)
# root logger, no __name__ as in submodules further down the hierarchy - this is very important - cost me a lot of time when I passed __name__ to the main logger
logger = logging.getLogger('root')
logger.info('*************************** INITIALIZING SIM ***************************')

# Check whether the specified path exists or not
if not os.path.exists('out/'):
    os.makedirs('out/')
    logger.info('out/ directory not present, created...')

########## TOP LEVEL SIM SETUP ##########
meshfile = 'mesh/' + 'boeing_plane_final'     # No file extension!
scale_factor =  # INPUT SCALE FACTOR HERE!!!
porder = 3
ndim = 3
solver = 'cg'

# CHANGE THIS FOR THE NEXT SIM! And make the meshes go here too!
outdir = 'out/'
vis_filename = 'boeing_plane_Ex'
build_mesh = True
buildAF = True
use_preconditioning = True
compute_sol = True
call_pv = False

vis_filename = outdir+vis_filename

########## BCs ##########
# Dirichlet
dbc = {
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
    6: 0,
    7: 0,
    8: 0,
    9: 0,
    10: 0,
    11: 0,
    12: 0,
    13: 0,
    14: 0,
    15: 0,
    16: 0,
    17: 0,
    18: 0,
    19: 0,
    20: 0,
    21: 0,
    22: 0,
    23: 0,
    24: 0,
    25: 0,
    26: 0,
    27: 0,
    28: 0,
    29: 0,
    30: 0,
    31: 0,
    32: 0,
    33: 0,
    34: 0,
    35: 0,
    36: 0,
    37: 0,
    38: 0,
    39: 0,
    40: 0,
    41: 0,
    42: 0,
    43: 0,
    44: 0,
    45: 0,
    46: 0,
    47: 0,
    48: 0,
    49: 0,
    50: 0,
    51: 0,
    52: 0,
    53: 0,
    54: 0,
    55: 0,
    56: 0,
    57: 0,
    58: 0,
    59: 0,
    60: 0,
    61: 0,
    62: 0,
    63: 0,
    64: 0,
    65: 0,
    66: 0,
    67: 0,
    68: 0,
    69: 0,
    70: 0,
    71: 0,
    72: 0,
    73: 0,
    74: 0,
    75: 0,
    76: 0,
    77: 0,
    78: 0,
    79: 0,
    80: 0,
    81: 0,
    82: 0,
    83: 0,
    84: 0,
    85: 0,
    86: 0,
    87: 0,
    88: 0,
    89: 0,
    90: 0,
    91: 0,
    92: 0,
    93: 0,
    94: 0,
    95: 0,
    96: 0,
    97: 0,
    98: 0,
    99: 0,
    100: 0,
    101: 0,
    102: 0,
    103: 0,
    104: 0,
    105: 0,
    106: 0,
    107: 0,
    108: 0,
    109: 0,
    110: 0,
    111: 0,
    112: 0,
    113: 0,
    114: 0,
    115: 0,
    116: 0,
    117: 0,
    118: 0,
    119: 0,
    120: 0,
    121: 0,
    122: 0,
    123: 0,
    124: 0,
    125: 0,
    126: 0,
    127: 0,
    128: 0,
    129: 0,
    130: 0,
    131: 0,
    132: 0,
    133: 0,
    134: 0,
    135: 0,
    136: 0,
    137: 0,
}

# Neumann
nbc = {
    138: -1,   # -X
    139: 1,   # +X

    140: 0,   # -Y
    141: 0,   # +Y

    142: 0,   # -Z
    143: 0,   # +Z
}


# Eventually: add support for the neumann condition being entered as a vector dot product
# Long term: neumann robin BC

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


if compute_sol:
    ########## CREATE MESH ##########
    mesh = mkmesh_cube.mkmesh_cube(porder, ndim, meshfile, build_mesh, scale_factor)
    logger.info('Converting high order mesh to CG...')
    mesh = cgmesh(mesh)
    mesh['dbc'] = dbc
    mesh['nbc'] = nbc

    logger.info('Preparing master data structure...')
    master = mkmaster.mkmaster(mesh, ndim=3, pgauss=2*mesh['porder'])

    # ########## CALCULATE APPROX SOLUTION ##########
    # logger.info('Computing approximate solution')
    # approx_x = np.zeros((mesh['pcg'].shape[0]))
    # for i in np.arange(mesh['tcg'].shape[0]):
    #     approx_x[mesh['tcg'][i, :]] = approx_sol_charge(mesh['pcg'][mesh['tcg'][i, :], :])

    ########## SOLVE ##########

    sol, x0 = cg_solve.cg_solve(master, mesh, forcing_zero, param, ndim, outdir, None, buildAF, solver)

    ########## SAVE DATA ##########

    # NOTE: in the future uh will need to be reshaped into a nplocal x numvisfields x numel when the derivatives are added
    with open(outdir+'mesh', 'wb') as file:
        pickle.dump(mesh, file)
    with open(outdir+'master', 'wb') as file:
        pickle.dump(master, file)
    with open(outdir+'sol', 'wb') as file:
        pickle.dump(sol, file)
    with open(outdir+'x0', 'wb') as file:
        pickle.dump(x0, file)
    logger.info('Wrote solution to file...')
else:
    ########## LOADING SOLUTION ##########

    logger.info('Reading solution from file...')

    with open(outdir+'mesh', 'rb') as file:
        mesh = pickle.load(file)
    with open(outdir+'master', 'rb') as file:
        master = pickle.load(file)
    with open(outdir+'sol', 'rb') as file:
        sol = pickle.load(file)
    with open(outdir+'x0', 'rb') as file:
        x0 = pickle.load(file)

# approx_reshaped = np.zeros((mesh['plocal'].shape[0], mesh['t'].shape[0]))
# for i in np.arange(mesh['t'].shape[0]):
#     if i%10000 == 0:
#         logger.info(str(i)+'/'+str(mesh['t'].shape[0]))
#     approx_reshaped[:, i] = approx_x[mesh['tcg'][i, :]]     # Change back to x0!!
# approx_reshaped = approx_reshaped[:,None,:]

########## CALC DERIVATIVES ##########
logger.info('Calculating derivatives')
# Reshape into DG high order data structure
sol_reshaped = np.zeros((mesh['plocal'].shape[0], mesh['t'].shape[0]))
for i in np.arange(mesh['t'].shape[0]):          # Set dirichlet BC
    sol_reshaped[:, i] = sol[mesh['tcg'][i, :], 0]

grad = calc_derivative.calc_derivatives(mesh, master, sol_reshaped, ndim)
result = np.concatenate((sol_reshaped[:,None,:], grad.transpose(1,2,0)), axis=1)

########## VISUALIZE SOLUTION ##########
logger.info('Running visualization script...')
viz_driver.viz_driver(mesh, master, result, vis_filename, call_pv)