import sys
from pathlib import Path

# Finding the sim root directory
cwd = Path.cwd()
for dirname in tuple(cwd.parents):
    if dirname.name == '3D-CG':
        sim_root_dir = dirname
        continue

sys.path.append(str(sim_root_dir.joinpath('util')))
sys.path.append(str(sim_root_dir.joinpath('mesh')))
sys.path.append(str(sim_root_dir.joinpath('master')))
sys.path.append(str(sim_root_dir.joinpath('viz')))
sys.path.append(str(sim_root_dir.joinpath('CG')))
sys.path.append(str(sim_root_dir.joinpath('logging')))
import logger_cfg
import logging
import datetime
import yaml

########## INITIALIZE LOGGING ##########
SLURM_JOB_ID = sys.argv[1]
SLURM_JOB_NAME = sys.argv[2]
SLURM_CPUS_ON_NODE = sys.argv[3]
SLURM_JOB_NUM_NODES = sys.argv[4]
SLURM_JOB_NODELIST = sys.argv[5]
SLURM_NTASKS = sys.argv[6]
SLURM_SUBMIT_DIR = sys.argv[7]
config_file = sys.argv[8]

with open(config_file, 'r') as stream:
    config_dict = yaml.load(stream, Loader=yaml.loader.FullLoader)

casename = config_dict['casename']
meshfile = config_dict['meshfile']
case_select = config_dict['case_select']
outdir = config_dict['outdir']
process_mesh = config_dict['process_mesh']
buildAF = config_dict['buildAF']
compute_sol = config_dict['compute_sol']
call_pv = config_dict['call_pv']
ndim = int(config_dict['ndim'])
porder = int(config_dict['porder'])
solver = config_dict['solver']
solver_tol = float(config_dict['solver_tol'])
visorder = int(config_dict['visorder'])
viz_labels = config_dict['viz_labels']
fuselage_dia = float(config_dict['fuselage_dia'])
fuselage_pts = config_dict['fuselage_pts']
x_minus_face = config_dict['x_minus_face_index']
x_plus_face = config_dict['x_plus_face_index']
y_minus_face = config_dict['y_minus_face_index']
y_plus_face = config_dict['y_plus_face_index']
z_minus_face = config_dict['z_minus_face_index']
z_plus_face = config_dict['z_plus_face_index']
phys_param = config_dict['phys_param']

logger = logger_cfg.initialize_logger(casename)
logger.info('*************************** INITIALIZING SIM ' +str(datetime.datetime.now())+' ***************************')
logger.info('Starting imports...')

import numpy as np
import viz
from cgmesh import cgmesh
import mkmesh_cube
import mkmaster
import cg_solve
import pickle
import calc_derivative
import os
import helper
import domain_helper_fcns

logger.info('SLURM_JOB_ID: ' + SLURM_JOB_ID)
logger.info('SLURM_JOB_NAME: ' + SLURM_JOB_NAME)
logger.info('SLURM_CPUS_ON_NODE: ' + SLURM_CPUS_ON_NODE)
logger.info('SLURM_JOB_NODELIST: ' + SLURM_JOB_NODELIST)
logger.info('SLURM_NTASKS: ' + SLURM_NTASKS)
logger.info('SLURM_SUBMIT_DIR: ' + SLURM_SUBMIT_DIR)
logger.info('Config file path: ' + config_file)

try:
    # Check whether the specified path exists or not
    if not os.path.exists('out/'):
        os.makedirs('out/')
        logger.info('out/ directory not present, created...')

    ########## TOP LEVEL SIM SETUP ##########
    d_fuselage_msh= np.linalg.norm(np.asarray(fuselage_pts[0])-np.asarray(fuselage_pts[1]))
    scale_factor=  fuselage_dia/d_fuselage_msh    # Normalize mesh by the fuselage radius and rescale so that mesh dimensions are in meters

    vis_filename = outdir+casename

    bdry = (x_minus_face, x_plus_face, y_minus_face, y_plus_face, z_minus_face, z_plus_face)
    surf_faces = np.arange(min(bdry)-1)+1         # Number of faces on aircraft body is implied from the value of the min face, which is assigned after the aircraft surfaces in the mesh generator script
    nbc = {face:0 for face in bdry}

    if case_select == 'Phi':
        dbc = {face:1 for face in surf_faces}
        dbc.update(nbc)     # Concatenating two dictionaries together
        nbc = {}
    elif case_select == 'Ex':
        dbc = {face:0 for face in surf_faces}
        nbc[x_minus_face] = -1
        nbc[x_plus_face] = 1
    elif case_select == 'Ey':
        dbc = {face:0 for face in surf_faces}
        nbc[y_minus_face] = -1
        nbc[y_plus_face] = 1
    elif case_select == 'Ez':
        dbc = {face:0 for face in surf_faces}
        nbc[z_minus_face] = -1
        nbc[z_plus_face] = 1

    ########## LOGGING SIM PARAMETERS ##########
    logger.info('Case type: '+case_select)
    logger.info('Dim: '+str(ndim))
    logger.info('porder: '+str(porder))
    logger.info('Solver: ' + solver)
    logger.info('Solver tolerance: ' + str(solver_tol))
    logger.info('Mesh scale factor: ' + str(scale_factor))
    logger.info('Dirichlet BCs: ' + str(dbc))
    logger.info('Neumann BCs: ' + str(nbc))
    logger.info('Physics parameters: ' + str(phys_param))
    logger.info('Build mesh y/n: ' +str(process_mesh))
    logger.info('Construct A and F y/n: ' +str(buildAF))
    logger.info('Compute solution y/n: ' +str(compute_sol))
    logger.info('Call paraview when done y/n: ' +str(call_pv))
    logger.info('Visualization porder: ' + str(visorder))
    logger.info('Visualization filename: ' + vis_filename + '.vtu')
    logger.info('Visualization labels: ' + str(viz_labels))
    logger.info('Mesh file: '+meshfile)

    if compute_sol:
        ########## CREATE MESH ##########
        mesh = mkmesh_cube.mkmesh_cube(porder, ndim, meshfile, process_mesh, scale_factor)
        mesh['dbc'] = dbc
        mesh['nbc'] = nbc

        logger.info('Degrees of freedom: ' + str(mesh['pcg'].shape[0]))

        logger.info('Preparing master data structure...')
        master = mkmaster.mkmaster(mesh, ndim=3, pgauss=2*mesh['porder'])

        ########## SOLVE ##########
        
        sol = cg_solve.cg_solve(master, mesh, domain_helper_fcns.forcing_zero, phys_param, ndim, outdir, buildAF, solver, solver_tol)

        ########## SAVE DATA ##########

        # NOTE: in the future uh will need to be reshaped into a nplocal x numvisfields x numel when the derivatives are added
        with open(outdir+'mesh', 'wb') as file:
            pickle.dump(mesh, file)
        with open(outdir+'master', 'wb') as file:
            pickle.dump(master, file)
        with open(outdir+'sol', 'wb') as file:
            pickle.dump(sol, file)
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

    ########## CALC DERIVATIVES ##########
    logger.info('Calculating derivatives')

    # Reshape into DG high order data structure
    sol_reshaped = helper.reshape_field(mesh, sol, 'to_array', 'scalars')

    grad = calc_derivative.calc_derivatives(mesh, master, sol_reshaped, ndim)[None,:,:]

    grad_reshaped = helper.reshape_field(mesh, grad, 'to_column', 'vectors')

    result_out = np.concatenate((sol, np.squeeze(grad_reshaped)), axis=1)    # Column 0 is the solution vector at the pcg nodes, and cols [1-3] are the gradient at the pcg nodes

    with open(vis_filename + '_solution.npy', 'wb') as file:
        np.save(file, result_out)
    logger.info('Wrote solution to /out')

    ########## VISUALIZE SOLUTION ##########

    logger.info('Generating .VTU file of solution...')
    viz.visualize(mesh, visorder, viz_labels, vis_filename, call_pv, scalars=sol, vectors=grad)
except Exception as e:
    logger.exception('message')