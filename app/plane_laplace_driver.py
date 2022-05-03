import sys
from pathlib import Path
import numpy as np
import logging
import datetime
import yaml
import gc

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
sys.path.append(str(sim_root_dir.joinpath('postprocessing')))

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
# case_select = config_dict['case_select']
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

x_minus_face = config_dict['x_minus_face_index']
x_plus_face = config_dict['x_plus_face_index']
y_minus_face = config_dict['y_minus_face_index']
y_plus_face = config_dict['y_plus_face_index']
z_minus_face = config_dict['z_minus_face_index']
z_plus_face = config_dict['z_plus_face_index']
phys_param = config_dict['phys_param']

if 'scale_factor' in config_dict:   # Allows the mesh scale factor to be entered manually instead of from a set of points in the mesh
    scale_factor = config_dict['scale_factor']
else:
    fuselage_dia = float(config_dict['fuselage_dia'])
    fuselage_pts = config_dict['fuselage_pts']
    d_fuselage_msh= np.linalg.norm(np.asarray(fuselage_pts[0]).astype(np.float)-np.asarray(fuselage_pts[1]).astype(np.float))
    scale_factor = fuselage_dia/d_fuselage_msh    # Normalize mesh by the fuselage radius and rescale so that mesh dimensions are in meters

bdry = (x_minus_face, x_plus_face, y_minus_face, y_plus_face, z_minus_face, z_plus_face)
if 'surface_index' in config_dict:  # Allows for manual entry of the face index of the aircraft surface instead of automatically calculating it from the boundary fields
    surf_face_pg = np.array([config_dict['surface_index']]).astype(np.int32)
else:
    surf_face_pg = np.arange(min(bdry)-1)+1         # Number of faces on aircraft body is implied from the value of the min face, which is assigned after the aircraft surfaces in the mesh generator script

if 'surf_viz_labels' in config_dict:
    surf_viz_labels = config_dict['surf_viz_labels']
else:
    surf_viz_labels = None

import logger_cfg
logger = logger_cfg.initialize_logger(casename)
logger.info('*************************** INITIALIZING SIM ' +str(datetime.datetime.now())+' ***************************')
logger.info('Starting imports...')

# These imports take a while, so we want to log the initialization first
import viz
from cgmesh import cgmesh
import mkmesh_cube
import mkmesh_falcon
import mkmaster
import cg_solve
import pickle
import calc_derivative
import os
import helper
import domain_helper_fcns
import cg_gradient
import extract_surface

logger.info('SLURM_JOB_ID: ' + SLURM_JOB_ID)
logger.info('SLURM_JOB_NAME: ' + SLURM_JOB_NAME)
logger.info('SLURM_CPUS_ON_NODE: ' + SLURM_CPUS_ON_NODE)
logger.info('SLURM_JOB_NODELIST: ' + SLURM_JOB_NODELIST)
logger.info('SLURM_NTASKS: ' + SLURM_NTASKS)
logger.info('SLURM_SUBMIT_DIR: ' + SLURM_SUBMIT_DIR)
logger.info('Config file path: ' + config_file)

try:
    # Check whether the specified path exists or not
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        logger.info(outdir+' directory not present, created...')

    outdir_top = outdir
    ########## TOP LEVEL SIM SETUP ##########

    cases = ['Phi', 'Ex', 'Ey', 'Ez']
    first_time = True

    for case_select in cases:      # Top level loop through each case
        logger.info('----------------------------- CASE: ' + case_select +' -----------------------------')

        outdir = outdir_top + case_select + '/'
        # Check whether the specified path exists or not
        if not os.path.exists(outdir):
            os.makedirs(outdir)
            logger.info(outdir + ' directory not present, created...')

        vis_filename = outdir+casename+'_'+case_select

        nbc = {face:0 for face in bdry}

        if case_select == 'Phi':
            dbc = {face:1 for face in surf_face_pg}
            dbc.update(nbc)     # Concatenating two dictionaries together
            nbc = {}
        elif case_select == 'Ex':
            dbc = {face:0 for face in surf_face_pg}
            nbc[x_minus_face] = -1
            nbc[x_plus_face] = 1
        elif case_select == 'Ey':
            dbc = {face:0 for face in surf_face_pg}
            nbc[y_minus_face] = -1
            nbc[y_plus_face] = 1
        elif case_select == 'Ez':
            dbc = {face:0 for face in surf_face_pg}
            nbc[z_minus_face] = -1
            nbc[z_plus_face] = 1

        ########## LOGGING SIM PARAMETERS ##########
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
        logger.info('Volume Visualization labels: ' + str(viz_labels))
        logger.info('Mesh file: '+meshfile)

        if compute_sol:
            ########## CREATE MESH ##########
            if first_time:      # Only build the mesh the first time
                if 'falcon' in meshfile:
                    mesh = mkmesh_falcon.import_falcon_mesh(porder, ndim, meshfile, process_mesh)
                else:
                    mesh = mkmesh_cube.mkmesh_cube(porder, ndim, meshfile, process_mesh, scale_factor)
                logger.info('Preparing master data structure...')
                master = mkmaster.mkmaster(mesh, ndim=3, pgauss=2*mesh['porder'])

            mesh['dbc'] = dbc
            mesh['nbc'] = nbc

            logger.info('Degrees of freedom: ' + str(mesh['pcg'].shape[0]))

            ########## SOLVE ##########
            
            sol = cg_solve.cg_solve(master, mesh, domain_helper_fcns.forcing_zero, phys_param, ndim, outdir, case_select, buildAF, solver, solver_tol)

            ########## SAVE DATA ##########

            # NOTE: in the future uh will need to be reshaped into a nplocal x numvisfields x numel when the derivatives are added
            with open(outdir+'mesh_'+case_select, 'wb') as file:
                pickle.dump(mesh, file)
            with open(outdir+'master', 'wb') as file:
                pickle.dump(master, file)
            with open(outdir+'sol_'+case_select, 'wb') as file:
                pickle.dump(sol, file)
            logger.info('Wrote solution to file...')
        else:
            ########## LOADING SOLUTION ##########

            logger.info('Reading solution from file...')

            with open(outdir+'mesh_'+case_select, 'rb') as file:
                mesh = pickle.load(file)
            with open(outdir+'master', 'rb') as file:
                master = pickle.load(file)
            with open(outdir+'sol_'+case_select, 'rb') as file:
                sol = pickle.load(file)

        ########## CALC DERIVATIVES ##########
        logger.info('Calculating derivatives')

        grad, grad_mag = cg_gradient.calc_gradient(mesh, master, sol, ndim, solver, solver_tol)

        e_field = -grad # Sign flip, E = -grad(potential)
        result_out = np.concatenate((sol, e_field), axis=1)

        with open(vis_filename + '_solution.npy', 'wb') as file:
            np.save(file, result_out)
        logger.info('Wrote solution to /out')

        ########## VISUALIZE SOLUTION ##########

        logger.info('Generating .VTU file of solution...')
        viz.visualize(mesh, visorder, viz_labels, vis_filename, call_pv, scalars=sol, vectors=e_field)

        if surf_viz_labels is not None:
            logger.info('Visualizing aircraft surface fields')

            mesh_face, face_scalars = extract_surface.extract_surfaces(mesh, master, surf_face_pg, 'pg', sol)
            __, face_field_dot_normal = extract_surface.extract_surfaces(mesh, master, surf_face_pg, 'pg', e_field, return_normal_quantity=True)
            
            face_scalars = np.concatenate((face_scalars, face_field_dot_normal), axis=1)

            viz.visualize(mesh_face, visorder, surf_viz_labels, vis_filename+'_surface', call_pv, face_scalars, None, type='surface_mesh') # Can only have scalars on a surface mesh

            logger.info('Saving surface mesh to disk')
            with open(vis_filename + 'surface_mesh', 'w+b') as file:
                pickle.dump(mesh_face, file)

            with open(vis_filename + '_surface_scalars.npy', 'wb') as file:
                np.save(file, face_scalars)
        logger.info('')

        del(sol)
        del(grad)
        del(grad_mag)
        gc.collect()

        if first_time:
            first_time = False

except Exception as e:
    logger.exception('message')