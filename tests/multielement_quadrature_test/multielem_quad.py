import numpy as np
import pickle
import logging
import time

logger = logging.getLogger(__name__)
# Finding the sim root directory
import sys
from pathlib import Path
cwd = Path.cwd()
for dirname in tuple(cwd.parents):
    if dirname.name == '3D-CG':
        sim_root_dir = dirname
        continue

sys.path.append(str(sim_root_dir.joinpath('util')))
sys.path.append(str(sim_root_dir.joinpath('CG')))
sys.path.append(str(sim_root_dir.joinpath('mesh')))
sys.path.append(str(sim_root_dir.joinpath('master')))
import mkmaster
import quadrature


def volume_unit_cube():
    with open('./cube_mesh/mesh', 'rb') as file:
        mesh = pickle.load(file)
    # Commenting this out because the old master structure had the wrong quadrature weights - will need to recompute for all legacy imported meshes
    # with open('./cube_mesh/master', 'rb') as file:
    #     master = pickle.load(file)

    master = mkmaster.mkmaster(mesh, 3, pgauss=2*mesh['porder'])
    ndof = mesh['pcg'].shape[0]
    scalar_test_ones = np.ones((ndof, 1))
    elements= np.arange(mesh['t'].shape[0])

    mesh['f'] = np.concatenate((np.arange(mesh['f'].shape[0])[:,None]+1, mesh['f']), axis=1)

    # Volume of the unit cube should == 1
    return quadrature.volume_integral(mesh, master, scalar_test_ones, elements)


def area_unit_square_one_face():
    with open('./cube_mesh/mesh', 'rb') as file:
        mesh = pickle.load(file)
    # Commenting this out because the old master structure had the wrong quadrature weights - will need to recompute for all legacy imported meshes
    # with open('./cube_mesh/master', 'rb') as file:
    #     master = pickle.load(file)

    master = mkmaster.mkmaster(mesh, 3, pgauss=2*mesh['porder'])

    nnodes_per_face = mesh['gmsh_mapping'][mesh['elemtype']]['nnodes'][mesh['ndim']-1]
    mesh['f'] = np.concatenate((np.arange(mesh['f'].shape[0])[:,None]+1, mesh['f']), axis=1)

    bdry_faces = mesh['f'][mesh['f'][:, -1] < 0, :]     # All the faces on the cube
    faces_on_lwall = bdry_faces[-bdry_faces[:,-1]==mesh['pg']['lwall']['idx'], :]   # Filter faces by physical group - here we have filtered by name string 'lwall'

    # These should both == 1
    return quadrature.surface_integral_serial(mesh, master, np.ones((mesh['pcg'].shape[0], 1)), faces_on_lwall, nnodes_per_face)
    # print(quadrature.surface_integral_parallel(mesh, master, np.ones((mesh['pcg'].shape[0], 1)), faces_on_lwall, nnodes_per_face))

def unit_square_total_surf_area():
    with open('./cube_mesh/mesh', 'rb') as file:
        mesh = pickle.load(file)
    # Commenting this out because the old master structure had the wrong quadrature weights - will need to recompute for all legacy imported meshes
    # with open('./cube_mesh/master', 'rb') as file:
    #     master = pickle.load(file)

    master = mkmaster.mkmaster(mesh, 3, pgauss=2*mesh['porder'])
    nnodes_per_face = mesh['gmsh_mapping'][mesh['elemtype']]['nnodes'][mesh['ndim']-1]
    mesh['f'] = np.concatenate((np.arange(mesh['f'].shape[0])[:,None]+1, mesh['f']), axis=1)

    bdry_faces = mesh['f'][mesh['f'][:, -1] < 0, :]     # All the faces on the cube

    # This is the total surface area of all the faces - should == 6
    return quadrature.surface_integral_parallel(mesh, master, np.ones((mesh['pcg'].shape[0], 1)), bdry_faces, nnodes_per_face)