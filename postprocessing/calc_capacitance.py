import numpy as np
import pickle
import logging
import time
import yaml

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
# import mkmaster
# import helper
import quadrature
# import mkmesh_tet

# # Integration tests on the unit cube test - should == 1
# with open('./cube_mesh/mesh', 'rb') as file:
#     mesh = pickle.load(file)
# # Commenting this out because the old master structure had the wrong quadrature weights - will need to recompute for all legacy imported meshes
# # with open('./cube_mesh/master', 'rb') as file:
# #     master = pickle.load(file)

# master = mkmaster.mkmaster(mesh, 3, pgauss=2*mesh['porder'])
# ndof = mesh['pcg'].shape[0]
# scalar_test_ones = np.ones((ndof, 1))
# elements= np.arange(mesh['t'].shape[0])

# nnodes_per_face = mesh['gmsh_mapping'][mesh['elemtype']]['nnodes'][mesh['ndim']-1]
# mesh['f'] = np.concatenate((np.arange(mesh['f'].shape[0])[:,None]+1, mesh['f']), axis=1)
# bdry_faces_start_idx = np.where(mesh['f'][:, -1] < 0)[0][0]

# bdry_faces = mesh['f'][mesh['f'][:, -1] < 0, :]     # All the faces on the cube
# faces_on_lwall = bdry_faces[-bdry_faces[:,-1]==mesh['pg']['lwall']['idx'], :]   # Filter faces by physical group - here we have filtered by name string 'lwall'


# # Volume of the unit cube should == 1
# print(quadrature.volume_integral(mesh, master, scalar_test_ones, elements))

# # These should both == 1
# print(quadrature.surface_integral_serial(mesh, master, np.ones((mesh['pcg'].shape[0], 1)), faces_on_lwall, nnodes_per_face))
# print(quadrature.surface_integral_parallel(mesh, master, np.ones((mesh['pcg'].shape[0], 1)), faces_on_lwall, nnodes_per_face))

# # This is the total surface area of all the faces - should == 6
# print(quadrature.surface_integral_parallel(mesh, master, np.ones((mesh['pcg'].shape[0], 1)), bdry_faces, nnodes_per_face))
# print('***')


# # Fetch exact solution
# with open('./430K_Phi_out/mesh', 'rb') as file:
#     mesh = pickle.load(file)
#     mesh['pcg']/=2
#     mesh['p']/=2
#     mesh['dgnodes']/=2

# bbx = mesh['bbox_after_scale']
# l1 = bbx['x'][1] - bbx['x'][0]
# l2 = bbx['y'][1] - bbx['y'][0]
# l3 = bbx['z'][1] - bbx['z'][0]

# # print(l1/5)
# # print(l1*l2*l3)
# # exit()
# # with open('./430K_Phi_out/master', 'rb') as file:
# #     master = pickle.load(file)
# master = mkmaster.mkmaster(mesh, 3, pgauss=2*mesh['porder'])

# with open('./430K_Phi_out/boeing_430K_Phi_solution.npy', 'rb') as file:
#     phi_sol = np.load(file)

# deltaV = 1  # Voltage potential between aircraft and far-field from simulation
# epsilon_0 = 8.85418782e-12

# # Compute magnitude of E-field on surface
# E_field_mag = np.linalg.norm(phi_sol[:,1:], axis=1)[:,None]*2
# mesh['f'] = np.concatenate((np.arange(mesh['f'].shape[0])[:,None]+1, mesh['f']), axis=1)

# bdry_faces_start_idx = np.where(mesh['f'][:, -1] < 0)[0][0]
# nnodes_per_face = mesh['gmsh_mapping'][mesh['elemtype']]['nnodes'][mesh['ndim']-1]
# bdry_faces = mesh['f'][mesh['f'][:, -1] < 0, :]     # This can be extended to inputting a list of arbitrary boundary faces, doesn't have to be *all* the faces on the boundary
# bdry_faces = bdry_faces[-bdry_faces[:,-1]<138,:]
# print(bdry_faces[:,-1])

#scaling fo the fuselage
#quadrature weights
#integrating over the farfield in addition to the aircraft surface


def calc_capacitance(solution, phys_param):

    # vol_mesh = solution['vol_mesh']
    # f_vol = np.concatenate((np.arange(vol_mesh['f'].shape[0])[:,None]+1, vol_mesh['f']), axis=1)
    # nnodes_per_face = vol_mesh['gmsh_mapping'][vol_mesh['elemtype']]['nnodes'][vol_mesh['ndim']-1]
    # bdry_faces = f_vol[f_vol[:, -1] < 0, :]     # This can be extended to inputting a list of arbitrary boundary faces, doesn't have to be *all* the faces on the boundary
    # bdry_faces = bdry_faces[-bdry_faces[:,-1]<max_surf_face_idx,:]
    # E_field_mag = np.linalg.norm(solution['Phi_grad_vol'], axis=1)
    # Q = quadrature.surface_integral(vol_mesh, solution['master'], E_field_mag, bdry_faces, nnodes_per_face)*phys_param['eps0']



    surf_mesh = solution['surf_mesh']

    faces = np.arange(surf_mesh['tcg'].shape[0])
    Q = quadrature.surface_integral(surf_mesh, solution['master'], solution['Phi_grad_normal_surf'], faces)*phys_param['eps0']

    deltaV = 1  # Voltage potential between aircraft and far-field from simulation
    C = Q/deltaV
    return C

if __name__ == '__main__':
    sol_fname = '/media/homehd/saustin/lightning_research/3D-CG/postprocessing/fem_solutions/d8/d8_electrostatic_solution'
    with open(sol_fname, 'rb') as file:
        solution = pickle.load(file)

    with open('physical_constants.yaml', 'r') as stream:
        phys_param = yaml.load(stream, Loader=yaml.loader.FullLoader)

    print(calc_capacitance(solution, phys_param, 114))