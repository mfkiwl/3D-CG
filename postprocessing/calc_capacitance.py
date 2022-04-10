from unittest import result
import numpy as np
import pickle
import logging
import multiprocessing as mp
from functools import partial
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
import helper
import quadrature
import mkmesh_tet

def surface_integral_serial(mesh, master, scalar_field, faces, nnodes_per_face, abs_face_idx=0):
   
    #NOTE: Eventually delete the nnodes_per_face param when the sims are re-run with the mesh supporting that field
    # Loop over each element like in the cgsolve BC assignment and perform the numerical integration 
    integral_qty = 0
    
    for bdry_face_num, face in enumerate(faces):
        facenum = bdry_face_num + abs_face_idx + 1      # t2f uses 1-indexing for the faces
        bdry_elem = face[nnodes_per_face]

        # Collect nodes of faces on boundary ONLY
        # Find the index that the face is in in t2f
        loc_face_idx = np.where(mesh['t2f'][bdry_elem, :] == facenum)[0][0]
        # Pull the local nodes on that face from permface - we don't want to include all the nodes in the element
        loc_face_nodes = master['perm'][:, loc_face_idx]

        # Use the perm indices to grab the face nodes from tcg
        face_nodes = mesh['tcg'][bdry_elem][loc_face_nodes]

        field_vals = scalar_field[face_nodes]

        # Also try pts_on_face = mesh['dgnodes'][bdry_elem, loc_face_nodes, :] instead of mesh['pcg'][:,face_nodes], but they should be the same
        integral_qty += quadrature.elem_surface_integral(mesh['pcg'][face_nodes,:], master, field_vals, mesh['ndim'])

    return integral_qty

def surface_integral_parallel(mesh, master, scalar_field, faces, nnodes_per_face, abs_face_idx=0):
   
    #NOTE: Eventually delete the nnodes_per_face param when the sims are re-run with the mesh supporting that field
    # Loop over each element like in the cgsolve BC assignment and perform the numerical integration 
    
    with mp.Pool(mp.cpu_count()) as pool:
        result = pool.map(partial(get_surface_dQ, mesh, master, scalar_field, nnodes_per_face, abs_face_idx, faces), np.arange(faces.shape[0]))
    return sum(result)

def get_surface_dQ(mesh, master, scalar_field, nnodes_per_face, abs_face_idx, faces, face_idx):
    face = faces[face_idx]
    facenum = face_idx + abs_face_idx + 1      # t2f uses 1-indexing for the faces
    bdry_elem = face[nnodes_per_face]

    # Collect nodes of faces on boundary ONLY
    # Find the index that the face is in in t2f
    loc_face_idx = np.where(mesh['t2f'][bdry_elem, :] == facenum)[0][0]
    # Pull the local nodes on that face from permface - we don't want to include all the nodes in the element
    loc_face_nodes = master['perm'][:, loc_face_idx]

    # Use the perm indices to grab the face nodes from tcg
    face_nodes = mesh['tcg'][bdry_elem][loc_face_nodes]

    field_vals = scalar_field[face_nodes]

    # Also try pts_on_face = mesh['dgnodes'][bdry_elem, loc_face_nodes, :] instead of mesh['pcg'][:,face_nodes], but they should be the same
    dQ = elemmat_integrals.elemmat_surface_integral(mesh['pcg'][face_nodes,:], master, field_vals, mesh['ndim'])
    
    return dQ

def volume_integral(mesh, master, scalar_field, elements):
    """
    elements is a list of all the elements that are in the domain of integration

    """

    integral_qty = 0

    for element in elements:
        dgpts = mesh['dgnodes'][element,:,:]
        field_vals = scalar_field[mesh['tcg'][element]]

        # print(dgpts)
        # print(field_vals)
        integral_qty += quadrature.elem_volume_integral(dgpts, master, field_vals, mesh['ndim'])

    return integral_qty



# # Fetch exact solution
# with open('./430K_Phi_out/mesh', 'rb') as file:
#     mesh = pickle.load(file)
# with open('./430K_Phi_out/master', 'rb') as file:
#     master = pickle.load(file)
# with open('./430K_Phi_out/boeing_430K_Phi_solution.npy', 'rb') as file:
#     phi_sol = np.load(file)

# deltaV = 1  # Voltage potential between aircraft and far-field from simulation
# epsilon_0 = 8.85418782e-12

# # Compute magnitude of E-field on surface
# E_field_mag = np.linalg.norm(phi_sol[:,1:], axis=1)[:,None]

# # Pull surfaces that are on aircraft (will want to break this out into a separate function that takes a list of nodes to integrate over)
# # This can be done with mesh.f[-1] etc
# logger.info('Assigning boundary conditions')
# bdry_faces_start_idx = np.where(mesh['f'][:, -1] < 0)[0][0]
# # Copied from cg_solve.assign_bcs
# nnodes_per_face = mesh['gmsh_mapping'][mesh['elemtype']]['nnodes'][mesh['ndim']-1]
# bdry_faces = mesh['f'][mesh['f'][:, -1] < 0, :]     # This can be extended to inputting a list of arbitrary boundary faces, doesn't have to be *all* the faces on the boundary

# start = time.perf_counter()
# Q = surface_integral_parallel(mesh, master, E_field_mag, bdry_faces, nnodes_per_face, bdry_faces_start_idx)*epsilon_0
# print(time.perf_counter()-start)

# C = Q/deltaV

# print(C)


# Fetch exact solution
# with open('./cube_mesh/mesh', 'rb') as file:
#     mesh = pickle.load(file)
# # with open('./cube_mesh/master', 'rb') as file:
# #     master = pickle.load(file)

# master = mkmaster.mkmaster(mesh, 3, pgauss=2*mesh['porder'])
# ndof = mesh['pcg'].shape[0]
# scalar_test_ones = np.ones((ndof, 1))
# elements= np.arange(mesh['t'].shape[0])

# print(volume_integral(mesh, master, scalar_test_ones, elements))



mesh, master, uh = mkmesh_tet.mkmesh_tet(porder=2)
ndof = mesh['pcg'].shape[0]

print(quadrature.elem_volume_integral(mesh['dgnodes'][0,:,:], master, np.ones((ndof, 1)), mesh['ndim']))


mesh, master, uh = mkmesh_tet.mkmesh_tet(2)
ndof = mesh['pcg'].shape[0]
scalar_test_ones = np.ones((ndof, 1))
elements= np.arange(mesh['t'].shape[0])

print(volume_integral(mesh, master, scalar_test_ones, elements))