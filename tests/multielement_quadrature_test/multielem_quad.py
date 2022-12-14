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
sys.path.append(str(sim_root_dir.joinpath('postprocessing')))
import mkmaster
import quadrature
import extract_surface

def volume_unit_cube():
    with open('./cube_mesh/mesh', 'rb') as file:
        mesh = pickle.load(file)
    # Commenting this out because the old master structure had the wrong quadrature weights - will need to recompute for all legacy imported meshes
    # with open('./cube_mesh/master', 'rb') as file:
    #     master = pickle.load(file)

    master = mkmaster.mkmaster(mesh, 3, pgauss=2*mesh['porder'])
    ndof = mesh['pcg'].shape[0]
    scalar_test_ones = np.ones((ndof))
    elements= np.arange(mesh['t'].shape[0])

    mesh['f'] = np.concatenate((np.arange(mesh['f'].shape[0])[:,None]+1, mesh['f']), axis=1)

    # Volume of the unit cube should == 1
    return quadrature.volume_integral(mesh, master, scalar_test_ones, elements)


def area_unit_square_one_face():
    with open('./cube_mesh/mesh', 'rb') as file:
        mesh = pickle.load(file)

    master = mkmaster.mkmaster(mesh, 3, pgauss=2*mesh['porder'])

    # These should both == 1

    mesh_face, face_scalars = extract_surface.extract_surfaces(mesh, master, np.array([1]), 'pg', np.ones((mesh['pcg'].shape[0])))
    bdry_faces = np.arange(mesh_face['tcg'].shape[0])
    return quadrature.surface_integral(mesh_face, master, face_scalars, bdry_faces)


def unit_square_total_surf_area():
    # The total surface area of the unit square should be 6.
    with open('./cube_mesh/mesh', 'rb') as file:
        mesh = pickle.load(file)

    master = mkmaster.mkmaster(mesh, 3, pgauss=2*mesh['porder'])

    # NOTE: surf_pg is just the list of surface index physical groups on the boundary, basically the "-(num)" that shows up on the right column of mesh['f']

    mesh_face, face_scalars = extract_surface.extract_surfaces(mesh, master, np.arange(6)+1, 'pg', np.ones((mesh['pcg'].shape[0])))
    bdry_faces = np.arange(mesh_face['tcg'].shape[0])
    return quadrature.surface_integral(mesh_face, master, face_scalars, bdry_faces)

def unit_cube_normals():
    with open('./cube_mesh/mesh', 'rb') as file:
        mesh = pickle.load(file)
    with open('./cube_mesh/master', 'rb') as file:
        master = pickle.load(file)

    master['phi_inv'] = np.linalg.pinv(master['shapvol'][:, :, 0])

    print(mesh['f'].shape)
    bdry_faces = mesh['f'][mesh['f'][:, -1] < 0, :]     # All the faces on the cube
    bdry_faces_start_idx = np.where(mesh['f'][:, -1] < 0)[0][0]
    print(bdry_faces_start_idx)
    print(mesh['p'][mesh['t'][91,:]])
    print()
    print(mesh['dgnodes'][91])
    print()
    print(mesh['t2f'][91])

    n = quadrature.get_elem_face_normals(mesh['dgnodes'], master, 3, (91, 1))

if __name__ == '__main__':
    unit_cube_normals()