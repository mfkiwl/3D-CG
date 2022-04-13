import numpy as np
import sys
from pathlib import Path
cwd = Path.cwd()
for dirname in tuple(cwd.parents):
    if dirname.name == '3D-CG':
        sim_root_dir = dirname
        continue

sys.path.append(str(sim_root_dir.joinpath('master')))
sys.path.append(str(sim_root_dir.joinpath('CG')))
sys.path.append(str(sim_root_dir.joinpath('mesh')))
sys.path.append(str(sim_root_dir.joinpath('viz')))
sys.path.append(str(sim_root_dir.joinpath('util')))

import viz
import mkmaster
import quadrature
import mkmesh_tet
import gmshwrite
from mkf_parallel2 import mkt2f_new
import normals

def single_tet(porder):
    # Test case for a single tetrahedron
    mesh, master = mkmesh_tet.mkmesh_tet(porder)
    ndof_vol = mesh['plocal'].shape[0]
    ndof_face = master['perm'].shape[0]
    call_pv = False

    bdry_faces = mesh['f'][mesh['f'][:, -1] < 0, :]
    bdry_faces_start_idx = np.where(mesh['f'][:, -1] < 0)[0][0]

    # Single out only nonzero entries in sparse matrix for dirichlet BC
    nnodes_per_face = 3


    for bdry_face_num, face in enumerate(bdry_faces):
        facenum = bdry_face_num + bdry_faces_start_idx + 1      # t2f uses 1-indexing for the faces
        bdry_elem = face[nnodes_per_face]

        # Collect nodes of faces on boundary ONLY
        # Find the index that the face is in in t2f
        loc_face_idx = np.where(mesh['t2f'][bdry_elem, :] == facenum)[0][0]     #try np.abs here?

        # Pull the local nodes on that face from permface - we don't want to include all the nodes in the element

        n = quadrature.get_elem_face_normals(mesh['dgnodes'][bdry_elem,:,:], master, mesh['ndim'], loc_face_idx)
        print(n)
        print()

        # exit()

    # np.set_printoptions(suppress=True, linewidth=np.inf, precision=10)

    surf_int = quadrature.elem_surface_integral(mesh['dgnodes'][0,:,:][master['perm'][:,2]], master, np.ones((ndof_face, 1)), mesh['ndim'],returnType='scalar')
    # This is the area of the x-y face of the tet, which should be 0.5

    vol_int = quadrature.elem_volume_integral(mesh['dgnodes'][0,:,:], master, np.ones((ndof_vol, 1)), mesh['ndim'])
    # This should be 1/6=0.16666666666666666, but instead I am getting (1/6)

    error = np.linalg.norm(np.array([surf_int, vol_int])-np.array([0.5, 0.16666666666666666]))

    # Writing local mesh to file
    # flocal, _ = mkt2f_new(mesh['tlocal'], 3)
    # gmshwrite.gmshwrite(mesh['plocal'], mesh['tlocal'], 'local_mesh', flocal[flocal[:, -1] < 0, :], 'individual')
    # gmshwrite.gmshwrite(mesh['p'], mesh['t'], 'total_linear_mesh', mesh['f'][mesh['f'][:, -1] < 0, :], 'individual')

    # viz_labels = {'scalars': {0: 'Solution'}, 'vectors': {0: 'Solution Gradient'}}
    # viz.visualize(mesh, mesh['porder'], viz_labels, 'vis_tet', call_pv, scalars=mesh['pcg'][:,0][:,None])
    return error

if __name__ == '__main__':
    # print(single_tet(1))
    print(single_tet(2))
    # print(single_tet(3))
    # print(single_tet(6))