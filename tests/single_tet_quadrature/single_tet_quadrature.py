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
import helper
import domain_helper_fcns
import cg_gradient
from numpy.polynomial import Polynomial as poly
from pascalindex import pascalindex3d
from functools import partial
from pathos.multiprocessing import ProcessingPool as Pool


def single_tet_quadrature(porder):

    # Test case for a single tetrahedron
    mesh, master = mkmesh_tet.mkmesh_tet(porder)
    ndof_vol = mesh['plocal'].shape[0]
    ndof_face = master['perm'].shape[0]

    # Quadrature
    surf_int = quadrature.elem_surface_integral(mesh['dgnodes'][0,:,:][master['perm'][:,2]], master, np.ones((ndof_face, 1)), mesh['ndim'],returnType='scalar')
    # This is the area of the x-y face of the tet, which should be 0.5

    vol_int = quadrature.elem_volume_integral(mesh['dgnodes'][0,:,:], master, np.ones((ndof_vol, 1)), mesh['ndim'])
    # This should be 1/6

    error = np.linalg.norm(np.array([surf_int, vol_int])-np.array([0.5, 1/6]))

    return error

def single_tet_gradients(porder):

    # Test case for a single tetrahedron
    mesh, master = mkmesh_tet.mkmesh_tet(porder)
    # call_pv = True
    call_pv = False

    # With polynomial

    np.set_printoptions(suppress=True, linewidth=np.inf, precision=4)

    x=mesh['pcg'][:,0]
    y=mesh['pcg'][:,1]
    z=mesh['pcg'][:,2]

    poly_indices = pascalindex3d(porder)
    errors = np.zeros((poly_indices.shape[0]))

    for idx, (m, n, l) in enumerate(poly_indices):

        # Solution is a polynomial with random coefficients between -10 and 10
        poly_x = poly(np.random.rand(m+1)*20-10)
        poly_y = poly(np.random.rand(n+1)*20-10)
        poly_z = poly(np.random.rand(l+1)*20-10)

        dpoly_x = poly_x.deriv(1)
        dpoly_y = poly_y.deriv(1)
        dpoly_z = poly_z.deriv(1)

        sol = poly_x(x) * poly_y(y) * poly_z(z)
        sol = sol[:,None]
        grad, __ = cg_gradient.calc_gradient(mesh, master, sol, 3, 'direct', 1e-10)

        grad_exact = np.zeros((mesh['pcg'].shape[0], 3))
        grad_exact[:,0] = dpoly_x(x)*poly_y(y)*poly_z(z)
        grad_exact[:,1] = poly_x(x)*dpoly_y(y)*poly_z(z)
        grad_exact[:,2] = poly_x(x)*poly_y(y)*dpoly_z(z)
    
        errors[idx] = np.linalg.norm((grad-grad_exact).ravel(), np.inf)

    # viz_grad = np.concatenate((grad, grad_exact, (grad-grad_exact)), axis=1)

    # viz_labels = {'scalars': {0: 'Solution'}, 'vectors': {0: 'Computed Gradient', 1: 'Exact Gradient', 2: 'Gradient Error'}}
    # viz.visualize(mesh, mesh['porder'], viz_labels, 'vis_tet', call_pv, scalars=sol, vectors=viz_grad)
  
    return np.allclose(errors, 0, rtol=0, atol=5e-10)

def single_tet_normals(porder):

    # Test case for a single tetrahedron
    mesh, master = mkmesh_tet.mkmesh_tet(porder)
    ndof_face = master['perm'].shape[0]

    bdry_faces = mesh['f'][mesh['f'][:, -1] < 0, :]
    bdry_faces_start_idx = np.where(mesh['f'][:, -1] < 0)[0][0]

    normals_exact = np.ones((4, ndof_face, 3))
    normals_exact[0,:,:] *= np.array([0, 0, -1])
    normals_exact[1,:,:] *= np.array([0, -1, 0])
    normals_exact[2,:,:] *= np.array([-1, 0, 0])
    normals_exact[3,:,:] *= np.array([1, 1, 1])/np.linalg.norm(np.array([1, 1, 1]))

    errors = np.zeros_like(normals_exact)

    # Normal vectors
    nnodes_per_face = 3
    for bdry_face_num, face in enumerate(bdry_faces):
        facenum = bdry_face_num + bdry_faces_start_idx + 1      # t2f uses 1-indexing for the faces
        bdry_elem = face[nnodes_per_face]

        # Collect nodes of faces on boundary ONLY
        # Find the index that the face is in in t2f
        loc_face_idx = np.where(mesh['t2f'][bdry_elem, :] == facenum)[0][0]     #try np.abs here?

        # Pull the local nodes on that face from permface - we don't want to include all the nodes in the element
        idx_tup=(bdry_elem, loc_face_idx)
        errors[bdry_face_num, :,:] = normals_exact[bdry_face_num,:,:] - quadrature.get_elem_face_normals(mesh['dgnodes'], master, mesh['ndim'], idx_tup)

    return np.linalg.norm(errors.ravel(), np.inf)


if __name__ == '__main__':
    # print(single_tet_quadrature(1))
    # print(single_tet_quadrature(2))
    # print(single_tet_quadrature(3))
    # print(single_tet_quadrature(4))
    # print(single_tet(6))
    # print(single_tet(7))

    # print(single_tet_normals(1))
    # print(single_tet_normals(2))
    print(single_tet_normals(3))
    # print(single_tet_normals(4))

    # print(single_tet_gradients(1))
    # print(single_tet_gradients(2))
    # print(single_tet_gradients(3))
    # print(single_tet_gradients(4))