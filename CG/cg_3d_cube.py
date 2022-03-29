import sys
import numpy as np

sys.path.insert(0, '../util')
sys.path.insert(0, '../mesh')
sys.path.insert(0, '../master')
sys.path.insert(0, '../viz')
from import_util import load_mat
from viz_driver import viz_driver
from cgmesh import cgmesh
from mkmesh_cubemesh_gmsh import mkmesh_cube
from mkmaster import mkmaster
from cg_solve import cg_solve
# from viz_driver import viz
import pickle

def cg3d_cube(porder, meshfile):
    ndim = 3

    mesh = mkmesh_cube(porder, ndim, meshfile)
    print('Converting high order mesh to CG...')
    mesh = cgmesh(mesh)

    print('Preparing master data structure...')
    master = mkmaster(mesh, ndim=3, pgauss=2*mesh['porder'])

    param = {'kappa': 1, 'c': np.array([0, 0, 0]), 's': 0}

    m=1
    n=1
    l=1

    def exact_cube(p):
        x = p[:, 0]
        y = p[:, 1]
        z = p[:, 2]
        exact = np.sin(m*np.pi*x) * np.sin(n*np.pi*y) * np.sin(l*np.pi*z)
        return exact

    def forcing_cube(p):
        # Note: doesn't take kappa into account, might add in later
        forcing_cube = (m**2+n**2+l**2)*np.pi**2*exact_cube(p)        # We can do this beacuse of the particular sin functions chosen for the exact solution
        return forcing_cube   # returns as column vector

    def forcing_zero(p):
        return np.zeros((p.shape[0], 1))

    uh = cg_solve(master, mesh, forcing_cube, param, ndim, 'sparse')

    # NOTE: in the future uh will need to be reshaped into a nplocal x numvisfields x numel when the derivatives are added
    with open('mesh_dump', 'wb') as file:
        pickle.dump(mesh, file)
    with open('master_dump', 'wb') as file:
        pickle.dump(master, file)
    with open('uh_dump', 'wb') as file:
        pickle.dump(uh, file)
    print('Wrote solution to file...')

    exact = exact_cube(mesh['pcg'])[:,None]


    # Reshape into DG high order data structure
    uh_reshaped = np.zeros((mesh['plocal'].shape[0], mesh['t'].shape[0]))
    exact_reshaped = np.zeros((mesh['plocal'].shape[0], mesh['t'].shape[0]))
    for i in np.arange(mesh['t'].shape[0]):          # Set dirichlet BC
        uh_reshaped[:, i] = uh[mesh['tcg'][i, :], 0]
        exact_reshaped[:, i] = exact[mesh['tcg'][i, :], 0]

    error = exact_reshaped.ravel()-uh_reshaped.ravel()

    viz_driver(mesh, master, uh_reshaped)

    return error

if __name__ == '__main__':
    # error = cg3d_cube(3, '3D/h1.0_tets24')
    # error = cg3d_cube(3, '3D/h0.5_tets101')
    error = cg3d_cube(3, '3D/h0.1_tets4686')
    # error = cg3d_cube(3, '3D/h0.05_tets37153')

    # print(np.linalg.norm(error, 1))
    # print(np.linalg.norm(error, 2))
    # print(np.linalg.norm(error, np.inf))
