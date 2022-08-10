import sys
import numpy as np

sys.path.insert(0, '../util')
sys.path.insert(0, '../mesh')
sys.path.insert(0, '../master')
sys.path.insert(0, '../viz')
from import_util import load_mat
from viz_driver import viz_driver
from cgmesh import cgmesh
from mkplanemesh import mkmesh_plane
from mkmaster import mkmaster
from cg_solve import cg_solve
# from viz_driver import viz
import pickle

def cg_plane(porder, meshfile):
    ndim = 3

    mesh = mkmesh_plane(porder, ndim, meshfile)
    print('Converting high order mesh to CG...')
    mesh = cgmesh(mesh)
    
    #Scaling
    print('Scaling...')
    scale = 700
    # mesh['p'] /= scale
    # mesh['pcg'] /= scale
    # mesh['dgnodes'] /= scale

    print('Preparing master data structure...')
    master = mkmaster(mesh, ndim=3, pgauss=2*mesh['porder'])

    mesh['dirichlet_bdrys'] = []
    mesh['neumann_bdrys'] = {}

    for pg in mesh['pg'].keys():
        if 'normals' in mesh['pg'][pg].keys():
            mesh['neumann_bdrys'][mesh['pg'][pg]['idx']] = mesh['pg'][pg]['normals'][0,:]
            # print(pg)
            # print(mesh['neumann_bdrys'][mesh['pg'][pg]['idx']])
            # print()
        else:
            mesh['dirichlet_bdrys'].append(mesh['pg'][pg]['idx'])
            # print(pg)
            # print()

    param = {'kappa': 1, 'c': np.array([0, 0, 0]), 's': 0}

    E_field = np.array([1, 0, 0])   # E-field in x

    def forcing_zero(p):
        return np.zeros((p.shape[0], 1))

    uh = cg_solve(master, mesh, forcing_zero, param, ndim, E_field, 'sparse')

    # Reshape into DG high order data structure
    uh_reshaped = np.zeros((mesh['plocal'].shape[0], mesh['t'].shape[0]))
    for i in np.arange(mesh['t'].shape[0]):          # Set dirichlet BC
        uh_reshaped[:, i] = uh[mesh['tcg'][i, :], 0]

    # NOTE: in the future uh will need to be reshaped into a nplocal x numvisfields x numel when the derivatives are added
    with open('mesh_plane_dump', 'wb') as file:
        pickle.dump(mesh, file)
    with open('master_plane_dump', 'wb') as file:
        pickle.dump(master, file)
    with open('uh_plane_dump', 'wb') as file:
        pickle.dump(uh, file)
    print('Wrote solution to file...')


    viz_driver(mesh, master, uh_reshaped)

    # return error
    return
if __name__ == '__main__':
    # error = cg3d_cube(3, '3D/h1.0_tets24')
    # error = cg3d_cube(3, '3D/h0.5_tets101')
    # error = cg3d_cube(3, '3D/h0.1_tets4686')
    # error = cg3d_cube(3, '3D/h0.05_tets37153')

    cg_plane(3, '../../data/797_coarse')