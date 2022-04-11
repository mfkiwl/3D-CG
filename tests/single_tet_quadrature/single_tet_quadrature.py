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
import viz
import mkmaster
import quadrature
import mkmesh_tet

def single_tet(porder):
    # Test case for a single tetrahedron
    mesh, master = mkmesh_tet.mkmesh_tet(porder)
    ndof_vol = mesh['plocal'].shape[0]
    ndof_face = master['perm'].shape[0]

    # np.set_printoptions(suppress=True, linewidth=np.inf, precision=10)

    surf_int = quadrature.elem_surface_integral(mesh['dgnodes'][0,:,:][master['perm'][:,2]], master, np.ones((ndof_face, 1)), mesh['ndim'],returnType='scalar')
    # This is the area of the x-y face of the tet, which should be 0.5

    vol_int = quadrature.elem_volume_integral(mesh['dgnodes'][0,:,:], master, np.ones((ndof_vol, 1)), mesh['ndim'])
    # This should be 1/6=0.16666666666666666, but instead I am getting (1/6)

    error = np.linalg.norm(np.array([surf_int, vol_int])-np.array([0.5, 0.16666666666666666]))

    # Write the mesh to a file to visualize in gmsh
    sys.path.append(str(sim_root_dir.joinpath('util')))
    import gmshwrite
    gmshwrite.gmshwrite(mesh['p'], mesh['t'], 'mesh_out')
    print('wrote mesh to file')

    viz_labels = {'scalars': {0: 'Solution'}, 'vectors': {0: 'Solution Gradient'}}

    viz.visualize(mesh, mesh['porder'], viz_labels, 'vis_tet', False, scalars=mesh['pcg'][:,0][:,None])

    return error

if __name__ == '__main__':
    # print(single_tet(1))
    # print(single_tet(2))
    print(single_tet(3))
    # print(single_tet(4))