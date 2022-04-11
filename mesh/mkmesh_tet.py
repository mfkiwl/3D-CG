# Finding the sim root directory
import sys
from pathlib import Path
cwd = Path.cwd()
for dirname in tuple(cwd.parents):
    if dirname.name == '3D-CG':
        sim_root_dir = dirname
        continue

sys.path.append(str(sim_root_dir.joinpath('CG')))
sys.path.append(str(sim_root_dir.joinpath('master')))
import masternodes
import create_dg_nodes
import cgmesh
import numpy as np
import mkmaster
from mkf_parallel2 import mkt2f_new

def mkmesh_tet(porder, sol_axis=None):
    # Test case for a single tetrahedron
    mesh = {}
    mesh['ndim'] = 3
    mesh['porder'] = porder
    # mesh['cgmesh'] = cgmesh.cgmesh(mesh)
    mesh['p'] = np.array([[0, 1, 0],
                        [0, 0, 0],
                        [1, 0, 0],
                        [0, 0, 1]])

    mesh['t'] = np.array([0, 1, 2, 3])[None,:]
    mesh['plocal'], mesh['tlocal'], _, _, _, _, _ = masternodes.masternodes(mesh['porder'], 3)
    mesh['dgnodes'] = create_dg_nodes.create_dg_nodes(mesh, 3)
    mesh['cgmesh'] = cgmesh.cgmesh(mesh)
    mesh['f'], mesh['t2f'] = mkt2f_new(mesh['t'], 3)

    master = mkmaster.mkmaster(mesh, ndim=3, pgauss=2*porder)

    if sol_axis is not None:
        if sol_axis == 'x':
            uh = mesh['pcg'][:,0]  # uh = x coord
        if sol_axis == 'y':
            uh = mesh['pcg'][:,1]  # uh = x coord
        if sol_axis == 'z':
            uh = mesh['pcg'][:,2]  # uh = x coord
        return mesh, master, uh
    else:
        return mesh, master