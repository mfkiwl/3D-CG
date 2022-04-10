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

import mkmaster
import quadrature

# Set up tet mesh - hardcoding the master node points for a porder=2 for simplicity

# Test case for a single tetrahedron
mesh = {}
mesh['ndim'] = 3
mesh['porder'] = 2
mesh['p'] = np.array([[0, 1, 0],
                    [0, 0, 0],
                    [1, 0, 0],
                    [0, 0, 1]])

mesh['t'] = np.array([0, 1, 2, 3])[None,:]

mesh['plocal'] = np.array([[0.,  0.,  0.],
                            [0.5, 0.,  0.],
                            [1.,  0.,  0.],
                            [0.,  0.5, 0.],
                            [0.5, 0.5, 0.],
                            [0.,  1.,  0.],
                            [0.,  0.,  0.5],
                            [0.5, 0.,  0.5],
                            [0.,  0.5, 0.5],
                            [0.,  0.,  1.]])

mesh['tlocal'] = np.array([[0, 1, 3, 6],
                            [7, 6, 8, 3],
                            [6, 7, 1, 3],
                            [1, 2, 4, 7],
                            [4, 7, 8, 3],
                            [4, 1, 7, 3],
                            [3, 4, 5, 8],
                            [6, 7, 8, 9]])

# For the master tetrahedron, the plocal pts are the same as the DG nodes. However, I have broadcast to a (1 x nplocal x 3) array in keeping with the dimensionality of this data structure for >1 element meshes
mesh['dgnodes'] = mesh['plocal'][None,:,:]
master = mkmaster.mkmaster(mesh, ndim=3, pgauss=2*mesh['porder'])
ndof_vol = mesh['plocal'].shape[0]
ndof_face = master['perm'].shape[0]

np.set_printoptions(suppress=True, linewidth=np.inf, precision=10)

print(quadrature.face_surface_integral(mesh['dgnodes'][0,:,:][master['perm'][:,2]], master, np.ones((ndof_face, 1)), mesh['ndim']))
# This is the area of the x-y face of the tet, which should be 0.5 and indeed it is.

print(quadrature.elem_volume_integral(mesh['dgnodes'][0,:,:], master, np.ones((ndof_vol, 1)), mesh['ndim']))
# This should be 1/6=0.16666666666666666, but instead I am getting (1/6)/6, or 1/36=0.027777777777776944