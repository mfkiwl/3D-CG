import numpy as np

# Test case for a single tetrahedron

mesh = {}
mesh['ndim'] = 3
mesh['porder'] = 2
mesh['p'] = np.array([[0, 1, 0],
                       [0, 0, 0],
                       [1, 0, 0],
                       [0, 0, 1]])

mesh['t'] = np.array([0, 1, 2, 3])[None,:]
mesh['plocal'], mesh['tlocal'], _, _, _, _, _ = master_nodes.master_nodes(mesh['porder'], 3)
mesh['dgnodes'] = create_dg_nodes.create_dg_nodes(mesh, 3)
mesh = cgmesh.cgmesh(mesh)

uh = mesh['pcg'][:,0]  # uh = x coord