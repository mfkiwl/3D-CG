import vtk
import pyvista as pv
import numpy as np
import os

# Plotting
def generate_vtu(p, t, u):

    theta = np.concatenate((4*np.ones((t.shape[0], 1)), t), axis=1)
    theta = np.hstack(theta).astype(int)

    # each cell is a VTK_TETRA
    celltypes = np.empty(t.shape[0], dtype=np.uint8)
    celltypes[:] = vtk.VTK_TETRA
    # print(celltypes.shape)

    # Build mesh
    mesh = pv.UnstructuredGrid(theta, celltypes, p)

    # Generate plot
    plot = pv.Plotter(window_size=[2000, 1500])
    # plot.set_background('white')

    # For now, set the "solution" field to be the point's x-value and associate with the mesh
    mesh.point_data['sol_data'] = u

    # Normal vector for clipping
    normal = (0, -1, 0)
    origin = (0, 264, 0)

    fname = 'viz_test.vtu'
    mesh.save(fname, binary=False)
    print('opening paraview')
    os.system("paraview --data=" + fname + " &")

    # clipped_mesh = mesh.clip(normal, origin)

    # plot.add_mesh(clipped_mesh, scalars='sol_data',show_edges=False,line_width=0.75,)
    # plot.add_mesh(mesh, scalars='sol_data',show_edges=False,line_width=0.75,)

    # plot.show(cpos='xy')
