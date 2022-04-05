import vtk
import pyvista as pv
import numpy as np
import os

def generate_vtu(p, t, scalars, vectors, labels, viz_filename, call_pv):

    theta = np.concatenate((4*np.ones((t.shape[0], 1)), t), axis=1)
    theta = np.hstack(theta).astype(int)

    # each cell is a VTK_TETRA
    celltypes = np.empty(t.shape[0], dtype=np.uint8)
    celltypes[:] = vtk.VTK_TETRA

    # Build mesh
    mesh = pv.UnstructuredGrid(theta, celltypes, p)

    for field_index, scalar_field in enumerate(scalars.T):  # Enumerate through rows
        mesh.point_data[labels['scalars'][field_index]] = scalar_field
    for field_index, vector_field in enumerate(vectors.T):  # Enumerate through pages
        mesh.point_data[labels['vectors'][field_index]] = vector_field
    
    mesh.set_active_scalars('sol_data')
    mesh.set_active_vectors('vec')

    fname = viz_filename+'.vtu'
    mesh.save(fname, binary=False)
    if call_pv:
        print('Opening paraview!')
        os.system("paraview --data=" + fname + " &")
    else:
        print('Completed writing .vtu, closing...')


if __name__ == '__main__':
    pass

    # Generate plot
    # plot = pv.Plotter(window_size=[2000, 1500])
    # plot.set_background('white')

    # Normal vector for clipping
    # normal = (0, -1, 0)
    # origin = (0, 264, 0)
    # clipped_mesh = mesh.clip(normal, origin)

    # plot.add_mesh(clipped_mesh, scalars='sol_data',show_edges=False,line_width=0.75,)
    # plot.add_mesh(mesh, scalars='sol_data',show_edges=False,line_width=0.75,)

    # plot.show(cpos='xy')
