import sys
from convert_cg_mesh import convert_cg_mesh
sys.path.insert(0, '../util')
import pickle
import numpy as np
from viz import plot_sol

def viz(mesh, sol):
    mesh = convert_cg_mesh(mesh)
    plot_sol(mesh['pcg'], mesh['tcg_connected'], sol)

if __name__ == '__main__':
    # print('Reading solution from file...')
    with open('/media/homehd/saustin/lightning_research/3D-CG/tests/cube_sine_neumann/out/mesh', 'rb') as file:
        mesh = pickle.load(file)
    with open('/media/homehd/saustin/lightning_research/3D-CG/tests/cube_sine_neumann/out/master', 'rb') as file:
        master = pickle.load(file)
    with open('/media/homehd/saustin/lightning_research/3D-CG/tests/cube_sine_neumann/out/sol', 'rb') as file:
        uh = pickle.load(file)  # Note that for this application, uh is a column vector (not in the high order format)

        # NOTE: in the future uh will need to be reshaped into a nplocal x numvisfields x numel when the derivatives are added

    # visscalars = ["temperature", 0]; # list of scalar fields for visualization
    # visvectors = ["temperature gradient", np.array([1, 2, 3]).astype(int)]; # list of vector fields for visualization

    # x = mesh['dgnodes'][:, :, 0].T
    viz(mesh, uh)