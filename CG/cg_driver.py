import numpy as np
from cg_2d_square import cg2d_square
from cg_3d_cube import cg3d_cube
import matplotlib.pyplot as plt


def plot_convergence(error, porder, h, norm, dim):
    fig, ax = plt.subplots()
    ax.invert_xaxis()
    for i, series in enumerate(error.T):
        ax.loglog(h, series, label='porder='+str(porder[i]))
    ax.legend()
    plt.xlabel('Element width, h')
    plt.ylabel(norm + '_Error')
    plt.title('Convergence plot for {:d}D Continuous Galerkin'.format(dim))
    # plt.show()
    plt.savefig('./out/{:d}d_convergence_{}-norm.png'.format(dim, norm))
    plt.close('all')


dim = 3

if dim == 2:
    meshfilenames = ['2D/square0', '2D/square1',
                        '2D/square2', '2D/square3', '2D/square4']
    numel = np.array([24, 66, 242, 944, 5828])    # Taken manually from gmsh
    h = (1/numel)**0.5
    porder_vec = [1, 2, 3, 4]

elif dim == 3:
    # meshfilenames = ['3D/h1.0_tets24', '3D/h0.5_tets101',
    #                     '3D/h0.1_tets4686', '3D/h0.05_tets37153', '3D/h0.02_tets570547']
    # meshfilenames = ['3D/h0.1_tets4686']
    # meshfilenames = ['3D/h0.05_tets37153']
    meshfilenames = ['3D/h1.0_tets24']

    numel = np.array([24, 100, 4591, 36538, 561173])    # Taken manually from gmsh
    h = (1/numel)**(1/3)
    porder_vec = [3]

l1_error = np.zeros((len(meshfilenames), len(porder_vec)))
l2_error = np.zeros_like(l1_error)
linf_error = np.zeros_like(l1_error)

for imesh, mesh in enumerate(meshfilenames):
    for jporder, porder in enumerate(porder_vec):

        if dim == 2:
            error = cg2d_square(porder, mesh)
        elif dim == 3:
            error = cg3d_cube(porder, mesh)

        np.save('out/' + mesh + '_error.npy', error)

        l1_error[imesh, jporder]=np.linalg.norm(error, 1)
        l2_error[imesh, jporder] = np.linalg.norm(error, 2)
        linf_error[imesh, jporder] = np.linalg.norm(error, np.inf)

        print('mesh:', mesh)
        print('porder:', porder)
        print('L-inf error:', linf_error[imesh, jporder])
        print(linf_error)
        print()


np.save('out/l1_error.npy', l1_error)
np.save('out/l2_error.npy', l2_error)
np.save('out/linf_error.npy', linf_error)



# dim=2
# numel = np.array([26, 66, 242, 944, 5828])    # Taken manually from gmsh
# h = (1/numel)**0.5
# porder_vec = [1, 2, 3, 4]
# l1_error = np.load('out/l1_error.npy')
# l2_error = np.load('out/l2_error.npy')
# linf_error = np.load('out/linf_error.npy')

# plot_convergence(l1_error, porder_vec, h, 'L1', dim)
# plot_convergence(l2_error, porder_vec, h, 'L2', dim)
# plot_convergence(linf_error, porder_vec, h, 'Linf', dim)
