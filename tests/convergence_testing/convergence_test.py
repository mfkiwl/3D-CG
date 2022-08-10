import numpy as np
import os
import sys
sys.path.append('../')
from cube_sine_neumann.main import test_3d_cube_sine_neumann
from cube_sine_homogeneous_dirichlet.main import test_3d_cube_sine_homoegeneous_dirichlet
from cube_linear_dirichlet.main import test_3d_cube_linear_dirichlet
from cube_sine_dirichlet.main import test_3d_cube_sine_dirichlet
from cube_linear_neumann.main import test_3d_cube_linear_neumann
import matplotlib.pyplot as plt


def run_convergence_test(func, rundir):

    # if 'cube_sine_homogeneous_dirichlet' not in rundir:
    #     continue
    
    numel_vec = np.array([24, 100, 4591, 36538]).astype(np.int)

    with open('./old_out/{}.npy'.format(rundir), 'rb') as file:
        errors = np.load(file, allow_pickle=True)

    print('Running convergence test for {} in ../{}'.format(func.__name__, rundir))

    porder_vec = np.array([1, 2, 3, 4]).astype(np.int)
    # numel_vec = np.array([24, 100, 4591]).astype(np.int)
    h_vec = (6/numel_vec)**(1/3)

    # errors = np.zeros((porder_vec.shape[0], h_vec.shape[0]))

    # os.chdir('../{}'.format(rundir))
    # for ip, porder in enumerate(porder_vec):
    #     for inum, numel in enumerate(numel_vec):
    #         print(porder)
    #         print(numel)
    #         print()
    #         if porder == 1:
    #             solver = 'direct'
    #         else:
    #             solver = 'gmres'
    #         errors[ip, inum] = func(porder, 'cube'+str(numel), solver)
    # os.chdir('../convergence_testing')

    log_errors = np.log10(errors)[:,np.array([0,-1])]
    log_h = np.log10(h_vec)[np.array([0,-1])]
    alpha = (log_errors[:,0]-log_errors[:,1])/(log_h[0]-log_h[1])
    print('Convergence rate for {}:'.format('case_name'))
    print('p=1: {:.3f}'.format(alpha[0]))
    print('p=2: {}'.format(alpha[1]))
    print('p=3: {}'.format(alpha[2]))
    print('p=4: {}'.format(alpha[3]))

    fig, ax = plt.subplots()
    ax.invert_xaxis()
    for ip, porder in enumerate(porder_vec):
        ax.loglog(h_vec, errors[ip,:], label='porder = {}, rate: {:.2f}'.format(porder, alpha[ip]))
    plt.xlabel('Mesh size h)')
    plt.ylabel('error')
    plt.legend()
    plt.title("Convergence plot")
    plt.savefig('./out/{}.png'.format(rundir))
    plt.show()

    with open('./out/{}.npy'.format(rundir), 'wb') as file:
        np.save(file, errors)

    # with open('./out/{}.npy'.format(rundir), 'rb') as file:
    #     errors = np.load(file)

    # exit()
    return

if __name__ == '__main__':
    funcs = [test_3d_cube_sine_neumann, test_3d_cube_sine_homoegeneous_dirichlet, test_3d_cube_linear_dirichlet, test_3d_cube_sine_dirichlet, test_3d_cube_linear_neumann]
    rundirs = ['cube_sine_neumann', 'cube_sine_homogeneous_dirichlet', 'cube_linear_dirichlet', 'cube_sine_dirichlet', 'cube_linear_neumann']

    for func, rundir in zip(funcs, rundirs):
        run_convergence_test(func, rundir)