import sys
import numpy as np

sys.path.insert(0, '../../util')
sys.path.insert(0, '../../mesh')
sys.path.insert(0, '../../master')
sys.path.insert(0, '../../CG')

from cgmesh import cgmesh
from cg_solve import cg_solve
from mkmaster import mkmaster
from mkmesh_sqmesh_gmsh import mkmesh_square
from import_util import load_mat



ndim = 2
porder=3
meshfile = 'square0'  # Don't include the file extension


########## BCs ##########
# Dirichlet
dbc = {
    1: 0,
    2: 0,
    3: 0,
    4: 0,
}

# Neumann
nbc = {}


mesh = mkmesh_square(porder, ndim, meshfile)
mesh = cgmesh(mesh)
mesh['dbc'] = dbc
mesh['nbc'] = nbc

master = mkmaster(mesh, 2, 2*mesh['porder'])

param = {'kappa': 0.1, 'c': np.array([1, -2]), 's': 1}

n = 3
m = 2

def exact_square(p):
    x = p[:, 0]
    y = p[:, 1]
    return np.sin(n*np.pi*x) * np.sin(m*np.pi*y)


def forcing_square(p):
    x = p[:, 0]
    y = p[:, 1]
    forcing_square = param['kappa']*(n**2+m**2)*np.pi*np.pi*(np.sin(n*np.pi*x)*np.sin(m*np.pi*y)) + param['c'][0]*n*np.pi*(np.cos(
        n*np.pi*x)*np.sin(m*np.pi*y)) + param['c'][1]*m*np.pi*(np.sin(n*np.pi*x)*np.cos(m*np.pi*y)) + param['s']*(np.sin(n*np.pi*x)*np.sin(m*np.pi*y))
    return forcing_square[:, None]   # returns as column vector


np.set_printoptions(suppress=True, linewidth=np.inf, precision=16)


uh = cg_solve(master, mesh, forcing_square, param, 2, './out')

# Reshape into DG high order data structure
uh_reshaped = np.zeros((mesh['plocal'].shape[0], mesh['t'].shape[0]))
for i in np.arange(mesh['t'].shape[0]):          # Set dirichlet BC
    uh_reshaped[:, i] = uh[mesh['tcg'][i, :], 0]
    
# Compare to matlab
uh_mat = load_mat('uh2d', 'uh')
print('Python solution matches Matlab:', np.allclose(uh_reshaped, uh_mat, rtol=1e-13, atol=4e-15))
