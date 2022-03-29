import sys
import numpy as np

sys.path.insert(0, '../../util')
sys.path.insert(0, '../../mesh')
sys.path.insert(0, '../../master')
sys.path.insert(0, '../../viz')
sys.path.insert(0, '../../CG')
from import_util import load_mat
from viz_driver import viz_driver
from cgmesh import cgmesh
from mkmesh_cube import mkmesh_cube
from mkmaster import mkmaster
from cg_solve import cg_solve
import pickle
from calc_derivative import calc_derivatives

########## TOP LEVEL SIM SETUP ##########
# meshfile = 'mesh/h1.0_tets24'  # Don't include the file extension
# meshfile = 'mesh/h0.5_tets101'  # Don't include the file extension
meshfile = 'mesh/h0.1_tets4686'
# meshfile = 'mesh/h0.05_tets37153'

case_name = 'dirichlet_cube'
outdir = 'out/'
vis_filename = 'cube_sol'
call_pv = False
build_mesh = True
buildAF = True
use_preconditioning = True
compute_error = True
vis_filename = outdir+vis_filename
porder = 3
ndim = 3

########## BCs ##########
# Dirichlet
dbc = {
1: 0,   # -X
2: 0,   # +X
3: 0,   # +z
4: 0,   # -Z
5: 0,   # -Y
6: 0,   # -Z
}

# Neumann
nbc = {}

# Eventually: add support for the neumann condition being entered as a vector dot product
# Long term: neumann robin BC

########## CREATE MESH ##########

mesh = mkmesh_cube(porder, ndim, meshfile, build_mesh)
print('Converting high order mesh to CG...')
mesh = cgmesh(mesh)
mesh['dbc'] = dbc
mesh['nbc'] = nbc

print('Preparing master data structure...')
master = mkmaster(mesh, ndim=3, pgauss=2*mesh['porder'])

########## PHYSICS PARAM ##########
param = {'kappa': 1, 'c': np.array([0, 0, 0]), 's': 0}

m = 1
n = 1
l = 1

def exact_cube(p):
    x = p[:, 0]
    y = p[:, 1]
    z = p[:, 2]
    exact = np.sin(m*np.pi*x) * np.sin(n*np.pi*y) * np.sin(l*np.pi*z)
    return exact

def forcing_cube(p):
    # Note: doesn't take kappa into account, might add in later
    forcing_cube = (m**2+n**2+l**2)*np.pi**2*exact_cube(p)        # We can do this because of the particular sin functions chosen for the exact solution
    return forcing_cube[:, None]   # returns as column vector

def forcing_zero(p):
    return np.zeros((p.shape[0], 1))

########## SOLVE ##########

# sol = cg_solve(master, mesh, forcing_cube, param, ndim, outdir, buildAF, use_preconditioning)
sol, _ = cg_solve(master, mesh, forcing_cube, param, ndim, outdir, approx_sol=None, buildAF=True, solver='amg', case_name=case_name)

########## SAVE DATA ##########

# NOTE: in the future uh will need to be reshaped into a nplocal x numvisfields x numel when the derivatives are added
with open(outdir+'mesh', 'wb') as file:
    pickle.dump(mesh, file)
with open(outdir+'master', 'wb') as file:
    pickle.dump(master, file)
with open(outdir+'sol', 'wb') as file:
    pickle.dump(sol, file)
print('Wrote solution to file...')


########## ERROR CALCULATION ##########
exact = exact_cube(mesh['pcg'])
# Reshape into DG high order data structure
sol_reshaped = np.zeros((mesh['plocal'].shape[0], mesh['t'].shape[0]))
exact_reshaped = np.zeros((mesh['plocal'].shape[0], mesh['t'].shape[0]))
for i in np.arange(mesh['t'].shape[0]):          # Set dirichlet BC
    sol_reshaped[:, i] = sol[mesh['tcg'][i, :]]
    exact_reshaped[:, i] = exact[mesh['tcg'][i, :]]

error = exact_reshaped.ravel()-sol_reshaped.ravel()
print('L-inf error: '+str(round(np.linalg.norm(error, np.inf), 5)))
########## CALC DERIVATIVES ##########

# Verify this derivative calc
print('Calculating derivatives')
grad = calc_derivatives(mesh, master, sol_reshaped, ndim)
result = np.concatenate((sol_reshaped[:,None,:], grad.transpose(1,2,0)), axis=1)

# ########## VISUALIZE SOLUTION ##########
viz_driver(mesh, master, result, vis_filename, call_pv)
