import numpy as np
import pickle
from pathlib import Path
import sys
import pyvista as pv

# Finding the sim root directory
cwd = Path.cwd()
for dirname in tuple(cwd.parents):
    if dirname.name == '3D-CG':
        sim_root_dir = dirname
        continue

sys.path.append(str(sim_root_dir.joinpath('util')))
sys.path.append(str(sim_root_dir.joinpath('mesh')))
sys.path.append(str(sim_root_dir.joinpath('master')))
sys.path.append(str(sim_root_dir.joinpath('viz')))
import viz
import helper

# Read in solution array
soln = 'd8'

mu_e = 2.7e-4
sign_q = -1

q = .1e-3   # aircraft charge, in mC
C = 840e-12 # aircarft capacitance, in pF
E_inf = 1000
u_inf = 262

V = q/C

print('Reading in data')
with open('./fem_solutions/d8/d8_electrostatic_solution', 'rb') as file:
    solution = pickle.load(file)
flow = pv.read('./fem_solutions/d8/flow.vtu')

E = -V*solution['Phi_grad_vol']  #+E_inf*0 # Add external electric field here
E_surf = -V*solution['Phi_grad_normal_surf']  #+E_inf*0 # Add external electric field here

mesh = solution['vol_mesh']
vE = sign_q*mu_e*E
vE_surf = sign_q*mu_e*E_surf

momentum = flow.point_data['Momentum']
rho = flow.point_data['Density'][:,None]
vFlow = momentum/rho * u_inf

print('Interpolating flowfield to high order mesh')
vectors = helper.reshape_field(mesh, vFlow, 'to_array', 'scalars', porder=1)
_, vectors = helper.interpolate_high_order(1, mesh['porder'], mesh['ndim'], lo_scalars=None, lo_vectors=vectors)
# Reshape back into the column vector of high order
vFlowHO = helper.reshape_field(mesh, vectors, 'to_column', 'scalars')

combined_v = np.concatenate((vFlowHO, vE, vFlowHO+vE, vFlowHO-vE), axis=1)

print('Visualizing')
labels = {'vectors':{0: 'V field - flow', 1: 'V field - electrostatic', 2: 'Combined - pos charge', 3: 'Combined - neg charge'}}
viz.visualize(mesh, 2, labels, 'combined', True, scalars=None, vectors=combined_v)

viz.visualize(solution['surf_mesh'], 2, {'scalars':{0: 'E dot n'}}, 'surface_plot', False, vE_surf[:,None], None, type='surface_mesh') # Can only have scalars on a surface mesh