import numpy as np
import sys
sys.path.append('../../util')
import pickle
import helper

with open('mesh_save', 'rb') as file:
    mesh = pickle.load(file)

with open('approx_charge_vec.npy', 'rb') as file:
    approx_charge=np.load(file)

approx_charge = helper.reshape_field(mesh, approx_charge, 'to_array', 'scalars', porder=1)
