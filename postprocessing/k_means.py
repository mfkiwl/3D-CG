import numpy as np
import pickle
from pathlib import Path
import sys
from sklearn.cluster import KMeans

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

# Read in solution array

with open('./test_data/d8_Phi_surface_scalars.npy', 'rb') as file:
    surf_field = np.load(file)[:,1]     # Surface E-field is the second column, 1D array

with open('./test_data/d8_Phisurface_mesh', 'rb') as file:
    surf_mesh = pickle.load(file)

# Threshold on points that are above a certain value
thresh_val = -0.3

surf_field[surf_field>=thresh_val] = 0      # Null out all values that are don't meet the threshold
# surf_field[surf_field<thresh_val] = 1     # Binary black/white thresholding

val_idx = np.nonzero(surf_field)[0]

# Find their (x,y,z coords)
coords=surf_mesh['pcg'][val_idx,:]


print(coords.shape[0], 'points meet the threshold of ', thresh_val)
print('running k means')

# Compute k-means
n_clusters=7
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=300, random_state=None, algorithm='elkan').fit(coords)

surf_field[val_idx] = kmeans.labels_+1

# Visualize
print('visualizing')
viz.visualize(surf_mesh, 2, {'scalars':{0: 'Threshold Normal Field'}}, 'surface_plot', True, surf_field[:,None], None, type='surface_mesh') # Can only have scalars on a surface mesh
