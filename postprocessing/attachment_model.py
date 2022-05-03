import numpy as np
import pickle
from pathlib import Path
import sys
from sklearn.cluster import KMeans
from sympy import centroid
import field_orientation
# import calc_capacitance

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
import masternodes
import multiprocessing as mp
from functools import partial
from pathos.multiprocessing import ProcessingPool as Pool

# Read in solution array

def attachment_preprocessing(soln_file, rad_fuselage):
    """
    Sets up data structures for computing the attachment points. Does not perform the integration or evaluate the leader inception criteria

    """

    with open(soln_file, 'rb') as file:
        solution = pickle.load(file)

    surf_mesh = solution['surf_mesh']
    __, __, __, __, surf_mesh['corner'], _, _ = masternodes.masternodes(surf_mesh['porder'], surf_mesh['ndim'])      # Adding in the corner indexing because that wasn't added in the main solver - was just added but the sims would have to be re-run

    alpha_vec = np.linspace(0, 360, num=3, endpoint=False)
    phi_vec = np.linspace(0, 360, num=3, endpoint=False)
    alpha, phi = np.meshgrid(alpha_vec, phi_vec, indexing='ij')
    # capacitance = calc_capacitance.capacitance(solution)
    capacitance = 1

    # Each element in data_dict is an E field orientation, stored as a tuple representing an index pair into the master array that will be constructed after the fact

    # Precompute element centroids to help speed up the process of searching for which elements should be integrated over
    print('Computing element centroids')
    elem_centroids = np.zeros((surf_mesh['tcg'].shape[0], 3))   # 3D
    for eidx, elem in enumerate(surf_mesh['tcg']):
        if eidx %100000 == 0:
            print(eidx,'/',surf_mesh['tcg'].shape[0])
        elem_centroids[eidx,:] = np.mean(surf_mesh['pcg'][elem,:], axis=0)

    pool = Pool(mp.cpu_count())

    angles = list(zip(alpha.ravel(), phi.ravel()))

    iter_idx = np.arange(alpha.ravel().shape[0])
    print(iter_idx)
    result = pool.map(partial(attachment_preprocessing_parallel, solution, angles, surf_mesh, capacitance, elem_centroids), iter_idx)
    # result = list(map(partial(attachment_preprocessing_parallel, solution, angles, surf_mesh, capacitance, elem_centroids), iter_idx))

    print('Loading data dictionary')
    data_dict = {}
    for i in np.arange(len(result)):
        data_dict[angles[i]] = result[i]

    return data_dict

def attachment_preprocessing_parallel(solution, angles, surf_mesh, capacitance, elem_centroids, ang_idx):
    ##################### Setting up E field paramters for angle iteration #####################
    alpha, phi = angles[ang_idx]
    print(alpha)
    print(phi)
    print()
    E_Q_vol, E_ext_vol, E_Q_surf, E_ext_surf = field_orientation.get_E(solution, alpha, phi, 'degrees')            

    # TODO: Implement binary search to optimize this
    E_inf = 1
    Q_ac = 0

    E_Q_vol *= Q_ac/capacitance     # Omitting Rf
    E_Q_surf *= Q_ac/capacitance     # Omitting Rf

    E_vol = E_Q_vol + E_inf*E_ext_vol
    E_surf = E_Q_surf + E_inf*E_ext_surf

    ##################### Attachment Model #####################
    # Don't forget to do this for the positive and negative leaders!!!!

    # Get the master list of points that will be tested for attachment
    # Get the top 5000 pts and cluster them
    cutoff_idx = 5000
    pts_per_region = 30
    n_clusters=7

    # Threshold on points that are above a certain value
    thresh_val = np.sort(E_ext_surf)[-(cutoff_idx+1)]

    surface_labels = np.copy(E_surf)
    surface_labels[E_surf<=thresh_val] = 0      # Null out all values that are don't meet the threshold

    val_idx = np.nonzero(surface_labels)[0]

    # Find their (x,y,z coords)
    coords=surf_mesh['pcg'][val_idx,:]

    # print(coords.shape[0], 'points meet the threshold of ', thresh_val)
    # print('running k means')

    # Compute k-means
    # TODO: knee analysis to find optimum number of clusters
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=300, random_state=0, algorithm='elkan').fit(coords)

    surface_labels[val_idx] = kmeans.labels_+1

    # Get the top 50 for each region and plot
    surf_labels_top_30 = np.copy(surface_labels)
    pcg_idx = np.arange(surface_labels.shape[0])
    for cluster in np.arange(n_clusters)+1:
        if surface_labels[surface_labels==cluster].shape[0] > pts_per_region:
            cluster_thresh = np.sort(E_surf[surf_labels_top_30 == cluster])[-(pts_per_region+1)]    # For positive vector

            cluster_thresh_mask = E_surf[surface_labels==cluster]<cluster_thresh

            cut_idx = pcg_idx[surface_labels==cluster][cluster_thresh_mask]
            surf_labels_top_30[cut_idx] = 0
            surf_labels_top_30[surface_labels==cluster][cluster_thresh_mask] = 0

        else:
            continue

    # Visualize
    # print('visualizing')
    # concat = np.concatenate((E_surf[:,None], surface_labels[:,None], surf_labels_top_50[:,None]), axis=1)
    # viz.visualize(solution['surf_mesh'], 2, {'scalars':{0: 'Normal Field', 1: 'Thresh top 5000', 2: 'Thresh top 50 per region'}}, 'surface_plot', True, concat, None, type='surface_mesh') # Can only have scalars on a surface mesh

    # Combine top 50 in each region to a boolean mask across the entire domain
    # pts_eval_boolean_mask = np.int64(surf_labels_top_50>0)

    surf_labels_top_30_nonzero = surf_labels_top_30[surf_labels_top_30>0]
    coords_eval = surf_mesh['pcg'][surf_labels_top_30>0,:]
    coords_elem_dict = {}
    
    # print('evaluating coords:', coords_eval.shape[0])
    for ipt, pt in enumerate(coords_eval):

        __, pts_idx, __ = get_pts_within_radius(pt, surf_mesh['pcg'], rad_fuselage)  # Returns a list of point indices
        __, elem_idx, elem_centroid_radii = get_pts_within_radius(pt, elem_centroids, rad_fuselage)  # Returns a list of element indices that are within the radius

        # indicator = np.int64(pts_idx>0)[:,None]
        # viz.visualize(surf_mesh, 2, {'scalars':{0: 'pts'}}, 'surface_plot', True, radius[:,None], None, type='surface_mesh') # Can only have scalars on a surface mesh

        # Partition elements into clearly within the radius, and possibly near the boundary
        inner_elems = elem_idx[elem_centroid_radii < 0.9*rad_fuselage]
        outer_elems = np.setdiff1d(elem_idx, inner_elems)

        # print(elem_idx.shape)
        # print(inner_elems.shape)
        # print(outer_elems.shape)

        # print('------------')
        # Figure out which elements are completely contained within those points - get a list of the elements that will be integrated over at each point
        pts_idx = pts_idx[pts_idx>0]   # Extracting smaller vector from the global index array
        
        elem_list = np.copy(inner_elems).tolist()
        pt_dict = {str(pt):None for pt in pts_idx}
        for eidx, elem in enumerate(surf_mesh['tcg'][outer_elems]):
            # if eidx % 100 == 0:
                # print(eidx)
            elem0 = elem[surf_mesh['corner'][0]]
            elem1 = elem[surf_mesh['corner'][1]]
            elem2 = elem[surf_mesh['corner'][2]]

            if (str(elem0) in pt_dict) and (str(elem1) in pt_dict) and (str(elem2) in pt_dict):
                elem_list.append(outer_elems[eidx])

        # # Visualize
        # elem_indicator = np.zeros((surf_mesh['t'].shape[0], 1))
        # elem_indicator[elem_idx] = 1
        # elem_indicator[elem_list] = 2
        # elem_indicator[inner_elems] = 3
        # viz.generate_vtu(surf_mesh['p'], surf_mesh['t'], None, None, {'cell_data': {0: 'Radius'}}, 'test_radius', True, cell_data=elem_indicator)

        coords_elem_dict[surf_labels_top_30_nonzero[ipt]] = elem_list

    # Store the global list of nodes in the dictionary
    return coords_elem_dict
    

def get_pts_within_radius(pt, pts, r):
    diff = pts-pt
    radius = np.sum(diff**2, axis=1)**0.5 # 1D array
    pts_idx = np.arange(pts.shape[0])

    pts_out = pts[radius <= r]
    pts_idx = pts_idx[radius <= r]
    radius = radius[radius<=r]

    # Uncomment to output format required for plotting
    # pts_idx[radius > r] = 0
    # radius_orig = np.copy(radius)
    # radius=(1/(radius+.01))
    # radius[radius_orig>r] = 0
    # return pts_out, pts_idx, radius

    return pts_out, pts_idx, radius

def eval_inception_criteria(mesh, elem_list, E_crit, E_inception):
    pass

def compute_attachment():
    # Choose an initial E field
    E_0 = 50e6

    # Can now enter the iteration loop to compute the field inception criteria - binary search

    eval_inception_criteria



    # Return the point at which the first attachment will occur


if __name__ == '__main__':
    rad_fuselage = 1.75  # m
    sol_fname = '/media/homehd/saustin/lightning_research/3D-CG/postprocessing/fem_solutions/d8/d8_electrostatic_solution'

    preprocessed_data_dict = attachment_preprocessing(sol_fname, rad_fuselage)

    with open('/media/homehd/saustin/lightning_research/3D-CG/postprocessing/fem_solutions/d8/' + 'attachment_preprocessed_data_dict', 'wb') as file:
        pickle.dump(preprocessed_data_dict, file)
        