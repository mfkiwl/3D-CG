import numpy as np
import pickle
from pathlib import Path
import sys
from sklearn.cluster import KMeans
from sympy import centroid
import field_orientation
# import calc_capacitance
import leader_inception
import yaml
import matplotlib.pyplot as plt

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
from functools import partial
import attachment

# Read in solution array

def attachment_preprocessing(soln_file, rad_fuselage, vtu_dir, num_1d_angle_pts):
    """
    Sets up data structures for computing the attachment points. Does not perform the integration or evaluate the leader inception criteria

    """

    with open(soln_file, 'rb') as file:
        solution = pickle.load(file)

    surf_mesh = solution['surf_mesh']
    __, __, __, __, surf_mesh['corner'], _, _ = masternodes.masternodes(surf_mesh['porder'], surf_mesh['ndim'])      # Adding in the corner indexing because that wasn't added in the main solver - was just added but the sims would have to be re-run

    alpha_vec = np.linspace(0, 360, num=num_1d_angle_pts, endpoint=False)
    phi_vec = np.linspace(0, 360, num=num_1d_angle_pts, endpoint=False)
    alpha, phi = np.meshgrid(alpha_vec, phi_vec, indexing='ij')

    # Each element in data_dict is an E field orientation, stored as a tuple representing an index pair into the master array that will be constructed after the fact

    # Precompute element centroids to help speed up the process of searching for which elements should be integrated over
    print('Computing element centroids')
    elem_centroids = np.zeros((surf_mesh['tcg'].shape[0], 3))   # 3D
    for eidx, elem in enumerate(surf_mesh['tcg']):
        if eidx %100000 == 0:
            print(eidx,'/',surf_mesh['tcg'].shape[0])
        elem_centroids[eidx,:] = np.mean(surf_mesh['pcg'][elem,:], axis=0)

    angles = list(zip(alpha.ravel(), phi.ravel()))

    iter_idx = np.arange(alpha.ravel().shape[0])
    print(iter_idx[-1]+1, 'angle pairs to process')
    result = list(map(partial(attachment_preprocessing_per_angle, solution, angles, surf_mesh, elem_centroids, vtu_dir), iter_idx))

    print('Loading data dictionary')
    data_dict = {}
    for i in np.arange(len(result)):
        data_dict[angles[i]] = result[i]

    return data_dict

def attachment_preprocessing_per_angle(solution, angles, surf_mesh, elem_centroids, vtu_dir, ang_idx):
    ##################### Setting up E field paramters for angle iteration #####################
    alpha, phi = angles[ang_idx]
    print(alpha)
    print(phi)
    __, E_surf = field_orientation.get_E(solution, alpha, phi, 'degrees', 'surf')            

    ##################### Attachment Model #####################
    # Don't forget to do this for the positive and negative leaders!!!!

    # Get the master list of points that will be tested for attachment
    # Get the top 5000 pts and cluster them
    cutoff_idx = 5000
    pts_per_region = 50
    n_clusters=7

    # Threshold on points that are above a certain value
    thresh_val = np.sort(E_surf)[-(cutoff_idx+1)]

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
            cluster_thresh = np.sort(E_surf[surf_labels_top_30 == cluster])[-(pts_per_region)]    # For positive vector

            cluster_thresh_mask = E_surf[surface_labels==cluster]<cluster_thresh

            cut_idx = pcg_idx[surface_labels==cluster][cluster_thresh_mask]
            surf_labels_top_30[cut_idx] = 0
            # surf_labels_top_30[surface_labels==cluster][cluster_thresh_mask] = 0

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
    
    # Will want to insert the feature here to visualize the result points - kick out a VTU
    viz.visualize(surf_mesh, 2, {'scalars':{0: 'top30', 1: 'E dot n'}}, '{}/top30_{:03d}'.format(vtu_dir, ang_idx), False, np.concatenate((surf_labels_top_30[:,None], E_surf[:,None]), axis=1), None, type='surface_mesh') # Can only have scalars on a surface mesh
    
    coords_elem_dict = {}

    print('evaluating coords:', coords_eval.shape[0])
    for ipt, pt in enumerate(coords_eval):

        __, pts_idx, __ = get_pts_within_radius(pt, surf_mesh['pcg'], rad_fuselage)  # Returns a list of point indices
        __, elem_idx, elem_centroid_radii = get_pts_within_radius(pt, elem_centroids, rad_fuselage)  # Returns a list of element indices that are within the radius

        # indicator = np.int64(pts_idx>0)[:,None]
        # viz.visualize(surf_mesh, 2, {'scalars':{0: 'pts'}}, 'surface_plot', True, radius[:,None], None, type='surface_mesh') # Can only have scalars on a surface mesh

        # Partition elements into clearly within the radius, and possibly near the boundary
        inner_elems = elem_idx[elem_centroid_radii < 0.9*rad_fuselage]
        outer_elems = np.setdiff1d(elem_idx, inner_elems)

        # Figure out which elements are completely contained within those points - get a list of the elements that will be integrated over at each point
        pts_idx = pts_idx[pts_idx>0]   # Extracting smaller vector from the global index array
        
        elem_list = np.copy(inner_elems).tolist()
        pt_dict = {str(pt):None for pt in pts_idx}
        for eidx, elem in enumerate(surf_mesh['tcg'][outer_elems]):
            elem0 = elem[surf_mesh['corner'][0]]
            elem1 = elem[surf_mesh['corner'][1]]
            elem2 = elem[surf_mesh['corner'][2]]

            if (str(elem0) in pt_dict) and (str(elem1) in pt_dict) and (str(elem2) in pt_dict):
                elem_list.append(outer_elems[eidx])

        coords_elem_dict[surf_labels_top_30_nonzero[ipt]] = elem_list

    # Store the global list of nodes in the dictionary
    print()

    return coords_elem_dict
    
def get_pts_within_radius(pt, pts, r):
    diff = pts-pt
    radius = np.sum(diff**2, axis=1)**0.5 # 1D array
    pts_idx = np.arange(pts.shape[0])

    pts_out = pts[radius <= r]
    pts_idx = pts_idx[radius <= r]
    radius = radius[radius<=r]

    return pts_out, pts_idx, radius

def compute_attachment(sol_fname, preprocessed_data_dict, integral_type, num_1d_angle_pts):
    """
    Computes the first and second attachment (usually pos then neg leader) points for a set of given input angles


    """

    # Setup
    # Thresholds for leader inception (chosen at standard conditions) and universal constants
    with open('physical_constants.yaml', 'r') as stream:
        phys_param = yaml.load(stream, Loader=yaml.loader.FullLoader)
    phys_param['Einf_0'] = phys_param['pos_corona_stability_field']/30  # Initial background field strength, V/m
    phys_param['Q_neg_crit_leader_incep'] = phys_param['Q_pos_crit_leader_incep']*phys_param['Q_neg_crit_leader_incep_factor']

    with open(sol_fname, 'rb') as file:
        solution = pickle.load(file)

    surf_mesh = solution['surf_mesh']
    __, __, __, __, surf_mesh['corner'], _, _ = masternodes.masternodes(surf_mesh['porder'], surf_mesh['ndim'])      # Adding in the corner indexing because that wasn't added in the main solver - was just added but the sims would have to be re-run

    attach_point_mat = np.zeros((num_1d_angle_pts, num_1d_angle_pts))
    q_opt_mat = np.zeros((num_1d_angle_pts, num_1d_angle_pts))
    q_opt_pos_attach_mat = np.zeros((num_1d_angle_pts, num_1d_angle_pts))
    q_opt_neg_attach_mat = np.zeros((num_1d_angle_pts, num_1d_angle_pts))
    E_field_amplification_mat = np.zeros((num_1d_angle_pts, num_1d_angle_pts))
    # /Setup

    phys_param['capacitance'] = calc_capacitance.calc_capacitance(aircraft)

    attachment_point_dict = {}
    for orientation in preprocessed_data_dict:
        alpha = orientation[0]
        phi = orientation[1]

        # unitE, unitEQ is the electric field solution on the surface given an ambient E field amplitude of 1 and an aircraft charge of 1 V, respectively
        # We'll need the numerical capacitance to determine how the voltage converts to a charge - V = Q/C, or, Q = V/C
        # The variable unitEQ doesn't change 
        unitEQ, unitE = field_orientation.get_E(solution, alpha, phi, 'degrees', integral_type)
        unitEQ *= -1    ##### NOTE!!!!!!!! This is because I messed up the sign on the normal vector calculation and needed to flip the sign! Only relevant for sims that were note re-run!!

        attach_pt1, attach_pt2, Efield_attach, leader1_sign, leader2_sign = attachment.compute_attachment_points(unitE, unitEQ, capacitance, integral_type, phys_param)

        attachment_point_dict[orientation] = {'attachment_pt_1': attach_pt1, 'attachment_pt_2': attach_pt2, 'E_amp_attach': Efield_attach}
    
        optimum_qty = attachment.optimum_charge()

    # Next, with a grab bag of all the attachment points, use k-means clustering to identify the attachment zones - might have to use a looping algo to find the knee
    # TODO

    # Next, associate each point with the attachment zone using the kmeans.labels
    # TODO

    # Plot the first attachment zone array

    # For each orientation, find the optimum Q - put this in a separate function, or possibly even integrate with the first function
    # For a given orientation and Q_ac =0, find the two attachment points and their electric fields
    # If the electric fields are asymmetrical, adjust Q_ac with successive approximation until they are equal to within a tolerance
    # Watch out for things not converging if different attachment points are chosen

    # E_field_amplification_mat = E_field_Q_opt/E_field_baseline

    # Plot the rest of the data
    # Baseline case
    # plt.imshow(attach_point_mat, interpolation='none', extent=[0, 360, 0, 360])
    # Additonally, plot the locations of the second attachment for the baseline case too and show on a different plot or something, the charge of the incepted leader for each
    # ^ Employ some cross-hatching or other indication of which locations the negative leader was incepted first

    # Q optimum case
    # plt.imshow(E_field_amplification_mat, interpolation='none', extent=[0, 360, 0, 360])
    # plt.imshow(q_opt_mat, interpolation='none', extent=[0, 360, 0, 360])
    # plt.imshow(q_opt_pos_attach_mat, interpolation='none', extent=[0, 360, 0, 360])
    # plt.imshow(q_opt_neg_attach_mat, interpolation='none', extent=[0, 360, 0, 360])

if __name__ == '__main__':
    rad_fuselage = 1.75  # m
    num_1d_angle_pts = 4
    integral_type = 'surf'

    sol_fname = '/media/homehd/saustin/lightning_research/3D-CG/postprocessing/fem_solutions/d8/d8_electrostatic_solution'
    vtu_dir = '/media/homehd/saustin/lightning_research/3D-CG/postprocessing/fem_solutions/d8/top30_vtus'

    preprocessed_data_dict = attachment_preprocessing(sol_fname, rad_fuselage, vtu_dir, num_1d_angle_pts)

    # with open('/media/homehd/saustin/lightning_research/3D-CG/postprocessing/fem_solutions/d8/' + 'attachment_preprocessed_data_dict', 'wb') as file:
    #     pickle.dump(preprocessed_data_dict, file)

    with open('/media/homehd/saustin/lightning_research/3D-CG/postprocessing/fem_solutions/d8/' + 'attachment_preprocessed_data_dict', 'rb') as file:
        preprocessed_data_dict = pickle.load(file)

    compute_attachment(sol_fname, preprocessed_data_dict, integral_type, num_1d_angle_pts)