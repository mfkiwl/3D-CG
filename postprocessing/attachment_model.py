import numpy as np
import pickle
from pathlib import Path
import sys
from sklearn.cluster import KMeans
from sympy import centroid
import field_orientation
# import calc_capacitance
import yaml
import matplotlib.pyplot as plt
import matplotlib as mpl

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
sys.path.append(str(sim_root_dir.joinpath('CG')))
sys.path.append(str(sim_root_dir.joinpath('logging')))
import calc_capacitance
import masternodes
import viz
from functools import partial
import attachment
import logger_cfg
import datetime

# Read in solution array
# def attachment_preprocessing(soln_file, vtu_dir, num_1d_angle_pts):
#     """
#     Sets up data structures for computing the attachment points. Does not perform the integration or evaluate the leader inception criteria

#     THIS FUNCTION IS NOW DEPRECATED!!!!!!!!! WE ARE NO LONGER PRECOMPUTING THE POSSIBLE ATTACHMENT POINTS BECAUSE IT'S NOT COMPATIBLE WITH CHANGING THE AIRCRAFT CHARGE
#     """

#     with open(soln_file, 'rb') as file:
#         solution = pickle.load(file)

#     surf_mesh = solution['surf_mesh']
#     r_fuselage = surf_mesh['r_fuselage']
#     __, __, __, __, surf_mesh['corner'], _, _ = masternodes.masternodes(surf_mesh['porder'], surf_mesh['ndim'])      # Adding in the corner indexing because that wasn't added in the main solver - was just added but the sims would have to be re-run

#     theta_vec = np.linspace(0, 360, num=num_1d_angle_pts, endpoint=False)
#     phi_vec = np.linspace(0, 360, num=num_1d_angle_pts, endpoint=False)
#     theta, phi = np.meshgrid(theta_vec, phi_vec, indexing='ij')
#     angles = list(zip(theta.ravel(), phi.ravel()))
#     iter_idx = np.arange(theta.ravel().shape[0])

#     # Each element in data_dict is an E field orientation, stored as a tuple representing an index pair into the master array that will be constructed after the fact

#     # Precompute element centroids to help speed up the process of searching for which elements should be integrated over
#     logger.info('Computing element centroids')
#     elem_centroids = np.zeros((surf_mesh['tcg'].shape[0], 3))   # 3D
#     for eidx, elem in enumerate(surf_mesh['tcg']):
#         if eidx %100000 == 0:
#             logger.info(eidx,'/',surf_mesh['tcg'].shape[0])
#         elem_centroids[eidx,:] = np.mean(surf_mesh['pcg'][elem,:], axis=0)

#     logger.info(iter_idx[-1]+1, 'angle pairs to process')
#     result = list(map(partial(attachment_preprocessing_per_angle, solution, angles, surf_mesh, elem_centroids, vtu_dir), iter_idx))

#     logger.info('Loading data dictionary')
#     data_dict = {}
#     for i in np.arange(len(result)):
#         data_dict[angles[i]] = result[i]

#     return data_dict

def compute_possible_attachment_points(E_surf, surf_mesh, elem_centroids, vtu_dir, sign_flag, r_limit):
    """
    Computes points on which it may be possible to see an attachment - these will be monitored during the attachment processing in the next step.

    For each point, identifies a list of elements surrounding the point - these elements in the sphere of influence will be integrated over in the attachment script.

    Possible tests: Look at visualized result to see that the sign is correct (positive points marked for positive, negative for negative surface E dot n, this can be done for any angle)
    Also check to make sure the sphere of influence is being computed correctly.

    r_limit has to be dimensional!!

    """

    ##################### Attachment Model #####################
    # Don't forget to do this for the positive and negative leaders!!!!

    # Get the master list of points that will be tested for attachment
    # Get the top 5000 pts and cluster them
    cutoff_idx = 5000
    pts_per_region = 3
    n_clusters=7

    # Threshold on points that are above a certain value
    if sign_flag == 'pos':
        thresh_val = np.sort(E_surf)[-cutoff_idx]
        # logger.info(thresh_val)
        # logger.info(np.max(E_surf))
    elif sign_flag == 'neg':
        thresh_val = np.sort(E_surf)[cutoff_idx]
        # logger.info(thresh_val)
        # logger.info(np.min(E_surf))

    surface_labels = np.copy(E_surf)

    if sign_flag == 'pos':
        surface_labels[E_surf<thresh_val] = 0      # Null out all values that don't meet the threshold
    elif sign_flag == 'neg':
        surface_labels[E_surf>thresh_val] = 0      # Null out all values that don't meet the threshold

    val_idx = np.nonzero(surface_labels)[0]

    # Find their (x,y,z coords)
    coords=surf_mesh['pcg'][val_idx,:]

    # logger.info(coords.shape[0], 'points meet the threshold of ', thresh_val)
    # logger.info('running k means')

    # Compute k-means
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=300, random_state=0, algorithm='elkan').fit(coords)

    surface_labels[val_idx] = kmeans.labels_+1
    # logger.info(kmeans.cluster_centers_)
    # viz.visualize(surf_mesh, 2, {'scalars':{0: 'top30', 1: 'E dot n'}}, 'out'+sign_flag, True, np.concatenate((surface_labels[:,None], E_surf[:,None]), axis=1), None, type='surface_mesh') # Can only have scalars on a surface mesh

    # Get the top 50 for each region and plot
    surf_labels_top_30 = np.copy(surface_labels)
    pcg_idx = np.arange(surface_labels.shape[0])
    for cluster in np.arange(n_clusters)+1:
        if surface_labels[surface_labels==cluster].shape[0] > pts_per_region:
            if sign_flag == 'pos':
                cluster_thresh = np.sort(E_surf[surf_labels_top_30 == cluster])[-pts_per_region]
                cluster_thresh_mask = E_surf[surface_labels==cluster]<cluster_thresh
            elif sign_flag == 'neg':
                cluster_thresh = np.sort(E_surf[surf_labels_top_30 == cluster])[pts_per_region]
                cluster_thresh_mask = E_surf[surface_labels==cluster]>=cluster_thresh

            cut_idx = pcg_idx[surface_labels==cluster][cluster_thresh_mask]
            surf_labels_top_30[cut_idx] = 0
        else:
            continue
    # viz.visualize(surf_mesh, 2, {'scalars':{0: 'top30', 1: 'E dot n'}}, 'out'+sign_flag, True, np.concatenate((surf_labels_top_30[:,None], E_surf[:,None]), axis=1), None, type='surface_mesh') # Can only have scalars on a surface mesh
    # exit()

    # Pull points corresponding to a cluster
    # logger.info(np.where(surf_labels_top_30 == 6))
    # exit()

    coords_eval = surf_mesh['pcg'][surf_labels_top_30>0,:]
    pcg_idx_global = pcg_idx[surf_labels_top_30>0]
    coords_elem_dict = {}

    # logger.info('evaluating coords:', coords_eval.shape[0])
    for ipt, pt in enumerate(coords_eval):

        __, pts_idx, __ = get_pts_within_radius(pt, surf_mesh['pcg'], r_limit)  # Returns a list of point indices
        __, elem_idx, elem_centroid_radii = get_pts_within_radius(pt, elem_centroids, r_limit)  # Returns a list of element indices that are within the radius




        # # Partition elements into clearly within the radius, and possibly near the boundary
        # inner_elems = elem_idx[elem_centroid_radii < 0.9*r_limit]
        # outer_elems = np.setdiff1d(elem_idx, inner_elems)

        # # Figure out which elements are completely contained within those points - get a list of the elements that will be integrated over at each point
        # pts_idx = pts_idx[pts_idx>0]   # Extracting smaller vector from the global index array
        
        # elem_list = np.copy(inner_elems).tolist()
        # pt_dict = {str(pt):None for pt in pts_idx}
        # for eidx, elem in enumerate(surf_mesh['tcg'][outer_elems]):
        #     elem0 = elem[surf_mesh['corner'][0]]
        #     elem1 = elem[surf_mesh['corner'][1]]
        #     elem2 = elem[surf_mesh['corner'][2]]

        #     if (str(elem0) in pt_dict) and (str(elem1) in pt_dict) and (str(elem2) in pt_dict):
        #         elem_list.append(outer_elems[eidx])

        # coords_elem_dict[pcg_idx_global[ipt]] = elem_list


        coords_elem_dict[pcg_idx_global[ipt]] = elem_idx

        # if pcg_idx_global[ipt] == 229096 or pcg_idx_global[ipt] == 227827 or pcg_idx_global[ipt] == 12633:
        #     # Visualize
        #     elem_indicator = np.zeros((surf_mesh['t'].shape[0], 1))
        #     elem_indicator[elem_idx] = 1
        #     # elem_indicator[elem_list] = 2
        #     # elem_indicator[inner_elems] = 3
        #     viz.generate_vtu(surf_mesh['p'], surf_mesh['t'], None, None, {'cell_data': {0: 'Radius'}}, 'test_radius{}'.format(pcg_idx_global[ipt]), True, cell_data=elem_indicator)
        #     # logger.info('exiting')
        #     # exit()

    # Store the global list of nodes in the dictionary
    return coords_elem_dict
    
def get_pts_within_radius(pt, pts, r):
    diff = pts-pt
    radius = np.sum(diff**2, axis=1)**0.5 # 1D array
    pts_idx = np.arange(pts.shape[0])

    pts_out = pts[radius <= r]
    pts_idx = pts_idx[radius <= r]
    radius = radius[radius<=r]

    return pts_out, pts_idx, radius

def compute_attachment(solution, integral_type, vtu_dir, eps, summaries_dname, phi_ang_start, phi_ang_end, numpts_theta, numpts_phi, r_fuselage=None):
    """
    Computes the first and second attachment (usually pos then neg leader) points for a set of given input angles

    """
    logger = logger_cfg.initialize_logger('{}attachment_d8_{}_{}'.format(summaries_dname, phi_ang_start, phi_ang_end))
    logger.info('*************************** INITIALIZING ATTACHMENT ANALYSIS ' +str(datetime.datetime.now())+' ***************************')

    theta_idx_vec = np.arange(numpts_theta)
    phi_idx_vec = np.arange(numpts_phi)
    theta_vec = np.linspace(0, 180, num=numpts_theta, endpoint=False)
    phi_vec = np.linspace(phi_ang_start, phi_ang_end, num=numpts_phi, endpoint=False)
    theta_idx_mat, theta_idx_mat = np.meshgrid(theta_idx_vec, phi_idx_vec, indexing='ij')
    angle_idxs = list(zip(theta_idx_mat.ravel(), theta_idx_mat.ravel()))

    logger.info('Plan:')
    logger.info('Phi min/max: {} {}'.format(phi_vec, phi_vec))
    logger.info('Theta min/max: {} {}'.format(theta_vec, theta_vec))
    logger.info('')

    # Setup
    # Thresholds for leader inception (chosen at standard conditions) and universal constants
    with open('physical_constants.yaml', 'r') as stream:
        phys_param = yaml.load(stream, Loader=yaml.loader.FullLoader)

    phys_param['Einf_0'] = phys_param['pos_corona_stability_field']/30  # Initial background field strength, V/m
    phys_param['Q_neg_crit_leader_incep'] = phys_param['Q_pos_crit_leader_incep']*phys_param['Q_neg_crit_leader_incep_factor']

    surf_mesh = solution['surf_mesh']
    __, __, __, __, surf_mesh['corner'], _, _ = masternodes.masternodes(surf_mesh['porder'], surf_mesh['ndim'])      # Adding in the corner indexing because that wasn't added in the main solver - was just added but the sims would have to be re-run

    if r_fuselage is None:  # For when the sims support the fuselage radius outputting
        r_fuselage = surf_mesh['r_fuselage']

    r_limit = r_fuselage*phys_param['d_R']    # Dimensional, in meters, the radius to include for integration of the corona charge criteria

    baseline_Eattach_mat = np.zeros_like(theta_idx_mat)
    q_opt_mat = np.zeros_like(theta_idx_mat)
    q_opt_Eattach_mat = np.zeros_like(theta_idx_mat)

    logger.info('Computing element centroids')
    elem_centroids = attachment.get_avg_field_on_elements(surf_mesh, surf_mesh['pcg'])

    logger.info('Calculating capacitance')
    # phys_param['capacitance'] = calc_capacitance.calc_capacitance(solution, phys_param)
    phys_param['capacitance'] = 1.022050723270232e-09   # F
    logger.info('Capacitance: '.format(phys_param['capacitance']))

    baseline_attach_pt1_vec = np.zeros((len(angle_idxs)))
    baseline_attach_pt2_vec = np.zeros((len(angle_idxs)))
    baseline_leader_sign_vec = np.zeros((len(angle_idxs), 2))

    q_opt_pos_attach_vec = np.zeros((len(angle_idxs)))
    q_opt_neg_attach_vec = np.zeros((len(angle_idxs)))

    for flattened_ang_idx, orientation in enumerate(angle_idxs):
        theta_idx = orientation[0]
        phi_idx = orientation[1]

        theta = theta_vec[theta_idx]
        phi = phi_vec[phi_idx]

        logger.info('------------- theta: {}, phi: {} -------------'.format(theta, phi))

        # unitE, unitEQ is the electric field solution on the surface given an ambient E field amplitude of 1 and an aircraft potential of 1 V, respectively
        # We'll need the numerical capacitance to determine how the voltage converts to a charge - V = Q/C
        # The variable unitEQ isn't modified inside get_E, it's just returned from 'solution'
        unitEQ, unitE = field_orientation.get_E(solution, theta, phi, 'degrees', integral_type)

        # viz.visualize(solution['surf_mesh'], 2, {'scalars':{0: 'Field'}}, 'surface_plot', True, solution['Ex_grad_normal_surf'][:,None], None, type='surface_mesh') # Can only have scalars on a surface mesh
        # exit()

        # Note that we only care about the E-field from the external field, and not due to the aircraft charge for the baseline case
        # TODO: change this so it uses the surface integral quantites for the possible attachment points but then can use either surf or volume integrals for the integration
        candidate_attach_pts_pos = compute_possible_attachment_points(unitE, surf_mesh, elem_centroids, vtu_dir, 'pos', r_limit)
        candidate_attach_pts_neg = compute_possible_attachment_points(unitE, surf_mesh, elem_centroids, vtu_dir, 'neg', r_limit)

        # logger.info()
        # logger.info('visualizing')
        # surf_pts = np.zeros((surf_mesh['pcg'].shape[0], 1))
        # surf_pts[list(candidate_attach_pts_pos.keys())] = 1
        # surf_pts[list(candidate_attach_pts_neg.keys())] = 2
        # data = np.concatenate((surf_pts, unitE[:,None]), axis=1)
        # viz.visualize(solution['surf_mesh'], 2, {'scalars':{0: 'Pts', 1: 'Field'}}, 'surface_plot', True, data, None, type='surface_mesh') # Can only have scalars on a surface mesh
        # logger.info('exiting')
        # exit()

        # logger.info(candidate_attach_pts_pos[229096])
        # logger.info()
        # logger.info(candidate_attach_pts_pos[229099])
        # logger.info()
        # logger.info(candidate_attach_pts_pos[229129])
        # logger.info()
        # logger.info(candidate_attach_pts_pos[229132])
        # logger.info()
        # logger.info(candidate_attach_pts_pos[229155])
        # logger.info()
        # logger.info(candidate_attach_pts_pos[229167])
        # logger.info()
        # logger.info(candidate_attach_pts_pos[229419])
        # logger.info()
        # logger.info(candidate_attach_pts_pos[229270])
        # logger.info()
        # exit()

        # Computing baseline attachment
        attach_pt1_baseline, attach_pt2_baseline, Efield_attach_baseline, leader1_sign_baseline, leader2_sign_baseline = attachment.compute_bidir_attachment_points(solution, unitE, unitEQ, integral_type, phys_param, candidate_attach_pts_pos, candidate_attach_pts_neg, eps)
        logger.info('Baseline attachment summary')
        logger.info('Attach pt 1: {}'.format(surf_mesh['pcg'][attach_pt1_baseline]))
        logger.info('Attach pt 1: {}'.format(attach_pt1_baseline))
        logger.info('Leader 1 sign {}:'.format(leader1_sign_baseline))
        logger.info('Attach pt 2: {}'.format(surf_mesh['pcg'][attach_pt2_baseline]))
        logger.info('Attach pt 2: {}'.format(attach_pt2_baseline))
        logger.info('Leader 2 sign: {}'.format(leader2_sign_baseline))
        logger.info('E field attach: {} kV'.format(Efield_attach_baseline/1000))
        # exit()

        # ##############
        # Can be cut out
        # logger.info('Visualizing')
        # surf_pts = np.zeros((surf_mesh['pcg'].shape[0], 1))
        # surf_pts[attach_pt1_baseline] = 100
        # surf_pts[attach_pt2_baseline] = 200
        # data = np.concatenate((surf_pts, unitE[:,None]), axis=1)
        # viz.visualize(solution['surf_mesh'], 2, {'scalars':{0: 'Pts', 1: 'Field'}}, 'surface_plot', True, data, None, type='surface_mesh') # Can only have scalars on a surface mesh

        # elem_indicator = np.zeros((surf_mesh['t'].shape[0], 1))
        # elem_indicator[candidate_attach_pts_neg[attach_pt1_baseline]] = 1
        # elem_indicator[candidate_attach_pts_pos[attach_pt2_baseline]] = 1
        # viz.generate_vtu(surf_mesh['p'], surf_mesh['t'], None, None, {'cell_data': {0: 'Radius'}}, 'pos_attach_pt', True, cell_data=elem_indicator)
        # exit()
        # ##############

        # Loading arrays
        baseline_attach_pt1_vec[flattened_ang_idx] = attach_pt1_baseline
        baseline_attach_pt2_vec[flattened_ang_idx] = attach_pt2_baseline
        baseline_leader_sign_vec[flattened_ang_idx] = np.array([leader1_sign_baseline, leader2_sign_baseline])
        baseline_Eattach_mat[theta_idx, phi_idx] = Efield_attach_baseline
    
        # Computing aircraft optimum charge and attachment under these conditions
        attach_pt_pos_opt, attach_pt_neg_opt, Qac_opt, Efield_attach_opt = attachment.optimum_charge(solution, unitE, unitEQ, integral_type, eps, phys_param, candidate_attach_pts_pos, candidate_attach_pts_neg)

        logger.info('Optimum charge summary')
        logger.info('Pos attach point: {}'.format(surf_mesh['pcg'][attach_pt_pos_opt]))
        logger.info('Pos attach point: {}'.format(attach_pt_pos_opt))
        logger.info('Neg attach point: {}'.format(surf_mesh['pcg'][attach_pt_neg_opt]))
        logger.info('Neg attach point: {}'.format(attach_pt_neg_opt))
        logger.info('E field attach: {} kV'.format(Efield_attach_opt/1000))
        logger.info('Qac opt: {}'.format(Qac_opt))
        # exit()
        
        # Loading arrays
        q_opt_pos_attach_vec[flattened_ang_idx] = attach_pt_pos_opt
        q_opt_neg_attach_vec[flattened_ang_idx] = attach_pt_neg_opt
        q_opt_Eattach_mat[theta_idx, phi_idx] = Efield_attach_opt
        q_opt_mat[theta_idx, phi_idx] = Qac_opt
        logger.info('')

    # Save output data
    logger.info('Saving data summaries to disk')
    with open('{}baseline_attach_pt1_vec_{}_{}.npy'.format(summaries_dname, phi_ang_start, phi_ang_end), 'wb') as file:
        np.save(file, baseline_attach_pt1_vec)
    with open('{}baseline_attach_pt2_vec_{}_{}.npy'.format(summaries_dname, phi_ang_start, phi_ang_end), 'wb') as file:
        np.save(file, baseline_attach_pt2_vec)
    with open('{}baseline_leader_sign_vec_{}_{}.npy'.format(summaries_dname, phi_ang_start, phi_ang_end), 'wb') as file:
        np.save(file, baseline_leader_sign_vec)
    with open('{}baseline_Eattach_mat_{}_{}.npy'.format(summaries_dname, phi_ang_start, phi_ang_end), 'wb') as file:
        np.save(file, baseline_Eattach_mat)
    with open('{}q_opt_pos_attach_vec_{}_{}.npy'.format(summaries_dname, phi_ang_start, phi_ang_end), 'wb') as file:
        np.save(file, q_opt_pos_attach_vec)
    with open('{}q_opt_neg_attach_vec_{}_{}.npy'.format(summaries_dname, phi_ang_start, phi_ang_end), 'wb') as file:
        np.save(file, q_opt_neg_attach_vec)
    with open('{}q_opt_Eattach_mat_{}_{}.npy'.format(summaries_dname, phi_ang_start, phi_ang_end), 'wb') as file:
        np.save(file, q_opt_Eattach_mat)
    with open('{}q_opt_mat_{}_{}.npy'.format(summaries_dname, phi_ang_start, phi_ang_end), 'wb') as file:
        np.save(file, q_opt_mat)
        
    logger.info('Done!')
    return

def plot_summaries(baseline_first_attach_zones, baseline_leader_sign_vec, baseline_second_attach_zones, E_margin_optimal_charging, q_opt_mat, q_opt_pos_attach_zones, q_opt_neg_attach_zones):
    # To indicate which ones have a negative leader incepted first, try the cross-hatching here: https://stackoverflow.com/questions/14045709/selective-patterns-with-matplotlib-imshow (in crosshatch_test.py)
    ax = plt.gca()
    ax.imshow(baseline_first_attach_zones, interpolation='bicubic', extent=[0, 360, 0, 360])
    for angle_idx, leader_sign_pair in enumerate(baseline_leader_sign_vec):
        theta_idx = angle_idx%num_1d_angle_pts      # Row index, 'i' in output array
        phi_idx = angle_idx//num_1d_angle_pts      # Column index, 'j' in output array
        if leader_sign_pair[0] == -1: # First leader was negative
            ax.add_patch(mpl.patches.Rectangle((theta_idx-.5, phi_idx-.5), 1, 1, hatch='///////', fill=False, snap=False))

    plt.show()
    exit()
    plt.imshow(baseline_second_attach_zones, interpolation='bicubic', extent=[0, 360, 0, 360])

    # DON'T forget to mirror the endpoints for both dimensions
    # Q optimum case
    plt.imshow(E_margin_optimal_charging, interpolation='bicubic', extent=[0, 360, 0, 360])
    plt.imshow(q_opt_mat, interpolation='bicubic', extent=[0, 360, 0, 360])
    plt.imshow(q_opt_pos_attach_zones, interpolation='bicubic', extent=[0, 360, 0, 360])
    plt.imshow(q_opt_neg_attach_zones, interpolation='bicubic', extent=[0, 360, 0, 360])


def kmeans_attach_zoning(attach_pts):
    """
    Use the elbow method here to determine the attachment points: https://stackoverflow.com/questions/19197715/scikit-learn-k-means-elbow-criterion
    attach_pts is an array of (x, y, z) attachment coordinates
    """

    # FIll in the optimization loop
    attachment_zones = KMeans(n_clusters=5, init='k-means++', n_init=10, max_iter=300, random_state=0, algorithm='elkan').fit(attach_pts)

    return attachment_zones

def analysis_postprocessing(solution, summaries_dirname):
    # Read from disk
    with open('{}baseline_attach_pt1_vec.npy'.format(summaries_dirname), 'rb') as file:
        baseline_attach_pt1_vec = np.load(file)
    with open('{}baseline_attach_pt2_vec.npy'.format(summaries_dirname), 'rb') as file:
        baseline_attach_pt2_vec = np.load(file)
    with open('{}baseline_leader_sign_vec.npy'.format(summaries_dirname), 'rb') as file:
        baseline_leader_sign_vec = np.load(file)
    with open('{}baseline_Eattach_mat.npy'.format(summaries_dirname), 'rb') as file:
        baseline_Eattach_mat = np.load(file)

    with open('{}q_opt_pos_attach_vec.npy'.format(summaries_dirname), 'rb') as file:
        q_opt_pos_attach_vec = np.load(file)
    with open('{}q_opt_neg_attach_vec.npy'.format(summaries_dirname), 'rb') as file:
        q_opt_neg_attach_vec = np.load(file)
    with open('{}q_opt_Eattach_mat.npy'.format(summaries_dirname), 'rb') as file:
        q_opt_Eattach_mat = np.load(file)
    with open('{}q_opt_mat.npy'.format(summaries_dirname), 'rb') as file:
        q_opt_mat = np.load(file)

    surf_mesh = solution['surf_mesh']

    # For now, temporary fix
    baseline_attach_pt1_vec = baseline_attach_pt1_vec[:,0].astype(np.int64)
    baseline_attach_pt2_vec = baseline_attach_pt2_vec[:,0].astype(np.int64)

    # surf_pts = np.zeros((surf_mesh['pcg'].shape[0]))
    # surf_pts[baseline_attach_pt1_vec] = 1
    # surf_pts[baseline_attach_pt2_vec] = 2
    # viz.visualize(solution['surf_mesh'], 2, {'scalars':{0: 'Pts'}}, 'surface_plot', True, surf_pts[:,None], None, type='surface_mesh') # Can only have scalars on a surface mesh
    # exit()

    E_margin_optimal_charging = np.abs((q_opt_Eattach_mat-baseline_Eattach_mat)/baseline_Eattach_mat)
    print(E_margin_optimal_charging)
    plt.imshow(E_margin_optimal_charging.T, origin='lower')
    plt.colorbar()
    plt.show()
    exit()

    q_opt_mat*=1e3
    print(q_opt_mat)
    plt.imshow(q_opt_mat.T,origin="lower")
    plt.colorbar()
    plt.show()
    exit()

    # Using the baseline attachment points (combined positive and negative), use k-means clustering to identify the attachment zones
    baseline_attach_pts = np.concatenate((baseline_attach_pt1_vec, baseline_attach_pt2_vec), axis=0)

    attachment_zones = kmeans_attach_zoning(baseline_attach_pts)

    baseline_first_attach_zones = np.reshape(attachment_zones.labels_[:num_1d_angle_pts], num_1d_angle_pts, num_1d_angle_pts)   # Undoing the concatenation from the line before the call to kmeans
    baseline_second_attach_zones = np.reshape(attachment_zones.labels_[num_1d_angle_pts:], num_1d_angle_pts, num_1d_angle_pts)

    # Next, associate each point with the attachment zone and reshape into array
    q_opt_pos_attach_zones = np.reshape(attachment_zones.predict(q_opt_pos_attach_vec), num_1d_angle_pts, num_1d_angle_pts)
    q_opt_neg_attach_zones = np.reshape(attachment_zones.predict(q_opt_neg_attach_vec), num_1d_angle_pts, num_1d_angle_pts)

    plot_summaries(baseline_first_attach_zones, baseline_leader_sign_vec, baseline_second_attach_zones, E_margin_optimal_charging, q_opt_mat, q_opt_pos_attach_zones, q_opt_neg_attach_zones)


if __name__ == '__main__':

    phi_ang_start = float(sys.argv[1])
    phi_ang_end = float(sys.argv[2])

    rad_fuselage = 1.75  # m
    numpts_theta = 120
    numpts_phi = 12

    integral_type = 'surf'
    eps=0.05

    sol_fname = '/media/homehd/saustin/lightning_research/3D-CG/postprocessing/fem_solutions/d8/d8_electrostatic_solution'
    vtu_dir = '/media/homehd/saustin/lightning_research/3D-CG/postprocessing/fem_solutions/d8/top30_vtus'

    summaries_dname = 'attachment_data_d8/'

    with open(sol_fname, 'rb') as file:
        solution = pickle.load(file)

    solution['Phi_grad_normal_surf'] *= -1
    ##### NOTE!!!!!!!! This is because I messed up the sign on the normal vector calculation and needed to flip the sign! Only relevant for sims that were note re-run!!

    compute_attachment(solution, integral_type, vtu_dir, eps, summaries_dname, phi_ang_start, phi_ang_end, numpts_theta, numpts_phi, r_fuselage=rad_fuselage)

    # print(solution['surf_mesh']['p'].shape)
    # exit()
    # analysis_postprocessing(solution, summaries_dname)