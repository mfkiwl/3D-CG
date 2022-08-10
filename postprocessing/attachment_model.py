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
    theta_idx_mat, phi_idx_mat = np.meshgrid(theta_idx_vec, phi_idx_vec, indexing='ij')
    angle_idxs = list(zip(theta_idx_mat.ravel(), phi_idx_mat.ravel()))

    logger.info('Plan:')
    logger.info('Number of theta points: {}'.format(numpts_theta))
    logger.info('Number of phi points: {}'.format(numpts_phi))
    logger.info('Phi min/max: {} {}'.format(phi_vec, phi_vec))
    logger.info('Theta min/max: {} {}'.format(np.min(theta_vec), np.max(theta_vec)))
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
    q_opt_mat = np.zeros_like(theta_idx_mat).astype(float)
    q_opt_Eattach_mat = np.zeros_like(theta_idx_mat)

    theta_mat = np.zeros_like(theta_idx_mat)
    phi_mat = np.zeros_like(theta_idx_mat)

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

        # if phi != 333:
        #     continue

        logger.info('------------- theta: {}, phi: {} -------------'.format(theta, phi))
        logger.info('Processing angle pair number {}'.format(flattened_ang_idx))

        # unitE, unitEQ is the electric field solution on the surface given an ambient E field amplitude of 1 and an aircraft potential of 1 V, respectively
        # We'll need the numerical capacitance to determine how the voltage converts to a charge - V = Q/C
        # The variable unitEQ isn't modified inside get_E, it's just returned from 'solution'
        unitEQ, unitE = field_orientation.get_E(solution, theta, phi, 'degrees', integral_type)

        # viz.visualize(solution['surf_mesh'], 2, {'scalars':{0: 'Field'}}, 'surface_plot', True, solution['Ex_grad_normal_surf'][:,None], None, type='surface_mesh') # Can only have scalars on a surface mesh
        # exit()

        # Note that we only care about the E-field from the external field, and not due to the aircraft charge for the baseline case
        # TODO: change this so it uses the surface integral quantites for the possible attachment points but then can use either surf or volume integrals for the integration
        # candidate_attach_pts_pos = compute_possible_attachment_points(unitE, surf_mesh, elem_centroids, vtu_dir, 'pos', r_limit)
        # candidate_attach_pts_neg = compute_possible_attachment_points(unitE, surf_mesh, elem_centroids, vtu_dir, 'neg', r_limit)

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
        # attach_pt1_baseline, attach_pt2_baseline, Efield_attach_baseline, leader1_sign_baseline, leader2_sign_baseline = attachment.compute_bidir_attachment_points(solution, unitE, unitEQ, integral_type, phys_param, candidate_attach_pts_pos, candidate_attach_pts_neg, eps)
        # logger.info('Baseline attachment summary')
        # logger.info('Attach pt 1: {}'.format(surf_mesh['pcg'][attach_pt1_baseline]))
        # logger.info('Attach pt 1: {}'.format(attach_pt1_baseline))
        # logger.info('Leader 1 sign {}:'.format(leader1_sign_baseline))
        # logger.info('Attach pt 2: {}'.format(surf_mesh['pcg'][attach_pt2_baseline]))
        # logger.info('Attach pt 2: {}'.format(attach_pt2_baseline))
        # logger.info('Leader 2 sign: {}'.format(leader2_sign_baseline))
        # logger.info('E field attach: {} kV'.format(Efield_attach_baseline/1000))
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

        # # Loading arrays
        # baseline_attach_pt1_vec[flattened_ang_idx] = attach_pt1_baseline
        # baseline_attach_pt2_vec[flattened_ang_idx] = attach_pt2_baseline
        # baseline_leader_sign_vec[flattened_ang_idx] = np.array([leader1_sign_baseline, leader2_sign_baseline])
        # baseline_Eattach_mat[theta_idx, phi_idx] = Efield_attach_baseline
    
        # # Computing aircraft optimum charge and attachment under these conditions
        # attach_pt_pos_opt, attach_pt_neg_opt, Qac_opt, Efield_attach_opt = attachment.optimum_charge(solution, unitE, unitEQ, integral_type, eps, phys_param, candidate_attach_pts_pos, candidate_attach_pts_neg)

        # logger.info('Optimum charge summary')
        # logger.info('Pos attach point: {}'.format(surf_mesh['pcg'][attach_pt_pos_opt]))
        # logger.info('Pos attach point: {}'.format(attach_pt_pos_opt))
        # logger.info('Neg attach point: {}'.format(surf_mesh['pcg'][attach_pt_neg_opt]))
        # logger.info('Neg attach point: {}'.format(attach_pt_neg_opt))
        # logger.info('E field attach: {} kV'.format(Efield_attach_opt/1000))
        # logger.info('Qac opt: {}'.format(Qac_opt))
        # # exit()
        
        # # Loading arrays
        # q_opt_pos_attach_vec[flattened_ang_idx] = attach_pt_pos_opt
        # q_opt_neg_attach_vec[flattened_ang_idx] = attach_pt_neg_opt
        Efield_attach_opt = 5
        q_opt_Eattach_mat[theta_idx, phi_idx] = Efield_attach_opt
        Qac_opt = -0.0006374999999999999
        q_opt_mat[theta_idx, phi_idx] = Qac_opt

        theta_mat[theta_idx, phi_idx] = theta
        phi_mat[theta_idx, phi_idx] = phi

        logger.info('')
        print(q_opt_mat.T)
        print(q_opt_Eattach_mat.T)

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

    with open('{}theta_mat_{}_{}.npy'.format(summaries_dname, phi_ang_start, phi_ang_end), 'wb') as file:
        np.save(file, theta_mat)
    with open('{}phi_mat_{}_{}.npy'.format(summaries_dname, phi_ang_start, phi_ang_end), 'wb') as file:
        np.save(file, phi_mat)
        
    logger.info('Done!')
    return

if __name__ == '__main__':

    # phi_ang_start = float(sys.argv[1])
    # phi_ang_end = float(sys.argv[2])

    rad_fuselage = 1.75  # m
    numpts_theta = 50
    numpts_phi = 2
    num_phi_div = 20

    integral_type = 'surf'
    eps=0.05

    sol_fname = '/media/homehd/saustin/lightning_research/3D-CG/postprocessing/fem_solutions/d8/d8_electrostatic_solution'
    # vtu_dir = '/home/gridsan/saustin/research/3D-CG/postprocessing/fem_solutions/d8/attachment_coarse2/attachment_vtus'
    # summaries_dname = '/home/gridsan/saustin/research/3D-CG/postprocessing/fem_solutions/d8/attachment_coarse2/'
    summaries_aggregate_dname = '/media/homehd/saustin/lightning_research/3D-CG/postprocessing/fem_solutions/d8/attachment_data/analysis_out/'

    with open(sol_fname, 'rb') as file:
        solution = pickle.load(file)

    solution['Phi_grad_normal_surf'] *= -1
    ##### NOTE!!!!!!!! This is because I messed up the sign on the normal vector calculation and needed to flip the sign! Only relevant for sims that were note re-run!!

    # compute_attachment(solution, integral_type, vtu_dir, eps, summaries_dname, phi_ang_start, phi_ang_end, numpts_theta, numpts_phi, r_fuselage=rad_fuselage)

    # composite_attachment_data(summaries_dname, numpts_theta, num_phi_div, summaries_aggregate_dname)

    analysis_postprocessing(solution, summaries_aggregate_dname)