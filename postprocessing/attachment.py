import numpy as np
# Finding the sim root directory
import sys
from pathlib import Path

from sympy import centroid
cwd = Path.cwd()
for dirname in tuple(cwd.parents):
    if dirname.name == '3D-CG':
        sim_root_dir = dirname
        continue

sys.path.append(str(sim_root_dir.joinpath('CG')))
import quadrature
import viz
import pickle
import logging

logger = logging.getLogger(__name__)

def optimum_charge(solution, unitE, unitEQ, integral_type, eps, phys_param, candidate_attach_pts_pos, candidate_attach_pts_neg):
    """
    Note that both leaders don't have to be incepted - only one does. The reason is that if they are both "close" to inception, if one has crossed the threshold, the second one is soon to follow, after gaining epsilon in charge

    """

    # Watch out for things not converging if different attachment points are chosen

    bidir_E_symmetric = False

    Qac = 0
    last_error_sign = 0
    increment_baseline = 1e-4
    num_sign_flips = 0

    while not bidir_E_symmetric:

        attach_pt1, leader1_sign, Efield_attach, error1 = compute_first_attachment(solution, unitE, unitEQ, integral_type, phys_param, Qac, candidate_attach_pts_pos, candidate_attach_pts_neg, eps)

        E = Efield_attach*unitE + Qac*unitEQ/phys_param['capacitance']

        if leader1_sign > 0:    # If first leader was positive
            attach_pt_pos = attach_pt1
            pos_error = error1
            leader2_sign = -1
            candidate_attach_pts2 = candidate_attach_pts_neg    # Only evaluate the negative attachment points
        else:
            attach_pt_neg = attach_pt1
            neg_error = error1
            leader2_sign = 1
            candidate_attach_pts2 = candidate_attach_pts_pos    # Only evaluate the positive attachment points if the first leader was negative
        
        _, pt2_max_error, error2, candidate_attach_pts2 = inception_evaluation(solution, E, integral_type, leader2_sign, phys_param, candidate_attach_pts2)

        if len(pt2_max_error) > 1:
            pt2_max_error = reduce_attach_pts(solution, pt2_max_error)

        if leader2_sign > 0:    # If second leader was positive
            pos_error = error2
            attach_pt_pos = pt2_max_error
        else:
            neg_error = error2
            attach_pt_neg = pt2_max_error

        diff = pos_error - neg_error  # This is the difference between the nondimensionalized charges for the (+) and (-) coronas with maximum charge

        # logger.info('Diff: {}'.format(diff))

        if np.abs(diff) < eps:
            bidir_E_symmetric = True
            # logger.info('DONE with optimization, stopping')
        else:
            # if diff > 0:   # Aircraft has too positive of a charge for the negative leader to be incepted simultaneously
            # if diff < 0, aircraft has too negative of a charge for the positive leader to be incepted simultaneously

            current_error_sign = np.sign(diff)

            if last_error_sign != current_error_sign:
                if last_error_sign != 0:     # not first time
                    num_sign_flips += 1                    
                last_error_sign = current_error_sign

            # Each time the sign of the error flips, the amount that E_amp is incremented or decremented is halved
            # logger.info('Incrementing Q by {} C'.format(-current_error_sign*increment_baseline*0.5**num_sign_flips))
            Qac -= current_error_sign*increment_baseline*0.5**num_sign_flips

    return attach_pt_pos, attach_pt_neg, Qac, Efield_attach

def compute_bidir_attachment_points(solution, unitE, unitEQ, integral_type, phys_param, candidate_attach_pts_pos, candidate_attach_pts_neg, eps, Qac=0):

    attach_pt1, leader1_sign, Efield_attach, __ = compute_first_attachment(solution, unitE, unitEQ, integral_type, phys_param, Qac, candidate_attach_pts_pos, candidate_attach_pts_neg, eps)
    
    # with open('attach_pt1', 'wb') as file:
    #     pickle.dump(attach_pt1, file)
    # with open('leader1_sign', 'wb') as file:
    #     pickle.dump(leader1_sign, file)
    # with open('Efield_attach', 'wb') as file:
    #     pickle.dump(Efield_attach, file)

    # with open('attach_pt1', 'rb') as file:
    #     attach_pt1 = pickle.load(file)
    # with open('leader1_sign', 'rb') as file:
    #     leader1_sign = pickle.load(file)
    # with open('Efield_attach', 'rb') as file:
    #     Efield_attach = pickle.load(file)

    if leader1_sign > 0:    # If first leader was positive
        leader2_sign = -1
        candidate_attach_pts2 = candidate_attach_pts_neg    # Only evaluate the negative attachment points
    else:
        leader2_sign = 1
        candidate_attach_pts2 = candidate_attach_pts_pos    # Only evaluate the positive attachment points if the first leader was negative
    
    E_background = unitE*Efield_attach

    # logger.info('First leader sign: {}'.format(leader1_sign))
    # logger.info('')

    attach_pt2 = compute_second_attachment(solution, E_background, unitEQ, Qac, integral_type, phys_param, leader2_sign, candidate_attach_pts2, eps)

    return attach_pt1, attach_pt2, Efield_attach, leader1_sign, leader2_sign

def compute_first_attachment(solution, unitE, unitEQ, integral_type, phys_param, Qac, candidate_attach_pts_pos, candidate_attach_pts_neg, stopping_tol):
    """
    Given an aircraft charge and E field orientation, compute the location, and required E field of the FIRST attachment point
    Base leader inception function

    returns:
    attach_pt: coord of attachment
    E_amp: amplitude of external E field required for attachment

    """

    E_amp = phys_param['Einf_0']  # Initial background field strength, V/m

    stop_iter_bool = False
    # kP_E = 10e3  # constant of proportionality in "PID" - play with this value

    last_error_sign = 0
    increment_baseline = phys_param['Einf_0']/2
    num_sign_flips = 0
    # logger.info('-------------Computing first attachment-------------')

    focus_sign = None
    iter=0
    while not stop_iter_bool:
        # logger.info('Iteration {}'.format(iter))
        # logger.info('E amplitude {} kV'.format(E_amp/1000))
        E = E_amp*unitE + unitEQ*Qac/phys_param['capacitance']

        # # Compute average E field on each surface element
        # E_avg = get_avg_field_on_elements(mesh, E)

        # # Compute corona charges for all possible points - call a surface or volume integral of the E field surrounding the points
        # global_elem_idx = np.arange(mesh['tcg'].shape[0])
        # pos_corona_elem_mask = (E_avg >= phys_param['pos_corona_stability_field'])
        # neg_corona_elem_mask = (E_avg <= phys_param['neg_corona_stability_field'])

        # pos_corona_elem = global_elem_idx[pos_corona_elem_mask]     # List of elements that meet the corona stabilization field criteria
        # neg_corona_elem = global_elem_idx[neg_corona_elem_mask]

        # # Compute corona charges at the candidate points, with the modification that if more than one leader is above the inception threshold, then only look at those corona and don't bother integrating over the rest of the points.
        # Q_pos_vec, pcg_idx_vec_pos = corona_integral(mesh, candidate_attach_pts_pos, pos_corona_elem, integral_type)  # Returns a vector of corona charges for each point classified as a candidate point given the E field orientation
        # Q_neg_vec, pcg_idx_vec_neg = corona_integral(mesh, candidate_attach_pts_neg, neg_corona_elem, integral_type)  # Returns a vector of corona charges for each point classified as a candidate point given the E field orientation

        # # pcg_idx_vec_xxx contain the indices of the mesh.pcg points that correspond to the centers of the corona charge evaluation integral.
        # if focus_sign == 'pos':
        #     pos_inception_pts, max_pos_error, candidate_attach_pts_pos, pt_max_pos_error = check_inception_unidir(1, Q_pos_vec, pcg_idx_vec_pos, phys_param, candidate_attach_pts_pos)
        #     num_pos_leaders = len(pos_inception_pts)
        # elif focus_sign == 'neg':
        #     neg_inception_pts, max_neg_error, candidate_attach_pts_neg, pt_max_neg_error = check_inception_unidir(-1, Q_neg_vec, pcg_idx_vec_neg, phys_param, candidate_attach_pts_neg)
        #     num_neg_leaders = len(neg_inception_pts)
        # else:
        #     pos_inception_pts, max_pos_error, candidate_attach_pts_pos, pt_max_pos_error = check_inception_unidir(1, Q_pos_vec, pcg_idx_vec_pos, phys_param, candidate_attach_pts_pos)
        #     neg_inception_pts, max_neg_error, candidate_attach_pts_neg, pt_max_neg_error = check_inception_unidir(-1, Q_neg_vec, pcg_idx_vec_neg, phys_param, candidate_attach_pts_neg)
        #     num_pos_leaders = len(pos_inception_pts)
        #     num_neg_leaders = len(neg_inception_pts)

        num_pos_leaders, pt_max_pos_error, max_pos_error, candidate_attach_pts_pos = inception_evaluation(solution, E, integral_type, 1, phys_param, candidate_attach_pts_pos)
        num_neg_leaders, pt_max_neg_error, max_neg_error, candidate_attach_pts_neg = inception_evaluation(solution, E, integral_type, -1, phys_param, candidate_attach_pts_neg)

        # logger.info('Positive leader info')
        # logger.info('leaders incepted: {}'.format(num_pos_leaders))
        # logger.info('max pos error: {}'.format(max_pos_error))

        # logger.info('Negative leader info')
        # logger.info('leaders incepted: {}'.format(num_neg_leaders))
        # logger.info('max neg error: {}'.format(max_neg_error))

        # Leader inception switch
        if focus_sign == 'pos':
            # logger.info('FOCUS SIGN {}'.format(focus_sign))
            if num_pos_leaders >= 1:
                if max_pos_error < stopping_tol:    # Handles both the cases of 1 and 2+ positive leaders incepted, with the max below the threshold
                    stop_iter_bool = True
                    leader_sign = 1
                    attach_pt1 = pt_max_pos_error
                    error = max_pos_error
                else:   # One or more (+) leader out of tolerance
                    error = max_pos_error
            else:       # No (+) leaders incepted
                error = max_pos_error

        elif focus_sign == 'neg':
            # logger.info('FOCUS SIGN {}'.format(focus_sign))
            if num_neg_leaders >= 1:
                if max_neg_error < stopping_tol:    # Handles both the cases of 1 and 2+ positive leaders incepted, with the max below the threshold
                    stop_iter_bool = True
                    leader_sign = -1
                    attach_pt1 = pt_max_neg_error
                    error = max_neg_error
                else:   # One or more (+) leader out of tolerance
                    error = max_neg_error
            else:       # No (+) leaders incepted
                error = max_neg_error
            
        # Now, need to write the rest of the 25 cases
        else:
            if num_pos_leaders == 0:
                if num_neg_leaders == 0:         # Leaders of neither sign has been incepted
                    error = max(max_pos_error, max_neg_error)    # Remember that the error is signed relative to 1!
                elif num_neg_leaders >= 1:
                    if max_neg_error < stopping_tol:
                        stop_iter_bool = True
                        leader_sign = -1
                        attach_pt1 = pt_max_neg_error
                        error = max_neg_error
                    else:
                        error = max_neg_error
                        focus_sign = 'neg'

            else:   # There are one or more positive leaders incepted
                if num_neg_leaders == 0:
                    if max_pos_error < stopping_tol:
                        stop_iter_bool = True
                        leader_sign = 1
                        attach_pt1 = pt_max_pos_error
                        error = max_pos_error
                    else:   # One (+) leader out of tolerance
                        error = max_pos_error
                        focus_sign = 'pos'

                else:
                    max_error = max(max_pos_error, max_neg_error)
                    if max_error < stopping_tol:
                        if max_error == max_pos_error:
                            stop_iter_bool = True
                            leader_sign = 1
                            attach_pt1 = pt_max_pos_error
                            error = max_pos_error
                        elif max_error == max_neg_error:
                            stop_iter_bool = True
                            leader_sign = -1
                            attach_pt1 = pt_max_neg_error
                            error = max_neg_error
                        else:
                            raise ValueError('Error not in either pos or neg')
                    else:
                        error = max_error

                        # The logic being, if one is OOT and the other is below the threshold, then the smallest one will likely drop below the threshold at the next iteration
                        if max_error == max_pos_error and max_neg_error < stopping_tol:
                            focus_sign == 'pos'
                        elif max_error == max_neg_error and max_pos_error < stopping_tol:
                            focus_sign == 'neg'

        if not stop_iter_bool:   # Stopping criteria not met - use successive approximation to dial in
            current_error_sign = np.sign(error)

            if last_error_sign != current_error_sign:
                if last_error_sign != 0:     # not first time
                    num_sign_flips += 1                    
                last_error_sign = current_error_sign

            # Each time the sign of the error flips, the amount that E_amp is incremented or decremented is halved
            # logger.info('Incrementing E_amp by {} kV'.format(-current_error_sign*increment_baseline*0.5**num_sign_flips))
            E_amp -= current_error_sign*increment_baseline*0.5**num_sign_flips

            # Decide how much to increment E based on how far away the corona integral with the max charge is away from Q/Q_cr = 1
            # E_amp -= kP_E*error     # Going to have to tune dQ_proportional - this is like the "P" term in a PID
            # If this isn't converging fast enough, try either an I term or resorting to a naive incremental decrease each time
        # logger.info('')
        iter += 1
    # logger.info('DONE computing first attachment')

    if len(attach_pt1) > 1:
        attach_pt1 = reduce_attach_pts(solution, attach_pt1)

    return attach_pt1, leader_sign, E_amp, error

def compute_second_attachment(solution, Efield_background, unitEQ, Qac, integral_type, phys_param, leader2_sign, candidate_attach_pts, stopping_tol):
    """
    Iterates on the aircraft charge until the leader of the opposite charge is incepted
    """

    stop_iter_bool = False


    last_error_sign = 0
    increment_baseline = 1e-4
    num_sign_flips = 0

    # # If the first leader was positive, the initial error in the negative leader inception is going to be negative. Thus, we need to decrease Qac, but this involves flipping the sign of kP_Q.
    # # If the first leader was negative, the initial error in the positive leader inception is going to be negative. Thus, we need to increase Qac.
    # kP_Q = leader2_sign*10e-6  # signed constant of proportionality in "PID" - play with this value

    # logger.info('-------------Computing second attachment-------------')
    # logger.info('Looking for a leader with sign {}'.format(leader2_sign))
    iter=0
    while not stop_iter_bool:

        E = Efield_background + unitEQ*Qac/phys_param['capacitance']
        # logger.info('Qac: {}'.format(Qac))
        # if Qac == .0002:
            # debug_flag=True
            # logger.info('debugging')
        # else:

        debug_flag=False
        num_leaders, pt_max_error, max_error, candidate_attach_pts = inception_evaluation(solution, E, integral_type, leader2_sign, phys_param, candidate_attach_pts, debug_flag)
        
        # logger.info('Leader info')
        # logger.info('leaders incepted: {}'.format(num_leaders))
        # logger.info('max error: {}'.format(max_error))

        # Leader inception switch
        if num_leaders > 0 and max_error < stopping_tol:
            stop_iter_bool = True
            attach_pt2 = pt_max_error
        else:       # No (+) leaders incepted
            error = max_error

        if not stop_iter_bool:   # Stopping criteria not met
            # # Decide how much to increment E based on how far away the corona integral with the max charge is away from Q/Q_cr = 1
            # Qac -= kP_Q*error     # Going to have to tune dQ_proportional - this is like the "P" term in a PID
            # # If this isn't converging fast enough, try either an I term or resorting to a naive incremental decrease each time

            current_error_sign = np.sign(error)

            if last_error_sign != current_error_sign:
                if last_error_sign != 0:     # not first time
                    num_sign_flips += 1                    
                last_error_sign = current_error_sign

            # Each time the sign of the error flips, the amount that E_amp is incremented or decremented is halved
            # logger.info('Incrementing Qac by {} C'.format(-leader2_sign*current_error_sign*increment_baseline*0.5**num_sign_flips))
            Qac -= leader2_sign*current_error_sign*increment_baseline*0.5**num_sign_flips

            # Decide how much to increment E based on how far away the corona integral with the max charge is away from Q/Q_cr = 1
            # E_amp -= kP_E*error     # Going to have to tune dQ_proportional - this is like the "P" term in a PID
            # If this isn't converging fast enough, try either an I term or resorting to a naive incremental decrease each time
        # logger.info('')
        iter += 1
    # logger.info('DONE computing second attachment')

    if len(attach_pt2) > 1:
        attach_pt2 = reduce_attach_pts(solution, attach_pt2)
        
    return attach_pt2

def corona_integral(solution, E_avg, candidate_attach_pts, corona_elem, integral_type, phys_param, debug_flag=False):
    if integral_type == 'surf':
        return corona_surface_integral(solution, E_avg, candidate_attach_pts, corona_elem, phys_param, debug_flag)
    elif integral_type == 'vol':
        return corona_volume_integral(solution, E_avg, candidate_attach_pts, corona_elem, phys_param)
    else:
        raise ValueError('integral_type must be either "surf" or "vol"')

def corona_surface_integral(solution, E, candidate_attach_pts, corona_elem, phys_param, debug_flag=False):
    """
    Surface integral around the points that are marked as possible attachment candidates
    
    """
    pcg_idx_vec = np.zeros((len(candidate_attach_pts.keys())), dtype=np.int64)
    Q_vec = np.zeros((len(candidate_attach_pts.keys())))

    for i, pt in enumerate(candidate_attach_pts):
        pcg_idx_vec[i] = pt
        elem_above_thresh_in_radius = np.intersect1d(candidate_attach_pts[pt], corona_elem)

        # if debug_flag:
        #     pts_list = [229096, 229099, 229129, 229132, 229155, 229167, 229168, 229171, 229176, 229189, 229190, 229191, 229195, 229196, 229201, 229213, 229214, 229217, 229218, 229220, 229233, 229234, 229238, 229241, 229244, 229245, 229261, 229270, 229286, 229419]
        #     elem_indicator = np.zeros((solution['surf_mesh']['t'].shape[0], len(pts_list)))
        #     if pt in pts_list:
        #         logger.info('visualizing')
        #         logger.info(pt)
        #         elem_indicator = np.zeros((solution['surf_mesh']['t'].shape[0], 1))
        #         # elem_indicator[candidate_attach_pts[pt]] = 1
        #         elem_indicator[elem_above_thresh_in_radius] = 1
        #         viz.generate_vtu(solution['surf_mesh']['p'], solution['surf_mesh']['t'], None, None, {'cell_data': {0: 'E threshold'}}, 'test_radius{}'.format(pt), False, cell_data=elem_indicator)

        Q_vec[i] = phys_param['eps0']*quadrature.surface_integral(solution['surf_mesh'], solution['master'], E, elem_above_thresh_in_radius)

    return Q_vec, pcg_idx_vec

def corona_volume_integral(solution, E_avg, candidate_attach_pts, corona_elem, phys_param):
    """
    Surface integral around the corona points
    
    """
    raise NotImplementedError

def get_avg_field_on_elements(mesh, field):
    """
    Computes the average of a quantity defined at points, on elements. Essentially converts from a pointwise representation of the data to an elementwise one
    Assumes the field is defined at the high order nodes
    Can handle both scalar and vector fields
    Vectorized instead of for loop
    """

    if len(field.shape) > 1:    # Vector field
        elem_avg = np.zeros((mesh['tcg'].shape[0], field.shape[1]))
        for idx, field_dim_data in enumerate(field.T):     # Iterate through columns of vector field
            elem_avg[:, idx] = np.mean(field_dim_data[mesh['tcg']], axis=1)
    else:   # Scalar field
        elem_avg = np.mean(field[mesh['tcg']], axis=1)

    return elem_avg

def inception_evaluation(solution, E, integral_type, leader_sign, phys_param, candidate_attach_pts, debug_flag=False):
    # Compute average E field on each surface element

    E_avg = get_avg_field_on_elements(solution['surf_mesh'], E)

    # Compute corona charges for all possible points - call a surface or volume integral of the E field surrounding the points
    global_elem_idx = np.arange(solution['surf_mesh']['tcg'].shape[0])

    if leader_sign < 0:    # Look for negative leader 2 inception
        # print('neg', phys_param['neg_corona_stability_field'])
        corona_elem_mask = (E_avg <= phys_param['neg_corona_stability_field'])
    else:    # Look for positive leader 2 inception
        # print('pos', phys_param['pos_corona_stability_field'])
        corona_elem_mask = (E_avg >= phys_param['pos_corona_stability_field'])

    corona_elem = global_elem_idx[corona_elem_mask]     # List of elements that meet the corona stabilization field criteria

    # if leader_sign < 0:
    #     print('visualizing')
    #     elem_indicator = np.zeros((solution['surf_mesh']['t'].shape[0], 1))
    #     elem_indicator[corona_elem] = 1
    #     print(len(corona_elem))
    #     viz.generate_vtu(solution['surf_mesh']['p'], solution['surf_mesh']['t'], None, None, {'cell_data': {0: 'E threshold'}}, 'corona_indicator_neg', True, cell_data=elem_indicator)    
    #     exit()

    # Compute corona charges at the candidate points, with the modification that if more than one leader is above the inception threshold, then only look at those corona and don't bother integrating over the rest of the points.
    Q_vec, pcg_idx_vec = corona_integral(solution, E, candidate_attach_pts, corona_elem, integral_type, phys_param, debug_flag)  # Returns a vector of corona charges for each point classified as a candidate point given the E field orientation

    # pcg_idx_vec_xxx contain the indices of the mesh.pcg points that correspond to the centers of the corona charge evaluation integral.
    return check_inception_unidir(leader_sign, Q_vec, pcg_idx_vec, phys_param, candidate_attach_pts)

def check_inception_unidir(sign, Q_vec, pcg_idx_vec, phys_param, candidate_attach_pts):
    """
    Checks to see if any leaders are incepted at the given condition, and if so, how many and what sign
    
    """

    if sign > 0:
        # Find the max Q/Q_cr of all the points - nondimensionalize
        errors = Q_vec/phys_param['Q_pos_crit_leader_incep'] - 1
    else:
        errors = np.abs(Q_vec/phys_param['Q_neg_crit_leader_incep']) - 1

    max_error = np.max(errors)  # Positive if Q above corona inception threshold (E field too high), negative if below (E field too low) - is negative if max charge is below the corona inception threshold
    pt_max_error = pcg_idx_vec[errors == max_error]

    # Add in logic here to deal with more than one point having the same max error due to overlap
    # logger.info('')
    # logger.info('Inside check_inception_unidir, debugging')
    # logger.info('errors:', errors)
    # logger.info('Sign:', sign, ', pt_max_error', pt_max_error)

    # if pt_max_error.shape[0] > 1:   # More than one point with the same max error, means the same points share the same regions where E > E_corona_stab

    # Find the number of attachment points - (unsigned nondim charge > 1)
    if max_error > 0:     # At least one inception point
        incepted_leaders_idx = np.argwhere(errors > 0).ravel()
        # Remove points from candidate_attach_pts that aren't incepted
        inception_pts = pcg_idx_vec[incepted_leaders_idx]
        non_incepted_pts = np.setdiff1d(list(candidate_attach_pts.keys()), inception_pts)
        # logger.info('incepted points {}'.format(inception_pts))
        # logger.info('non incepted points', non_incepted_pts)

        # Remove those keys from the dictionary
        for pt in non_incepted_pts:
            # logger.info(pt)
            # logger.info(candidate_attach_pts[pt])
            del candidate_attach_pts[pt]
    else:
        inception_pts = []

    num_leaders = len(inception_pts)
    # Make sure that this function only returns a single point for the max error!
    return num_leaders, pt_max_error, max_error, candidate_attach_pts

def reduce_attach_pts(solution, attach_pt):
    # We need to return only one attachment point. Sometimes, it may be the case that multiple leader inception points have the same error because they include the same points that are integrated over in the corona stability region. Thus, they are valid leader inceptions points but we can only return one.
    # We get around this by taking the point that is nearest the centroid of all the points.

    attach_coords = solution['surf_mesh']['pcg'][attach_pt]
    attach_centroid = np.mean(attach_coords, axis=0)
    closest_to_centroid = np.argmin(np.linalg.norm(attach_coords-attach_centroid, axis=1)).ravel()[0]
    return attach_pt[closest_to_centroid]