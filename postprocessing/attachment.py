import numpy as np

def optimum_charge(unitE, unitEQ, integral_type, eps, phys_param):
    """
    Note that both leaders don't have to be incepted - only one does. The reason is that if they are both "close" to inception, if one has crossed the threshold, the second one is soon to follow, after gaining epsilon in charge

    """

    bidir_E_symmetrical = False

    Qac = phys_param['Qac_0']
    dQ_proportional = phys_param['dQ_0']

    while not bidir_E_symmetrical:

        attach_pt1, leader1_sign, Efield_attach, Q_corona_attach = compute_first_attachment(unitE, unitEQ, Qac, integral_type, phys_param)  # Don't need capacitance or charge information for the baseline case

        # Take maximum of the oppositely charged regions
        if leader1_sign > 0:    # First incepted leaders is positive
            Q_pos = Q_corona_attach
            attach_pt2, Q_neg = get_max_corona_charge_other_sign_given_current_E()
        else:       # First incepted leader is negative
            Q_neg = Q_corona_attach
            attach_pt2, Q_pos = get_max_corona_charge_other_sign_given_current_E()

        diff = Q_pos/phys_param['Q_pos_crit_leader_incep'] + Q_neg/phys_param['Q_neg_crit_leader_incep']

        if np.abs(diff) < eps:
            bidir_E_symmetrical = True
        else:
            dQ = dQ_proportional*diff     # Going to have to tune dQ_proportional - this is like the "P" term in a PID - we need to get to 0
            Qac -= dQ

    return attach_pt1, attach_pt2, Qac, Efield_attach


def compute_attachment_points(unitE, unitEQ, Qac, integral_type, phys_param):

    attach_pt1, leader1_sign, Efield_attach, = compute_first_attachment(unitE, unitEQ, integral_type, phys_param, Qac)  # Don't need capacitance or charge information for the baseline case
    
    attach_pt2, leader2_sign, __, __ = compute_second_attachment(unitE, unitEQ, integral_type, phys_param, Qac, leader1_sign, Efield_attach)  # Don't need capacitance or charge information for the baseline case

    return attach_pt1, attach_pt2, Efield_attach, leader1_sign, leader2_sign


def compute_first_attachment(unitE, unitEQ, integral_type, phys_param, Qac):
    """
    Given an aircraft charge and E field orientation, compute the location, and required E field of the FIRST attachment point
    Base leader inception function

    returns:
    attach_pt: coord of attachment
    E_amp: amplitude of external E field required for attachment

    """

    E_amp = phys_param['Einf_0']  # Initial background field strength, V/m

    stopping_condition = False
    dE_proportional = 10e3  # constant of proportionality in "PID" - play with this value
    stopping_tol = .01  # Play with this value

    while not stopping_condition:

        E = E_amp*unitE + unitEQ*Qac/phys_param['capacitance']


        # Compute corona charges for all possible points - call a surface or volume integral of the E field surrounding the points



        # Find the max Q/Q_cr of all the points
        Q_pos_nondim = Q_pos/phys_param['Q_pos_crit_leader_incep']
        Q_neg_nondim = Q_neg/phys_param['Q_neg_crit_leader_incep']

        # Group the two together
        corona_charges = combine(Q_pos_nondim, Q_neg_nondim)

        # Find the number of attachment points - (unsigned nondim charge > 1)
        # FILL ME IN

        max_charge = np.max(corona_charges)

        if num_attach_pts == 1 and np.abs(max_charge) < stopping_tol
            stopping_condition = True
        else:
            dE = dE_proportional*max_charge     # Going to have to tune dQ_proportional - this is like the "P" term in a PID - we need to get to 0
            Efield += dE

    return attach_pt1, attach_pt2, Efield_attach



    return attach_pt1, leader_sign, E_field_attach, Q_corona_attach

def compute_second_attachment(unitE, unitEQ, integral_type, phys_param, leader1_sign, Efield_attach, Qac):
    """
    Iterates on the aircraft charge until the leader of the opposite charge is incepted
    """

    # Iterate on aircraft Q until second leader is incepted, hold E field constant
    unitE*Efield_attach + unitEQ*Qac/phys_param['capacitance']


    return attach_pt2, leader_sign

def check_inception(phys_param):
    """
    Checks to see if any leaders are incepted at the given condition, and if so, how many and what sign
    
    """

    return num_leaders_incepted, sign_leaders_incepted


def corona_integral_surface():
    """
    Surface integral around the corona points
    
    """

    pass


def corona_integral_volume():
    """
    Surface integral around the corona points
    
    """



    pass