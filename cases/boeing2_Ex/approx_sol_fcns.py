import numpy as np
import sys
sys.path.append('../../utils')
import gmsh
from gmsh import model as gm

gm.occ.importShapes('boeing_plane_no_landing_gear.STEP')
gm.occ.synchronize()

def forcing_zero(p):
    return np.zeros((p.shape[0], 1))

def approx_sol_x(p):
    # bbox = gm.occ.getBoundingBox()
    x = p[:, 0]
    u1 = -240
    u2 = 390
    x1 = -100
    x2 = 600
    m = (u2-u1)/(x2-x1)
    return m*(x-x1)+u1

def approx_sol_y(p):
    y = p[:, y]
    return 'not implemented'

def approx_sol_z(p):
    z = p[:, 2]
    return 'not implemented'

def approx_sol_charge(p):
    # Assumes the plane is aligned with the x-direction

    def is_in_wing(pfr, pbr, pfl, point):
        is_in_wing_bool = True
        # Check x:
        if point[0] > pfr[0] or point[0] < pbr[0]:
            is_in_wing_bool = False

        # Check y:
        if point[1] < pfr[1] or point[1] > pfl[1]:
            is_in_wing_bool = False

        # No need to check z because if it is above or below the wing in the x-y plane, is it guaranteed to lie outside the wing surface itself.
        return is_in_wing_bool

    approx_potential = np.zeros((p.shape[0]))

    # Defining cylinder
    rear_p1 = np.array([-50, 292, 403])
    rear_p2 = np.array([-50, 241, 403])
    front_p1 = np.array([400, 292, 403])    # Don't need second front point because all we need are the cylinder radius and center axis

    # Top of wing
    wing_front_right = np.array([217, 103, 405])
    wing_back_right = np.array([166, 103, 405])
    wing_front_left = np.array([217, 433, 405])
    thickness = 5

    center = (rear_p1+rear_p2)/2    # Center point of rear circular cross section
    r_fuselage = (rear_p1[1]-rear_p2[1])/2

    # Compute r
    for i, pt in enumerate(p):
        x = pt[0]
        y = pt[1]
        z = pt[2]

        if x<rear_p1[0] or x > front_p1[0]:         # Check to see if the point is in front of or behind the fuselage in x
            r = ((y-center[1])**2+(z-center[2])**2)**0.5 # r = sqrt((y-b)^2 + (z-c)^2), pythagorean theorem
            if r<r_fuselage:        # Point is directly in front of or behind the plane's cross section, contained inside cross section in y and z
                if x < rear_p1[0]:
                    dist = np.abs(rear_p1[0]-x)
                elif x > front_p1[0]:
                    dist = np.abs(front_p1[0]-x)

                approx_potential[i] = 1/((dist+r_fuselage)/r_fuselage)**0.8   # Scale length is r_fuselage, and the distance from the surface starts at 1/fuselage

        elif x>rear_p1[0] and x < front_p1[0]:        # Check to see if the point is next to the fuselage in y and z
            r = ((y-center[1])**2+(z-center[2])**2)**0.5 # r = sqrt((y-b)^2 + (z-c)^2), pythagorean theorem
            approx_potential_fuselage = 1/(r/r_fuselage)**0.8     # Will = 1 on surface of fuselage, and anything > 1 is due to the fact that the element surfaces aren't curved and some of them lie inside the fuselage. These points closest to the surface will get taken out by setting the dirichlet condition anyway.

            if is_in_wing(wing_front_right, wing_back_right, wing_front_left, pt):
                if pt[2] > wing_front_right[2]:      # On top of wing
                    dist = pt[2] - wing_front_right[2]
                else:      # On bottom of wing
                    dist = wing_front_right[2]-thickness - pt[2]

                approx_potential_wing = 1/((dist+r_fuselage)/r_fuselage)**0.8   # Scale length is r_fuselage, and the distance from the surface starts at 1/fuselage
                approx_potential[i] = max([approx_potential_fuselage, approx_potential_wing])
            else:
                approx_potential[i] = approx_potential_fuselage

    return approx_potential
