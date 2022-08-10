########## TOP LEVEL SIM SETUP ##########
meshfile: 'mesh/' + 'boeing_plane_final'     # No file extension!
stepfile: 'mesh/boeing_plane_no_landing_gear.STEP'

case_select: 'Ex'
# umin: None     # Fill these in with the max and min values of the potential when computing the external E field solutions
# umax: None

porder: 2
ndim: 3
solver: 'gmres'
solver_tol: 1e-7

outdir: 'out/'
vis_filename: 'boeing_plane_'+case_select
build_mesh: False
buildAF: False
compute_sol: False
call_pv: False
vis_filename: outdir+vis_filename
visorder: porder
viz_labels: {'scalars': {0: 'Potential', 1: 'x0'}, 'vectors': {0: 'Potential Gradient'}}

fuselage_dia: 3.76     # This is the fuselage of the 737 in m
# stabilizers: [20, 26, 51, 85, 72, 95, 34, 38, 87, 108, 97, 116]
# nose: [39, 78, 33, 48, 99, 118, 84, 106, 77, 100, 49, 83]
# fuselage: [107, 117, 122, 130, 131, 134]
# engines: [16, 17, 18, 19, 31, 32, 59, 60, 57, 58, 89, 90]
# wings: [121, 119, 101, 103, 79, 82, 41, 45, 27, 30, 6, 11, 2, 3, 132, 137, 126, 136, 123, 124, 109, 114, 88, 93, 56, 69, 35, 36]
# body_surfs: stabilizers + nose + fuselage + engines + wings

########## GEOMETRY SETUP ##########
pt_1_fuselage: np.array([8547.42, 1505.00, 5678.37])
pt_2_fuselage: np.array([8547.42, -1505.00, 5678.37])

r_fuselage_msh: np.linalg.norm(pt_1_fuselage-pt_2_fuselage)/2
scale_factor:  fuselage_dia/r_fuselage_msh    # Normalize mesh by the fuselage radius and rescale so that mesh dimensions are in meters

########## BCs ##########
surf_faces: np.arange(137)+1   # Faces are 1-indexed
x_minus_face: 138
x_plus_face: 139
y_minus_face: 140
y_plus_face: 141
z_minus_face: 142
z_plus_face: 143