import sys
sys.path.append('../')
import gmsh_4_10 as gmsh
import sys
import numpy as np
from gmsh_4_10 import model as gm

#############################################
volume_gen=True
# volume_gen=False

case_name = 'bwb'

# Global scale factor - base linear point density
base_factor = 150/35257

# Only need if specifying a distance field in the volume refinement section
ref_pt1 = 30
ref_pt2 = 18

#############################################

gmsh.initialize()
gm.add(case_name)

############## CREATE PLANE GEOMETRY ##############

gm.occ.importShapes('bwb_no_engines_vol_bodies.STEP')
gm.occ.synchronize()
# gmsh.fltk.run()
# exit()

# Exraction of the bbox parameters
xmin, ymin, zmin, xmax, ymax, zmax = gm.occ.getBoundingBox(3, 1)

# Mesh resolution along curves
# print('Assigning transfinite curves')
for curve in gm.occ.getEntities(1):
    curve_idx = curve[1]

    length = gm.occ.getMass(1, curve[1])
    numpts = max(int(length*base_factor), 3)

    if curve_idx in [16, 26, 68, 70, 39, 61]:     # stabilizers + wingtips
        numpts *= 6

    elif curve_idx in [58, 60]:
        gm.mesh.setTransfiniteCurve(curve_idx, int(numpts*3), coef=1.02)
        continue

    gm.mesh.setTransfiniteCurve(curve_idx, numpts)

# Assigning farfield boundary box
################## Make box ##################

xlength = xmax-xmin
ylength = ymax-ymin
zlength = zmax-zmin

ref_len = max(xlength, ylength, zlength)

xoffset = 2*ref_len
yoffset = 2*ref_len
zoffset = 2*ref_len

x0 = xmin - xoffset
y0 = ymin - yoffset
z0 = zmin - zoffset

dx = xlength + 2*xoffset
dy = ylength + 2*yoffset
dz = zlength + 2*zoffset

if volume_gen:
    print('Assigning box')
    gm.occ.addBox(x0, y0, z0, dx, dy, dz)
    solids = gm.occ.getEntities(3)
    gm.occ.cut([solids[1]], [solids[0]], removeObject=False, removeTool=False)
    gm.occ.remove([(3, 1)], recursive=True)
    gm.occ.remove([(3, 2)], recursive=True)
    gm.occ.synchronize()

    # Mesh resolution on outer boundary of box
    hmax=dx/10
    gmsh.option.setNumber("Mesh.MeshSizeMax", hmax)


# gmsh.fltk.run()
# exit()

# Mesh resolution on exterior of aircraft - distance fields
ref_len = np.linalg.norm(gm.getValue(0, ref_pt1, [0]) - gm.getValue(0, ref_pt2, [0]))
# Distance field for nose
gm.mesh.field.add("Distance", 3)
gm.mesh.field.setNumbers(3, "PointsList", [53, 16, 56, 34])

gmsh.model.mesh.field.add("MathEval", 12)   # Nose + points on wingtips and stabilizers
gmsh.model.mesh.field.setString(12, "F", "9000*(F3/{})^2 + 10".format(ref_len/2.5))

# Let's use the minimum of all the fields as the background mesh field:
gm.mesh.field.add("Min", 20)
gm.mesh.field.setNumbers(20, "FieldsList", [12])

gm.mesh.field.setAsBackgroundMesh(20)

gm.mesh.setSmoothing(3, 3, 10)

# gmsh.fltk.run()
# exit()


if volume_gen:
    gm.mesh.generate(3)
else:
    gm.mesh.generate(2)

if volume_gen:
    gmsh.write(case_name+'_processed_no_engines.msh3')
else:
    gmsh.write(case_name+'_surface_processed.msh3')

if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()
