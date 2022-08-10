import sys
sys.path.append('../')
import gmsh_4_10 as gmsh
import sys
import numpy as np
from gmsh_4_10 import model as gm

#############################################
# volume_gen=True
volume_gen=False

case_name = 'CASE_NAME'

# Global scale factor - base linear point density
base_factor = 100/35257

# Only need if specifying a distance field in the volume refinement section
# ref_pt1 = 85
# ref_pt2 = 212
# ref_len = np.linalg.norm(gm.getValue(0, ref_pt1, [0]) - gm.getValue(0, ref_pt2, [0]))
#############################################

gmsh.initialize()
gm.add(case_name)

############## CREATE PLANE GEOMETRY ##############

gm.occ.importShapes('MESH_FMANE.STEP')
gm.occ.synchronize()
# gmsh.fltk.run()
# exit()

# print(gm.occ.getEntities())
# exit()
# Exraction of the bbox parameters
xmin, ymin, zmin, xmax, ymax, zmax = gm.occ.getBoundingBox(3, 1)

# Curve mesh refinement - translates to surface refinement
for curve in gm.occ.getEntities(1):
    curve_idx = curve[1]

    length = gm.occ.getMass(1, curve[1])
    numpts = max(int(length*base_factor), 3)

    # # Custom setting the element densities for important surfaces
    # if curve_idx in [307, 308, 136, 133, 135, 137, 134, 138, 306, 311, 309, 310, 88, 254]:     # engine pylons
    #     numpts *= 5

    # elif curve_idx in [289, 84, 346, 250]:
    #     gm.mesh.setTransfiniteCurve(curve_idx, int(numpts*5), coef=1.02)
    #     continue

    gm.mesh.setTransfiniteCurve(curve_idx, numpts)

aircraft_surfaces = [a[1] for a in gm.getEntities(2)]

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

    # Mesh resolution at box boundary - here we have 10 elements along the longest edge
    hmax=dx/10
    gmsh.option.setNumber("Mesh.MeshSizeMax", hmax)


# Volume mesh refinement
# Distance field for nose
# gm.mesh.field.add("Distance", 3)
# gm.mesh.field.setNumbers(3, "PointsList", [66, 82, 67, 81])

# gm.mesh.field.add("Distance", 4)
# gm.mesh.field.setNumbers(4, "PointsList", [9,2,103,42, 12, 142])

# # Distance field for leading edges - short
# gm.mesh.field.add("Distance", 5)
# gm.mesh.field.setNumbers(5, "CurvesList", [56, 96, 156, 152, 153, 205, 210, 103, 165, 305, 336, 339, 347, 361, 5, 16, 118, 226, 227, 284, 327])
# gm.mesh.field.setNumber(5, "Sampling", 80)
# gm.mesh.field.setNumbers(5, "PointsList", [211, 188])

# # Distance field for leading edges - long
# gm.mesh.field.add("Distance", 6)
# gm.mesh.field.setNumbers(6, "CurvesList", [74, 252])
# gm.mesh.field.setNumber(6, "Sampling", 200)

# gm.mesh.field.add("Distance", 7)    # Trailing edges
# gm.mesh.field.setNumbers(7, "CurvesList", [359, 354, 351, 349, 341, 345, 321, 312, 317, 260, 265, 177, 181, 184, 187, 190, 193, 196, 199, 253, 163,
#                                             330, 292, 295, 286, 231, 233, 124, 128, 129, 80, 82, 35, 38, 41, 44, 22, 25, 28, 31, 73, 19,
#                                             149, 211, 60, 65, 68, 70, 10])
# gm.mesh.field.setNumber(7, "Sampling", 100)

# gm.mesh.field.add("Distance", 8)    # APU/tail
# gm.mesh.field.setNumbers(8, "SurfacesList", [96])

# gmsh.model.mesh.field.add("MathEval", 12)   # Nose
# gmsh.model.mesh.field.setString(12, "F", "6000*(F3/{})^2 + 5".format(ref_len/1.5))

# gmsh.model.mesh.field.add("MathEval", 13)   # Points on wingtips and stabilizers
# gmsh.model.mesh.field.setString(13, "F", "5500*(F4/{})^2 + 12".format(ref_len/1.5))

# gmsh.model.mesh.field.add("MathEval", 14)   # Leading edges short
# gmsh.model.mesh.field.setString(14, "F", "15000*(F5/{})^2 + 9".format(ref_len))

# gmsh.model.mesh.field.add("MathEval", 15)   # Leading edges long
# gmsh.model.mesh.field.setString(15, "F", "15000*(F6/{})^2 + 8".format(ref_len))

# gmsh.model.mesh.field.add("MathEval", 16)   # Trailing edges
# gmsh.model.mesh.field.setString(16, "F", "20000*(F7/{})^2 + 25".format(ref_len))

# gmsh.model.mesh.field.add("MathEval", 17)   # APU
# gmsh.model.mesh.field.setString(17, "F", "10000*(F8/{})^2 + 15".format(ref_len))

# # Let's use the minimum of all the fields as the background mesh field:
# gm.mesh.field.add("Min", 20)
# gm.mesh.field.setNumbers(20, "FieldsList", [12, 13, 14, 15, 16, 17])

# gm.mesh.field.setAsBackgroundMesh(20)

gm.mesh.setSmoothing(3, 3, 10)

############################################################
# Assign physical groups -  when I enable this block, no tetrahedra are written to the .SU2 file!
assign_pg = 0
if assign_pg:
    surfaces= [a[1] for a in gm.getEntities(2)]
    farfield_surf = np.setdiff1d(surfaces, aircraft_surfaces)

    gm.addPhysicalGroup(2, aircraft_surfaces, -1, 'aircraft')
    gm.addPhysicalGroup(2, [farfield_surf[0]], -1, 'neg_X')
    gm.addPhysicalGroup(2, [farfield_surf[1]], -1, 'pos_X')
    gm.addPhysicalGroup(2, [farfield_surf[2]], -1, 'neg_Y')
    gm.addPhysicalGroup(2, [farfield_surf[3]], -1, 'pos_Y')
    gm.addPhysicalGroup(2, [farfield_surf[4]], -1, 'neg_Z')
    gm.addPhysicalGroup(2, [farfield_surf[5]], -1, 'pos_Z')
############################################################

if volume_gen:
    gm.mesh.generate(3)
else:
    gm.mesh.generate(2)

if volume_gen:
    if assign_pg:
        gmsh.write(case_name+'PG.msh3')
        gmsh.write(case_name+'PG.su2')

        # Copy physical groups from one SU2 file to the other - workaround due to gmsh bug - HAS TO BE DONE AFTER THE MESH W/O PHYSICAL GROUPS HAS BEEN GENERATED
        with open(case_name+'PG.su2', 'r') as file:
            lines = file.readlines()

        marker_start = 'NMARK= 7\n'

        idx = lines.index(marker_start)

        pg = lines[idx:]

        with open(case_name+'.su2', 'a') as file:
            file.writelines(pg)
    else:
        gmsh.write(case_name+'.msh3')
        gmsh.write(case_name+'.su2')
else:
    gmsh.write(case_name+'_surface_processed.msh3')

if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()
