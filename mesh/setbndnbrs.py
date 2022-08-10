import numpy as np

def setbndnbrs(p, f, bndexpr, nnodes_per_face):
    # Extract faces that are on the boundary
    bdry_faces = np.argwhere(f<0)[:,0]

    # For each boundary face:
    for face in bdry_faces:
        # Take the mesh points at the ends of each face
        pts = p[f[face][:nnodes_per_face], :]   # Not sure why it was p[f[face][:2], :-1] before    - update: this was because in 2D the nodes are represented as 3D points with z=0 for all points which might mess with some of the expression evaluations.

        found_flag = False
        # Evaluate each boundary expression on the pair of points
        for bdry_idx, expr in enumerate(bndexpr):
            if eval(expr):
                f[face, nnodes_per_face+1] = -(bdry_idx+1)  # If found, update element in f - 1 indexed
                found_flag = True
                break        # Break out of the boundaryexpr loop
        
        if not found_flag:
            print('Error, boundary expression not matched for node ' + str(bdry_idx) + '!')
    return f


    #TODO: Update to use physical groups instead of the boundary expressions! Consolidate this assignment into the cg.py main driver script!