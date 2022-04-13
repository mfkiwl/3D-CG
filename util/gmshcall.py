import time
import shelve
# from memory_profiler import profile
import os
import numpy as np
import pandas as pd
import sys
# cdir = '/media/homehd/saustin/Exasim'
# exec(open('/media/homehd/saustin/Exasim/Installation/setpath.py').read())

# import Preprocessing


def find(list, substr):
    for idx, string in enumerate(list):
        if substr in string:
            return idx
    return None

def bisection_search(input_lines, elemtype1, global_first_elem_idx):
    """
    Returns the 0-indexed first index where the element type changes
    """

    elemtype_last = int(input_lines[-1].split()[1])   # 0-indexed

    # Catches the case where there is no index change in all the lines
    if elemtype1 == elemtype_last:
        return len(input_lines)

    numel = len(input_lines)

    if numel == 1:
        return global_first_elem_idx

    half_idx = numel//2
    elemtype = int(input_lines[half_idx - 1].split()[1])   # 0-indexed

    if elemtype == elemtype1:
        global_first_elem_idx += half_idx
        return bisection_search(input_lines[half_idx:], elemtype1, global_first_elem_idx)
    elif elemtype != elemtype1:
        return bisection_search(input_lines[:half_idx], elemtype1, global_first_elem_idx)


# @profile
def gmshcall(filename, ndim, elemtype, elemtype_face=None):
    """
    inputs:
    -pde: Exasim pde dictionary object
    -filename: .geo mesh filename
    -nd: dimensionality of geometry
    -elemtype: tri, quad, tet, hex, etc. See http://gmsh.info/dev/doc/texinfo/gmsh.pdf 9.1 for a complete list of possibilities

    According to the .msh version 3 standard, the ASCII mesh file will conform to the following format. See Sec 9.1 of http://gmsh.info/dev/doc/texinfo/gmsh.pdf for information on version 4.1 (can't find a reference for V3).
    $MeshFormat
    <Mesh format information>
    $EndMeshFormat
    $PhysicalNames
    <Physical groups and names>
    $EndPhysicalNames
    $Entities
    <Physical entity informatin>
    $EndEntities
    $Nodes
    <Node coordinates>
    $EndNodes
    $Elements
    <Mesh elements, includes both surface (2D) and volume (3D) elements
    $EndElements

    """

    tb0 = time.perf_counter()

    if ('.geo' not in filename) and ('.msh' not in filename) and ('.msh3' not in filename):
        raise ValueError(
            'Please specify a valid mesh file extension, either .geo or .msh, exiting!')

    if '.geo' in filename:
        # Format .msh version 3, silent output
        opts = "-format msh3 -v 0"

        # # find gmsh executable
        # gmsh = Preprocessing.findexec(pde['gmsh'], pde['version'])

        # print("Gmsh mesh generator...\n")
        mystr = "gmsh " + filename + ".geo -" + str(ndim) + " " + opts
        os.system(mystr)
        filename.replace('.geo', '.msh')

    with open(filename, 'r') as meshfile:
        lines = meshfile.readlines()

    # Version check
    if float(lines[1].split()[0]) != 3:
        raise ValueError(
            'The Gmsh parser only accepts .msh files of version 3, exiting!')

    ptr = 0     # "Pointer" to current location in file line list - prevents searching through the whole file

    # Physical group names
    if elemtype_face is not None:
        ptr_tmp = find(lines, '$PhysicalNames')

        if ptr_tmp != None:
            num_phys_grps = int(lines[ptr_tmp + 1].strip())
            phys_grp_dict = {}

            for physgrp in lines[ptr_tmp+2: ptr_tmp+2+num_phys_grps]:
                phys_grp = physgrp.split()
                if not int(phys_grp[0]) == ndim:
                    phys_grp_dict[phys_grp[2].strip('"')] = {'dim': int(
                        phys_grp[0]), 'idx': int(phys_grp[1])}

            # Update ptr to end of block
            ptr = ptr_tmp + num_phys_grps + 2

        else:   # No physical group names specified
            ptr_tmp = find(lines, '$Entities')
            if ptr_tmp:
                l = np.asarray(
                    lines[ptr_tmp+1].strip().split(), dtype=np.int32)
                num_phys_grps = l[ndim-1]
                phys_grp_dict = {key: {'idx': key, 'dim': ndim-1}
                                 for key in np.arange(1, num_phys_grps+1)}
            else:
                print('No physical groups found')

    # Nodes
    # Update ptr to the start of the nodes list
    ptr += find(lines[ptr:], '$Nodes')
    nnodes = int(lines[ptr + 1].strip())

    nodes = pd.read_csv(filename, delimiter=' ', header=None, dtype=np.float64,
                        skiprows=ptr+2, nrows=nnodes).to_numpy()[:, 1:-1]

    # Elements
    # Update ptr to the start of the nodes list
    ptr = ptr + nnodes + 2 + find(lines[ptr+nnodes+2:], '$Elements')
    numel = int(lines[ptr + 1].strip())
    first_elem_line = ptr + 2

    elemtype1 = int(lines[first_elem_line].split()[1])

    elem_change_idx1 = bisection_search(
        lines[first_elem_line:first_elem_line+numel], elemtype1, 0)


    # Assumption: Pulls the first two types of elements listed - this is usually the highest two dimensions.
    elem1 = pd.read_csv(filename, delimiter=' ', header=None, dtype=np.int64,
                        skiprows=first_elem_line, nrows=elem_change_idx1).to_numpy()

    elemtype2 = int(lines[first_elem_line+elem_change_idx1].split()[1])

    elem_change_idx2 = bisection_search(lines[first_elem_line+elem_change_idx1:first_elem_line+numel], elemtype2, 0)

    elem2 = pd.read_csv(filename, delimiter=' ', header=None, dtype=np.int64,
                        skiprows=first_elem_line+elem_change_idx1, nrows=elem_change_idx2).to_numpy()

    elemtype2 = elem2[0, 1]

    if elemtype1 == elemtype:
        mesh_t = elem1[:, 4:] - 1    # Subtracting 1 to make nodes 0-indexed
    elif elemtype2 == elemtype:
        mesh_t = elem2[:, 4:] - 1
    

    if elemtype_face:
        if elemtype1 == elemtype_face:
            surf_elem = np.delete(elem1, [0, 1, 3], axis=1)
        if elemtype2 == elemtype_face:
            surf_elem = np.delete(elem2, [0, 1, 3], axis=1)
        
        surf_elem[:,1:] -= 1 # Nodes must be 0-indexed

        if ndim == 2:
            # Will contain the indices of the surfaces of constant x, y, or z (0, 1, or 2 respectively) at the extrema of the cube (can only handle cubic domains)
            bdry_surf_idx = {0: [], 1: []}
        elif ndim == 3:
            # Will contain the indices of the surfaces of constant x, y, or z (0, 1, or 2 respectively) at the extrema of the cube (can only handle cubic domains)
            bdry_surf_idx = {0: [], 1: [], 2: []}

        for key in phys_grp_dict:       # Key is the surface # (1-indexed)
            nodes_on_surf = surf_elem[surf_elem[:, 0] == phys_grp_dict[key]['idx'], :]      # This list includes the face index and normal vectors
            unique_nodes_on_surf = np.unique(nodes_on_surf[:, 1:4]).astype(np.int32)        # surface node info is contained in cols 1, 2, and 3 of surf_elem
            
            # phys_grp_dict[key]['nodes'] = unique_nodes_on_surf
            phys_grp_dict[key]['nodes'] = {key.tobytes():None for key in unique_nodes_on_surf}

            pts = nodes[unique_nodes_on_surf, :]
            # Find bounding box of domain
            # Finds the column that has repeated coordinates - this means that it's on a planar surface

            col_idx = np.where(np.all(pts == pts[0, :], axis=0))
            # print(col_idx[0].shape[0])
            if col_idx[0].shape[0] != 0:     # If the surface is planar
                col_idx = col_idx[0][0]
                bdry_surf_idx[col_idx].append([pts[0, col_idx], phys_grp_dict[key]['idx']])

        for axis in bdry_surf_idx:
            axis_pts = np.asarray(bdry_surf_idx[axis]).T
            sort_idx = np.argsort(axis_pts[0, :])
            # Take the first and last columns - max and min - allows for multiple planar surfaces in the domain, but only takes the max and min. Now we have the faces that we need to set the bondary on.
            axis_pts = axis_pts[:, sort_idx][:, [0, -1]]

            neg_normal_vec = np.zeros((3))
            neg_normal_vec[axis] = -1

            # Take surface of minimum axis val
            for key in phys_grp_dict:       # Key is the surface # (1-indexed)
                if phys_grp_dict[key]['idx'] == axis_pts[1,0]:
                    # Create the normal array and set the normals to -1
                    phys_grp_dict[key]['normals'] = np.tile(neg_normal_vec, (len(phys_grp_dict[key]['nodes']), 1))     # Tiles the normal vector the number of times as there are elements on the surface

            # Take surface of maximum axis val
            for key in phys_grp_dict:       # Key is the surface # (1-indexed)
                if phys_grp_dict[key]['idx'] == axis_pts[1,1]:
                    # Create the normal array and set the normals to -1
                    phys_grp_dict[key]['normals'] = np.tile(-neg_normal_vec, (len(phys_grp_dict[key]['nodes']), 1))     # Tiles the normal vector the number of times as there are elements on the surface

        return nodes, mesh_t, phys_grp_dict
    else:
        return nodes, mesh_t



if __name__ == '__main__':
    # pde, mesh = Preprocessing.initializeexasim()

    # p, t, pg = gmshcall(pde, "h1.0_tets24.msh", 3, 4, 2)
    # p, t = gmshcall(pde, "h1.0_tets24.msh", 3, 4)

    # p, t, pg = gmshcall(pde, "h0.01_tets4504962.msh", 3, 4, 2)
    # p, t = gmshcall(pde, "h0.01_tets4504962.msh", 3, 4)
    # p, t = gmshcall(pde, "h0.5_tets101.msh", 3, 4)

    filename = sys.argv[1]
    p, t, pg = gmshcall(filename, 3, 4, 2)

    # print(p.T)
    # print()
    # print(t.T)
    # print()
    # print(pg)

    # print(p.transpose())
    # print(t.transpose())

    # for key in pg:
    #     print(key)
    #     print(pg[key]['nodes_and_normals'])
    #     print()

    with open(filename + 'ORIG.npy', 'wb') as f:
        np.save(f, p)
        np.save(f, t)
        np.save(f, pg)

    print(p)
    print(t)

    print('Saved to ' + filename + 'ORIG.npy')
