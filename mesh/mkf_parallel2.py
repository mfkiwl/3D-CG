import sys
import numpy as np
from sympy import Eijk
import multiprocessing as mp
from functools import partial
import time
import logging

logger = logging.getLogger(__name__)

# Given a face with a certain node pattern, the element opposite it will have that face with the nodes going in the opposite direction.

def find_first(a, b):
    result = np.where(np.all(a == b, axis=1))
    result = result[0][0] if result[0].shape[0]>0 else -1
    return result

def get_face(t, face_array, f_idx_template, idx):
        numel, num_nodes_per_elem = t.shape
        elnum = idx // t.shape[1]
        nodenum = idx%t.shape[1]

        if elnum % 10000 == 0 and nodenum == 0:
            logger.info(str(elnum)+'/'+str(t.shape[0]))
        elem = t[elnum, :]

        # This is the face
        nodes_on_face = elem[f_idx_template[nodenum, :]]

        # The face will match going backwards on the opposite element. The "opposite element idx" is what is returned by find_first
        idx = find_first(face_array, np.flip(nodes_on_face))

        if idx == -1:
            # Insert face and elnum into the bdry_faces list
            face = np.concatenate((nodes_on_face, np.array([elnum, -1])))
        else:
            # magic number 3 is the number of face combintations per face - we didn't have this with line segment faces
            opp_elnum = idx // (num_nodes_per_elem*3)
            # Figure out the parity of the nodes in the face permutation

            sign = np.sign(
                Eijk(nodes_on_face[0], nodes_on_face[1], nodes_on_face[2]))

            # Insert into faces list
            if sign > 0:    # Face nodes going CCW around element match going in order of increasing node number
                face = np.concatenate(
                    (np.sort(nodes_on_face), np.array([elnum, opp_elnum])))
            elif sign < 0:    # Face nodes going CCW around OPPOSITE element match going in order of increasing node number
                face = np.concatenate(
                    (np.sort(nodes_on_face), np.array([opp_elnum, elnum])))
            else:
                raise ValueError('Cannot have a repeated index')
        return face

def mkt2f_new(t, ndim):

    if ndim == 2:
        t2t = np.zeros_like(t, dtype=np.int32)

        rolled = np.roll(t, -1, axis=1)     # Rolling so that the faces opposite the first node is naturally the first pair in the list - isnt' the case for 3D and isn't important
        t_ext = np.concatenate((rolled, rolled[:, 0][:, None]), axis=1)    # Accounts for the fact that the third face in the triangle will include the first and last points in the array and thus the face array needs to be wrapped around.
        # ^ Will look different in 3D because there are more options for the faces
        t_flattened = np.ravel(t_ext)  # Using C ordering (row major)

        window = 2  # number of nodes per face  - will need to be changed in 3D

        # For a array with an empty element, like (104,), array[-1] is the last non-empty element, so the [-1]th element would be 103 and array[:-1] would be empty
        # Shape of the resultant array of faces for all elements - includes duplicates. Idea: for 3D, include all permutations of node numbers on each face separated by -1 so that it can't match across permutations
        shape = t_flattened.shape[:-1] + (t_flattened.shape[-1] - window + 1, window)       # Number in index 0 is the number of rows of faces, accounts for the width of the window, etc

        strides = t_flattened.strides + (t_flattened.strides[-1],)  # This is tuple addition, increading the dimension of the eventual array by 1 but keeping the same strides as the original flattened array
        # ^ This is straight out of exercise 14: slide a 1D window - in the stride medium post. We are sliding a 1D window across the flattened array.
        # Note that this creates the issue where rows are included that have the lines between elements in the flattened array. This should not be the case and are removed in 

        # Again, see "sliding 1D" window example in stride post. This is just that.
        face_array = np.lib.stride_tricks.as_strided(t_flattened, shape=shape, strides=strides)

        # Removes inter-element artifact faces of the reshaping process
        face_array = face_array[np.mod(np.arange(1, face_array.shape[0]+1), t.shape[1]+1) != 0, :]

        num_nodes_per_elem = t.shape[1]

        # Everything up to this point has just been preparing the data structures for the population process

        for elnum, elem in enumerate(t_ext):
            for i, node in enumerate(elem[:-1]):
                idx = find_first(face_array, np.flip(elem[i:i+2]))      # The face will match going backwards on the opposite element. The "opposite element idx" is what is returned by find_first
                idx = idx // num_nodes_per_elem
                t2t[elnum, i] = idx

    if ndim == 3:
        # t2t = np.zeros_like(t, dtype=np.int32)
        numel, num_nodes_per_elem = t.shape

        # Indices of the nodes in the element corresponding to each of the four faces - face i is across from node i. These orderings are hardcoded from the Gmsh standard node numbering convention.
        f_idx_template = np.array([[1, 3, 2],    # Nodes on face 0
                                [2, 3, 0],      # Nodes on face 1
                                [0, 3, 1],      # Nodes on face 2
                                [0, 1, 2]])     # Nodes on face 3

        f_idx = np.tile(f_idx_template, (numel, 1))
        col_idx = np.tile(np.arange(f_idx.shape[0])[:, None], (1, f_idx.shape[1]))
        t_ext = np.repeat(t, 4, axis=0)
        face_array = t_ext[col_idx, f_idx]

        # Keep going to expand the possibilities for each face: (1, 2, 3), (2, 3, 1), (3, 1, 2)

        face_array = np.repeat(face_array, 3, axis=0)   # 3 is the number of possibilities for each face (see above)
        # Roll the second and third elements
        face_array[1::3, :] = np.roll(face_array[1::3, :], -1, axis=1)
        face_array[2::3, :] = np.roll(face_array[2::3, :], -2, axis=1)

        # Need to number the boundary nodes correctly after being added at the end of the array in t2f
        # Need to account for taking duplicates out 

        # bdry_faces = np.asarray(bdry_faces)
        # interior_faces = np.unique(np.asarray(interior_faces), axis=0)
        # f = np.concatenate((interior_faces, bdry_faces), axis=0)

        pool = mp.Pool(mp.cpu_count())
        start = time.perf_counter()

        result = pool.map(partial(get_face, t, face_array, f_idx_template), np.arange(t.size))
        faces = np.asarray(result)

        # faces = np.asarray(list(map(partial(get_face, t, face_array, f_idx_template), np.arange(t.size))))

        logger.info(str(time.perf_counter()-start))

        sort_idx = np.argsort(faces[:,-1])
        faces = faces[sort_idx[::-1], :]      # Separate boundary faces to end of array

        bdry_faces_start_idx = np.where(faces[:, -1] < 0)[0][0]
        interior_faces = faces[:bdry_faces_start_idx, :]
        bdry_faces = faces[bdry_faces_start_idx:, :]
        f = np.concatenate((np.unique(interior_faces, axis=0), bdry_faces), axis=0)

        t2f = np.zeros_like(t)
        for iface, face in enumerate(f):
            # if iface != 3:
            #     continue

            nodes_on_face = face[:3]
            elem1 = face[3]

            elem1_nodes = t[elem1, :]
            mask1 = np.in1d(elem1_nodes, nodes_on_face)
            elem1_face_nodes = t[elem1, :][f_idx_template[~mask1,:]].ravel()
            # print(elem1_face_nodes)

            # print(nodes_on_face)
            # print(elem1)

            bdry_flag = False
            elem2 = face[4]
            if elem2 < 0:
                bdry_flag = True
            else:
                elem2_nodes = t[elem2, :]
                mask2 = np.in1d(elem2_nodes, nodes_on_face)
                # elem2_face_nodes = t[elem2, :][f_idx_template[~mask2, :]]

            if np.sign(Eijk(nodes_on_face[0], nodes_on_face[1], nodes_on_face[2])) == np.sign(Eijk(elem1_face_nodes[0], elem1_face_nodes[1], elem1_face_nodes[2])):
                t2f[elem1,:][~mask1] = iface+1        # ~mask gives us the node index for the face
                if not bdry_flag:
                    t2f[elem2,:][~mask2] = -(iface+1)        # ~mask gives us the node index for the face
            else:
                t2f[elem1,:][~mask1] = -(iface+1)        # ~mask gives us the node index for the face
                if not bdry_flag:
                    t2f[elem2,:][~mask2] = iface+1        # ~mask gives us the node index for the face
            # print(t2f)
            # exit()

    return f, t2f


if __name__ == '__main__':
    sys.path.insert(0, '../util')
    from import_util import load_processed_mesh

    mesh = load_processed_mesh('../data/3D/h1.0_tets24')
    # mesh = load_processed_mesh('../data/3D/h0.1_tets4686')
    f, t2f = mkt2f_new(mesh['t'].T, 3)
    # print(f)
    # print(t2f)

    np.savetxt('f_test2.txt', f, fmt='%d')
    np.savetxt('t2f_test2.txt', t2f, fmt='%d')
