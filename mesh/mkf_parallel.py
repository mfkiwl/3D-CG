import sys
import numpy as np
import time
import multiprocessing

# Guiding light principle: given a face with a certain node pattern, the element opposite it will have that face with the nodes going in the opposite direction.

def find_first(a, b):
    result = np.where(np.all(a == b, axis=1))
    result = result[0][0] if result[0].shape[0]>0 else -1
    return result

# @profile
def mkt2f_new(t, ndim):
    def build_faces3d():

        interior_f_idx = 0  # Starting out with 1-indexing

        # "Alias" for number of face combos per_elem
        nfcpe = num_nodes_per_elem*3    # magic number 3 is the number of face combintations per face - we didn't have this with line segment faces

        for elnum in range(numel):
            if elnum%1000 == 0:
                print('elem', elnum, '/', numel)
            elem = t[elnum,:]
            # for i, node in enumerate(elem): # The ith node in the element
            for i in range(len(elem)): # The ith node in the element
                node = elem[i]
                nodes_on_face = elem[f_idx_template[i,:]]   # This is the face

                # The face will match going backwards on the opposite element. The "opposite element idx" is what is returned by find_first
                # idx = find_first(face_array, np.flip(nodes_on_face))      # Too slow!!!

                # eidx = np.flatnonzero(t-nodes_on_face[0]==0)//num_nodes_per_elem
                eidx = np.where(t==nodes_on_face[0])[0]


                fidx = eidx*nfcpe    # Narrow down the search to the elements that contain at least one of the nodes in the face - could have used any of the nodes but just picked the first one here

                # r=np.tile(np.arange(nfcpe), fidx.shape[0])
                # arrays = np.arange(nfcpe).reshape(nfcpe, -1)
                # r = np.repeat(arrays,fidx.shape[0], axis=0).ravel()
                # l=tuple([np.arange(nfcpe)]*fidx.shape[0])
                # r=np.hstack(l)
                r = np.arange(nfcpe*fidx.shape[0])%nfcpe

                faces_idx = r+np.repeat(fidx, nfcpe)      # All in 1D arrays

                found_idx = np.where(np.all(face_array[faces_idx, :] == np.flip(nodes_on_face), axis=1))
                # found_idx = np.nonzero(np.sum(np.abs(face_array[faces_idx, :]-np.flip(nodes_on_face)), axis=1)==0)    # This was actually found to be slower than the above line
                found_idx = found_idx[0][0] if found_idx[0].shape[0]>0 else -1

                if found_idx == -1:
                    # Insert face and elnum into the bdry_faces list
                    bdry_faces.append(np.concatenate((nodes_on_face, np.array([elnum, -1]))))
                    # t2f_bdry[elnum, i] = bdry_f_idx + 1
                    # bdry_f_idx += 1

                else:
                    # opp_elnum = idx // num_face_combos_per_elem

                    opp_elnum = eidx[found_idx // nfcpe]
                    # Figure out the parity of the nodes in the face permutation
                    
                    node1 = nodes_on_face[0]
                    node2 = nodes_on_face[1]
                    node3 = nodes_on_face[2]

                    # sign = np.sign(Eijk(nodes_on_face[0], nodes_on_face[1], nodes_on_face[2]))

                    sign = 1
                    if (node1 > node2) and (node2 > node3):
                        sign  = -1
                    elif (node2 > node3) and (node2 > node1):
                        sign = -1
                    elif (node3 > node1) and (node3 > node2):
                        sign = -1

                    # Insert into faces list
                    if sign > 0:    # Face nodes going CCW around element match going in order of increasing node number
                        interior_faces.append(np.concatenate((np.sort(nodes_on_face), np.array([elnum, opp_elnum]))))
                        t2f_int[elnum, i] = interior_f_idx + 1      # 1-indexed
                    elif sign < 0:    # Face nodes going CCW around OPPOSITE element match going in order of increasing node number
                        interior_faces.append(np.concatenate((np.sort(nodes_on_face), np.array([opp_elnum, elnum]))))
                        t2f_int[elnum, i] = -(interior_f_idx+1)     # 1-indexed
                    else:
                        raise ValueError('Cannot have a repeated index')

                    interior_f_idx += 1
        # print(time.perf_counter()-start)
        return (interior_faces, bdry_faces)


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


        # Make a serial version of what will eventually become the multiprocessing

        t2f_int = np.zeros_like(t)
        t2f_bdry = np.zeros_like(t)



        total_elem = np.arange(t.size)



        result = build_faces3d()
        interior_faces, bdry_faces,t2f_int, t2f_bdry = result






        # Need to number the boundary nodes correctly after being added at the end of the array in t2f
        # Need to account for taking duplicates out 

        bdry_faces = np.asarray(bdry_faces)
        interior_faces, inv_idx = np.unique(np.asarray(interior_faces), axis=0, return_inverse=True)

        sign_arry = np.sign(t2f_int)
        idx_arry = np.abs(t2f_int) - 1

        t2f_int = sign_arry * (inv_idx[idx_arry]+1)

        t2f_bdry[t2f_bdry != 0] += interior_faces.shape[0]
        t2f = t2f_int + t2f_bdry

        f = np.concatenate((interior_faces, bdry_faces), axis=0)
    return f, t2f


if __name__ == '__main__':
    sys.path.insert(0, '../util')
    from import_util import load_processed_mesh

    global mesh
    mesh = load_processed_mesh('../data/3D/h0.1_tets4686')
    # mesh = load_processed_mesh('../data/3D/h0.05_tets37153')
    f, t2f = mkt2f_new(mesh['t'].T, 3)
