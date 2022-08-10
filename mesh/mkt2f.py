import sys
from re import L
import numpy as np
from mkt2t import mkt2t, find_first
import logging

def mkt2f(t, ndim):
    # Think about it this way: f and t2f are basically inverse mappings. One maps the row of f (face #) to the elements to the right and left of it.
    # The other maps the row of t (element #) to the face numbers that bound it.

    # NOTE: t2f is 1-indexed because the faces are signed.
    logging.basicConfig(filename='test.log', encoding='utf-8', level=logging.DEBUG)

    if ndim == 2:
        t2t, face_array, t_ext = mkt2t(t, ndim)

        # Returns (elem, in) index of points across from boundaries
        bdry_pts = np.asarray(np.where(t2t < 0)).T

        # Next: make a list of all the faces that are on the boundary. These will be the last rows of t2f.
        bdry_f = np.delete(t[bdry_pts[:,0],:], bdry_pts[:,1], axis=1)
        bdry_f = np.concatenate((bdry_f, bdry_pts[:, 0][:,None], np.ones_like(bdry_pts[:, 0][:,None])*-1), axis=1)
        bdry_face_idx = 3*bdry_pts[:, 0] + bdry_pts[:, 1]

        # Remove boundary faces from the face list
        face_array = np.delete(face_array, bdry_face_idx, axis=0)

        # Distill all the unique interior faces
        face_array = np.unique(np.sort(face_array, axis=1), axis=0)

        # Initialize face array
        f = np.zeros((face_array.shape[0]+bdry_f.shape[0], 4), dtype=np.int32)
        f[:face_array.shape[0], :face_array.shape[1]] = face_array
        f[face_array.shape[0]:, :] = bdry_f

        # Initialize t array
        # Note that by convention, t2f is 1-indexed because the faces are signed - repeated note above.
        t2f = np.zeros_like(t)

        # Loop through all the elements in t2t
        for i_elem, elem in enumerate(t2t):
            for i_node, opp_elem_num in enumerate(elem):     # i_node is 0, 1 2 (column in mesh.t)
                if opp_elem_num >= 0:    # Already took care of the boundary faces above
                    # We can kill two bird with one stone here - if the current element is found going the right way, then we can immediately input the opposite element as going the other way and vice versa.

                    # Pull node numbers for face opposite node from mesh.t GOING COUNTER-CLOCKWISE
                    face_ccw = t_ext[i_elem, i_node:i_node+2]

                    # Check to see if face is in face array and pull index
                    face_idx = find_first(face_array, face_ccw)
                    face_of_opp_elem = np.where(t2t[opp_elem_num, :] == i_elem)[0][0]

                    # If it is, then assign i_elem to face_array[if, 2] and the opposite element to face_array[if, 3]
                    if face_idx >= 0:
                        f[face_idx, face_array.shape[1]:] = np.array([i_elem, opp_elem_num])
                        t2f[i_elem, i_node] = face_idx + 1  # 1-indexed
                        t2f[opp_elem_num, face_of_opp_elem] = -(face_idx+1) # 1-indexed

                    # If not, then the CW face has to match
                    else:
                        face_idx = find_first(face_array, np.flip(face_ccw))
                        if face_idx >= 0:
                            f[face_idx, face_array.shape[1]:] = np.array([opp_elem_num, i_elem])
                            t2f[i_elem, i_node] = -(face_idx+1)  # 1-indexed
                            t2f[opp_elem_num, face_of_opp_elem] = face_idx + 1  # 1-indexed

        # Populate t2f with bdry nodes
        for ibdry_face, bdry_face in enumerate(f[face_array.shape[0]:, :]):
            elnum = bdry_face[2]
            face_idx_in_elem = np.argwhere(t2f[elnum, :] == 0)[0]   # One less index required for np.argwhere
            t2f[elnum, face_idx_in_elem] = face_array.shape[0] + ibdry_face + 1     # 1-indexed


        # print(np.concatenate((np.arange(f.shape[0])[:,None]+1, f+1), axis=1))
        # print(t+1)

    if ndim == 3:

        # Guiding light principle: Make an array of the unique faces in the mesh. Each face will list the nodes on each face going in ascending order (inside the domain), and on the boundaries, such that the element going CCW on the left.

        t2t, face_array, t_ext = mkt2t(t, ndim)

        # Returns (elem, in) index of points across from boundaries
        bdry_pts = np.asarray(np.where(t2t < 0)).T

        # Next: make a list of all the faces that are on the boundary. These will be the last rows of t2f.
        bdry_f = np.delete(t[bdry_pts[:, 0], :], bdry_pts[:, 1], axis=1)        # BUG: This only works for 
        bdry_f = np.concatenate((bdry_f, bdry_pts[:, 0][:, None], np.ones_like(bdry_pts[:, 0][:, None])*-1), axis=1)
        bdry_face_idx = 3*bdry_pts[:, 0] + bdry_pts[:, 1]

        # Remove boundary faces from the face list
        face_array = np.delete(face_array, bdry_face_idx, axis=0)

        # Distill all the unique interior faces
        face_array = np.unique(np.sort(face_array, axis=1), axis=0)

        # Initialize face array
        f = np.zeros((face_array.shape[0]+bdry_f.shape[0], 4), dtype=np.int32)
        f[:face_array.shape[0], :face_array.shape[1]] = face_array
        f[face_array.shape[0]:, :] = bdry_f

        # Initialize t array
        # Note that by convention, t2f is 1-indexed because the faces are signed - repeated note above.
        t2f = np.zeros_like(t)

        # Loop through all the elements in t2t
        for i_elem, elem in enumerate(t2t):
            # i_node is 0, 1 2 (column in mesh.t)
            for i_node, opp_elem_num in enumerate(elem):
                if opp_elem_num >= 0:    # Already took care of the boundary faces above
                    # We can kill two bird with one stone here - if the current element is found going the right way, then we can immediately input the opposite element as going the other way and vice versa.

                    # Pull node numbers for face opposite node from mesh.t GOING COUNTER-CLOCKWISE
                    face_ccw = t_ext[i_elem, i_node:i_node+2]

                    # Check to see if face is in face array and pull index
                    face_idx = find_first(face_array, face_ccw)
                    face_of_opp_elem = np.where(
                        t2t[opp_elem_num, :] == i_elem)[0][0]

                    # If it is, then assign i_elem to face_array[if, 2] and the opposite element to face_array[if, 3]
                    if face_idx >= 0:
                        f[face_idx, face_array.shape[1]:] = np.array(
                            [i_elem, opp_elem_num])
                        t2f[i_elem, i_node] = face_idx + 1  # 1-indexed
                        t2f[opp_elem_num, face_of_opp_elem] = - \
                            (face_idx+1)  # 1-indexed

                    # If not, then the CW face has to match
                    else:
                        face_idx = find_first(face_array, np.flip(face_ccw))
                        if face_idx >= 0:
                            f[face_idx, face_array.shape[1]:] = np.array(
                                [opp_elem_num, i_elem])
                            t2f[i_elem, i_node] = -(face_idx+1)  # 1-indexed
                            # 1-indexed
                            t2f[opp_elem_num, face_of_opp_elem] = face_idx + 1

        # Populate t2f with bdry nodes
        for ibdry_face, bdry_face in enumerate(f[face_array.shape[0]:, :]):
            elnum = bdry_face[2]
            # One less index required for np.argwhere
            face_idx_in_elem = np.argwhere(t2f[elnum, :] == 0)[0]
            t2f[elnum, face_idx_in_elem] = face_array.shape[0] + \
                ibdry_face + 1     # 1-indexed

        # print(np.concatenate((np.arange(f.shape[0])[:,None]+1, f+1), axis=1))
        # print(t+1)

    return f, t2f

if __name__ == '__main__':
    sys.path.insert(0, '../util')
    from import_util import load_processed_mesh

    mesh = load_processed_mesh('../data/square')
    mkt2f(mesh['t'])

