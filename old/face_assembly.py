import numpy as np
import multiprocessing as mp
from functools import partial
import time
# from mpi4py import MPI
withMPI = False
import os

def Eijk(p1, p2, p3):
    if (p1 < p2) and (p2 < p3):    # (1, 2, 3)
        return 1
    elif (p1<p3) and (p3<p2):    # (1, 3, 2)
        return -1
    elif (p2<p1) and (p1<p3):    # (2, 1, 3)
        return -1
    elif (p3<p1) and (p1<p2):    # (2, 3, 1)
        return 1
    elif (p2<p3) and (p3<p1):    # (3, 1, 2)
        return 1
    elif (p3<p2) and (p2<p1):    # (3, 2, 1)
        return -1

def find_first(a, b):
    # result = np.where(np.all(a == b, axis=1))
    # result = result[0][0] if result[0].shape[0]>0 else -1
    key = b.tobytes()
    if key in a:
        return a[key]
    else:
        return -1

def get_face(t, face_array, f_idx_template, idx):
        numel, num_nodes_per_elem = t.shape
        elnum = idx // t.shape[1]
        nodenum = idx%t.shape[1]

        if elnum % 1000 == 0 and nodenum == 0:
            print(str(elnum)+'/'+str(t.shape[0]))
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

if __name__ == '__main__':

    if withMPI:

        comm = MPI.COMM_WORLD
        myrank = comm.Get_rank()
        nprocs = comm.Get_size()
        host = MPI.Get_processor_name()
        print('I am rank','%4d'%(myrank),'of',nprocs,'executing on',host)

        with open('setup_arrays.npy', 'rb') as f:
            t = np.load(f).astype(np.int32)
            face_array = np.load(f).astype(np.int32)
            f_idx_template = np.load(f)


        print('Converting face_array to dict')
        face_array_dict = {face_entry.tobytes():idx for idx, face_entry in enumerate(face_array)}

        print('done converting')

        # "Domain decomposition" - mapping from global to local indices that will be handled by this rank
        numjobs = t.shape[0]*t.shape[1]
        nbase = numjobs//nprocs
        rem = numjobs%nprocs

        if myrank < rem:
            start_idx = (nbase+1)*myrank
            end_idx = start_idx + nbase + 1
        else:
            start_idx = (nbase+1)*rem + (myrank-rem)*nbase
            end_idx = start_idx + nbase

        loc_idx = np.arange(start_idx, end_idx)
        # print('Proc {}: handling {}'.format(myrank, loc_idx))

        # print(loc_idx)
        print(len(os.sched_getaffinity(0)))
        if myrank == 0:
            start = time.perf_counter()
        with mp.Pool(mp.cpu_count()) as pool:
            result = pool.map(partial(get_face, t, face_array_dict, f_idx_template), loc_idx)    # All I'm changing here is the indices that are fed into the multiprocessing routine

        # print(type(result))

        gathered_result = comm.gather(result, root=0)

        if myrank == 0:
            print('Time ', time.perf_counter()-start)

            faces = np.asarray(gathered_result).reshape((-1, 5))
            # print('here')
            # print(faces.shape)
            # for face in faces:
            #     print(face)
            with open('faces.npy', 'wb') as f:
                np.save(f, faces)
            
            print('Saved faces to file, done...')

    else:
        with open('setup_arrays.npy', 'rb') as f:
            t = np.load(f)
            face_array = np.load(f)
            f_idx_template = np.load(f)

        start = time.perf_counter()
        with mp.Pool(mp.cpu_count()) as pool:
            result = pool.map(partial(get_face, t, face_array, f_idx_template), np.arange(t.size))

        print('Time ', time.perf_counter()-start)

        faces = np.asarray(result)
        # print(t.shape)
        # print(faces.shape)
        # for face in faces:
        #     print(face)
        # exit()
        with open('faces.npy', 'wb') as f:
            np.save(f, faces)
        
        print('Saved faces to file, done...')

