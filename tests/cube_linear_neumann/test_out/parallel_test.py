import numpy as np
import multiprocessing as mp
from functools import partial
import time
import os

# UNCOMMENT the following line and change 'withMPI' to 'True' to lines to run with MPI on multiple nodes
# from mpi4py import MPI
withMPI = False


if __name__ == '__main__':

    if withMPI:

        comm = MPI.COMM_WORLD
        myrank = comm.Get_rank()
        nprocs = comm.Get_size()
        host = MPI.Get_processor_name()
        print('I am rank','%4d'%(myrank),'of',nprocs,'executing on',host,'. Python can use',len(os.sched_getaffinity(0)),'of',mp.cpu_count(),'processors.')

    else:
        print('Python can use',len(os.sched_getaffinity(0)),'of',mp.cpu_count(),'processors.')
