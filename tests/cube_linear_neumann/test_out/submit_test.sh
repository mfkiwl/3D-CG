#!/bin/bash

#SBATCH --exclusive
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --nodes 4

module load anaconda/2021b mpi

mpiexec --mca btl ^openib -np 4 python parallel_test.py



# Swap with the above to run with Python MP only on one node

#SBATCH --exclusive

# module load anaconda/2021b mpi    # Note that I am still loading the mpi module for a fair comparison (even though it's likely not doing anything)
# python parallel_test.py


