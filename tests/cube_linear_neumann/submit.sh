#!/bin/bash

#SBATCH --exclusive
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --nodes 4

module load anaconda/2021b mpi

mpiexec --mca btl ^openib -np 4 python face_assembly.py
# python face_assembly.py



# Swap with the above to run with Python MP only

#SBATCH --exclusive

# module load anaconda/2021b mpi
# python face_assembly.py


