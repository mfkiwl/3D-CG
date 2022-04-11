#!/bin/bash

#SBATCH --exclusive
#SBATCH -N 1
#SBATCH -n 48

python main.py
