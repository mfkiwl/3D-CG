#!/bin/bash

#SBATCH --exclusive
#SBATCH -N 1
#SBATCH -n 48

config_path="./config_falcon_Ez.yaml"

python /home/gridsan/saustin/research/3D-CG/app/plane_laplace_driver.py $SLURM_JOB_ID $SLURM_JOB_NAME $SLURM_CPUS_ON_NODE $SLURM_JOB_NUM_NODES $SLURM_JOB_NODELIST $SLURM_NTASKS $SLURM_SUBMIT_DIR $config_path
