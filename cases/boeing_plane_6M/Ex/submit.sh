#!/bin/bash

#SBATCH -N 1
#SBATCH -n 48

config_path="./config_boeing_6M_Ex.yaml"

python main.py $SLURM_JOB_ID $SLURM_JOB_NAME $SLURM_CPUS_ON_NODE $SLURM_JOB_NUM_NODES $SLURM_JOB_NODELIST $SLURM_NTASKS $SLURM_SUBMIT_DIR $config_path

#SBATCH --exclusive
