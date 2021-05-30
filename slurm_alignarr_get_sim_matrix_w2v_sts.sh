#!/bin/bash
#SBATCH -p gpu20
#SBATCH --gres gpu:1
#SBATCH -t 10:00:00
#SBATCH -o /scratch/inf0/user/paramita/slurm-alignarr-w2v-sts.log
#SBATCH -a 0-9

python -u alignarr.py $SLURM_ARRAY_TASK_ID sts

