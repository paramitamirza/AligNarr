#!/bin/bash
#SBATCH -p gpu20
#SBATCH --gres gpu:1
#SBATCH -t 5:00:00
#SBATCH -o /scratch/inf0/user/paramita/slurm-alignarr-scriptbase.log
#SBATCH -a 0-913

python -u -W ignore alignarr.py scriptbase /GW/D5data-14/scriptbase_extracted/ $SLURM_ARRAY_TASK_ID sim_matrix

