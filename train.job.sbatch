#!/bin/bash

#SBATCH --gres=gpu:gtx:1
#SBATCH --partition=gpu
#SBATCH --time=96:00:00
#SBATCH --output=slurm/output-%j.out

#SBATCH --mail-user=
#SBATCH --mail-type=BEGIN,END

mkdir -p slurm

srun train.job.sh "$@"
