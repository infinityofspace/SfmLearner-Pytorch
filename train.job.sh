#!/bin/bash

module load cuda/10.0
CUDA_DEVICE=$(echo "$CUDA_VISIBLE_DEVICES," | cut -d',' -f $((SLURM_LOCALID + 1)) );
T_REGEX='^[0-9]$';
if ! [[ "$CUDA_DEVICE" =~ $T_REGEX ]]; then
        echo "error no reserved gpu provided"
        exit 1;
fi
echo "Process $SLURM_PROCID of Job $SLURM_JOBID with the local id $SLURM_LOCALID using gpu id $CUDA_DEVICE (we may use gpu: $CUDA_VISIBLE_DEVICES on $(hostname))"
echo "computing on $(nvidia-smi --query-gpu=gpu_name --format=csv -i $CUDA_DEVICE | tail -n 1)"

venv/bin/python train.py "$@"

echo "done"
