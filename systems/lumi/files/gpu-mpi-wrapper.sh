#!/usr/bin/env bash

export GPU_MAP=(0 1 2 3 4 5 6 7)
export GPU=${GPU_MAP[$SLURM_LOCALID]}
export HIP_VISIBLE_DEVICES=$GPU
unset ROCR_VISIBLE_DEVICES
echo "RANK $SLURM_LOCALID using GPU $GPU"
exec "$@"
