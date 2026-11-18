#!/bin/bash
export LOCAL_RANK=$SLURM_LOCALID
export GLOBAL_RANK=$SLURM_PROCID
export GPUS=(0 1 2 3)
NUMA_NODE=$(echo "$LOCAL_RANK % 4" | bc)
export NUMA_NODE
export CUDA_VISIBLE_DEVICES=${GPUS[$NUMA_NODE]}
echo "local rank ${LOCAL_RANK} global rank ${GLOBAL_RANK} NUMA ${NUMA_NODE} GPU ${CUDA_VISIBLE_DEVICES}"
numactl --cpunodebind="$NUMA_NODE" --membind="$NUMA_NODE" "$@"