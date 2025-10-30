#!/usr/bin/env bash

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MPICH_GPU_SUPPORT_ENABLED=1
export FI_MR_CACHE_MONITOR=disabled
