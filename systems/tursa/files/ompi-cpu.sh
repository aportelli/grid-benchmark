#!/usr/bin/env bash

# OpenMP/OpenMPI/UCX environment ###############################################
export OMP_NUM_THREADS=64
export OMP_DISPLAY_AFFINITY=true
export OMPI_MCA_btl=^uct,openib
export OMPI_MCA_pml=ucx
export UCX_TLS=rc,sm,self
export UCX_RNDV_THRESH=16384
export UCX_MEMTYPE_CACHE=n
export UCX_NET_DEVICES=mlx5_0:1

# IO environment ###############################################################
export OMPI_MCA_BTL_SM_USE_KNEM=1
export OMPI_MCA_coll_hcoll_enable=1
export OMPI_MCA_coll_hcoll_np=0
