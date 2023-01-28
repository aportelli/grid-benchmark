#!/usr/bin/env bash
# shellcheck disable=SC1091,SC2050,SC2170

# using options from https://github.com/paboyle/Grid/tree/develop/systems/Tursa

#SBATCH -J @job-name@
#SBATCH -A @budget@
#SBATCH -t 48:00:00
#SBATCH --nodes=@nnodes@
#SBATCH --ntasks=@ntasks@
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --partition=@partition@
#SBATCH --gres=gpu:4
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --qos=standard
#SBATCH --no-requeue

set -e

# OpenMP/OpenMPI/UCX environment ###############################################
export OMP_NUM_THREADS=8
export OMPI_MCA_btl=^uct,openib
export OMPI_MCA_pml=ucx
export UCX_TLS=gdr_copy,rc,rc_x,sm,cuda_copy,cuda_ipc
export UCX_RNDV_SCHEME=put_zcopy
export UCX_RNDV_THRESH=16384
export UCX_IB_GPU_DIRECT_RDMA=yes
export UCX_MEMTYPE_CACHE=n

# IO environment ###############################################################

if [ @nnodes@ -eq 1 ]; then
	export OMPI_MCA_io=ompio
else
	export OMPI_MCA_io=romio321
fi
export OMPI_MCA_btl_openib_allow_ib=true
export OMPI_MCA_btl_openib_device_type=infiniband
export OMPI_MCA_btl_openib_if_exclude=mlx5_1,mlx5_2,mlx5_3

# load environment #############################################################
env_dir="$(readlink -f @env-dir@)"
source "${env_dir}/env-base.sh"
if [ "${SLURM_JOB_PARTITION}" = 'gpu' ]; then
	source "${env_dir}/env-gpu.sh"
else
	echo "error: partition ${SLURM_JOB_PARTITION} not supported for this template" 1>&2
  exit 1
fi

# application and parameters ###################################################
app='@application@'
opt=('--comms-overlap' '--comms-concurrent')
par='@par@'

# collect job information ######################################################
job_info_dir=job/${SLURM_JOB_NAME}.${SLURM_JOB_ID}
mkdir -p "${job_info_dir}"

date                         > "${job_info_dir}/start-date"
set                          > "${job_info_dir}/env"
ldd ${app}                   > "${job_info_dir}/ldd"
md5sum ${app}                > "${job_info_dir}/app-hash"
readelf -a ${app}            > "${job_info_dir}/elf"
echo "${SLURM_JOB_NODELIST}" > "${job_info_dir}/nodes"
cp "${BASH_SOURCE[0]}"       "${job_info_dir}/script"
if [ -n "${par}" ]; then cp "${par}" "${job_info_dir}/par"; fi

# run! #########################################################################
mpirun -np "${SLURM_NTASKS}" -x LD_LIBRARY_PATH --bind-to none \
	./gpu-mpi-wrapper.sh \
  ${app} "${par}" "${opt[@]}" \
	--mpi @mpi-geom@ \
  --accelerator-threads 8 \
	--grid @grid-geom@ \
	--shm 2048 &> "${job_info_dir}/log"

# if we reach that point the application exited successfully ###################
touch "${job_info_dir}/success"
date > "${job_info_dir}/end-date"

################################################################################
