#!/usr/bin/env bash
# shellcheck disable=SC1091,SC2050,SC2170

#SBATCH -J benchmark-grid-16
#SBATCH -t 1:00:00
#SBATCH --nodes=32
#SBATCH --ntasks=128
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --qos=standard
#SBATCH --no-requeue
#SBATCH --gpu-freq=1410

set -euo pipefail

# load environment #############################################################
env_cfg="${HOME}/.config/lattice-benchmarks/grid-env"
if [ ! -f "${env_cfg}" ]; then
	echo "error: ${env_cfg} does not exists, did you execute 'source env.sh' with your user account?"
	exit 1
fi
env_dir="$(readlink -f "$(cat "${env_cfg}")")"
source "${env_dir}/env.sh"      # load base Spack environment
source "${env_dir}/env-gpu.sh"  # load GPU-sepcific packages
source "${env_dir}/ompi-gpu.sh" # set GPU-specific OpenMPI variables

# application and parameters ###################################################
app="${env_dir}/prefix/gridbench_gpu/bin/Benchmark_Grid"

# collect job information ######################################################
job_info_dir=job/${SLURM_JOB_NAME}.${SLURM_JOB_ID}
mkdir -p "${job_info_dir}"

date                         > "${job_info_dir}/start-date"
set                          > "${job_info_dir}/env"
ldd "${app}"                 > "${job_info_dir}/ldd"
md5sum "${app}"              > "${job_info_dir}/app-hash"
readelf -a "${app}"          > "${job_info_dir}/elf"
echo "${SLURM_JOB_NODELIST}" > "${job_info_dir}/nodes"
cp "${BASH_SOURCE[0]}"       "${job_info_dir}/script"

# run! #########################################################################
mpirun -np "${SLURM_NTASKS}" -x LD_LIBRARY_PATH --bind-to none \
	"${env_dir}/gpu-mpi-wrapper.sh" \
  "${app}" \
	--json-out "${job_info_dir}/result.json" \
	--mpi 2.4.4.4 \
  --accelerator-threads 8 \
	--threads 8 \
	--shm 2048 &> "${job_info_dir}/log"

# if we reach that point the application exited successfully ###################
touch "${job_info_dir}/success"
date > "${job_info_dir}/end-date"

################################################################################
