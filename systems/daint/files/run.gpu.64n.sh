#!/usr/bin/env bash
# shellcheck disable=SC1091,SC2050,SC2170

#SBATCH -J benchmark-grid-64
#SBATCH -t 1:00:00
#SBATCH --nodes=64
#SBATCH --ntasks=256
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=single:1
#SBATCH --partition=normal
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --no-requeue
#SBATCH --exclusive
#SBATCH --uenv=prgenv-gnu/25.6:v1

set -euo pipefail

# load environment #############################################################
env_cfg="${HOME}/.config/grid-benchmark/grid-env"
if [ ! -f "${env_cfg}" ]; then
	echo "error: ${env_cfg} does not exists"
	exit 1
fi
env_dir="$(readlink -f "$(cat "${env_cfg}")")"
source "${env_dir}/env.sh" gpu  # load environment
source "${env_dir}/mpich-gpu.sh" # set GPU-specific OpenMPI variables

# application and parameters ###################################################
app="${env_dir}/prefix/gridbench_gpu/bin/Benchmark_Grid"

# collect job information ######################################################
job_info_dir=job/${SLURM_JOB_NAME}.${SLURM_JOB_ID}
mkdir -p "${job_info_dir}"

date                         > "${job_info_dir}/start-date"
echo "epoch $(date '+%s')" >> "${job_info_dir}/start-date"
set                          > "${job_info_dir}/env"
ldd "${app}"                 > "${job_info_dir}/ldd"
md5sum "${app}"              > "${job_info_dir}/app-hash"
readelf -a "${app}"          > "${job_info_dir}/elf"
echo "${SLURM_JOB_NODELIST}" > "${job_info_dir}/nodes"
cp "${BASH_SOURCE[0]}"       "${job_info_dir}/script"

# start GPU telemetry ##########################################################
tmp=$(mktemp)
coproc nvidia-smi dmon -o DT &> "${tmp}"

# run! #########################################################################
srun --uenv=prgenv-gnu/25.6:v2 --view=default \
	"${env_dir}/gpu-mpi-wrapper.sh" \
  "${app}" \
	--json-out "${job_info_dir}/result.json" \
	--mpi 4.4.4.4 \
  --accelerator-threads 8 --comms-overlap --shm 8192 --shm-mpi 1 \
	--threads 72 &> "${job_info_dir}/log"

# process telemetry data #######################################################
kill -INT "${COPROC_PID}"
"${env_dir}/dmon-to-db.sh" "${tmp}" "${job_info_dir}/telemetry.db" 'nvidia_smi'

# if we reach that point the application exited successfully ###################
touch "${job_info_dir}/success"
date > "${job_info_dir}/end-date"
echo "epoch $(date '+%s')" >> "${job_info_dir}/end-date"

################################################################################
