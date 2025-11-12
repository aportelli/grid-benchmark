#!/usr/bin/env bash
# shellcheck disable=SC1091,SC2050,SC2170

set -euo pipefail

# load environment #############################################################
env_cfg="${HOME}/.config/grid-benchmark/grid-env"
if [ ! -f "${env_cfg}" ]; then
	echo "error: ${env_cfg} does not exists"
	exit 1
fi
env_dir="$(readlink -f "$(cat "${env_cfg}")")"
source "${env_dir}/env.sh" cpu  # load environment

# application and parameters ###################################################
app="${env_dir}/prefix/gridbench_cpu/bin/Benchmark_Grid"

# create output directory ######################################################
job_info_dir=job/cpu
mkdir -p "${job_info_dir}"

# run! #########################################################################
"${app}" \
  --json-out "${job_info_dir}/result.json" \
  --max-L 32 \
  --no-benchmark-deo-fp64 \
  --mpi 1.1.1.1 \
  --threads 8 \
  --shm 1
