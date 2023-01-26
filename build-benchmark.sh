#!/usr/bin/env bash
# shellcheck disable=SC1090,SC1091

set -euo pipefail

if (( $# != 2 )); then
    echo "usage: $(basename "$0") <environment directory> <config>" 1>&2
    exit 1
fi
env_dir=$1
cfg=$2

call_dir=$(pwd -P)
script_dir="$(dirname "$(readlink -f "${BASH_SOURCE:-$0}")")"
cd "${env_dir}"
env_dir=$(pwd -P)
cd "${call_dir}"
build_dir="${env_dir}/build/Grid-benchmarks/${cfg}"
mkdir -p "${build_dir}"
source "${env_dir}/env.sh"
entry=$(jq ".configs[]|select(.name==\"${cfg}\")" "${env_dir}"/grid-config.json)
env_script=$(echo "${entry}" | jq -r ".\"env-script\"")
cd "${build_dir}" || return
source "${env_dir}/${env_script}"
if [ ! -f Makefile ]; then
    "${script_dir}/configure" --with-grid="${env_dir}/prefix/grid_${cfg}" \
                            --prefix="${env_dir}/prefix/gridbench_${cfg}"
fi
make -j 128
make install
cd "${call_dir}"
