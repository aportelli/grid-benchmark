#!/usr/bin/env bash
# shellcheck disable=SC1090,SC1091

set -euo pipefail

if (( $# != 3 )); then
    echo "usage: $(basename "$0") <environment directory> <config> <njobs>" 1>&2
    exit 1
fi
env_dir=$1
cfg=$2
njobs=$3

if [ ! -d "${env_dir}" ]; then
  echo "error: environment directory '${env_dir}' does not exist." 1>&2
  exit 1
fi
if [ -f "${env_dir}/shell-wrapper.sh" ]; then
    if [ ! "${_grid_wrapped_+x}" ]; then
        echo "error: this envrionment requires to use a shell wrapper, please run" 1>&2
        echo '' 1>&2
        echo "${env_dir}/shell-wrapper.sh $0 $*" 1>&2
        exit 1
    fi
fi

call_dir=$(pwd -P)
script_dir="$(dirname "$(readlink -f "${BASH_SOURCE:-$0}")")"
cd "${env_dir}"
env_dir=$(pwd -P)
cd "${call_dir}"
build_dir="${env_dir}/build/Grid-benchmarks/${cfg}"
mkdir -p "${build_dir}"
source "${env_dir}/env.sh" "${cfg}"
entry=$(jq -e ".configs[]|select(.name==\"${cfg}\")" "${env_dir}"/grid-config.json)
cd "${script_dir}"
if [ ! -f configure ]; then
    ./bootstrap.sh
fi
cd "${build_dir}"
CXX="$("${env_dir}/prefix/grid_${cfg}/bin/grid-config" --cxx)"
export CXX
if [ ! -f Makefile ]; then
    "${script_dir}/configure" --with-grid="${env_dir}/prefix/grid_${cfg}" \
                            --prefix="${env_dir}/prefix/gridbench_${cfg}"
fi
make -j "${njobs}"
make install
cd "${call_dir}"
