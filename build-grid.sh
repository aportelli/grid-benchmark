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
cd "${env_dir}"
env_dir=$(pwd -P)
cd "${call_dir}"
build_dir="${env_dir}/build/Grid/${cfg}"
if [ -d "${build_dir}" ]; then
    echo "error: directory '${build_dir}' exists"
    exit 1
fi
mkdir -p "${build_dir}"
source "${env_dir}/env.sh"
entry=$(jq ".configs[]|select(.name==\"${cfg}\")" "${env_dir}"/grid-config.json)
IFS=" " read -r -a args <<< "$(echo "${entry}" | jq -r ".\"config-options\"")"
env_script=$(echo "${entry}" | jq -r ".\"env-script\"")
cd "${build_dir}" || return
source "${env_dir}/${env_script}"
extra_env=$(mktemp)
echo "${entry}" | jq -r '.env|to_entries|map("export \(.key)='\''\(.value|tostring)'\''")|.[]' > "${extra_env}"
commit=$(echo "${entry}" | jq -r ".commit")
git clone https://github.com/paboyle/Grid.git "${build_dir}"
cd "${build_dir}"
git checkout "${commit}"
./bootstrap.sh
mkdir build; cd build
source "${extra_env}"
../configure --prefix="${env_dir}/prefix/grid_${cfg}" "${args[@]}"
make -j128
make install
rm -rf "${extra_env}"
cd "${call_dir}"
