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
        echo "error: this environment requires a shell wrapper, please run" 1>&2
        echo '' 1>&2
        echo "${env_dir}/shell-wrapper.sh $0 $*" 1>&2
        exit 1
    fi
fi

call_dir=$(pwd -P)
cd "${env_dir}"
env_dir=$(pwd -P)
cd "${call_dir}"
build_dir="${env_dir}/build/Grid/${cfg}"
if [ -d "${build_dir}" ]; then
    echo "error: directory '${build_dir}' exists"
    exit 1
fi

entry=$(jq ".configs[]|select(.name==\"${cfg}\")" "${env_dir}"/grid-config.json)
if [ -z "$entry" ]; then
  echo "error: config \"${cfg}\" does not exist for system." 1>&2
  configs=$(jq -r ".configs[]|.name" "${env_dir}"/grid-config.json)
  echo "available configs:" 1>&2
  for cfgname in ${configs[@]}; do
    echo "  ${cfgname}" 1>&2
  done
  exit 1
fi

IFS=" " read -r -a args <<< "$(echo "${entry}" | jq -r ".\"config-options\"")"
source "${env_dir}/env.sh" "${cfg}"
mkdir -p "${build_dir}"
cd "${build_dir}" || return
extra_env=$(mktemp)
echo "${entry}" | jq -er '.env|to_entries|map("export \(.key)=\"\(.value|tostring)\"")|.[]' > "${extra_env}"
commit=$(echo "${entry}" | jq -er ".commit")
git clone https://github.com/paboyle/Grid.git "${build_dir}"
cd "${build_dir}"
git checkout "${commit}"
./bootstrap.sh
mkdir build; cd build
source "${extra_env}"
../configure --prefix="${env_dir}/prefix/grid_${cfg}" --enable-gparity=no \
    --enable-fermion-reps=no --enable-zmobius=no "${args[@]}"
make grid-config
make -j"${njobs}" -C Grid
make install -C Grid
mkdir -p "${env_dir}/prefix/grid_${cfg}/bin"
cp ./grid-config "${env_dir}/prefix/grid_${cfg}/bin"
rm -rf "${extra_env}"
cd "${call_dir}"
