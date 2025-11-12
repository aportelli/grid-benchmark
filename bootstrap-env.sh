#!/usr/bin/env bash

set -euo pipefail

config_dir="${HOME}/.config/grid-benchmark"

if (( $# != 2 )); then
    echo "usage: $(basename "$0") <environment directory> <system>" 1>&2
    exit 1
fi
dir=$1
sys=$2

call_dir=$(pwd -P)
script_dir="$(dirname "$(readlink -f "${BASH_SOURCE:-$0}")")"
if [ -d "${dir}" ]; then
    echo "error: directory '${dir}' exists"
    exit 1
fi
if [ ! -d "${script_dir}/systems/${sys}" ]; then
    echo "error: system directory '${sys}' does not exist"
    exit 1
fi
if [ -f "${script_dir}/systems/${sys}/files/shell-wrapper.sh" ]; then
    if [ ! "${_grid_wrapped_+x}" ]; then
        echo "error: system '${sys}' requires to use a shell wrapper, please run" 1>&2
        echo '' 1>&2
        echo "systems/${sys}/files/shell-wrapper.sh $0 $*" 1>&2
        exit 1
    fi
fi

mkdir -p "${dir}"
cd "${dir}"; dir=$(pwd -P); cd "${call_dir}"

echo "-- install Pixi"
export PIXI_HOME="${dir}/prefix/pixi"
export PIXI_NO_PATH_UPDATE=1
curl -fsSL https://pixi.sh/install.sh | sh
export PATH="${dir}/prefix/pixi/bin:${PATH}"
pixi_log='pixi -v --no-progress'

echo "-- install default environment"
cp "${script_dir}/pixi.toml" "${dir}"
cd "${dir}"
$pixi_log install
export PATH="${dir}/.pixi/envs/default/bin:${PATH}"
cp "${script_dir}/env-base.sh" "${script_dir}/env-pixi.sh" "${script_dir}/env.sh" "${dir}"

echo "-- install system specific files (system: ${sys})"
cp "${script_dir}/systems/${sys}/files"/* "${dir}"
cp "${script_dir}/systems/${sys}/grid-config.json" "${dir}"

echo "-- install system specific environment(s) (system: ${sys})"
for e in $(jq -r '.configs[]."pixi-env"' "${dir}/grid-config.json"); do
  echo "* install environment '${e}'"
  $pixi_log install -e "${e}"
done

echo "-- install C-LIME"
mkdir -p "${dir}/build"
cd "${dir}/build"
lime_url='http://usqcd-software.github.io/downloads/c-lime/lime-1.3.2.tar.gz'
wget ${lime_url}
lime_archive=$(basename ${lime_url})
lime_dir=$(tar -tzf "${lime_archive}" | cut -d/ -f1 | uniq )
tar -xzf "${lime_archive}"
cd "${lime_dir}"
mkdir build && cd build
../configure --prefix="${dir}/prefix/lime" CFLAGS='-fPIE'
make -j && make install

echo "-- save environment directory in '${config_dir}/grid-env'"
mkdir -p "${config_dir}"
echo "${dir}" > "${config_dir}/grid-env"

echo "-- done!"
cd "${call_dir}"
