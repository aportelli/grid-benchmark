#!/usr/bin/env bash

set -euo pipefail

if (( $# != 1 )); then
    echo "usage: $(basename "$0") <environment directory>" 1>&2
    exit 1
fi
dir=$1

call_dir=$(pwd -P)
script_dir="$(dirname "$(readlink -f "${BASH_SOURCE:-$0}")")"
if [ -d "${dir}" ]; then
    echo "error: directory '${dir}' exists"
    exit 1
fi
mkdir -p "${dir}"
cd "${dir}"
git clone https://github.com/spack/spack.git
cd "${call_dir}"
cp "${script_dir}"/files/* "${dir}"
cp "${script_dir}/env.sh" "${script_dir}/grid-config.json" "${dir}"
source "${dir}"/spack/share/spack/setup-env.sh
"${script_dir}"/spack-bootstrap.sh "${dir}"
