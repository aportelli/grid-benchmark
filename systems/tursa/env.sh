#!/usr/bin/env bash
# shellcheck disable=SC1091

env_dir="$(dirname "$(readlink -f "${BASH_SOURCE:-$0}")")"
source "${env_dir}/spack/share/spack/setup-env.sh"
spack load jq git
