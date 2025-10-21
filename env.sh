#!/usr/bin/env bash
# shellcheck disable=SC1090

env_dir="$(dirname "$(readlink -f "${BASH_SOURCE:-$0}")")"

if [ ! -d "${env_dir}/.pixi" ]; then
  echo "error: directory '${env_dir}/.pixi' does not exist." 1>&2
  exit 1
fi

if (( $# != 1 )); then
    echo "usage: $(basename "$0") <config>" 1>&2
    exit 1
fi
cfg=$1

source "${env_dir}/env-base.sh"
entry=$(jq -e ".configs[]|select(.name==\"${cfg}\")" "${env_dir}"/grid-config.json)
pixi_env=$(echo "${entry}" | jq -er ".\"pixi-env\"")
if [ -n "${pixi_env}" ]; then
    source "${env_dir}/env-pixi.sh" "${pixi_env}"
else
    echo "warning: 'pixi-env' is empty, please check this is intentional" 1>&2
fi
env_script=$(echo "${entry}" | jq -er ".\"env-script\"")
if [ -n "${env_script}" ]; then
    source "${env_dir}/${env_script}" 
fi
