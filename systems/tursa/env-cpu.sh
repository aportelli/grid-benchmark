#!/usr/bin/env bash
# shellcheck disable=SC2046

script_dir="$(dirname "$(readlink -f "${BASH_SOURCE:-$0}")")"
spack load $(cat "${script_dir}"/grid-cpu.spack)
