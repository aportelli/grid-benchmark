#!/usr/bin/env bash

__script_dir__="$(dirname "$(readlink -f "${BASH_SOURCE:-$0}")")"
if [ ! -d "${__script_dir__}/.pixi" ]; then
  echo "error: directory '${__script_dir__}/.pixi' does not exist." 1>&2
  exit 1
fi
GRIDENVDIR="${__script_dir__}"
export GRIDENVDIR
export PATH="${GRIDENVDIR}/prefix/pixi/bin:${PATH}"
export PIXI_HOME="${GRIDENVDIR}/prefix/pixi"
