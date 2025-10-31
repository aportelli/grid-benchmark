#!/usr/bin/env bash

if (( $# != 1 )); then
    echo "usage: $(basename "$0") <pixi env>" 1>&2
    exit 1
fi
env=$1

if [ ! -d "${GRIDENVDIR}" ]; then
    echo "error: directory '${GRIDENVDIR}' does not exist (if empty, was env-base.sh sourced?)." 1>&2
    exit 1
fi

export CONDA_PREFIX="${GRIDENVDIR}/.pixi/envs/${env}"
export CONDA_DEFAULT_ENV="grid-benchmark:${env}"
export GRID_CURRENT_ENV="${env}"
export PATH="${PATH:-}:${GRIDENVDIR}/prefix/lime/bin:${CONDA_PREFIX}/bin"
export LIBRARY_PATH="${LIBRARY_PATH:-}:${GRIDENVDIR}/prefix/lime/lib:${CONDA_PREFIX}/lib"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${GRIDENVDIR}/prefix/lime/lib:${CONDA_PREFIX}/lib"
export DYLD_FALLBACK_LIBRARY_PATH="${LD_LIBRARY_PATH}"
export INCLUDE="${INCLUDE:-}:${GRIDENVDIR}/prefix/lime/include:${CONDA_PREFIX}/include"
export C_INCLUDE_PATH="${C_INCLUDE_PATH:-}:${GRIDENVDIR}/prefix/lime/include:${CONDA_PREFIX}/include"
export CPLUS_INCLUDE_PATH="${CPLUS_INCLUDE_PATH:-}:${GRIDENVDIR}/prefix/lime/include:${CONDA_PREFIX}/include"
