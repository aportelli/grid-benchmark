#!/usr/bin/env bash
# shellcheck disable=SC1091

GRIDENVDIR="$(dirname "$(readlink -f "${BASH_SOURCE:-$0}")")"
export GRIDENVDIR
export PATH="${GRIDENVDIR}/prefix/base/bin:${PATH}"
export ACLOCAL_PATH="${GRIDENVDIR}/prefix/base/share/aclocal:${ACLOCAL_PATH}"
source "${GRIDENVDIR}"/spack/share/spack/setup-env.sh
