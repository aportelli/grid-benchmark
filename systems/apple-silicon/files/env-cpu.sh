#!/usr/bin/env bash
# shellcheck disable=SC1091

__script_dir__="$(dirname "$(readlink -f "${BASH_SOURCE:-$0}")")"

source "${__script_dir__}/env-pixi.sh" cpu-apple-silicon
export DYLD_FALLBACK_LIBRARY_PATH="${LD_LIBRARY_PATH}"
