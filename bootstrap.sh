#!/usr/bin/env bash

set -euo pipefail

json_url='https://github.com/nlohmann/json/blob/bc889afb4c5bf1c0d8ee29ef35eaaf4c8bef8a5d/single_include/nlohmann/json.hpp'

if [ ! -f json.hpp ]; then
  wget ${json_url}
fi
mkdir -p .buildutils/m4
autoreconf -fvi
