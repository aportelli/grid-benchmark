#!/usr/bin/env bash

set -euo pipefail

json_url='https://github.com/nlohmann/json/releases/download/v3.12.0/json.hpp'
json_sha256='aaf127c04cb31c406e5b04a63f1ae89369fccde6d8fa7cdda1ed4f32dfc5de63'
json_file="$(basename ${json_url})"

echo '-- downloading nlohmann/json'
if [ ! -f "${json_file}" ]; then
  wget ${json_url}
else
  echo 'already downloaded'
fi
echo "${json_sha256} ${json_file}" | sha256sum --check || exit 1
echo '-- generating configure script'
mkdir -p .buildutils/m4
autoreconf -fvi
