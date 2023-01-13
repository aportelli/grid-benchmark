#!/usr/bin/env bash

set -euo pipefail

mkdir -p .buildutils/m4
autoreconf -fvi
