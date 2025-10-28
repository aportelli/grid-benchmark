#!/usr/bin/env bash

module load CrayEnv LUMI/24.03 partition/G rocm/6.0.3
export LD_LIBRARY_PATH="/opt/rocm-6.0.3/lib/llvm/lib:${LD_LIBRARY_PATH:-}"
