#!/usr/bin/env bash

module load CrayEnv LUMI/25.09 partition/G rocm/6.4.4
export LD_LIBRARY_PATH="/opt/rocm-6.4.4/lib/llvm/lib:${LD_LIBRARY_PATH:-}"
