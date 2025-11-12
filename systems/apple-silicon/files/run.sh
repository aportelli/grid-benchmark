#!/usr/bin/env bash

env_dir=$(cat "${HOME}/.config/grid-benchmark/grid-env")
source "${env_dir}/env.sh" cpu
mpirun -np 2 "${env_dir}/prefix/gridbench_cpu/bin/Benchmark_Grid" --max-L 32 --mpi 1.1.1.2 --threads 5
