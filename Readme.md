# Grid benchmarks

This folder contains benchmarks for the [Grid](https://github.com/aportelli/) library.
The benchmarks can be summarised as follows

- `Benchmark_Grid`: This benchmark measure floating point performances for various fermion
matrices, as well as bandwidth measurement for different operations. Measurements are
performed for a fixed range of problem sizes.
- `Benchmark_IO`: Parallel I/O benchmark.

## TL;DR
Build and install Grid, all dependencies, and the benchmark with
```bash
systems/<system>/bootstrap-env.sh <env_dir> # build dependencies, takes a long time
./build-grid.sh <env_dir> <config>          # build Grid
./build-benchmark.sh <env_dir> <config>     # build benchmarks
```
where `<env_dir>` is an arbitrary directory where every product will be stored, `<system>`
is a sub-directory of `systems` containing system-specific scripts 
(an existing preset or your own), and finally `<config>` is the name of a build config
in `systems/<system>/grid-config.json`. After a successful execution the benchmark binaries
will be in `<env_dir>/prefix/gridbench_<config>`.

## Environment setup
A complete runtime environnement can be deploy using scripts from this repository. System-specific scripts are in the `systems` directory.

You should first deploy the environment for the specific system you are using, for example
```bash
systems/tursa/bootstrap-env.sh ./env
```
will deploy the relevant environment for the [Tursa](https://www.epcc.ed.ac.uk/hpc-services/dirac-tursa-gpu) supercomputer in `./env`. This step might compile from source a large set
of packages, and take some time to complete.

After that, the environment directory (`./env` in the example above) will contain a `env.sh` file that need to be sourced to activate the environment
```bash
source ./env/env.sh
```
Additional scripts `env-*.sh` can be sourced after to activate more specific environments,
this should be done after sourcing `env.sh` as above.

## Building the benchmarks
The environnement directory contains a `grid-config.json` file specifying compilation flag
configurations for Grid (please see Grid's repository for documentation). All entries have 
the form
```json
{
  "name": "foo",          // name of the configuration
  "env-script": "bar.sh", // script to source before building 
                          // (path relative to the environment directory)
  "commit": "...",        // Grid commit to use 
                          // (anything that can be an argument of git checkout)
  "config-options": "..." // options to pass to the configure script,
  "env" : {               // environment variables
    "VAR": "value"        // export VAR="value" before building
  }
}
```
Grid can then be built with
```
./build-grid.sh <env_dir> <config>
```
where `<env_dir>` is the environment directory and `<config>` is the build config name in 
`grid-config.json`. Similarly, the benchmarks can then be built with
```
./build-grid <env_dir> <config>
```

## Running the benchmarks
After building the benchmarks as above you can find the binaries in 
`<env_dir>/prefix/gridbench_<config>`. Depending on the system selected, the environment
directory might also contain batch script examples. More information about the benchmarks
is provided below.

### `Benchmark_Grid`
This benchmark performs flop/s measurement for typical lattice QCD sparse matrices, as
well as memory and inter-process bandwidth measurement using Grid routines. The benchmark
command accept any Grid flag (see complete list with `--help`), as well as a 
`--json-out <file>` flag to save the measurement results in JSON to `<file>`. The 
benchmarks are performed on a fix set of problem sizes, and the Grid flag `--grid` will
be ignored.

The resulting metrics are as follows, all data size units are in base 2 
(i.e. 1 kB = 1024 B).

*Memory bandwidth*

One sub-benchmark measure the memory bandwidth using a lattice version of the `axpy` BLAS
routine, in a similar fashion to the STREAM benchmark. The JSON entries under `"axpy"` 
have the form
```json
{
  "GBps": 215.80653375861607,   // bandwidth in GB/s/node
  "GFlops": 19.310041765757834, // FP performance (double precision)
  "L": 8,                       // local lattice volume
  "size_MB": 3.0                // memory size in MB/node
}
```

A second benchmark performs site-wise SU(4) matrix multiplication, and has a higher
arithmetic intensity than the `axpy` one (although it is still memory-bound). 
The JSON entries under `"SU4"` have the form
```json
{
  "GBps": 394.76639187026865,  // bandwidth in GB/s/node
  "GFlops": 529.8464820758512, // FP performance (single precision)
  "L": 8,                      // local lattice size
  "size_MB": 6.0               // memory size in MB/node
}
```

*Inter-process bandwidth*

This sub-benchmark measures the achieved bidirectional bandwidth in threaded halo exchange
using routines in Grid. The exchange is performed in each direction on the MPI Cartesian
grid which is parallelised across at least 2 processes. The resulting bandwidth is related
to node-local transfers (inter-CPU, NVLink, ...) or network transfers depending on the MPI
decomposition. he JSON entries under `"comms"` have the form
```json
{
  "L": 40,                       // local lattice size
  "bytes": 73728000,             // payload size in B/rank
  "dir": 2,                      // direction of the exchange, 8 possible directions
                                 // (0: +x, 1: +y, ..., 5: -x, 6: -y, ...)
  "rate_GBps": {
    "error": 6.474271894240327,  // standard deviation across measurements (GB/s/node)
    "max": 183.10546875,         // maximum measured bandwidth (GB/s/node)
    "mean": 175.21747026766676   // average measured bandwidth (GB/s/node)
  },
  "time_usec": 3135.055          // average transfer time (microseconds)
}
```

*Floating-point performances*

This sub-benchmark measures the achieved floating-point performances using the 
Wilson fermion, domain-wall fermion, and staggered fermion sparse matrices from Grid.
In the `"flops"` and `"results"` section of the JSON output are recorded the best 
performances, e.g.
```json
{
  "Gflops_dwf4": 366.5251173474483,       // domain-wall in Gflop/s/node (single precision)
  "Gflops_staggered": 7.5982861018529455, // staggered in Gflop/s/node (single precision)
  "Gflops_wilson": 15.221839719288932,    // Wilson in Gflop/s/node (single precision)
  "L": 8                                  // local lattice size
}
```
Here "best" means across a number of different implementations of the routines. Please
see the log of the benchmark for an additional breakdown. Finally, the JSON output
contains a "comparison point", which is the average of the L=24 and L=32 best
domain-wall performances.