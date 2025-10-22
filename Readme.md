# Grid benchmarks

This folder contains benchmarks for the [Grid](https://github.com/paboyle/Grid) library.
The benchmarks can be summarised as follows

- `Benchmark_Grid`: This benchmark measure floating point performances for various fermion
matrices, as well as bandwidth measurement for different operations. Measurements are
performed for a fixed range of problem sizes.
- `Benchmark_IO`: Parallel I/O benchmark.

## TL;DR
Build and install Grid, all dependencies, and the benchmark with
```bash
./bootstrap-env.sh <env_dir> <system>       # create benchmark environment
./build-grid.sh <env_dir> <config>          # build Grid
./build-benchmark.sh <env_dir> <config>     # build benchmarks
```
where `<env_dir>` is an arbitrary directory where every product will be stored, `<system>`
is a sub-directory of `systems` containing system-specific configuration files 
(an existing preset or your own), and finally `<config>` is the name of a build config
in `systems/<system>/grid-config.json`. After a successful execution the benchmark binaries
will be in `<env_dir>/prefix/gridbench_<config>`.



## Developing configurations for additional systems

### System-specific directory
You can create a configuration for a new system by creating a new subdirectory in the
`systems` directory that should at least contains:
- a `grid-config.json` file according to the specification described below;
- a `files` directory containing any files that need to be copied to the environment directory

### Configuration file
The system directory must contain a `grid-config.json` file specifying compilation flag
configurations for Grid. This file must contain a single `"configs"` JSON array, with all
elements of the form
```json
{
  "name": "foo",          // name of the configuration
  "env-script": "bar.sh", // additional script to source before building 
                          // (e.g. to load modules, path absolute or relative to the 
                          // environment directory, ignored if empty)
  "commit": "...",        // Grid commit to use 
                          // (anything that can be an argument of git checkout)
  "config-options": "..." // options to pass to the configure script,
  "env" : {               // environment variables
    "VAR": "value"        // export VAR="value" before building
  }
  "pixi-env": "..."       // Pixi environment to use for this configuration
}
```
Grid's dependencies are managed with [Pixi](https://pixi.sh/latest/) environments defined
in the [`pixi.toml`](pixi.toml) file. The following environments are available:
- `gpu-nvidia`: NVIDIA GPU Linux build
- `gpu-amd`: AMD GPU Linux build (TODO)
- `cpu-linux`: CPU Linux build with LLVM
- `cpu-apple-silicon`: Apple Silicon build on macOS
and one of these strings must be used as a value for `"pixi-env"` above.

Please refer to [Grid's repository](https://github.com/paboyle/Grid) 
for documentation

### Environment setup
One a complete system folder has been created as above, the associated environment can be
deployed with
```bash
./bootstrap-env.sh ./env <system>
```
where `<system>` is the name of the system directory in `systems`. Here, `./env` was
picked as an example of deployment location, but any writable path can be used. This
script will install Pixi and deploy the relevant environments in `./env`, as well as copy
all the files present in the system `files` directory. After successful completion,
`./env` will contain a `env.sh` file that can be sourced to activate a given environment 
```bash
source ./env/env.sh <config>
```
wherer `<config>` must be a value of a `"name"` field from the `grid-config.json` file.
This script will:
  1. make the embedded Pixi path available
  2. activate the Pixi environment specified in `"pixi-env"`
  3. source the (optional) additional script specified in `"env-script"`
The procedure above is executed in the scripts `build-grid.sh` and `build-benchmark.sh`,
which can be used to build Grid and the benchmark as described above.

## Running the benchmarks
After building the benchmarks as above you can find the binaries in 
`<env_dir>/prefix/gridbench_<config>`. Depending on the system selected, the environment
directory might also contain batch script examples. Each HPC system tends to have its own specific runtime characteristics, and it is not possible to automatise
determining the best runtime environment to run the Grid benchmark. Example of known supercomputing
environments can be found
- in the [`systems`](systems) directory of this repository
- in the [`systems`](https://github.com/paboyle/Grid/tree/develop/systems) directory of the Grid repository

More information about the benchmark results is provided below.

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