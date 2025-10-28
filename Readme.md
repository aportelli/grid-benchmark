# Grid benchmarks

This folder contains benchmarks for the [Grid](https://github.com/paboyle/Grid) library.
The benchmarks can be summarised as follows.

- `Benchmark_Grid`: This benchmark measures floating-point performance for various fermion
  matrices, as well as bandwidth measurements for different operations. Measurements are
  performed for a fixed range of problem sizes.
- `Benchmark_IO`: Parallel I/O benchmark.

## TL;DR
Build and install Grid, all dependencies, and the benchmark with:
```bash
./bootstrap-env.sh <env_dir> <system>           # create benchmark environment
./build-grid.sh <env_dir> <config> <njobs>      # build Grid
./build-benchmark.sh <env_dir> <config> <njobs> # build benchmarks
```
where `<env_dir>` is an arbitrary directory where every product will be stored, `<system>`
is a sub-directory of `systems` containing system-specific configuration files 
(an existing preset or your own), and finally `<config>` is the name of a build config
in `systems/<system>/grid-config.json`. After a successful execution, the benchmark binaries
will be in `<env_dir>/prefix/gridbench_<config>`. Build tasks are executed using `<njobs>`
parallel processes.

## Developing configurations for additional systems

### System-specific directory
You can create a configuration for a new system by creating a new subdirectory in the
`systems` directory that should contain at least:
- a `grid-config.json` file according to the specification described below;
- a `files` directory containing any files that need to be copied to the environment directory.

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
for documentation on the use of Grid's `configure` script.

### Environment setup
Once a complete system folder has been created as above, the associated environment can be
deployed with:
```bash
./bootstrap-env.sh ./env <system>
```
where `<system>` is the name of the system directory in `systems`. Here, `./env` is an
example of a deployment location, but any writable path can be used. This script will
install Pixi and deploy the relevant environments in `./env`, as well as copy all files
present in the system `files` directory. After successful completion, `./env` will contain
an `env.sh` file that can be sourced to activate a given environment:
```bash
source ./env/env.sh <config>
```
where `<config>` must match a `"name"` field from the `grid-config.json` file.

This script will:
1. make the embedded Pixi path available;
2. activate the Pixi environment specified in `"pixi-env"`; and
3. source the (optional) additional script specified in `"env-script"`.

The procedure above is used by the scripts `build-grid.sh` and `build-benchmark.sh`,
which can be used to build Grid and the benchmark as described above.

## Running the benchmarks
After building the benchmarks as described above, you can find the binaries in 
`<env_dir>/prefix/gridbench_<config>`. Depending on the system selected, the environment
directory might also contain example batch scripts. Each HPC system tends to have its own
runtime characteristics, and it is not possible to automate determining the best runtime
environment for the Grid benchmark. Examples of known supercomputing environments can be found:
- in the [`systems` directory](systems) of this repository; and
- in the [`systems` directory](https://github.com/paboyle/Grid/tree/develop/systems) of the Grid repository.

More information about the benchmark results is provided below.

### `Benchmark_Grid`
This benchmark performs flop/s measurements for typical lattice QCD sparse matrices, as
well as memory and inter-process bandwidth measurements using Grid routines. The benchmark
command accepts any Grid flag (see the complete list with `--help`), as well as a 
`--json-out <file>` flag to save the measurement results in JSON to `<file>`. The 
benchmarks are performed on a fixed set of problem sizes, and the Grid flag `--grid` will
be ignored.

The resulting metrics are as follows. All data-size units are in base 2 
(i.e. 1 kB = 1024 B).

*Memory bandwidth*

One sub-benchmark measures the memory bandwidth using a lattice version of the `axpy` BLAS
routine, similar to the STREAM benchmark. The JSON entries under `"axpy"` have the form:
```json
{
  "GBps": 215.80653375861607,   // bandwidth in GB/s/node
  "GFlops": 19.310041765757834, // FP performance (double precision)
  "L": 8,                       // local lattice volume
  "size_MB": 3.0                // memory size in MB/node
}
```

A second benchmark performs site-wise SU(4) matrix multiplication and has a higher
arithmetic intensity than the `axpy` test (although it is still memory-bound). 
The JSON entries under `"SU4"` have the form:
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
grid, which is parallelised across at least two processes. The resulting bandwidth is
related to node-local transfers (inter-CPU, NVLink, ...) or network transfers depending
on the MPI decomposition. The JSON entries under `"comms"` have the form:
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

This sub-benchmark measures the achieved floating-point performance for the Wilson-, 
domain-wall- and staggered-fermion sparse matrices provided by Grid. In the `"flops"`
and `"results"` sections of the JSON output the best performances are recorded, e.g.:
```json
{
  "Gflops_dwf4": 366.5251173474483,       // domain-wall in Gflop/s/node (single precision)
  "Gflops_staggered": 7.5982861018529455, // staggered in Gflop/s/node (single precision)
  "Gflops_wilson": 15.221839719288932,    // Wilson in Gflop/s/node (single precision)
  "L": 8                                  // local lattice size
}
```
Here "best" means the best result across the different implementations of the routines.
Please see the benchmark log for a detailed breakdown. Finally, the JSON output contains
a "comparison point", which is the average of the L=24 and L=32 best domain-wall performances.