# Grid benchmarks

This folder contains benchmarks for the [Grid](https://github.com/aportelli/) library.
The benchmarks can be summarised as follows

- `Benchmark_Grid`: This benchmark measure floating point performances for various fermion
matrices, as well as bandwidth measurement for different operations. Measurements are
performed for a fixed range of problem sizes.

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
of packages, and might take some time to complete.

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
`<env_dir>/prefix/gridbench_<config>`.