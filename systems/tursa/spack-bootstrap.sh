#!/usr/bin/env bash
# shellcheck disable=SC2016
set -euo pipefail

gcc_spec='gcc@9.4.0'
cuda_spec='cuda@11.4.0'
hdf5_spec='hdf5@1.10.7'

if (( $# != 1 )); then
    echo "usage: $(basename "$0") <env dir>" 1>&2
    exit 1
fi
dir=$1
cwd=$(pwd -P)
cd "${dir}"
dir=$(pwd -P)
cd "${cwd}"

# General configuration ########################################################
# build with 128 tasks
echo 'config:                                  
  build_jobs: 128
  build_stage:
    - $spack/var/spack/stage
  test_stage: $spack/var/spack/test
  misc_cache: $spack/var/spack/cache' > jobs.yaml
spack config --scope site add -f jobs.yaml
rm jobs.yaml

# add lustre as external package
echo 'packages:
  lustre:
    externals:
    - spec: "lustre@2.12.6_ddn36"
      prefix: /usr' > external.yaml
spack config --scope site add -f external.yaml
rm external.yaml

# Base compilers ###############################################################
# configure system base
spack compiler find --scope site

# install GCC, CUDA & LLVM
spack install ${gcc_spec} ${cuda_spec} llvm

spack load llvm
spack compiler find --scope site
spack unload llvm

spack load ${gcc_spec}
spack compiler find --scope site
spack unload ${gcc_spec}

# Manual compilation of OpenMPI & UCX ##########################################
# set build directories
mkdir -p "${dir}"/build
cd "${dir}"/build

spack load ${gcc_spec} ${cuda_spec}

cuda_path=$(spack find --format "{prefix}" cuda)
gdrcopy_path=/mnt/lustre/tursafs1/apps/gdrcopy/2.3.1

# Install ucx 1.12.0
ucx_url=https://github.com/openucx/ucx/releases/download/v1.12.0/ucx-1.12.0.tar.gz

echo "-- building UCX from source"
wget ${ucx_url}
ucx_ar=$(basename ${ucx_url})
tar -xvf "${ucx_ar}"
cd "${ucx_ar%.tar.gz}"

# ucx gpu build
mkdir -p build_gpu; cd build_gpu
../configure --build=x86_64-redhat-linux-gnu --host=x86_64-redhat-linux-gnu    \
             --disable-dependency-tracking --prefix="${dir}"/prefix/ucx_gpu    \
             --enable-devel-headers --enable-examples --enable-optimizations   \
             --with-gdrcopy=${gdrcopy_path} --with-verbs --disable-logging     \
             --disable-debug --disable-assertions --enable-cma                 \
             --with-knem=/opt/knem-1.1.4.90mlnx1/ --with-rdmacm                \
             --without-rocm --without-ugni --without-java                      \
             --enable-compiler-opt=3 --with-cuda="${cuda_path}" --without-cm   \
             --with-rc --with-ud --with-dc --with-mlx5-dv --with-dm            \
             --enable-mt --without-go LDFLAGS=-L${gdrcopy_path}/lib
make -j 128
make install
cd ..

# ucx cpu build
mkdir -p build_cpu; cd build_cpu
../configure --build=x86_64-redhat-linux-gnu --host=x86_64-redhat-linux-gnu    \
             --disable-dependency-tracking --prefix="${dir}"/prefix/ucx_cpu    \
             --enable-devel-headers --enable-examples --enable-optimizations   \
             --with-verbs --disable-logging --disable-debug                    \
             --disable-assertions --enable-mt --enable-cma                     \
             --with-knem=/opt/knem-1.1.4.90mlnx1/ --with-rdmacm                \
             --without-rocm --without-ugni --without-java                      \
             --enable-compiler-opt=3 --without-cm --without-ugni --with-rc     \
             --with-ud --with-dc --with-mlx5-dv --with-dm --enable-mt --without-go
make -j 128
make install

cd "${dir}"/build

# Install openmpi 4.1.1
ompi_url=https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.1.tar.gz

echo "-- building OpenMPI from source"

wget ${ompi_url}
ompi_ar=$(basename ${ompi_url})
tar -xvf "${ompi_ar}"
cd "${ompi_ar%.tar.gz}"
export AUTOMAKE_JOBS=128
./autogen.pl -f

# openmpi gpu build
mkdir build_gpu; cd build_gpu
../configure --prefix="${dir}"/prefix/ompi_gpu --without-xpmem    \
             --with-ucx="${dir}"/prefix/ucx_gpu                   \
             --with-ucx-libdir="${dir}"/prefix/ucx_gpu/lib        \
             --with-knem=/opt/knem-1.1.4.90mlnx1/                 \
             --enable-mca-no-build=btl-uct                        \
             --with-cuda="${cuda_path}" --disable-getpwuid        \
             --with-verbs --with-slurm --enable-mpi-fortran=all   \
             --with-pmix=internal --with-libevent=internal
make -j 128 
make install 
cd ..

# openmpi cpu build
mkdir build_cpu; cd build_cpu
../configure --prefix="${dir}"/prefix/ompi_cpu --without-xpmem    \
             --with-ucx="${dir}"/prefix/ucx_cpu                   \
             --with-ucx-libdir="${dir}"/prefix/ucx_cpu/lib        \
             --with-knem=/opt/knem-1.1.4.90mlnx1/                 \
             --enable-mca-no-build=btl-uct --disable-getpwuid     \
             --with-verbs --with-slurm --enable-mpi-fortran=all   \
             --with-pmix=internal --with-libevent=internal
make -j 128 
make install
cd "${dir}"

# Add externals to spack
echo "packages:
  ucx:
    externals:
    - spec: \"ucx@1.12.0.GPU%gcc@9.4.0\"
      prefix: ${dir}/prefix/ucx_gpu
    - spec: \"ucx@1.12.0.CPU%gcc@9.4.0\"
      prefix: ${dir}/prefix/ucx_cpu
    buildable: False
  openmpi:
    externals:
    - spec: \"openmpi@4.1.1.GPU%gcc@9.4.0\"
      prefix: ${dir}/prefix/ompi_gpu
    - spec: \"openmpi@4.1.1.CPU%gcc@9.4.0\"
      prefix: ${dir}/prefix/ompi_cpu
    buildable: False" > spack.yaml

spack config --scope site add -f spack.yaml
rm spack.yaml
spack install ucx@1.12.0.GPU%gcc@9.4.0 openmpi@4.1.1.GPU%gcc@9.4.0
spack install ucx@1.12.0.CPU%gcc@9.4.0 openmpi@4.1.1.CPU%gcc@9.4.0

cd "${cwd}"

# environments #################################################################
dev_tools=("autoconf" "automake" "libtool" "jq" "git")
ompi_gpu_hash=$(spack find --format "{hash}" openmpi@4.1.1.GPU)
ompi_cpu_hash=$(spack find --format "{hash}" openmpi@4.1.1.CPU)

spack env create grid-gpu
spack env activate grid-gpu
spack add ${gcc_spec} ${cuda_spec} "${dev_tools[@]}" 
spack add ucx@1.12.0.GPU%gcc@9.4.0 openmpi@4.1.1.GPU%gcc@9.4.0
spack add ${hdf5_spec}+cxx+threadsafe ^/"${ompi_gpu_hash}"
spack add fftw ^/"${ompi_gpu_hash}"
spack add openssl gmp mpfr c-lime
spack install
spack env deactivate

spack env create grid-cpu
spack env activate grid-cpu
spack add llvm "${dev_tools[@]}" 
spack add ucx@1.12.0.CPU%gcc@9.4.0 openmpi@4.1.1.CPU%gcc@9.4.0
spack add ${hdf5_spec}+cxx+threadsafe ^/"${ompi_cpu_hash}"
spack add fftw ^/"${ompi_cpu_hash}"
spack add openssl gmp mpfr c-lime
spack install
spack env deactivate

spack install jq git

# Final setup ##################################################################
spack clean
spack gc -y

# add more environment variables in module loading
spack config --scope site add 'modules:prefix_inspections:lib:[LD_LIBRARY_PATH,LIBRARY_PATH]'
spack config --scope site add 'modules:prefix_inspections:lib64:[LD_LIBRARY_PATH,LIBRARY_PATH]'
spack config --scope site add 'modules:prefix_inspections:include:[C_INCLUDE_PATH,CPLUS_INCLUDE_PATH,INCLUDE]'
spack module tcl refresh -y
