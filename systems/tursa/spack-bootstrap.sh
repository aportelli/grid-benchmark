#!/usr/bin/env bash
# shellcheck disable=SC2016
set -euo pipefail

GCC='gcc@9.4.0'
CUDA='cuda@11.4.0'
HDF5='hdf5@1.10.7'

if (( $# != 1 )); then
    echo "usage: $(basename "$0") <env dir>" 1>&2
    exit 1
fi
ENVDIR=$1
CWD=$(pwd -P)

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

# configure system base
spack compiler find --scope site

# Base packages ################################################################
# install GCC
spack install ${GCC}
spack load ${GCC}
spack compiler find --scope site
spack unload ${GCC}

# clean
spack clean
spack gc -y

# install CUDA
spack install ${CUDA}

# install development tools
dev_tools=("autoconf" "automake" "libtool" "git")
spack install "${dev_tools[@]}"

# create view for CLI & dev tools
spack view symlink -i "${ENVDIR}/prefix/base" "${dev_tools[@]}"

# install clang
spack install llvm@12.0.1

# locate new compilers
spack load llvm@12.0.1
spack compiler find --scope site
spack unload llvm@12.0.1

# Manual compilation of OpenMPI & UCX ##########################################
# set build directories
mkdir -p "${ENVDIR}"/build
cd "${ENVDIR}"/build

spack load ${GCC} ${CUDA}

CUDA_PATH=$(which nvcc | sed "s/bin/@/g" | cut -d "@" -f1)
GDRCOPY_PATH=/mnt/lustre/tursafs1/apps/gdrcopy/2.3.1

# Install ucx 1.12.0
UCX_URL=https://github.com/openucx/ucx/releases/download/v1.12.0/ucx-1.12.0.tar.gz

echo "-- building UCX from source"
wget ${UCX_URL}
UCX_AR=$(basename ${UCX_URL})
tar -xvf "${UCX_AR}"
cd "${UCX_AR%.tar.gz}"

# ucx gpu build
mkdir build_gpu; cd build_gpu
../configure --build=x86_64-redhat-linux-gnu --host=x86_64-redhat-linux-gnu    \
             --disable-dependency-tracking --prefix="${ENVDIR}"/prefix/ucx_gpu \
             --enable-devel-headers --enable-examples --enable-optimizations   \
             --with-gdrcopy=${GDRCOPY_PATH} --with-verbs --disable-logging     \
             --disable-debug --disable-assertions --enable-cma                 \
             --with-knem=/opt/knem-1.1.4.90mlnx1/ --with-rdmacm                \
             --without-rocm --without-ugni --without-java                      \
             --enable-compiler-opt=3 --with-cuda="${CUDA_PATH}" --without-cm   \
             --with-rc --with-ud --with-dc --with-mlx5-dv --with-dm            \
             --enable-mt LDFLAGS=-L${GDRCOPY_PATH}/lib
make -j 128
make install
cd ..

# ucx cpu build
mkdir build_cpu; cd build_cpu
../configure --build=x86_64-redhat-linux-gnu --host=x86_64-redhat-linux-gnu    \
             --disable-dependency-tracking --prefix="${ENVDIR}"/prefix/ucx_cpu \
             --enable-devel-headers --enable-examples --enable-optimizations   \
             --with-verbs --disable-logging --disable-debug                    \
             --disable-assertions --enable-mt --enable-cma                     \
             --with-knem=/opt/knem-1.1.4.90mlnx1/ --with-rdmacm                \
             --without-rocm --without-ugni --without-java                      \
             --enable-compiler-opt=3 --without-cm --without-ugni --with-rc     \
             --with-ud --with-dc --with-mlx5-dv --with-dm --enable-mt
make -j 128
make install

cd "${ENVDIR}"/build

# Install openmpi 4.1.1 (needs to be done on a gpu node)
OMPI_URL=https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.1.tar.gz

echo "-- building OpenMPI from source"

wget ${OMPI_URL}
OMPI_AR=$(basename ${OMPI_URL})
tar -xvf "${OMPI_AR}"
cd "${OMPI_AR%.tar.gz}"

# openmpi gpu build
mkdir build_gpu; cd build_gpu
../configure --prefix="${ENVDIR}"/prefix/ompi_gpu --without-xpmem \
             --with-ucx="${ENVDIR}"/prefix/ucx_gpu                \
             --with-ucx-libdir="${ENVDIR}"/prefix/ucx_gpu/lib     \
             --with-knem=/opt/knem-1.1.4.90mlnx1/                 \
             --enable-mca-no-build=btl-uct                        \
             --with-cuda="${CUDA_PATH}" --disable-getpwuid        \
             --with-verbs --with-slurm --enable-mpi-fortran=all   \
             --with-pmix=internal --with-libevent=internal
make -j 128 
make install 
cd ..

# openmpi cpu build
mkdir build_cpu; cd build_cpu
../configure --prefix="${ENVDIR}"/prefix/ompi_cpu --without-xpmem \
             --with-ucx="${ENVDIR}"/prefix/ucx_cpu                \
             --with-ucx-libdir="${ENVDIR}"/prefix/ucx_cpu/lib     \
             --with-knem=/opt/knem-1.1.4.90mlnx1/                 \
             --enable-mca-no-build=btl-uct --disable-getpwuid     \
             --with-verbs --with-slurm --enable-mpi-fortran=all   \
             --with-pmix=internal --with-libevent=internal
make -j 128 
make install
cd "${ENVDIR}"

# Add externals to spack
echo "packages:
  ucx:
    externals:
    - spec: \"ucx@1.12.0.GPU%gcc@9.4.0\"
      prefix: ${ENVDIR}/prefix/ucx_gpu
    - spec: \"ucx@1.12.0.CPU%gcc@9.4.0\"
      prefix: ${ENVDIR}/prefix/ucx_cpu
    buildable: False
  openmpi:
    externals:
    - spec: \"openmpi@4.1.1.GPU%gcc@9.4.0\"
      prefix: ${ENVDIR}/prefix/ompi_gpu
    - spec: \"openmpi@4.1.1.CPU%gcc@9.4.0\"
      prefix: ${ENVDIR}/prefix/ompi_cpu
    buildable: False" > spack.yaml

spack config --scope site add -f spack.yaml
rm spack.yaml
spack install ucx@1.12.0.GPU%gcc@9.4.0
spack install ucx@1.12.0.CPU%gcc@9.4.0
spack install openmpi@4.1.1.GPU%gcc@9.4.0
spack install openmpi@4.1.1.CPU%gcc@9.4.0

# Install Grid dependencies ####################################################
cd "${CWD}"

OPENMPIGPUHASH=$(spack find --format "{hash}" openmpi@4.1.1.GPU)
OPENMPICPUHASH=$(spack find --format "{hash}" openmpi@4.1.1.CPU)

spack install ${HDF5}+cxx+threadsafe ^/"${OPENMPIGPUHASH}"
spack install ${HDF5}+cxx+threadsafe ^/"${OPENMPICPUHASH}"
spack install fftw ^/"${OPENMPIGPUHASH}"
spack install fftw ^/"${OPENMPICPUHASH}"
spack install openssl gmp mpfr c-lime

# Final setup ##################################################################
spack clean

# add more environment variables in module loading
spack config --scope site add 'modules:prefix_inspections:lib:[LIBRARY_PATH]'
spack config --scope site add 'modules:prefix_inspections:include:[C_INCLUDE_PATH,CPLUS_INCLUDE_PATH,INCLUDE]'
spack module tcl refresh -y

# permission change for group access
chmod -R g+rw "${ENVDIR}/spack/var/spack/cache"
setfacl -d -R -m g::rwX "${ENVDIR}/spack/var/spack/cache"
