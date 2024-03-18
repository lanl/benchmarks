#!/bin/bash

set -x
set -e

# setup environment
source /projects/sems/modulefiles/utils/sems-modules-init.sh
module load sems-cmake/3.24.3 sems-gcc/10.1.0 sems-openmpi sems-cuda sems-boost sems-netcdf-c sems-parallel-netcdf sems-hdf5

# define key environment vars
export DIR_BASE="` pwd -P `"
export DIR_BUILD="${DIR_BASE}/trilinos-build-ascicgpu030"
export DIR_INSTALL="${DIR_BASE}/trilinos-install-ascicgpu030"
export DIR_TRILINOS="` git rev-parse --show-toplevel `/trilinos"
export DIR_KOKKOSP="` git rev-parse --show-toplevel `/kokkos-tools/profiling/space-time-stack"
export OMPI_CXX="${DIR_TRILINOS}/packages/kokkos/bin/nvcc_wrapper"

# build Trilinos
## setup directories
if test -d "${DIR_BUILD}" ; then
    rm -rf "${DIR_BUILD}"
fi
if test -d "${DIR_INSTALL}" ; then
    rm -rf "${DIR_INSTALL}"
fi
mkdir -p "${DIR_BUILD}" "${DIR_INSTALL}"
pushd "${DIR_BUILD}"

## do build
rm -rf CMakeFiles CMakeCache.txt
cmake \
    -D CMAKE_BUILD_TYPE:STRING="Release" \
    -D Trilinos_ENABLE_Amesos2=ON \
    -D Trilinos_ENABLE_AztecOO=ON \
    -D Trilinos_ENABLE_Epetra=ON \
    -D Trilinos_ENABLE_EpetraExt=ON \
    -D Trilinos_ENABLE_Ifpack=ON \
    -D Trilinos_ENABLE_Ifpack2=ON \
    -D Ifpack2_ENABLE_TESTS=ON \
    -D Ifpack2_ENABLE_EXAMPLES=ON \
    -D Trilinos_ENABLE_Kokkos=ON \
    -D Kokkos_ARCH_VOLTA70=ON \
    -D Kokkos_ENABLE_CUDA=ON \
    -D Kokkos_ENABLE_CUDA_UVM=ON \
    -D Kokkos_ENABLE_CUDA_LAMBDA=ON \
    -D Kokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE=OFF \
    -D Trilinos_ENABLE_ML=ON \
    -D Trilinos_ENABLE_MueLu=ON \
    -D MueLu_ENABLE_TESTS=ON \
    -D MueLu_ENABLE_EXAMPLES=ON \
    -D Trilinos_ENABLE_NOX=ON \
    -D NOX_ENABLE_EpetraExt=ON \
    -D NOX_ENABLE_LOCA=ON \
    -D NOX_ENABLE_ML=ON \
    -D Trilinos_ENABLE_Panzer=ON \
    -D Trilinos_ENABLE_PanzerMiniEM=ON \
    -D Trilinos_ENABLE_PanzerAdaptersSTK=ON \
    -D PanzerMiniEM_ENABLE_EXAMPLES=ON \
    -D PanzerMiniEM_ENABLE_TESTS=ON \
    -D Panzer_ENABLE_TESTS=ON \
    -D Panzer_ENABLE_EXAMPLES=ON \
    -D Trilinos_ENABLE_Piro=ON \
    -D Trilinos_ENABLE_Rythmos=ON \
    -D Trilinos_ENABLE_ShyLU=OFF \
    -D Trilinos_ENABLE_ShyLU_DD=OFF \
    -D Trilinos_ENABLE_ShyLU_Node=OFF \
    -D Trilinos_ENABLE_Stratimikos=ON \
    -D Trilinos_ENABLE_Stokhos=ON \
    -D Stokhos_ENABLE_CUDA=ON \
    -D Trilinos_ENABLE_Tpetra=ON \
    -D Tpetra_INST_SERIAL=OFF \
    -D Tpetra_INST_CUDA=ON \
    -D Tpetra_ENABLE_TESTS=ON \
    -D Tpetra_ENABLE_EXAMPLES=ON \
    -D Trilinos_ENABLE_Thyra=ON \
    -D Trilinos_ENABLE_ThyraEpetraAdapters=ON \
    -D Trilinos_ENABLE_Xpetra=ON \
    \
    -D TPL_ENABLE_HDF5=ON \
    -D HDF5_INCLUDE_DIRS=${HDF5_ROOT}/include \
    -D HDF5_LIBRARY_DIRS=${HDF5_ROOT}/lib \
    -D TPL_ENABLE_Netcdf=ON \
    -D Netcdf_INCLUDE_DIRS=${NETCDF_ROOT}/include \
    -D Netcdf_LIBRARY_DIRS=${NETCDF_ROOT}/lib \
    -D TPL_ENABLE_Boost=ON \
    -D Boost_INCLUDE_DIRS=${BOOST_ROOT}/include \
    -D TPL_ENABLE_BoostLib=ON \
    -D BoostLib_INCLUDE_DIRS=${BOOST_ROOT}/include \
    -D BoostLib_LIBRARY_DIRS=${BOOST_ROOT}/lib \
    \
    -D CMAKE_C_COMPILER:PATH=`which mpicc` \
    -D CMAKE_CXX_COMPILER:PATH=`which mpicxx` \
    -D CMAKE_CXX_FLAGS:STRING="-g" \
    -D CMAKE_C_FLAGS:STRING="-g" \
    -D CMAKE_INSTALL_PREFIX:PATH="${DIR_INSTALL}" \
    \
    -D TPL_ENABLE_MPI=ON \
    -D Trilinos_ENABLE_EXPLICIT_INSTANTIATION=ON \
    -D BUILD_SHARED_LIBS=ON \
    "${DIR_TRILINOS}"
make -j 32
make install
popd

# build Kokkos Tools for memory information (if desired)
pushd "${DIR_KOKKOSP}"
make CXX=mpicxx
popd

exit 0
