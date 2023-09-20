#!/bin/sh
set -e
NUM_JOBS=40
INSTALL_ROOT="$(pwd)"/install
mkdir -p "$INSTALL_ROOT"

module load \
    gcc/8.3.1 \
    cuda/11.2.0 \
    cde/v3/cmake \
    cde/v3/ninja \
    cde/v3/git \
    cde/v3/git-lfs \
    essl/6.3.0.1 \
    spectrum-mpi/rolling-release

# Note: semicolon seperated list of libraries/flags
NETLIB_OPTIMIZED_BLAS_LIBS="$ESSLLIBDIR64/libesslsmpcuda.so"

export OMPI_CXX=g++
export OMPI_CC=gcc
export OMPI_FC=gfortran
export OMPI_F77=gfortran
export OMPI_F90=gfortran

export AR=$(command -v gcc-ar)
export NM=$(command -v gcc-nm)
export RANLIB=$(command -v gcc-ranlib)
export LD=$(command -v ld.gold)

export CXX=mpic++
export CC=mpicc
export FC=mpifort
export F77=mpifort
export F90=mpifort

export CPPFLAGS=""
export CFLAGS="-fopenmp -O3 -pthread"
export CXXFLAGS="-fopenmp -O3 -pthread"
export FFLAGS="-fopenmp -O3 -pthread"
export FCFLAGS="-fopenmp -O3 -pthread"
export LDFLAGS="-fuse-ld=gold -Wl,--disable-new-dtags -lgfortran -mcmodel=large -Wl,-rpath,/usr/tce/packages/spectrum-mpi/ibm/spectrum-mpi-rolling-release/lib/pami_port"

export LLNL_USE_OMPI_VARS=y

mkdir zlib_build
pushd zlib_build

curl -L -O $'https://zlib.net/zlib-1.2.13.tar.xz'
tar xJf zlib-1.2.13.tar.xz

mkdir build
pushd build

mkdir tmp
export TMPDIR=$(pwd)/tmp
  LDFLAGS="$LDFLAGS -Wl,-rpath,'\$\$ORIGIN/../lib'" \
  CC="$OMPI_CC" \
  CFLAGS="$CFLAGS $LDFLAGS" \
    ../zlib-1.2.13/configure \
        --static \
        --prefix="$INSTALL_ROOT"
make -j${NUM_JOBS}
make install

popd
popd

mkdir hdf5_build
pushd hdf5_build

curl -L -O $'https://github.com/HDFGroup/hdf5/archive/refs/tags/hdf5-1_10_9.tar.gz'
tar xzf hdf5-1_10_9.tar.gz

mkdir build
pushd build

mkdir tmp
export TMPDIR=$(pwd)/tmp
  LDFLAGS="$LDFLAGS -Wl,-rpath,'\$\$ORIGIN/../lib'" \
  CC="mpicc" \
  CXX="mpic++" \
  FC="mpifort" \
  FCLIBS="$LDFLAGS" \
    ../hdf5-hdf5-1_10_9/configure \
        --with-zlib="$INSTALL_ROOT" \
        --enable-hl \
        --enable-hltools \
        --enable-tools \
        --disable-tests \
        --disable-shared \
        --enable-symbols \
        --enable-parallel \
        --with-default-api-version=v110 \
        --prefix="$INSTALL_ROOT"

make -j${NUM_JOBS}
make install

popd
popd

mkdir pnetcdf_build
pushd pnetcdf_build

curl -L -O $'https://parallel-netcdf.github.io/Release/pnetcdf-1.12.3.tar.gz'
tar xzf pnetcdf-1.12.3.tar.gz

mkdir build
pushd build

mkdir tmp
export TMPDIR=$(pwd)/tmp
  LDFLAGS="$LDFLAGS -Wl,-rpath,'\$\$ORIGIN/../lib'" \
  CC="mpicc" \
  CXX="mpic++" \
  CFLAGS="${CFLAGS}" \
  CXXFLAGS="${CXXFLAGS}" \
    ../pnetcdf-1.12.3/configure \
        --disable-fortran \
        --enable-static \
        --with-sysroot="$INSTALL_ROOT" \
        --prefix="$INSTALL_ROOT"
make -j${NUM_JOBS}
make install

popd
popd

mkdir netcdf_build
pushd netcdf_build

curl -L -O $'https://github.com/Unidata/netcdf-c/archive/refs/tags/v4.8.1.tar.gz'
tar xzf v4.8.1.tar.gz

mkdir build
pushd build

mkdir tmp
export TMPDIR=$(pwd)/tmp
  ZLIB_ROOT="$INSTALL_ROOT" \
  HDF5_ROOT="$INSTALL_ROOT" \
  PNETCDF_ROOT="$INSTALL_ROOT" \
  LDFLAGS="$LDFLAGS -Wl,-rpath,'\$ORIGIN/../lib'" \
    cmake \
        -G Ninja \
        -D CMAKE_INSTALL_PREFIX="$INSTALL_ROOT" \
        -D CMAKE_INSTALL_LIBDIR=lib \
        -D ENABLE_TESTS=OFF \
        -D ENABLE_DAP=OFF \
        -D ENABLE_PNETCDF=ON \
        -D BUILD_SHARED_LIBS=OFF \
        -D CURL_DIR='' \
        -D CURL_INCLUDE_DIR='' \
        -D CURL_LIBRARY_DEBUG='' \
        -D CURL_LIBRARY_RELEASE='' \
        ../netcdf-c-4.8.1
cmake --build . -j${NUM_JOBS}
cmake --install .

popd
popd

mkdir netlib_build
pushd netlib_build

curl -L -O $'https://github.com/Reference-LAPACK/lapack/archive/refs/tags/v3.10.1.tar.gz'
tar xzf v3.10.1.tar.gz

mkdir build
pushd build

mkdir tmp
export TMPDIR=$(pwd)/tmp
  LDFLAGS="$LDFLAGS -Wl,-rpath,'\$ORIGIN/../lib'" \
  CC="$OMPI_CC" \
  CXX="$OMPI_CXX" \
  FC="$OMPI_FC" \
    cmake \
        -G Ninja \
        -D CMAKE_INSTALL_PREFIX="$INSTALL_ROOT" \
        -D CMAKE_INSTALL_LIBDIR=lib \
        -D BUILD_DEPRECATED=ON \
        -D BUILD_SHARED_LIBS=ON \
        -D CBLAS=OFF \
        -D USE_OPTIMIZED_BLAS=ON \
        -D BLAS_LIBRARIES="$NETLIB_OPTIMIZED_BLAS_LIBS" \
        ../lapack-3.10.1
cmake --build . -j${NUM_JOBS}
cmake --install .

popd
popd

mkdir trilinos_build
pushd trilinos_build

git clone --branch trilinos-release-14-0-0 --depth 1 https://github.com/trilinos/Trilinos.git

mkdir build
pushd build

mkdir tmp
export TMPDIR=$(pwd)/tmp
  LDFLAGS="$LDFLAGS -Wl,-rpath,'\$ORIGIN/../lib'" \
  LDFLAGS="$LDFLAGS -L'$INSTALL_ROOT/lib'" \
  BLAS_ROOT="$INSTALL_ROOT" \
  LAPACK_ROOT="$INSTALL_ROOT" \
  CMAKE_PREFIX_PATH="$INSTALL_ROOT":"$CMAKE_PREFIX_PATH" \
  OMPI_CXX="$(pwd)/../Trilinos/packages/kokkos/bin/nvcc_wrapper" \
    cmake \
        -G Ninja \
        -D CMAKE_C_COMPILER=mpicc \
        -D CMAKE_CXX_COMPILER=mpic++ \
        -D CMAKE_CXX_FLAGS="-g" \
        -D CMAKE_BUILD_TYPE=RELEASE \
        -D Trilinos_ENABLE_Fortran:BOOL=OFF \
        -D CMAKE_INSTALL_PREFIX="$INSTALL_ROOT" \
        -D Trilinos_TPL_SYSTEM_INCLUDE_DIRS=TRUE \
        -D Trilinos_ENABLE_OpenMP:BOOL=ON \
        -D Trilinos_ENABLE_PanzerMiniEM:BOOL=ON \
        -D PanzerMiniEM_ENABLE_EXAMPLES:BOOL=ON \
        -D Kokkos_ENABLE_CUDA_UVM:BOOL=ON \
        -D KokkosKernels_INST_MEMSPACE_CUDAUVMSPACE:BOOL=ON \
        -D Tpetra_ENABLE_CUDA_UVM:BOOL=ON \
        -D TPL_ENABLE_MPI:BOOL=ON \
        -D TPL_ENABLE_CUDA:BOOL=ON \
        -D TPL_ENABLE_Netcdf:BOOL=ON \
        -D TPL_BLAS_LIBRARIES="$NETLIB_OPTIMIZED_BLAS_LIBS" \
        -D TPL_LAPACK_LIBRARIES="$INSTALL_ROOT/lib/liblapack.so;$NETLIB_OPTIMIZED_BLAS_LIBS" \
        -D TPL_Netcdf_LIBRARIES="$INSTALL_ROOT/lib/libnetcdf.a;$INSTALL_ROOT/lib/libpnetcdf.a;$INSTALL_ROOT/lib/libhdf5_hl.a;$INSTALL_ROOT/lib/libhdf5.a;-lcurl;-lz;-ldl;-lm" \
        -D Trilinos_ENABLE_ALL_OPTIONAL_PACKAGES:BOOL=OFF \
        -D Trilinos_ENABLE_ALL_PACKAGES:BOOL=OFF \
        -D Kokkos_ENABLE_CUDA:BOOL=ON \
        -D Kokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE:BOOL=ON \
        -D Kokkos_ARCH_POWER9:BOOL=ON \
        -D Kokkos_ARCH_VOLTA70:BOOL=ON \
        ../Trilinos
  OMPI_CXX="$(pwd)/../Trilinos/packages/kokkos/bin/nvcc_wrapper" \
    cmake --build . -j${NUM_JOBS}
cmake --install .

popd
popd

