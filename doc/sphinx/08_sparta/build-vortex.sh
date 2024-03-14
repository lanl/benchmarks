#!/bin/sh

umask 022
set -e
set -x

dir_root="`git rev-parse --show-toplevel`"
dir_src="${dir_root}/sparta"
dir_base="` pwd -P `"

module load cuda/11.2.0
module load gcc/8.3.1
module list

# do the build
pushd "${dir_src}"
git clean -fdx
git reset --hard
rm -f src/spa_*
popd
# cp -a Makefile.vortex_kokkos "${dir_src}/src/MAKE"
pushd "${dir_src}/src"
make clean-all
make yes-kokkos
make -j 16 vortex_kokkos
echo "Resultant build info:"
ls -lh "`pwd -P`/spa_vortex_kokkos"
cp -a "`pwd -P`/spa_vortex_kokkos" "${dir_base}"
popd

# build map_cpu for binding
pushd "${dir_root}/map_cpu"
gcc -O2 -o map_cpu map_cpu.c
popd

# build Kokkos Tools for memory information (if desired)
pushd "${dir_root}/kokkos-tools/profiling/space-time-stack"
make CXX=mpicxx
popd


exit 0
