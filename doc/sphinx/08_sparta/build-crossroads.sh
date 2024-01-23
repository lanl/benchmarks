#!/bin/sh

umask 022
set -e
set -x

dir_root="`git rev-parse --show-toplevel`"
dir_src="${dir_root}/sparta"
dir_base="` pwd -P `"

module list

# do the -march=sapphirerapids build
pushd "${dir_src}"
git clean -fdx
git reset --hard
popd
cp -a Makefile.crossroads_serial_spr "${dir_src}/src/MAKE"
pushd "${dir_src}/src"
make yes-kokkos
make -j 16 crossroads_serial_spr
echo "Resultant build info:"
ls -lh "`pwd -P`/spa_crossroads_serial_spr"
cp -a "`pwd -P`/spa_crossroads_serial_spr" "${dir_base}"
popd

# build map_cpu for binding
pushd "${dir_root}/map_cpu"
gcc -O2 -o map_cpu map_cpu.c
popd

# build Kokkos Tools for memory information (if desired)
pushd "${dir_root}/kokkos-tools/profiling/space-time-stack"
make CXX=CC
popd


exit 0
