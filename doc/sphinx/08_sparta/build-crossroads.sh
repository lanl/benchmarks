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
cp -a Makefile.crossroads_omp_spr "${dir_src}/src/MAKE"
pushd "${dir_src}/src"
make yes-kokkos
make -j 16 crossroads_omp_spr
echo "Resultant build info:"
ls -lh "`pwd -P`/spa_crossroads_omp_spr"
mv "`pwd -P`/spa_crossroads_omp_spr" "${dir_base}"
popd

# do the AVX512 build
pushd "${dir_src}"
git clean -fdx
git reset --hard
popd
cp -a Makefile.crossroads_omp_skx "${dir_src}/src/MAKE"
pushd "${dir_src}/src"
make yes-kokkos
make -j 16 crossroads_omp_skx
echo "Resultant build info:"
ls -lh "`pwd -P`/spa_crossroads_omp_skx"
mv "`pwd -P`/spa_crossroads_omp_skx" "${dir_base}"
popd


exit 0
