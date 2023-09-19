#!/bin/sh

umask 022
set -e
set -x

dir_root=`git rev-parse --show-toplevel`
dir_src="${dir_root}/sparta"

module list

pushd "${dir_src}"
git clean -fdx
git reset --hard
popd
cp -a Makefile.manzano_kokkos "${dir_src}/src/MAKE"

pushd "${dir_src}/src"
make yes-kokkos
make -j 16 manzano_kokkos
echo "Resultant build info:"
ls -lh `pwd -P`/spa_manzano_kokkos
popd


exit 0
