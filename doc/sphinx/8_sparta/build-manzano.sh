#!/bin/sh

umask 022
set -e
set -x

dir_root=`git rev-parse --show-toplevel`
dir_src="${dir_root}/sparta"

module unload intel
module unload openmpi-intel
module use /apps/modules/modulefiles-apps/cde/v3/
module load cde/v3/devpack/intel-ompi
module list

git clone https://github.com/sparta/sparta.git "${dir_src}"
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
