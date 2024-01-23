#!/bin/sh

export DIR_BASE=` pwd -P `

# clean up dirs
rm -rf install /tmp/$USER/spack* ~/.spack .spack* 

# clone spack
# git clone git@github.com:spack/spack

# get to appropriate checkout
# pushd spack
# git checkout v0.21.0
# popd

# apply patch
pushd spack
git apply "${DIR_BASE}/spack-fixes-v0.21.0.patch"
popd

# load Spack into environment
. ./spack/share/spack/setup-env.sh

# load Crossroads environment
spack env activate -p -d "${DIR_BASE}"

# concretize Spack
spack concretize

# install
spack install

exit 0
