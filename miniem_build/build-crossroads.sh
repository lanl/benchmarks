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

# create spack.yaml from template.yaml
if test -f "${DIR_BASE}/prep-spack-yaml.sh" ; then
    sh "${DIR_BASE}/prep-spack-yaml.sh"
fi

# exit if spack.yaml not present
if test ! -f "${DIR_BASE}/spack.yaml" ; then
    echo "ERROR: spack.yaml not present!"
    exit 1
fi

# load Spack into environment
. ./spack/share/spack/setup-env.sh

# load Crossroads environment
spack env activate -p -d "${DIR_BASE}"

# concretize Spack
spack concretize

# install
spack install

exit 0
