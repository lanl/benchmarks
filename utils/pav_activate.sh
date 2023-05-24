#!/bin/bash

#OPTIONAL ARG. Default pavilion.yaml is yellow.yaml in base_configs.
pavconf=${1:-"yellow.yaml"}

# Resolves all symlinks in the given or current (no arg) path.
realpath() {
    tpth=${1:-$(pwd)}
    echo $(python3 -c "import os; print(os.path.realpath('$tpth'))")
}

realpath() {
    tpth=${1:-$(pwd)}
	echo $(/usr/bin/realpath $tpth)
}

if [[ -f ${BASH_SOURCE[0]} ]]; then
	thisfile=${BASH_SOURCE[0]}
elif [[ -f $0 ]]; then
	thisfile=$0
else
	echo "THE SCRIPT YOU SOURCED CAN'T FIND ITSELF.  CONSULT YOUR HIGHER POWER."
	exit 1
fi

# Set PAV_CONFIG_DIR to the directory with this file.
# Set PAVBIN to the direcotry with the pavilion binaries in this repo.
export PAV_CONFIG_DIR=$(dirname $(realpath $thisfile))
PAVBIN="${PAV_CONFIG_DIR}/pav_src/bin"
echo "THISPATH: $(realpath $PWD)"
echo "PAVCPATH: $(realpath $PAV_CONFIG_DIR)"
echo "BASH_SOURCE: ${BASH_SOURCE[0]}"
echo "0: ${0}"

# Only prepend PAVBIN to path if it hasn't already been done.
# Error out if PAVBIN doesn't exist.
if [[ -d $PAVBIN ]]; then
	export PAVBIN
	if [[ ! ("${PATH}" =~ "${PAVBIN}") ]]; then
		export PATH="${PAVBIN}:${PATH}"
	fi
else
	echo "ERROR: PAVBIN NOT SET: ${PAVBIN} is not a directory."
    echo "       PERHAPS git submodule --init --recursive hasn't been run."
fi

# Symlink desired configuration yaml to top directory.
ln -sfn ${PAV_CONFIG_DIR}/base_configs/${pavconf} ${PAV_CONFIG_DIR}/pavilion.yaml

echo "Success:"
echo "  PAVBIN         -- ${PAVBIN}"
echo "  PAV_CONFIG_DIR -- ${PAV_CONFIG_DIR}"
