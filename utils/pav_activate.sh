#!/bin/bash
umask 002

_dir=$(readlink -f "$(dirname "${BASH_SOURCE[0]}")")
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

export BMARK_UTIL="$(dirname $(realpath $thisfile))"
export BMARK_TOP="$(dirname $BMARK_UTIL)"

# Set PAV_CONFIG_DIR to the directory with this file.
# Set PAVBIN to the direcotry with the pavilion binaries in this repo.
export PAV_CONFIG_DIR="${BMARK_UTIL}/pav_config"
PAVBIN="${BMARK_UTIL}/pavilion/bin"

echo "BENCHMARK PAVILION ACTIVATION PATHS:"
echo "  THISPATH:    $(realpath $PWD)"
echo "  PAVCPATH:    $(realpath $PAV_CONFIG_DIR)"
echo "  PAVBIN:      $(realpath $PAVBIN)"
echo "  BASH_SOURCE: ${BASH_SOURCE[0]}"
echo "  0:           ${0}"

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

echo "  -----------------------  "
echo "Success:"
echo "  PAVBIN         -- ${PAVBIN}"
echo "  PAV_CONFIG_DIR -- ${PAV_CONFIG_DIR}"
