#!/bin/bash

setvar () {
	while [[ $# -gt 0 ]]; do
		export $1
		shift
	done
}

# FOR LANL SYSTEMS
notloaded() {
    echo "$1 NOT LOADED"
    echo " PLEASE LOAD A COMPILER AND MPI BEFORE RUNNING THIS SCRIPT"
}

# SET THESE VARS FROM THE COMMAND LINE.
# USE KEY=VALUE AS ARGS.
# IF USING THE PROVIDED MODULEFILE, 
# ADJUST THE `softdir` VARIABLE IF CHANGING PRE_PREFIX.
PRE_PREFIX="${HOME}/proj/installs"
NAME="memory_recorder"

setvar $@

# Determine what compiler library we're using
if [[ -n $LCOMPILER ]]; then
    compstr=$LCOMPILER
elif [[ -n $LMOD_FAMILY_COMPILER ]]; then
    compstr=$LMOD_FAMILY_COMPILER
else
    notloaded "COMPILER"
    exit 1
fi

# Determine what MPI library we're using
if [[ -n $LMPI ]]; then
    mpistr=$LMPI
elif [[ -n $LMOD_FAMILY_MPI ]]; then
    mpistr=$LMOD_FAMILY_MPI
else
    notloaded "MPI"
    exit 1
fi

SYSNAME=$(/usr/projects/hpcsoft/utilities/bin/sys_name)
SYSOS=$(/usr/projects/hpcsoft/utilities/bin/sys_os)

SYS_PREFIX="${PRE_PREFIX}/${SYSNAME}/${NAME}"
PREFIX="${SYS_PREFIX}/${compstr}_${mpistr}"
mkdir -p $SYS_PREFIX

if [[ -z $CXX ]]; then
    if [[ ${SYSOS} =~ toss ]]; then
        export CXX=mpicxx
    else
        export CXX=CC
    fi
fi

make -j 1 PREFIX=$PREFIX install