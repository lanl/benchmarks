#!/bin/bash

##############
# IOR
##############
set -e

## SET VARS FROM CL INPUT KEY=VALUE
setvar() {
	while [[ $# -gt 0 ]]; do
		export $1
		shift
	done
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
    echo "THIS SCRIPT CAN'T FIND ITSELF."
    exit 1
fi

# XRDS AND ROCI
runpath=$(pwd)
export SCRATCH_HOME=$(find /lustre -maxdepth 5 -name $USER -type d  2> /dev/null | grep -v -m1 givedir)
export THISDIR=$(dirname $thisfile)
export IORLOG=${HOME}/logs/ior
export PREFIX=${HOME}/ior
export TPN=110
export SEGMENTS=16
export SIZE=2G
export NNODES=${SLURM_NNODES}
export BUILD_DIR=/tmp/${USER}/ior
export BUILD=true
export IORSRC=$THISDIR
separator="\t ------------------------------------------------- \t"

###################################################################
## ANY VARIABLES SET THAT APPEAR BEFORE THIS CAN BE
## SET ON THE COMMAND LINE WITH KEY=VALUE.
###################################################################
setvar $@

mkdir -p $IORLOG

export WORKING_DIR=${SCRATCH_HOME}/ior
mkdir -p ${WORKING_DIR}
find ${WORKING_DIR} -type f -delete

### IF CRAY
export MPICC=cc
export MPICXX=CC
export MPIFC=ftn
export CC=cc
export CXX=CC
export FC=ftn
export CFLAGS='-O3 -w'
prearg=''
postarg=''

runior() {
    # 1 is working dir output file
    # 2 is POSIX or MPIIO
    # 3 is PRE 2 args
    # 4 is POST 2 args
    echo srun -N ${NNODES} --ntasks-per-node=${TPN} ./bin/ior ${prearg} $2 ${postarg} -o ${NNODES}_${2}_${1}/a &>> ior_results
	srun -N ${NNODES} --ntasks-per-node=${TPN} ./bin/ior ${prearg} $2 ${postarg}} -o ${NNODES}_${2}_${1}/a  &>> ior_results
    sleep 3
}

###################################################################
# BUILD IOR
if [[ $BUILD != "false" ]]; then
    rm -rf $PREFIX
    mkdir -p $PREFIX
    mkdir -p $BUILD_DIR
    cd $BUILD_DIR
    $IORSRC/configure --prefix=$PREFIX
    make
    make install
    cd $runpath
fi
if [[ $BUILD == "only" ]]; then
    echo BUILT HERE: $BUILD_DIR
    echo INSTALLED HERE: $PREFIX
    exit 0
fi

###################################################################
# PRE
# -k -e -a
# -C -Q $TPN -k -E -a
# -k -e -E -a
# -C -Q $TPN -k -E -a

# lfs setstripe -c 4 /lustre/xrscratch1/aparga/ior/${numNodes}_nto1_posix
# lfs setstripe -c 4 /lustre/xrscratch1/aparga/ior/${numNodes}_nto1_MPIIO

###################################################################
# POST
# -F -v  $SIZE -s $SEGMENTS -t 1M -D 30 -r
# -F -v  $SIZE -s $SEGMENTS -t 1M -D 180 -w #WRITE
# -v  -b $SIZE -s $SEGMENTS -t 1M -D 180 -w
# -v  -b $SIZE -s $SEGMENTS -t 1M -D 45 -r

###################################################################
# PER NODE READ WRITE
###################################################################
cd $PREFIX

title="per_node"
echo -e "START $separator"
echo -e "$separator"
echo "WRITE: $title"
cd $WORKING_DIR

prearg='-k -e -a'
postarg='-F -v -b $SIZE -s $SEGMENTS -t 1M -D 180 -w'
runior $title "POSIX"
echo -e "$separator"
runior $title "MPIIO"

echo -e "$separator"
echo "READ: $title"
prearg="-C -Q ${TPN} -k -E -a"
postarg="-F -v -b $SIZE -s $SEGMENTS -t 1M -D 30 -r"
runior $title "POSIX" $prearg $postarg
echo -e "$separator"
runior $title "MPIIO" $prearg $postarg

###################################################################
# SHARED READ WRITE
###################################################################

title="shared"

echo -e "$separator"
echo -e "$separator"
echo "WRITE: $title"

prearg='-k -e -E -a'
postarg='-v -b ${SIZE} -s ${SEGMENTS} -t 1M -D 180 -w'
lfs setstripe -c 4 ${WORKING_DIR}/${NNODES}_POSIX_${title}
runior $title "POSIX" $prearg $postarg
echo -e "$separator"
lfs setstripe -c 4 ${WORKING_DIR}/${NNODES}_MPIIO_${title}
runior $title "MPIIO" $prearg $postarg

echo -e "$separator"
echo "READ: $title"
prearg="-C -Q ${TPN} -k -E -a"
postarg="-v -b $SIZE -s $SEGMENTS -t 1M -D 45 -r"
runior $title "POSIX" $prearg $postarg
echo -e "$separator"
runior $title "MPIIO" $prearg $postarg

mv $WORKING_DIR ${HOME}/ior_current

echo "Results preserved:"
echo "  ${HOME}/ior_current"
echo $(date) >> ${HOME}/ior_current/run_summary
env >> ${HOME}/ior_current/run_summary

echo -e "END $separator"
cd $THISDIR
