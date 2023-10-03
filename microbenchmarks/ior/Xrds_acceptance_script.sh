#!/bin/bash

## SET VARS FROM CL INPUT KEY=VALUE
setvar () {
	while [[ $# -gt 0 ]]; do
		export $1
		shift
	done
}

export SCRATCH=/lustre/xrscratch1/${USER}
export WORKING_DIR=${SCRATCH}/ior
export IORLOG=${HOME}/logs/ior
export IORLOC=
export TPN=110
export SEGMENTS=16
export SIZE=2G
export NNODES=$SLURM_NNODES
separator="\t ------------------------------------------------- \t"

###################################################################
## ANY VARIABLES SET THAT APPEAR BEFORE THIS CAN BE
## SET ON THE COMMAND LINE WITH KEY=VALUE.
###################################################################
setvar $@

find ${WORKING_DIR} -type f -delete
mkdir -p $IORLOG

runior() {
    # 1 is working dir output file
    # 2 is POSIX or MPIIO
    # 3 is PRE 2 args
    # 4 is POST 2 args
    srun -N ${NNODES} --ntasks-per-node=${TPN} ${IORLOC}/ior "${3}" $2 "${4}" -o ${WORKING_DIR}/${NNODES}_${2}_${1}
    sleep 3
}

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
# -F -v  -b 4G -s 16 -t 1M -D 30 -r 
# -F -v  -b 4G -s 16 -t 1M -D 180 -w #WRITE 
# -v  -b $size -s $segments -t 1M -D 180 -w
# -v  -b $size -s $segments -t 1M -D 45 -r 

###################################################################
# PER NODE READ WRITE
###################################################################

title="per_node"
echo -e "START $separator"
echo -e "$separator"
echo "WRITE: $title"

prearg="-k -e -a"
postarg="-F -v -b 4G -s 16 -t 1M -D 180 -w"
runior $title "POSIX" $prearg $postarg
echo -e "$separator"
runior $title "MPIIO" $prearg $postarg

echo -e "$separator"
echo "READ: $title"
prearg="-C -Q ${TPN} -k -E -a"
postarg="-F -v -b 4G -s 16 -t 1M -D 30 -r"
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

prearg="-k -e -E -a"
postarg="-v -b $size -s $segments -t 1M -D 180 -w"
lfs setstripe -c 4 ${WORKING_DIR}/${NNODES}_POSIX_${title}
runior $title "POSIX" $prearg $postarg
echo -e "$separator"
lfs setstripe -c 4 ${WORKING_DIR}/${NNODES}_MPIIO_${title}
runior $title "MPIIO" $prearg $postarg

echo -e "$separator"
echo "READ: $title"
prearg="-C -Q ${TPN} -k -E -a"
postarg="-v -b $size -s $segments -t 1M -D 45 -r"
runior $title "POSIX" $prearg $postarg
echo -e "$separator"
runior $title "MPIIO" $prearg $postarg

echo -e "END $separator"

