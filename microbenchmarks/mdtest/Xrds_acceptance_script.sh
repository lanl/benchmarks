#!/bin/bash

##############
# MDTEST
##############

## SET VARS FROM CL INPUT KEY=VALUE
setvar () {
	while [[ $# -gt 0 ]]; do
		export $1
		shift
	done
}

export SCRATCH=/lustre/xrscratch1 # CROSSROADS SCRATCH
export MDTESTLOC=${HOME}/.local/bin # PATH TO MDTEST BINARY
export TPN=10
export NFILES=104857
export NNODES=$SLURM_NNODES
separator="\t ------------------------------------------------- \t"

###################################################################
## ANY VARIABLES SET THAT APPEAR BEFORE THIS CAN BE
## SET ON THE COMMAND LINE WITH KEY=VALUE.
###################################################################
setvar $@

export SCRATCH_HOME=${SCRATCH}/${USER}
export WORKING_DIR=${SCRATCH_HOME}/mdtest
taaskspn="--ntasks-per-node=${TPN}"

find ${WORKING_DIR} -type f -delete

runmdtest() {
    
    # Command line used: /users/aparga/xr/bin/mdtest '-F' '-I' '104857' '-d=/lustre/xrscratch1/aparga/mdtest'
    srun -N ${NNODES} $taskspn ${MDTESTLOC}/mdtest -F -I $1 -d=${WORKING_DIR}
    sleep 3
}

###################################################################
# PER NODE READ WRITE
###################################################################

title="per_node"
echo -e "START $separator"
echo -e "$separator"
echo "WRITE: $title"

runmdtest $title "POSIX" $prearg $postarg
echo -e "$separator"
runmdtest $title "MPIIO" $prearg $postarg

echo -e "$separator"
echo "READ: $title"
prearg="-C -Q ${TPN} -k -E -a"
postarg="-F -v -b 4G -s 16 -t 1M -D 30 -r"
runmdtest $title "POSIX" $prearg $postarg
echo -e "$separator"
runmdtest $title "MPIIO" $prearg $postarg

###################################################################
# SHARED READ WRITE
###################################################################

title="shared"

echo -e "$separator"
echo -e "$separator"
echo "WRITE: $title"

lfs setstripe -c 4 ${WORKING_DIR}/${NNODES}_POSIX_${title}
runmdtest $title "POSIX" $prearg $postarg
echo -e "$separator"
lfs setstripe -c 4 ${WORKING_DIR}/${NNODES}_MPIIO_${title}
runmdtest $title "MPIIO" $prearg $postarg

echo -e "$separator"
echo "READ: $title"
prearg="-C -Q ${TPN} -k -E -a"
postarg="-v -b $size -s $segments -t 1M -D 45 -r"
runmdtest $title "POSIX" $prearg $postarg
echo -e "$separator"
runmdtest $title "MPIIO" $prearg $postarg

mv $WORKING_DIR ${HOME}/mdtest_current

echo "Results preserved:"
echo "  ${HOME}/mdtest_current"
echo $(date) >> ${HOME}/mdtest_current/run_summary
env >> ${HOME}/mdtest_current/run_summary
echo -e "END $separator"

