#!/bin/bash
# This submits a full sweep

export NUM_DUPLICATES=8
export APP_REPEAT=8
export BSUB_OPTIONS="-W 04:00 -core_isolation 2 -J sparta"

for (( i=0 ; i<${NUM_DUPLICATES} ; i++ )) ; do

    export SPARTA_IS_KOKKOS_TOOLS="no"
    export DIR_TAG="single"
    export APP_NAME="spa_vortex_kokkos"

    export SPARTA_PPC=5
    export SPARTA_RUN=13937
    export SPARTA_STATS=300
    export SLURM_JOB_ID=${SPARTA_PPC}${RANDOM}
    sleep 0.2
    bsub ${BSUB_OPTIONS} run-vortex.sh

    export SPARTA_PPC=4
    export SPARTA_RUN=17299
    export SPARTA_STATS=400
    export SLURM_JOB_ID=${SPARTA_PPC}${RANDOM}
    sleep 0.2
    bsub ${BSUB_OPTIONS} run-vortex.sh

    export SPARTA_PPC=3
    export SPARTA_RUN=20536
    export SPARTA_STATS=400
    export SLURM_JOB_ID=${SPARTA_PPC}${RANDOM}
    sleep 0.2
    bsub ${BSUB_OPTIONS} run-vortex.sh

    export SPARTA_PPC=2
    export SPARTA_RUN=28583
    export SPARTA_STATS=600
    export SLURM_JOB_ID=${SPARTA_PPC}${RANDOM}
    sleep 0.2
    bsub ${BSUB_OPTIONS} run-vortex.sh

    export SPARTA_PPC=1
    export SPARTA_RUN=43543
    export SPARTA_STATS=1000
    export SLURM_JOB_ID=${SPARTA_PPC}${RANDOM}
    sleep 0.2
    bsub ${BSUB_OPTIONS} run-vortex.sh

done

exit 0
