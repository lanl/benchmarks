#!/bin/bash
# This submits a full sweep

export NUM_DUPLICATES=1
export APP_REPEAT=1
export SBATCH_OPTS="--core-spec=0 --partition=hbm"

for (( i=0 ; i<${NUM_DUPLICATES} ; i++ )) ; do

    export SPARTA_IS_KOKKOS_TOOLS="no"
    export DIR_TAG="single"
    export APP_NAME="spa_crossroads_serial_spr"

    export SPARTA_PPC=35
    export RANKS_PER_DOMAIN=14
    export SPARTA_RUN=6866
    export SPARTA_STATS=100
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mapcpu.sh

done

exit 0
