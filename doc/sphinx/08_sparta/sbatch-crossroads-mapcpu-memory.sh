#!/bin/bash
# This submits a full sweep

export NUM_DUPLICATES=1
export APP_REPEAT=1
export SBATCH_OPTS="--core-spec=0 --partition=hbm"

for (( i=0 ; i<${NUM_DUPLICATES} ; i++ )) ; do

    export SPARTA_IS_KOKKOS_TOOLS="no"
    export DIR_TAG="memory"

    export APP_NAME="spa_crossroads_serial_spr"

    export SPARTA_PPC=55
    export RANKS_PER_DOMAIN=14
    export SPARTA_RUN=4346
    export SPARTA_STATS=100
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mapcpu-memory.sh

    export RANKS_PER_DOMAIN=11
    export SPARTA_RUN=3924
    export SPARTA_STATS=90
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mapcpu-memory.sh

    export RANKS_PER_DOMAIN=7
    export SPARTA_RUN=3095
    export SPARTA_STATS=70
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mapcpu-memory.sh

    export RANKS_PER_DOMAIN=4
    export SPARTA_RUN=1981
    export SPARTA_STATS=40
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mapcpu-memory.sh

    export RANKS_PER_DOMAIN=1
    export SPARTA_RUN=500
    export SPARTA_STATS=10
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mapcpu-memory.sh

    export SPARTA_PPC=35
    export RANKS_PER_DOMAIN=14
    export SPARTA_RUN=6866
    export SPARTA_STATS=100
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mapcpu-memory.sh

    export RANKS_PER_DOMAIN=11
    export SPARTA_RUN=6406
    export SPARTA_STATS=100
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mapcpu-memory.sh

    export RANKS_PER_DOMAIN=7
    export SPARTA_RUN=4818
    export SPARTA_STATS=100
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mapcpu-memory.sh

    export RANKS_PER_DOMAIN=4
    export SPARTA_RUN=2882
    export SPARTA_STATS=60
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mapcpu-memory.sh

    export RANKS_PER_DOMAIN=1
    export SPARTA_RUN=790
    export SPARTA_STATS=10
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mapcpu-memory.sh

    export SPARTA_PPC=15
    export RANKS_PER_DOMAIN=14
    export SPARTA_RUN=15193
    export SPARTA_STATS=300
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mapcpu-memory.sh

    export RANKS_PER_DOMAIN=11
    export SPARTA_RUN=14405
    export SPARTA_STATS=300
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mapcpu-memory.sh

    export RANKS_PER_DOMAIN=7
    export SPARTA_RUN=11067
    export SPARTA_STATS=200
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mapcpu-memory.sh

    export RANKS_PER_DOMAIN=4
    export SPARTA_RUN=6824
    export SPARTA_STATS=100
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mapcpu-memory.sh

    export RANKS_PER_DOMAIN=1
    export SPARTA_RUN=1779
    export SPARTA_STATS=40
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mapcpu-memory.sh

done

exit 0
