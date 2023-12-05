#!/bin/bash
# This submits a full sweep

export NUM_DUPLICATES=1
export APP_REPEAT=1
# export SBATCH_OPTS="--core-spec=0 --exclude=nid006219,nid005850,nid003658,nid001813,nid001451,nid002233,nid005892,nid005896,nid003804,nid002065,nid001855,nid005912,nid006402,nid005723,nid006615,nid005851,nid005356"
export SBATCH_OPTS="--core-spec=0 --partition=hbm"

for (( i=0 ; i<${NUM_DUPLICATES} ; i++ )) ; do

    export SPARTA_IS_KOKKOS_TOOLS="no"

    export APP_NAME="spa_crossroads_omp_spr"

    export SPARTA_PPC=55
    export RANKS_PER_DOMAIN=14
    export SPARTA_RUN=4346
    export SPARTA_STATS=100
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mapcpu.sh

    export RANKS_PER_DOMAIN=11
    export SPARTA_RUN=3924
    export SPARTA_STATS=90
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mapcpu.sh

    export RANKS_PER_DOMAIN=7
    export SPARTA_RUN=3095
    export SPARTA_STATS=70
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mapcpu.sh

    export RANKS_PER_DOMAIN=4
    export SPARTA_RUN=1981
    export SPARTA_STATS=40
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mapcpu.sh

    export RANKS_PER_DOMAIN=1
    export SPARTA_RUN=500
    export SPARTA_STATS=10
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mapcpu.sh

    export SPARTA_PPC=35
    export RANKS_PER_DOMAIN=14
    export SPARTA_RUN=6866
    export SPARTA_STATS=100
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mapcpu.sh

    export RANKS_PER_DOMAIN=11
    export SPARTA_RUN=6406
    export SPARTA_STATS=100
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mapcpu.sh

    export RANKS_PER_DOMAIN=7
    export SPARTA_RUN=4818
    export SPARTA_STATS=100
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mapcpu.sh

    export RANKS_PER_DOMAIN=4
    export SPARTA_RUN=2882
    export SPARTA_STATS=60
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mapcpu.sh

    export RANKS_PER_DOMAIN=1
    export SPARTA_RUN=790
    export SPARTA_STATS=10
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mapcpu.sh

    export SPARTA_PPC=15
    export RANKS_PER_DOMAIN=14
    export SPARTA_RUN=15193
    export SPARTA_STATS=300
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mapcpu.sh

    export RANKS_PER_DOMAIN=11
    export SPARTA_RUN=14405
    export SPARTA_STATS=300
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mapcpu.sh

    export RANKS_PER_DOMAIN=7
    export SPARTA_RUN=11067
    export SPARTA_STATS=200
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mapcpu.sh

    export RANKS_PER_DOMAIN=4
    export SPARTA_RUN=6824
    export SPARTA_STATS=100
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mapcpu.sh

    export RANKS_PER_DOMAIN=1
    export SPARTA_RUN=1779
    export SPARTA_STATS=40
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mapcpu.sh

done

exit 0
