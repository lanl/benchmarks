#!/bin/bash
# This submits a full sweep

export NUM_DUPLICATES=1
export APP_REPEAT=1
# export SBATCH_OPTS="--core-spec=0 --exclude=nid006219,nid005850,nid003658,nid001813,nid001451,nid002233,nid005892,nid005896,nid003804,nid002065,nid001855,nid005912,nid006402,nid005723,nid006615,nid005851,nid005356"
export SBATCH_OPTS="--core-spec=0 --partition=hbm"

for (( i=0 ; i<${NUM_DUPLICATES} ; i++ )) ; do

    export MINIEM_IS_KOKKOS_TOOLS="no"

    export MINIEM_SIZE=70
    export RANKS_PER_DOMAIN=14
    export MINIEM_STEPS=1263
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mapcpu.sh

    export RANKS_PER_DOMAIN=11
    export MINIEM_STEPS=1541
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mapcpu.sh

    export RANKS_PER_DOMAIN=7
    export MINIEM_STEPS=1192
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mapcpu.sh

    export RANKS_PER_DOMAIN=4
    export MINIEM_STEPS=812
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mapcpu.sh

    export RANKS_PER_DOMAIN=1
    export MINIEM_STEPS=267
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mapcpu.sh

    export MINIEM_SIZE=60
    export RANKS_PER_DOMAIN=14
    export MINIEM_STEPS=2490
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mapcpu.sh

    export RANKS_PER_DOMAIN=11
    export MINIEM_STEPS=2349
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mapcpu.sh

    export RANKS_PER_DOMAIN=7
    export MINIEM_STEPS=2135
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mapcpu.sh

    export RANKS_PER_DOMAIN=4
    export MINIEM_STEPS=1285
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mapcpu.sh

    export RANKS_PER_DOMAIN=1
    export MINIEM_STEPS=421
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mapcpu.sh

    export MINIEM_SIZE=40
    export RANKS_PER_DOMAIN=14
    export MINIEM_STEPS=5855
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mapcpu.sh

    export RANKS_PER_DOMAIN=11
    export MINIEM_STEPS=6019
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mapcpu.sh

    export RANKS_PER_DOMAIN=7
    export MINIEM_STEPS=5804
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mapcpu.sh

    export RANKS_PER_DOMAIN=4
    export MINIEM_STEPS=4775
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mapcpu.sh

    export RANKS_PER_DOMAIN=1
    export MINIEM_STEPS=1563
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mapcpu.sh

done

exit 0
