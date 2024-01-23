#!/bin/bash
# This submits a full sweep

export NUM_DUPLICATES=8
export APP_REPEAT=8
export SBATCH_OPTS="--core-spec=0 --partition=hbm"

for (( i=0 ; i<${NUM_DUPLICATES} ; i++ )) ; do

    export SPARTA_IS_KOKKOS_TOOLS="no"

    export APP_NAME="spa_crossroads_serial_spr"
    export SPARTA_PPC=35
    export RANKS_PER_DOMAIN=14
    export SPARTA_RUN=6866
    export SPARTA_STATS=100

    export NODES=96
    export DIR_TAG="scale--` printf '%04d' $NODES `"
    export SPARTA_L=` echo "sqrt(${NODES})" | bc -l `
    sleep 0.2
    sbatch --nodes=${NODES} ${SBATCH_OPTS} scale-crossroads-mapcpu.sh

    export NODES=64
    export DIR_TAG="scale--` printf '%04d' $NODES `"
    export SPARTA_L=` echo "sqrt(${NODES})" | bc -l `
    sleep 0.2
    sbatch --nodes=${NODES} ${SBATCH_OPTS} scale-crossroads-mapcpu.sh

    export NODES=32
    export DIR_TAG="scale--` printf '%04d' $NODES `"
    export SPARTA_L=` echo "sqrt(${NODES})" | bc -l `
    sleep 0.2
    sbatch --nodes=${NODES} ${SBATCH_OPTS} scale-crossroads-mapcpu.sh

    export NODES=16
    export DIR_TAG="scale--` printf '%04d' $NODES `"
    export SPARTA_L=` echo "sqrt(${NODES})" | bc -l `
    sleep 0.2
    sbatch --nodes=${NODES} ${SBATCH_OPTS} scale-crossroads-mapcpu.sh

    export NODES=8
    export DIR_TAG="scale--` printf '%04d' $NODES `"
    export SPARTA_L=` echo "sqrt(${NODES})" | bc -l `
    sleep 0.2
    sbatch --nodes=${NODES} ${SBATCH_OPTS} scale-crossroads-mapcpu.sh

    export NODES=4
    export DIR_TAG="scale--` printf '%04d' $NODES `"
    export SPARTA_L=` echo "sqrt(${NODES})" | bc -l `
    sleep 0.2
    sbatch --nodes=${NODES} ${SBATCH_OPTS} scale-crossroads-mapcpu.sh

    export NODES=2
    export DIR_TAG="scale--` printf '%04d' $NODES `"
    export SPARTA_L=` echo "sqrt(${NODES})" | bc -l `
    sleep 0.2
    sbatch --nodes=${NODES} ${SBATCH_OPTS} scale-crossroads-mapcpu.sh

    export NODES=1
    export DIR_TAG="scale--` printf '%04d' $NODES `"
    export SPARTA_L=` echo "sqrt(${NODES})" | bc -l `
    sleep 0.2
    sbatch --nodes=${NODES} ${SBATCH_OPTS} scale-crossroads-mapcpu.sh

done

exit 0
