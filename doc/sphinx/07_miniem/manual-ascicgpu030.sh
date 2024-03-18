#!/bin/bash
# This submits a full sweep

# setup environment
source /projects/sems/modulefiles/utils/sems-modules-init.sh
module load sems-cmake/3.24.3 sems-gcc/10.1.0 sems-openmpi sems-cuda sems-boost sems-netcdf-c sems-parallel-netcdf sems-hdf5

export NUM_DUPLICATES=8
export APP_REPEAT=1

for (( i=0 ; i<${NUM_DUPLICATES} ; i++ )) ; do

    export MINIEM_IS_KOKKOS_TOOLS="no"

    export MINIEM_SIZE=40
    export RANKS_PER_DOMAIN=1
    export MINIEM_STEPS=3603
    export SLURM_JOB_ID="single${MINIEM_SIZE}at${MINIEM_STEPS}r${RANDOM}"
    sleep 0.2
    ./run-ascicgpu030.sh

    export MINIEM_SIZE=38
    export RANKS_PER_DOMAIN=1
    export MINIEM_STEPS=4254
    export SLURM_JOB_ID="single${MINIEM_SIZE}at${MINIEM_STEPS}r${RANDOM}"
    sleep 0.2
    ./run-ascicgpu030.sh

    export MINIEM_SIZE=35
    export RANKS_PER_DOMAIN=1
    export MINIEM_STEPS=4749
    export SLURM_JOB_ID="single${MINIEM_SIZE}at${MINIEM_STEPS}r${RANDOM}"
    sleep 0.2
    ./run-ascicgpu030.sh

    export MINIEM_IS_KOKKOS_TOOLS="no"
    export MINIEM_SIZE=31
    export RANKS_PER_DOMAIN=1
    export MINIEM_STEPS=5389
    export SLURM_JOB_ID="single${MINIEM_SIZE}at${MINIEM_STEPS}r${RANDOM}"
    sleep 0.2
    ./run-ascicgpu030.sh

    export MINIEM_SIZE=25
    export RANKS_PER_DOMAIN=1
    export MINIEM_STEPS=4454
    export SLURM_JOB_ID="single${MINIEM_SIZE}at${MINIEM_STEPS}r${RANDOM}"
    sleep 0.2
    ./run-ascicgpu030.sh

    export MINIEM_SIZE=20
    export RANKS_PER_DOMAIN=1
    export MINIEM_STEPS=6458
    export SLURM_JOB_ID="single${MINIEM_SIZE}at${MINIEM_STEPS}r${RANDOM}"
    sleep 0.2
    ./run-ascicgpu030.sh

done

exit 0
