#!/bin/bash
#SBATCH --job-name=sparta-L1
#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --core-spec=0  # this seemingly needs to be done with sbatch
##SBATCH --exclude=nid006219,nid005850,nid003658,nid001813,nid001451,nid002233,nid005892,nid005896,nid003804,nid002065,nid001855,nid005912,nid006402,nid005723,nid006615,nid005851,nid005356

umask 022
ulimit -c unlimited
set -e
set -x

# module!
module load cuda/11.2.0
module load gcc/8.3.1

# directory and log file setup
export SLURM_JOB_ID=${SLURM_JOB_ID:-424242}
export DIR_BASE="`pwd -P`"
export DIR_ROOT="`git rev-parse --show-toplevel`"
export DIR_SRC="${DIR_ROOT}/sparta"
export DIR_EXE="${DIR_SRC}/src"
export DIR_TAG="${DIR_TAG:-run}"
export DAYSTAMP="`date '+%Y%m%d'`"
export SECSTAMP="`date '+%Y%m%d_%H%M%S'`"
export FULLUNIQ="${SECSTAMP}_${RANDOM}"
export DIR_CASE="${SECSTAMP}_${SLURM_JOB_ID}"
export DIR_RUN="${DIR_TAG}-${DIR_CASE}"
export TMPDIR="/tmp/$SLURM_JOB_ID"  # set to avoid potential runtime error
export FILE_LOG="output-srun-${DIR_CASE}.log"
export FILE_METRICS="output-metrics-${DIR_CASE}.csv"
export FILE_ENV="output-environment-${DIR_CASE}.txt"
export FILE_STATE="output-state-${DIR_CASE}.log"
export FILE_TRY="output-script-${DIR_CASE}.log"
export FILE_TIME="output-time-${DIR_CASE}.txt"

# SPARTA setup
export SPARTA_PPC=${SPARTA_PPC:-5}
export SPARTA_RUN=${SPARTA_RUN:-1000}
export SPARTA_STATS=${SPARTA_STATS:-1000}
export SPARTA_IS_KOKKOS_TOOLS="${SPARTA_IS_KOKKOS_TOOLS:-no}"
export APP_REPEAT=${APP_REPEAT:-1}
export APP_NAME=${APP_NAME:-"spa_vortex_kokkos"}
export APP_EXE="${DIR_BASE}/${APP_NAME}"

# MPI & hardware setup
export SLURM_JOB_NUM_NODES=${SLURM_JOB_NUM_NODES:-1}
export NODES=${NODES:-$SLURM_JOB_NUM_NODES}
export RANKS_PER_DOMAIN=${RANKS_PER_DOMAIN:-1}
export SOCKETS_PER_NODE=${SOCKETS_PER_NODE:-2}
export DOMAINS_PER_SOCKET=${DOMAINS_PER_SOCKET:-2}
export RANKS_PER_SOCKET=$(( $RANKS_PER_DOMAIN * $DOMAINS_PER_SOCKET ))
export RANKS_PER_NODE=$(( $RANKS_PER_DOMAIN * $DOMAINS_PER_SOCKET * $SOCKETS_PER_NODE ))
export RANKS_PER_JOB=$(( $RANKS_PER_NODE * $NODES ))
export SPARTA_RANK_BIND="`${DIR_ROOT}/map_cpu/map_cpu ${RANKS_PER_NODE} 2>/dev/null`"

# Thread setup
export SWTHREADS_PER_RANK=${SWTHREADS_PER_RANK:-1}
export HWTHREADS_PER_CORE=${HWTHREADS_PER_CORE:-1}
export PLACEHOLDERS_PER_RANK=$(( $SWTHREADS_PER_RANK * $HWTHREADS_PER_CORE ))
export PLACEHOLDERS_PER_RANK=1
export PES_PER_NODE=$(( $RANKS_PER_NODE * $SWTHREADS_PER_RANK ))

# OpenMP setup
# export OMP_NUM_THREADS=${SWTHREADS_PER_RANK}
# export GOMP_CPU_AFFINITY=${SPARTA_RANK_BIND}  # "112-223"
# export OMP_PROC_BIND=spread #TRUE #spread
# export OMP_PLACES=threads #'{1,2}' #threads
# export SRUN_CPUS_PER_TASK=${SRUN_CPUS_PER_TASK:-28}
# export OMP_DISPLAY_ENV=VERBOSE
# export OMP_STACKSIZE=2G
# export MPICH_MAX_THREAD_SAFETY=multiple
# export FI_CXI_RX_MATCH_MODE=software

# Kokkos Tools
if test "${SPARTA_IS_KOKKOS_TOOLS}" = "yes" ; then
    export KOKKOS_TOOLS_LIBS="${DIR_ROOT}/kokkos-tools/profiling/space-time-stack/kp_space_time_stack.so"
else
    unset KOKKOS_TOOLS_LIBS
fi

# Create and populate run folder and edit input file
prep_toplevel_run()
{
    mkdir -p "${DIR_BASE}/${DIR_RUN}"
    cd "${DIR_BASE}/${DIR_RUN}"
    # cp -a "${DIR_SRC}/examples/cylinder/in.cylinder" ./
    awk "\$1 ~ /^run\$/ {\$2 = ${SPARTA_RUN}}1" "${DIR_SRC}/examples/cylinder/in.cylinder" \
        | awk "\$2 ~ /^ppc\$/ {\$4 = ${SPARTA_PPC}}1" \
        | awk "\$1 ~ /^stats\$/ {\$2 = ${SPARTA_STATS}}1" \
        > "./in.cylinder"
    cp -a "${DIR_SRC}/examples/cylinder/circle_R0.5_P10000.surf" ./
    cp -a "${DIR_SRC}"/examples/cylinder/air.* ./
    ln -s ../sparta_fom.py ./
}
export -f prep_toplevel_run
prep_toplevel_run

print_system_env_info()
{
    echo "################"
    echo "INFO: System and environment information"
    echo "    INFO: Date and Time"
    date
    echo "    INFO: modules"
    module list
    echo "    INFO: CPU info"
    # cat /proc/cpuinfo | tail -27
    lscpu
    echo "    INFO: memory info"
    cat /proc/meminfo
    echo "    INFO: SLURM info"
    env | grep -i slurm
    echo "    INFO: HOST info"
    hostname
    echo "    INFO: NUMA info"
    numactl --hardware
}
export -f print_system_env_info
print_system_env_info >"${FILE_ENV}" 2>&1

# do work
run_try()
{
    set -x
    i=$1
    echo "INFO: Perform Simulation #${i}"
    date
    dir_try="try-` printf '%02d' $i `"
    mkdir -p "${dir_try}"
    pushd "${dir_try}"
    ln -s ../air.* ../*.surf ../in.cylinder ./
    ln -s ../sparta_fom.py ./
    # /usr/bin/time --verbose --output="${FILE_TIME}" \
    # time \
    #         -r 1 \  # 1 instance per node
    #         -a 1 \  # MPI ranks
    #         -c 1 \  # cores/rank
    #         -g 1 \  # GPUs
    #         -l gpu-gpu \  # affinity
    #         -d packed  \  # affinity
        jsrun \
            -M "-gpu -disable_gdr" \
            -r 1 \
            -a 1 \
            -c 1 \
            -g 1 \
            -l gpu-gpu \
            -d packed  \
            -o "${FILE_LOG//.log/-${i}-stdout.log}" \
            -k "${FILE_LOG//.log/-${i}-stderr.log}" \
            "${APP_EXE}" \
                -k on g 1 -sf kk \
                -in "in.cylinder"
    popd
    date
}
export -f run_try

for (( i=0; i<${APP_REPEAT}; i++ )) ; do
    echo "INFO: Perform Simulation #${i} at ` date `"
    run_try $i >"${FILE_TRY//.log/-${i}.log}" 2>&1
done
                        
exit 0
