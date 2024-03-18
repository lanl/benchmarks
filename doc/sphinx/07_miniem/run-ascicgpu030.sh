#!/bin/bash
#SBATCH --job-name=miniem
#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --core-spec=0  # this seemingly needs to be done with sbatch
##SBATCH --exclude=nid006219,nid005850,nid003658,nid001813,nid001451,nid002233,nid005892,nid005896,nid003804,nid002065,nid001855,nid005912,nid006402,nid005723,nid006615,nid005851,nid005356

umask 022
ulimit -c unlimited
set -e
set -x

# directory and log file setup
export SLURM_JOB_ID=${SLURM_JOB_ID:-424242}
export DIR_BASE="`pwd -P`"
export DIR_ROOT="`git rev-parse --show-toplevel`"

export DAYSTAMP="`date '+%Y%m%d'`"
export SECSTAMP="`date '+%Y%m%d_%H%M%S'`"
export FULLUNIQ="${SECSTAMP}_${RANDOM}"
export DIR_CASE="${SECSTAMP}_${SLURM_JOB_ID}"
export DIR_RUN="run-${DIR_CASE}"
export TMPDIR="/tmp/$SLURM_JOB_ID"  # set to avoid potential runtime error
export FILE_LOG="output-mpirun-${DIR_CASE}.log"
export FILE_METRICS="output-metrics-${DIR_CASE}.csv"
export FILE_ENV="output-environment-${DIR_CASE}.txt"
export FILE_STATE="output-state-${DIR_CASE}.log"
export FILE_TRY="output-script-${DIR_CASE}.log"
export FILE_TIME="output-time-${DIR_CASE}.txt"

# MiniEM setup
export APP_NAME="PanzerMiniEM_BlockPrec.exe"
export APP_INPUT="maxwell-large.xml"
export APP_REPEAT=${APP_REPEAT:-1}
export APP_EXE="${DIR_BASE}/${APP_NAME}"
if test ! -f "${APP_EXE}" ; then
    # manage Spack's installation hierarchy
    APP_LIST=( ` find "${DIR_ROOT}/miniem_build" -name "${APP_NAME}" -type f ` )
    APP_TOCOPY=` ls -t1 ${APP_LIST[@]} | head -n 1 `
    cp -a "${APP_TOCOPY}" "${DIR_BASE}"
    cp -a ` dirname "${APP_TOCOPY}" `/*.xml "${DIR_BASE}"
fi
export MINIEM_SIZE="${MINIEM_SIZE:-40}"
export MINIEM_STEPS="${MINIEM_STEPS:-450}"
export MINIEM_IS_KOKKOS_TOOLS="${MINIEM_IS_KOKKOS_TOOLS:-no}"

# MPI & hardware setup
export NODES=${NODES:-1}
export RANKS_PER_DOMAIN=${RANKS_PER_DOMAIN:-14}
export SOCKETS_PER_NODE=${SOCKETS_PER_NODE:-2}
export DOMAINS_PER_SOCKET=${DOMAINS_PER_SOCKET:-4}
export RANKS_PER_SOCKET=$(( $RANKS_PER_DOMAIN * $DOMAINS_PER_SOCKET ))
export RANKS_PER_NODE=$(( $RANKS_PER_DOMAIN * $DOMAINS_PER_SOCKET * $SOCKETS_PER_NODE ))
export RANKS_PER_JOB=$(( $RANKS_PER_NODE * $NODES ))
export MINIEM_RANK_BIND="`${DIR_ROOT}/map_cpu/map_cpu ${RANKS_PER_NODE} 2>/dev/null`"

# Thread setup
export SWTHREADS_PER_RANK=${SWTHREADS_PER_RANK:-1}
export HWTHREADS_PER_CORE=${HWTHREADS_PER_CORE:-2}
export PLACEHOLDERS_PER_RANK=$(( $SWTHREADS_PER_RANK * $HWTHREADS_PER_CORE ))
export PLACEHOLDERS_PER_RANK=1
export PES_PER_NODE=$(( $RANKS_PER_NODE * $SWTHREADS_PER_RANK ))

# OpenMP setup
export OMP_NUM_THREADS=${SWTHREADS_PER_RANK}
# export GOMP_CPU_AFFINITY=${MINIEM_RANK_BIND}  # "112-223"
export OMP_PROC_BIND=spread #TRUE #spread
export OMP_PLACES=threads #'{1,2}' #threads
# export SRUN_CPUS_PER_TASK=${SRUN_CPUS_PER_TASK:-28}
# export OMP_DISPLAY_ENV=VERBOSE
export OMP_STACKSIZE=2G
export MPICH_MAX_THREAD_SAFETY=multiple
export FI_CXI_RX_MATCH_MODE=software

# Kokkos Tools
if test "${MINIEM_IS_KOKKOS_TOOLS}" = "yes" ; then
    export KOKKOS_TOOLS_LIBS="${DIR_ROOT}/kokkos-tools/profiling/space-time-stack/kp_space_time_stack.so"
else
    unset KOKKOS_TOOLS_LIBS
fi

# LDPXI setup if applicable
export LDPXI_INST="gru,io,mpi,rank,mem"
# export LDPXI_PERF_EVENTS="cpu-cycles,ref-cycles,dTLB-load-misses,dTLB-loads"
export LDPXI_OUTPUT="miniem-ldpxi.$SLURM_JOB_ID.csv"

# Create and populate run folder and edit input file
prep_toplevel_run()
{
    mkdir -p "${DIR_BASE}/${DIR_RUN}"
    cd "${DIR_BASE}/${DIR_RUN}"
    cp -a "${DIR_BASE}"/*.xml ./
    sed s?"<Parameter name=\"X Elements\" type=\"int\" value=\"40\" />"?"<Parameter name=\"X Elements\" type=\"int\" value=\"${MINIEM_SIZE}\" />"?g ${APP_INPUT} > tmp
    mv tmp ${APP_INPUT}
    sed s?"<Parameter name=\"Y Elements\" type=\"int\" value=\"40\" />"?"<Parameter name=\"Y Elements\" type=\"int\" value=\"${MINIEM_SIZE}\" />"?g ${APP_INPUT} > tmp
    mv tmp ${APP_INPUT}
    sed s?"<Parameter name=\"Z Elements\" type=\"int\" value=\"40\" />"?"<Parameter name=\"Z Elements\" type=\"int\" value=\"${MINIEM_SIZE}\" />"?g ${APP_INPUT} > tmp
    mv tmp ${APP_INPUT}
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
    ln -s ../*.xml ./
    dir_current_work="` pwd -P `"

    # single-thread case is nice
    export MINIEM_CPU_BIND=""
    if test ${SWTHREADS_PER_RANK} -eq 1 ; then
        export MINIEM_CPU_BIND=",map_cpu:${MINIEM_RANK_BIND}"
    fi

    # /usr/bin/time --verbose --output="${FILE_TIME}" \
    # time \
        mpirun \
            -np 1 \
            "${APP_EXE}" \
                --stacked-timer \
                --solver=MueLu \
                --numTimeSteps=${MINIEM_STEPS} \
                --linAlgebra=Tpetra \
                --inputFile="${dir_current_work}/${APP_INPUT}" \
            >"${FILE_LOG//.log/-${i}.log}"
    popd
    date
}
export -f run_try

for (( i=0; i<${APP_REPEAT}; i++ )) ; do
    echo "INFO: Perform Simulation #${i} at ` date `"
    run_try $i >"${FILE_TRY//.log/-${i}.log}" 2>&1
done
                        
# mv ../slurm-${SLURM_JOBID}.out .

exit 0
