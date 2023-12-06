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

# directory and log file setup
export SLURM_JOB_ID=${SLURM_JOB_ID:-424242}
export DIR_BASE="`pwd -P`"
export DIR_ROOT="`git rev-parse --show-toplevel`"
export DIR_SRC="${DIR_ROOT}/sparta"
export DIR_EXE="${DIR_SRC}/src"
export DAYSTAMP="`date '+%Y%m%d'`"
export SECSTAMP="`date '+%Y%m%d_%H%M%S'`"
export FULLUNIQ="${SECSTAMP}_${RANDOM}"
export DIR_CASE="${SECSTAMP}_${SLURM_JOB_ID}"
export DIR_RUN="run-${DIR_CASE}"
export TMPDIR="/tmp/$SLURM_JOB_ID"  # set to avoid potential runtime error
export FILE_LOG="output-srun-${DIR_CASE}.log"
export FILE_METRICS="output-metrics-${DIR_CASE}.csv"
export FILE_ENV="output-environment-${DIR_CASE}.txt"
export FILE_STATE="output-state-${DIR_CASE}.log"
export FILE_TRY="output-script-${DIR_CASE}.log"
export FILE_TIME="output-time-${DIR_CASE}.txt"

# SPARTA setup
# 130   14  1356    30
export SPARTA_RUN=${SPARTA_RUN:-1000}
export SPARTA_PPC=${SPARTA_PPC:-24}
export SPARTA_STATS=${SPARTA_STATS:-10}
export SPARTA_IS_KOKKOS_TOOLS="${SPARTA_IS_KOKKOS_TOOLS:-no}"
export APP_REPEAT=${APP_REPEAT:-1}
export APP_NAME=${APP_NAME:-"spa_crossroads_omp_skx"}
export APP_EXE="${DIR_BASE}/${APP_NAME}"

# MPI & hardware setup
export SLURM_JOB_NUM_NODES=${SLURM_JOB_NUM_NODES:-1}
export NODES=${NODES:-$SLURM_JOB_NUM_NODES}
export RANKS_PER_DOMAIN=${RANKS_PER_DOMAIN:-14}
export SOCKETS_PER_NODE=${SOCKETS_PER_NODE:-2}
export DOMAINS_PER_SOCKET=${DOMAINS_PER_SOCKET:-4}
export RANKS_PER_SOCKET=$(( $RANKS_PER_DOMAIN * $DOMAINS_PER_SOCKET ))
export RANKS_PER_NODE=$(( $RANKS_PER_DOMAIN * $DOMAINS_PER_SOCKET * $SOCKETS_PER_NODE ))
export RANKS_PER_JOB=$(( $RANKS_PER_NODE * $NODES ))
export SPARTA_RANK_BIND="`${DIR_ROOT}/map_cpu/map_cpu ${RANKS_PER_NODE} 2>/dev/null`"

# Thread setup
export SWTHREADS_PER_RANK=${SWTHREADS_PER_RANK:-1}
export HWTHREADS_PER_CORE=${HWTHREADS_PER_CORE:-2}
export PLACEHOLDERS_PER_RANK=$(( $SWTHREADS_PER_RANK * $HWTHREADS_PER_CORE ))
export PLACEHOLDERS_PER_RANK=1
export PES_PER_NODE=$(( $RANKS_PER_NODE * $SWTHREADS_PER_RANK ))

# OpenMP setup
export OMP_NUM_THREADS=${SWTHREADS_PER_RANK}
# export GOMP_CPU_AFFINITY=${SPARTA_RANK_BIND}  # "112-223"
export OMP_PROC_BIND=spread #TRUE #spread
export OMP_PLACES=threads #'{1,2}' #threads
# export SRUN_CPUS_PER_TASK=${SRUN_CPUS_PER_TASK:-28}
# export OMP_DISPLAY_ENV=VERBOSE
export OMP_STACKSIZE=2G
export MPICH_MAX_THREAD_SAFETY=multiple
export FI_CXI_RX_MATCH_MODE=software

# Kokkos Tools
if test "${SPARTA_IS_KOKKOS_TOOLS}" = "yes" ; then
    export KOKKOS_TOOLS_LIBS="${DIR_ROOT}/kokkos-tools/profiling/space-time-stack/kp_space_time_stack.so"
else
    unset KOKKOS_TOOLS_LIBS
fi

# LDPXI setup if applicable
# export LDPXI_INST="gru,io,mpi,rank,mem"
# export LDPXI_PERF_EVENTS="cpu-cycles,ref-cycles,dTLB-load-misses,dTLB-loads"
# export LDPXI_OUTPUT="sparta-ldpxi.$SLURM_JOB_ID.csv"

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

# address node starting state
fix_state()
{
    set -x
    srun --drop-caches=pagecache  -N $SLURM_NNODES -n $SLURM_NNODES --ntasks-per-node=1 true
    srun --drop-caches=slab       -N $SLURM_NNODES -n $SLURM_NNODES --ntasks-per-node=1 true
    srun --vm-compact-atom=enable -N $SLURM_NNODES -n $SLURM_NNODES --ntasks-per-node=1 true
    set +x
}
export -f fix_state
fix_state >"${FILE_STATE}" 2>&1

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

    # single-thread case is nice
    export SPARTA_CPU_BIND=""
    if test ${SWTHREADS_PER_RANK} -eq 1 ; then
        export SPARTA_CPU_BIND=",map_cpu:${SPARTA_RANK_BIND}"
    fi

    # export LD_PRELOAD=libldpxi_mpi.so
    # export LD_PRELOAD=/usr/projects/hpctest/amagela/ldpxi/ldpxi/install/ats3/ldpxi-1.0.1/intel+cray-mpich-8.1.25/lib/libldpxi_mpi.so.1.0.1
    # export LD_PRELOAD=/usr/projects/hpctest/amagela/ats-5/LDPXI/xr/libldpxi.so
    # /usr/bin/time --verbose --output="${FILE_TIME}" \
    # time \
        srun \
            --unbuffered \
            --ntasks=$RANKS_PER_JOB \
            --ntasks-per-node=$RANKS_PER_NODE \
            --cpu-bind=verbose${SPARTA_CPU_BIND} \
            --distribution=block:block \
            --output="${FILE_LOG//.log/-${i}.log}" \
            "${APP_EXE}" \
                -k on t ${OMP_NUM_THREADS} -sf kk \
                -in "in.cylinder"
    # unset LD_PRELOAD
#             --ntasks-per-socket=$RANKS_PER_SOCKET \
#             --hint=nomultithread \
#             --cpu-bind=verbose,ldoms \
#             --cpus-per-task=$SRUN_CPUS_PER_TASK \
#             --cpus-per-task=${PLACEHOLDERS_PER_RANK} \
#             --distribution=block:cyclic \
#             --cpu-bind="map_cpu:`${DIR_ROOT}/map_cpu/map_cpu ${PES_PER_NODE} 2>/dev/null`" \

    # post process
    l_fom=`./sparta_fom.py -a -f "log.sparta" 2>&1 | awk -F'FOM = ' '{print $2}'`
    # l_maxrss=`grep maxrss "${LDPXI_OUTPUT}" | awk -F',' '{print $2}'`
    # echo "FOM,RUN,PPC,RanksPerDomain,MaxRSS(KiB),AppName,Try,Dir" > "${FILE_METRICS}"
    # echo "${l_fom},${SPARTA_RUN},${SPARTA_PPC},${RANKS_PER_DOMAIN},${l_maxrss},${APP_NAME},${i},` pwd -P `" >> "${FILE_METRICS}"
    echo "FOM,RUN,PPC,RanksPerDomain,AppName,Try,Dir" > "${FILE_METRICS}"
    echo "${l_fom},${SPARTA_RUN},${SPARTA_PPC},${RANKS_PER_DOMAIN},${APP_NAME},${i},` pwd -P `" >> "${FILE_METRICS}"

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
