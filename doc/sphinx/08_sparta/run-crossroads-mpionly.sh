#!/bin/bash
#SBATCH --job-name=sparta-L2-S0
#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --core-spec=0  # this seemingly needs to be done with sbatch
#SBATCH --exclude=nid006219,nid005850,nid003658,nid001813,nid001451,nid002233,nid005892,nid005896,nid003804,nid002065,nid001855,nid005912,nid006402,nid005723,nid006615,nid005851,nid005356

umask 022
set -e
ulimit -c unlimited

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

# SPARTA setup
export SPARTA_RUN=${SPARTA_RUN:-4000}
export SPARTA_PPC=${SPARTA_PPC:-42}
export SPARTA_STATS=${SPARTA_STATS:-10}
export APP_NAME=${APP_NAME:-"spa_crossroads_omp_spr"}
export APP_EXE="${DIR_BASE}/${APP_NAME}"
export APP_REPEAT=${APP_REPEAT:-1}

# MPI setup
export NODES=1
export RANKS_PER_DOMAIN=${RANKS_PER_DOMAIN:-14}
export RANKS_PER_NODE=$(( $RANKS_PER_DOMAIN * 8  ))
export THREADS_PER_RANK=1
export CORES_PER_RANK=2
if test $RANKS_PER_DOMAIN -eq 14 ; then
    export SPARTACPUBIND="map_cpu:112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223"
elif test $RANKS_PER_DOMAIN -eq 11 ; then
    export SPARTACPUBIND="map_cpu:113,114,115,117,118,119,121,122,123,124,125,127,128,129,131,132,133,135,136,137,138,139,141,142,143,145,146,147,149,150,151,152,153,155,156,157,159,160,161,163,164,165,166,167,169,170,171,173,174,175,177,178,179,180,181,183,184,185,187,188,189,191,192,193,194,195,197,198,199,201,202,203,205,206,207,208,209,211,212,213,215,216,217,219,220,221,222,223"
elif test $RANKS_PER_DOMAIN -eq 7 ; then
    export SPARTACPUBIND="map_cpu:113,115,117,119,121,123,125,127,129,131,133,135,137,139,141,143,145,147,149,151,153,155,157,159,161,163,165,167,169,171,173,175,177,179,181,183,185,187,189,191,193,195,197,199,201,203,205,207,209,211,213,215,217,219,221,223"
elif test $RANKS_PER_DOMAIN -eq 4 ; then
    export SPARTACPUBIND="map_cpu:113,117,121,125,127,131,135,139,141,145,149,153,155,159,163,167,169,173,177,181,183,187,191,195,197,201,205,209,211,215,219,223"
elif test $RANKS_PER_DOMAIN -eq 1 ; then
    export SPARTACPUBIND="map_cpu:113,127,141,155,169,183,197,211"
else
    export SPARTACPUBIND="map_cpu:112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223"
fi

# OpenMP setup
export OMP_NUM_THREADS=${THREADS_PER_RANK}

# LDPXI setup if applicable
export LDPXI_INST="gru,io,mpi,rank,mem,perf"
export LDPXI_PERF_EVENTS="cpu-cycles,ref-cycles,dTLB-load-misses,dTLB-loads"
export LDPXI_OUTPUT="sparta-ldpxi.$SLURM_JOB_ID.csv"

# create and populate run folder and edit input file
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
    i=$1
    echo "INFO: Perform Simulation #${i}"
    date
    dir_try="try-` printf '%02d' $i `"
    mkdir -p "${dir_try}"
    pushd "${dir_try}"
    ln -s ../air.* ../*.surf ../in.cylinder ./
    ln -s ../sparta_fom.py ./

    if test $THREADS_PER_RANK -eq 1 ; then
        # export LD_PRELOAD=libldpxi_mpi.so
        export LD_PRELOAD=/usr/projects/hpctest/amagela/ldpxi/ldpxi/install/ats3/ldpxi-1.0.1/intel+cray-mpich-8.1.25/lib/libldpxi_mpi.so.1.0.1
        time srun \
            --unbuffered \
            --ntasks=$RANKS_PER_NODE \
            --cpu-bind="${SPARTACPUBIND}" \
            --output="${FILE_LOG//.log/-${i}.log}" \
            "${APP_EXE}" \
                -in "in.cylinder"
        unset LD_PRELOAD
        l_fom=`./sparta_fom.py -a -f "log.sparta" 2>&1 | awk -F'FOM = ' '{print $2}'`
        l_maxrss=`grep maxrss "${LDPXI_OUTPUT}" | awk -F',' '{print $2}'`
        echo "FOM,RUN,PPC,RanksPerDomain,MaxRSS(KiB),AppName,Try,Dir" > "${FILE_METRICS}"
        echo "${l_fom},${SPARTA_RUN},${SPARTA_PPC},${RANKS_PER_DOMAIN},${l_maxrss},${APP_NAME},${i},` pwd -P `" >> "${FILE_METRICS}"
    fi

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
