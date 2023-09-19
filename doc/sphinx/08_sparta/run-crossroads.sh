#!/bin/sh
#SBATCH --job-name=sparta-L2-S0
#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --core-spec=0
#SBATCH --exclude=nid006219,nid005850,nid003658,nid001813,nid001451,nid002233,nid005892,nid005896,nid003804,nid002065,nid001855,nid005912,nid006402,nid005723,nid006615,nid005851,nid005356

umask 022
set -e
set -x
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
export DIR_RUN="run-output-${DIR_CASE}"
export FILE_LOG="${DIR_RUN}/run-sparta-${DIR_CASE}.log"
export TMPDIR="/tmp/$SLURM_JOB_ID"  # set to avoid potential runtime error

# SPARTA setup
export APP_EXE="${DIR_EXE}/spa_crossroads_omp"
export APP_REPEAT=1

# MPI setup
export NODES=1
export RANKS_PER_NODE=112
export THREADS_PER_RANK=1

# OpenMP setup
export THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=true
export OMP_NUM_THREADS=${THREADS_PER_RANK}

# LDPXI setup if applicable
export LDPXI_INST="gru,rank"
export LDPXI_OUTPUT="sparta-ldpxi.$SLURM_JOB_ID.csv"
export LDPXI_RANK_OUTPUT="sparta-ldpxi.$SLURM_JOB_ID.%d.csv"

echo "################"
echo "INFO: System and environment information"
echo "    INFO: modules"
module list
echo "    INFO: CPU info"
cat /proc/cpuinfo | tail -27
echo "    INFO: memory info"
cat /proc/meminfo
echo "    INFO: SLURM info"
env | grep -i slurm
echo "    INFO: HOST info"
hostname
echo "    INFO: NUMA info"
numactl --hardware

# create and populate run folder
mkdir -p "${DIR_BASE}/${DIR_RUN}"
cd "${DIR_BASE}/${DIR_RUN}"
cp -a "${DIR_SRC}/examples/cylinder/in.cylinder" ./
cp -a "${DIR_SRC}/examples/cylinder/circle_R0.5_P10000.surf" ./
cp -a "${DIR_SRC}"/examples/cylinder/air.* ./

for (( i=0; i<${APP_REPEAT}; i++ )) ; do
    echo "################"
    echo "INFO: Perform Simulation #{i}"
    # export LD_PRELOAD=libldpxi_mpi.so
    time srun \
        --ntasks-per-node=$RANKS_PER_NODE \
        --cpus-per-task=2 \
        numactl --physcpubind=112-223 \  # this avoids all LNM conflicts
            "${APP_EXE}" \
            -in "in.cylinder" \
        >"${FILE_LOG}-${i}" 2>&1
    unset LD_PRELOAD
done
                        
mv ../slurm-${SLURM_JOBID}.out .

exit 0
