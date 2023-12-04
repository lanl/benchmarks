#!/bin/bash
#SBATCH --job-name=sparta-L2-S0
#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --core-spec=0  # this seemingly needs to be done with sbatch
##SBATCH --exclude=nid006219,nid005850,nid003658,nid001813,nid001451,nid002233,nid005892,nid005896,nid003804,nid002065,nid001855,nid005912,nid006402,nid005723,nid006615,nid005851,nid005356

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
export FILE_LOG="run-sparta-${DIR_CASE}.log"
export TMPDIR="/tmp/$SLURM_JOB_ID"  # set to avoid potential runtime error

# SPARTA setup
export SPARTA_RUN=${SPARTA_RUN:-4000}
export SPARTA_PPC=${SPARTA_PPC:-42}
export APP_EXE="${DIR_EXE}/spa_crossroads_omp"
export APP_REPEAT=1

# MPI setup
export NODES=1
export RANKS_PER_NODE=24
export THREADS_PER_RANK=2
export CORES_PER_RANK=2

# OpenMP setup
export OMP_NUM_THREADS=${THREADS_PER_RANK}

# LDPXI setup if applicable
export LDPXI_INST="gru,rank"
export LDPXI_OUTPUT="sparta-ldpxi.$SLURM_JOB_ID.csv"

echo "################"
echo "INFO: System and environment information"
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

# address node starting state
srun --drop-caches=pagecache  -N $SLURM_NNODES -n $SLURM_NNODES --ntasks-per-node=1 true
srun --drop-caches=slab       -N $SLURM_NNODES -n $SLURM_NNODES --ntasks-per-node=1 true
srun --vm-compact-atom=enable -N $SLURM_NNODES -n $SLURM_NNODES --ntasks-per-node=1 true

# create and populate run folder and edit input file
mkdir -p "${DIR_BASE}/${DIR_RUN}"
cd "${DIR_BASE}/${DIR_RUN}"
# cp -a "${DIR_SRC}/examples/cylinder/in.cylinder" ./
awk "\$1 ~ /^run\$/ {\$2 = ${SPARTA_RUN}}1" "${DIR_SRC}/examples/cylinder/in.cylinder" \
    | awk "\$2 ~ /^ppc\$/ {\$4 = ${SPARTA_PPC}}1" \
    > "./in.cylinder"
cp -a "${DIR_SRC}/examples/cylinder/circle_R0.5_P10000.surf" ./
cp -a "${DIR_SRC}"/examples/cylinder/air.* ./
cp -a ../libsparta_spr_setaff.so ../sparta_spr_setaff.sh ./  # affinity stuff

for (( i=0; i<${APP_REPEAT}; i++ )) ; do
    echo "################"
    echo "INFO: Perform Simulation #${i}"
    dir_try="try-` printf '%02d' $i `"
    mkdir -p "${dir_try}"
    pushd "${dir_try}"
    ln -s ../air.* ../*.surf ../in.cylinder ./
    ln -s ../libsparta_spr_setaff.so ../sparta_spr_setaff.sh ./  # affinity stuff

    if test $THREADS_PER_RANK -gt 1 ; then
        # export LD_PRELOAD=libldpxi_mpi.so
        # export LD_PRELOAD=/usr/projects/hpctest/amagela/ldpxi/ldpxi/install/ats3/ldpxi-1.0.1/intel+cray-mpich-8.1.25/lib/libldpxi_mpi.so.1.0.1
        time srun \
            --unbuffered \
            --ntasks=$RANKS_PER_NODE \
            --cpus-per-task=${CORES_PER_RANK} \
            --cpu-bind=no \
            --output="${FILE_LOG//.log/-${i}.log}" \
            "./sparta_spr_setaff.sh" \
                "${APP_EXE}" \
                    -k on t ${OMP_NUM_THREADS} -sf kk \
                    -in "in.cylinder"
        # unset LD_PRELOAD
    else
        # export LD_PRELOAD=libldpxi_mpi.so
        # export LD_PRELOAD=/usr/projects/hpctest/amagela/ldpxi/ldpxi/install/ats3/ldpxi-1.0.1/intel+cray-mpich-8.1.25/lib/libldpxi_mpi.so.1.0.1
        time srun \
            --ntasks=$RANKS_PER_NODE \
            --cpu-bind=map_cpu:112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223 \
            --output="${FILE_LOG//.log/-${i}.log}" \
            "${APP_EXE}" \
                -k on -sf kk \
                -in "in.cylinder"
        # unset LD_PRELOAD
    fi

    popd
done
                        
# mv ../slurm-${SLURM_JOBID}.out .

exit 0
