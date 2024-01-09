#!/bin/bash
#
#SBATCH -J memrecord_run
#SBATCH -o logs/out.%x.%j
#SBATCH -e logs/err.%x.%j

#######################################################
# MAKE 'logs' dir in your cwd before running
#
# RUNS memrecord in a slurm allocation
# with varying array sizes.
# Collects results.
#
# SET VARIABLES WITH KEY=VALUE ON
# COMMAND LINE.
# OPTIONS:
#  - EXE: Name of Executable
#  - SBATCH_NAME: Descriptive name of program
#       (for output dir naming)
#  - COLLECTOR_DIR: Start of name of collection
#       output directory.
#  - TPN: Procs per node
#  - COMPACT_MEM: #Run memory compaction
#       0=Don't run
#       1=Run once before loop
#       2=Run between each srun launch
#  - NMB: Array of # elements per array per node
#       (in Millions)
#######################################################

# SET CL OPTIONS
setvar () {
	while [[ $# -gt 0 ]]; do
		export $1
		shift
	done
}

# CLEAN UP THE RAM.
mem_compact() {
    srun --drop-caches=slab -N $SLURM_NNODES --ntasks-per-node=1 true
    srun --drop-caches=pagecache -N $SLURM_NNODES --ntasks-per-node=1 true
    srun --vm-compact-atom=enable -N $SLURM_NNODES --ntasks-per-node=1 true
    echo MEMORY_COMPACTED
}

npr=$(nproc)
nprr=$(( npr/2 ))

setvar $@

: ${EXE:=memrecord}
: ${SBATCH_NAME:="basicTests"}
: ${COLLECTOR_DIR:="collect_${SBATCH_NAME}"}
: ${TPN:=$nprr} # PROCS PER NODE, DEFAULT IS HALF AVAILABLE
: ${COMPACT_MEM:=1} #Run memory compaction
: ${NMB:="512 1024 2048 4096 8192"} # ARRAY OF NUMBER OF ELEMENTS (in Millions) PER NODE (3 ARRAYS)
# MNB default: PER NODE 6 GiB, 12 GiB, 24 GiB, 48GiB, 96GiB

# IF the Executable ($EXE) isn't in the PATH.
if ! which $EXE &> /dev/null; then
    echo "FAIL FAIL"
    echo "  LOAD THE $PREFIX/bin directory into your PATH before running this program."
    echo "  ALSO, loading the memory_recorder.tcl modulefile works."
    exit 1
fi

logfile_suffix="${SLURM_JOB_NAME}.${SLURM_JOB_ID}"
echo $logfile_suffix

NODEFACTOR=$(( 1000000/TPN ))
if [[ $COMPACT_MEM -gt 0 ]]; then mem_compact; fi

for N in $NMB; do
    MEM=$(( NODEFACTOR*N ))
    s0=${SECONDS}

    echo "   ---------------------------------------------"
    echo "srun -N $SLURM_NNODES --ntasks-per-node=${TPN} $EXE $MEM"
    srun -N $SLURM_NNODES --ntasks-per-node=${TPN} $EXE $MEM

    if [[ $COMPACT_MEM -gt 1 ]]; then mem_compact; fi

    sf=$(( SECONDS-s0 ))
    echo "Size: $N, NUM_NODES: $SLURM_NNODES, TOOK: $sf (s)"

    indir=${PWD}
    outdir="${indir}/${SBATCH_NAME}_NODES-${SLURM_NNODES}_SZPP-${MEM}"
    mem_collect -i $indir -o $outdir
done

Nbasic=$(ls | grep -c $COLLECTOR_DIR)
collect_dir="${COLLECTOR_DIR}_${Nbasic}"
mkdir -p "$collect_dir"


mv ${SBATCH_NAME}_NODES* $collect_dir
cp logs/out.${logfile_suffix} $collect_dir
cp logs/err.${logfile_suffix} $collect_dir