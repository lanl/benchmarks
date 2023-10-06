#!/bin/bash

set +x
set +e

NP=$1
NT=$2

if [ -z ${NP} ]; then
    echo "Unknown process count. Please specify process count."
    exit 1
fi



if [ -z ${NT} ]; then
    echo "Unknown thread count. Please specify thread count."
    exit 1
fi

count=${NP}

#BEGIN Do not change the following for official benchmarking 
NXB=16
NLIM=10
NLVL=3
#END

TIMING_FILE_NAME="cpu_throughput.csv"
#export OMP_PROC_BIND=spread
#export OMP_PLACES=threads

EXEC=./burgers-benchmark # executable
INP=../../../benchmarks/burgers/burgers.pin

HEADER="No. Cores, Actual"
echo "Saving timing to ${TIMING_FILE_NAME}"
echo "${HEADER}"
echo "${HEADER}" > ${TIMING_FILE_NAME}

# loop
i=0
IDEAL1=0
for NX in 32 64 96 128 160 192; do
	echo "Core count = ${count}"
	outfile=$(printf "strong-scale-%d-%d.out" ${NX} ${count})
	errfile=$(printf "strong-scale-%d-%d.err" ${NX} ${count})
	echo "saving to output file ${outfile}"
	ARGS="${EXEC} -i ${INP} parthenon/mesh/nx1=${NX} parthenon/mesh/nx2=${NX} parthenon/mesh/nx3=${NX} parthenon/meshblock/nx1=${NXB} parthenon/meshblock/nx3=${NXB} parthenon/meshblock/nx3=${NXB} parthenon/time/nlim=${NLIM} parthenon/mesh/numlevel=${NLVL}"
	CMD="srun --cpu-bind=ldoms -n ${NP} -c ${NT} -o ${outfile} -e ${errfile} ${ARGS}"
	echo ${CMD}
	${CMD}
	wait
	zc=$(grep 'zone-cycles/wallsecond = ' ${outfile} | cut -d '=' -f 2 | xargs)
	echo ${zc}
	OUTSTR="${count}, ${zc}"
	echo "${OUTSTR}"
	echo "${OUTSTR}" >> ${TIMING_FILE_NAME}
	i=$((${i} + 1))
done 
