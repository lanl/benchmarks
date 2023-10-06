#!/bin/bash

set +x
set +e

NP=$1

if (( ${NP} <= 0 )); then
    echo "Unknown process count. Please specify process count."
    exit 1
fi

#BEGIN Do not change the following for official benchmarking 
NXB=16
NLIM=250
NLVL=3
#END

TIMING_FILE_NAME="cpu_throughput.csv"

EXEC=./burgers-benchmark # executable
INP=../../../benchmarks/burgers/burgers.pin

HEADER="No. Cores, Actual, Ideal"
echo "Saving timing to ${TIMING_FILE_NAME}"
echo "${HEADER}"
echo "${HEADER}" > ${TIMING_FILE_NAME}

# loop
i=0
IDEAL1=0
for NX in 32 64 96 128 160 192; do
    for count in ${NP}; do
	echo "Core count = ${count}"
	outfile=$(printf "strong-scale-%d-%d.out" ${NX} ${count})
	errfile=$(printf "strong-scale-%d-%d.err" ${NX} ${count})
	echo "saving to output file ${outfile}"
	ARGS="${EXEC} -i ${INP} parthenon/mesh/nx{1,2,3}=${NX} parthenon/meshblock/nx{1,2,3}=${NXB} parthenon/time/nlim=${NLIM} parthenon/mesh/numlevel=${NLVL}"
	CMD="srun --hint=nomultithread -n ${count} -o ${outfile} -e ${errfile} ${ARGS}"
	echo ${CMD}
	${CMD}
	wait
	zc=$(grep 'zone-cycles/wallsecond = ' ${outfile} | cut -d '=' -f 2 | xargs)
	echo ${zc}
	if (( ${i} == 0 )); then
	    IDEAL1=$(echo "print(\"%.7e\" % (${zc}/4))" | python3)
	fi
	IDEAL=$(echo "print(\"%.7e\" % (${count}*${IDEAL1}))" | python3)
	OUTSTR="${count}, ${zc}, ${IDEAL}"
	echo "${OUTSTR}"
	echo "${OUTSTR}" >> ${TIMING_FILE_NAME}
	i=$((${i} + 1))
    done
done 
