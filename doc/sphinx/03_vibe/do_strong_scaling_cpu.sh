#!/bin/bash

set +x
set +e

FOOTPRINT=$1

#BEGIN Do not change the following for official benchmarking 
NXB=16
NLIM=250
NLVL=3
#END


if (( ${FOOTPRINT} == 20 )); then
    NX=64
elif (( ${FOOTPRINT} == 40 )); then
    NX=128
elif (( ${FOOTPRINT} == 60 )); then
    NX=160
else
    echo "Unknown footprint. Available footprints are 20, 40, 60."
    exit 1
fi
TIMING_FILE_NAME="cpu_${FOOTPRINT}.csv"

EXEC=./burgers-benchmark # executable
INP=../../../benchmarks/burgers/burgers.pin

HEADER="No. Cores, Actual, Ideal"
echo "Saving timing to ${TIMING_FILE_NAME}"
echo "${HEADER}"
echo "${HEADER}" > ${TIMING_FILE_NAME}

# loop
i=0
IDEAL1=0
for count in  8 32 56 88 112; do
    echo "Core count = ${count}"
    outfile=$(printf "strong-scale-%d.out" ${count})
    errfile=$(printf "strong-scale-%d.err" ${count})
    echo "saving to output file ${outfile}"
    ARGS="${EXEC} -i ${INP} parthenon/mesh/nx1=${NX} parthenon/mesh/nx2=${NX} parthenon/mesh/nx3=${NX} parthenon/meshblock/nx1=${NXB} parthenon/meshblock/nx3=${NXB} parthenon/meshblock/nx3=${NXB} parthenon/time/nlim=${NLIM} parthenon/mesh/numlevel=${NLVL}"
    CMD="mpirun -n ${count} -outfile-pattern ${outfile} -errfile-pattern ${errfile} ${ARGS}"
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
