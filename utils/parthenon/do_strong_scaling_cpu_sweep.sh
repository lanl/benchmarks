#!/bin/bash

set +x
set +e

FOOTPRINT=$1

#BEGIN Do not change the following for official benchmarking 
NXB=16
NLIM=250
NLVL=3
#END

NXs=(64 128 160)  

EXEC=./burgers-benchmark # executable
INP=../../../benchmarks/burgers/burgers.pin

# loop
for NX in ${NXs[@]}; do 
i=0
IDEAL1=0
TIMING_FILE_NAME="cpu_${NX}.csv"
HEADER="No. Cores, Actual, Ideal"
echo "Saving timing to ${TIMING_FILE_NAME}"
echo "${HEADER}"
echo "${HEADER}" > ${TIMING_FILE_NAME}
    
for count in  8 32 56 88 112; do
    echo "Core count = ${count}"
    L=`echo "((${NX}^3)/(${NXB}^3)) >= ${count}" | bc -l`
    if (($L == 0));  then
	continue; 
    fi
    outfile=$(printf "strong-scale-%d-%d.out" ${NX} ${count})
    errfile=$(printf "strong-scale-%d-%d.err" ${NX} ${count})
    echo "saving to output file ${outfile}"
    ARGS="${EXEC} -i ${INP} parthenon/mesh/nx1=${NX} parthenon/mesh/nx2=${NX} parthenon/mesh/nx3=${NX} parthenon/meshblock/nx1=${NXB} parthenon/meshblock/nx2=${NXB} parthenon/meshblock/nx3=${NXB} parthenon/time/nlim=${NLIM} parthenon/mesh/numlevel=${NLVL}"
    CMD="srun -n ${count} --hint=nomultithread  -o ${outfile} -e ${errfile} ${ARGS}"
    echo ${CMD}
    ${CMD}
    wait
    zc=$(grep 'zone-cycles/wallsecond = ' ${outfile} | cut -d '=' -f 2 | xargs)
    echo ${zc}
    if (( ${i} == 0 )); then
       IDEAL1=$(echo "print(\"%.7e\" % (${zc}/${count}))" | python3)
    fi
    IDEAL=$(echo "print(\"%.7e\" % (${count}*${IDEAL1}))" | python3)
    OUTSTR="${count}, ${zc}, ${IDEAL}"
    echo "${OUTSTR}"
    echo "${OUTSTR}" >> ${TIMING_FILE_NAME}
    i=$((${i} + 1))
done
done
