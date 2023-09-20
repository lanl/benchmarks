#!/bin/bash

set +x
set +e

TIMING_FILE_NAME="gpu.csv"

EXEC=./burgers-benchmark # executable
INP=../../../benchmarks/burgers/burgers.pin

HEADER="Mesh Base Size,  Actual"
echo "Saving timing to ${TIMING_FILE_NAME}"
echo "${HEADER}"
echo "${HEADER}" > ${TIMING_FILE_NAME}

# loop
NXB=32
NL=2
for NX in 32 64 96 128 160 192; do
    echo "Mesh base size = ${NX}"
    outfile=$(printf "gpu-throughput-%d.out" ${NX})
    echo "saving to output file ${outfile}"
    ARGS="${EXEC} -i ${INP} parthenon/mesh/nx1=${NX} parthenon/mesh/nx2=${NX} parthenon/mesh/nx3=${NX} parthenon/meshblock/nx1=${NXB} parthenon/meshblock/nx2=${NXB} parthenon/meshblock/nx3=${NXB} parthenon/time/nlim=250 parthenon/mesh/numlevel=${NL}"
    CMD="${ARGS} | tee ${outfile}"
    echo ${CMD}
    ${ARGS} | tee ${outfile}
    zc=$(grep 'zone-cycles/wallsecond = ' ${outfile} | cut -d '=' -f 2 | xargs)
    echo ${zc}
    OUTSTR="${NX}, ${zc}"
    echo "${OUTSTR}"
    echo "${OUTSTR}" >> ${TIMING_FILE_NAME}
done
