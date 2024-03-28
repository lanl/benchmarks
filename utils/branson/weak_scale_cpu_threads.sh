#!/bin/bash
echo "$@"
numas=8
cores=14
nphper=760000
mynodes=1
ht=--hint=nomultithread 
while getopts N:n:c:p:h opt; do
    case $opt in
        N) mynodes=${OPTARG}
        ;;
        n) numas=${OPTARG}
        ;;
        c) cores=${OPTARG}
        ;;
        p) nphper=${OPTARG}
        ;;
        h) ht=""
        ;;
    esac
done

echo "numas: ${numas} cores: ${cores} nphper: ${nphper} ht: ${ht} mynodes: ${mynodes}"



input=3D_hohlraum_multi_node.xml


cp ../inputs/${input} .

echo ${input}
sed -i 's/<t_stop>.*<\/t_stop>/<t_stop>'.010'\<\/t_stop>/g' ${input}
sed -i 's/<n_omp_threads>.*<\/n_omp_threads>/<n_omp_threads>'${cores}'\<\/n_omp_threads>/g' ${input}
sed -i 's/<use_gpu_transporter>.*<\/use_gpu_transporter>/<use_gpu_transporter>'FALSE'\<\/use_gpu_transporter>/g' ${input}

HEADER="No. Cores, Actual"
SUFFIX="numas_${numas}_cores_${cores}_nphper_${nphper}"
TIMING_FILE_NAME="cpu_${SUFFIX}.csv"
echo "Saving timing to ${TIMING_FILE_NAME}"
echo "${HEADER}"
echo "${HEADER}" > ${TIMING_FILE_NAME}

for nodes in 1 4 8 16 32 40 64 96; do
    if [ $nodes -gt $mynodes ];  then 
    exit 0
    fi
    ranks=$((${nodes}*${numas}))
    threads=$((${ranks}*${cores}))
    size=$((${nphper}*${threads}))
    
    echo "ranks: ${ranks} threads: ${threads} size: ${size}"
    sed -i 's/<photons>.*<\/photons>/<photons>'${size}'\<\/photons>/g' ${input}

    
    outfile=$(printf "weakscale_${SUFFIX}-%d-%d.out" ${size} ${threads})
    errfile=$(printf "weakscale_${SUFFIX}-%d-%d.err" ${size} ${threads})
    grep photons ${input}
    MPICH_SMP_SINGLE_COPY_MODE=NONE FI_CXI_RX_MATCH_MODE=software OMP_NUM_THREADS=14
    srun ${ht}  -N ${nodes} -n ${ranks} --ntasks-per-node=${numas} -c ${cores} -o ${outfile} -e ${errfile}  ./BRANSON ./3D_hohlraum_multi_node.xml  
    fom=$(grep 'Photons Per Second (FOM): ' ${outfile} | cut -d ':' -f 2 | xargs)
    echo ${fom}
    
    OUTSTR="${threads}, ${fom}"
    echo "${OUTSTR}"
    echo "${OUTSTR}" >> ${TIMING_FILE_NAME}
    
done
exit 0