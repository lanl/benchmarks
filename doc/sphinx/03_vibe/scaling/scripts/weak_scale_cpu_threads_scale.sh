#!/bin/bash
set +x
set +e
export CALI_CONFIG="spot,time.exclusive,profile.mpi"

numas=8
cores=14
mynodes=1
NX=96
ht=--hint=nomultithread 
while getopts N:n:c:x:h opt; do
    case $opt in
        N) mynodes=${OPTARG}
        ;;
        n) numas=${OPTARG}
        ;;
        c) cores=${OPTARG}
        ;;
	h) ht=""
        ;;
    esac
done

echo "numas: ${numas} cores: ${cores}  ht: ${ht} mynodes: ${mynodes}"


SUFFIX="numas_${numas}_cores_${cores}"
input=burgers.pin
workdir=${SUFFIX}_${SLURM_JOB_ID}_scale
mkdir ${workdir}
cd ${workdir}
cp ../../../../benchmarks/burgers/${input} . 
sed -i 's/pack_size = .*/pack_size = '${cores}'/g' ${input}
NXs=(208 256 320 400 512 640 800 1024 1280 1616 2048 2576)  
NXBs=(16) # 32)
NLIM=250
NLVL=3
NODES=(2 4 8 16 32 64 128 256 512 1024 2048 4096)
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
i=0
for N in ${NODES[@]}; do
    ranks=$((${N}*${numas}))
    threads=$((${ranks}*${cores}))
    if [ $N -gt ${mynodes} ]; then
	    exit 0
    fi 
    for NXB in ${NXBs[@]}; do     
	
    NX=${NXs[i]}
#            echo "ranks: ${ranks} threads: ${threads}"
            echo "${NXB} blocksize, ${NX} grid, ${N} nodes, ${ranks} ranks, ${threads} threads"
            if [ 0 -eq `echo "((${NX}^3) / (${NXB}^3)) >= ${ranks}" | bc ` ];  then            
                echo "not enough blocks! skipping" 
		continue
            fi
#            if [  0 -eq `echo "((${NX}^3) / ${N}) <= (190^3) " | bc ` ];  then            
#                echo "too much mesh for number of nodes!  skipping" 
#		        continue
#	        fi
            echo "ranks: ${ranks} threads: ${threads}"
            echo "running ${NXB} blocksize, ${NX} grid, ${N} nodes, ${ranks} ranks"
            OMP_NUM_THREADS=${cores} srun ${BINDOPTS}  --exclusive  --distribution=block:block ${ht} -n ${ranks} -N ${N}  --ntasks-per-node=${numas} -c ${cores} ../burgers-benchmark -i ./burgers.pin parthenon/mesh/nx1=${NX}  parthenon/mesh/nx2=${NX} parthenon/mesh/nx3=${NX} parthenon/meshblock/nx1=${NXB}  parthenon/meshblock/nx2=${NXB} parthenon/meshblock/nx3=${NXB} parthenon/time/nlim=${NLIM} parthenon/mesh/numlevel=${NLVL} |& tee scale-cpu.$NXB.$NX.$N.out
done
i=$((i+1))

done
