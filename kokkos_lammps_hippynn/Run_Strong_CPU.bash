#!/bin/bash

#On singe system run multiple strong scaling studies of different problem sizes
export lmpexec="${HOME}/lammps-kokkos-mliap/build/lmp"

#source exports.bash #configures environment and sets ${lmpexec}
export HIPPYNN_USE_CUSTOM_KERNELS="pytorch"
export HIPPYNN_WARN_LOW_DISTANCES="False"

for s in 16 #24 32 40 48 56 64
  do 
  for i in 8 32 56 88 112 
  do
    echo $i
    echo $lmpexec
    srun -n $i --hint=nomultithread  --cpus-per-task 1  ${lmpexec} -var xrep 2 -var yrep 2 -var zrep ${s} -in MD-HO-cpu.in -l Strong_Single_${s}_${i}.out  
  done 
done


