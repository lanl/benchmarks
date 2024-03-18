#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH -p gpu
#SBATCH --job-name=Ag-MD

#On 1 gpu, run systems of different sizes
export lmpexec="pathto/lammps-kokkos-mliap/build/lmp"

source exports.bash #configures environment and sets ${lmpexec}
export HIPPYNN_USE_CUSTOM_KERNELS="pytorch"
export HIPPYNN_WARN_LOW_DISTANCES="False"

for i in 2 4 8 12 16 24 32 40 48 56 64 
do
${lmpexec} -var xrep 2 -var yrep 2 -var zrep ${i} -in MD-HO.in -l Strong_Single_${i}.out -k on g 1 -sf kk -pk kokkos neigh full newton on 
done


