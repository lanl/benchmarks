#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --ntasks-per-node=4
#SBATCH -p gpu
#SBATCH --job-name=Ag-MD

#Keep system size constant, change number of resources

source exports.bash #configures environment and sets ${lmpexec}
export HIPPYNN_USE_CUSTOM_KERNELS="pytorch"
export HIPPYNN_WARN_LOW_DISTANCES="False"

srun bash -c "export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID; ${lmpexec} -var xrep 25 -var yrep 24 -var zrep 24 -in MD-HO.in -l Strong_Parallel_${SLURM_NNODES}.out -k on g 1 -sf kk -pk kokkos neigh full newton on "



