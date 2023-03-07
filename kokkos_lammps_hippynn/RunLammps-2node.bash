#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH -p gpu
#SBATCH --job-name=Ag-MD

#This script needs to be run for each node or gpu set
#As configured, this will run the 8GPU example. 

source activate ${virtenvpath} #This should be set during environment building
export HIPPYNN_USE_CUSTOM_KERNELS="pytorch"
export HIPPYNN_WARN_LOW_DISTANCES="False"

srun bash -c 'export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID; <path to lammps>/lmp -in MD-HO.in -k on g 1 -sf kk -pk kokkos neigh full newton on '



