#!/bin/bash

#Strong Parallel Scaling
for i in 1 2 4 8 16 32
do
  sbatch --nodes ${i} Run_Strong_Parallel.bash
done

#Strong single GPU scaling
sbatch Run_Strong_Single.bash

#Weak Parallel Scaling
for i in 1 2 4 8 16 32
do
  sbatch --nodes ${i} Run_Weak_Parallel.bash
done