#!/bin/bash
find /lustre/xrscratch1/${USER}/ior/ -type f -delete
for taskPerNode in 1 8 32 56 88 110; do
  segments=16
  size=2G
  if [ $numNodes -lt 3070 ];  then
          taskPerNode=20
  fi
  if [ $numNodes -lt 614 ];  then
          taskPerNode=50
  fi
        nTasks=$(( taskPerNode * numNodes ))
        srun -N $numNodes --ntasks=$nTasks /users/aparga/xr/bin/ior -k -e -a POSIX -F -v  -b 4G -s 16 -t 1M -D 180 -w -o /lustre/xrscratch1/aparga/ior/${numNodes}_fpr_posix
        sleep 3
        srun -N $numNodes --ntasks=$nTasks /users/aparga/xr/bin/ior -k -e -a MPIIO -F -v  -b 4G -s 16 -t 1M -D 180 -w -o /lustre/xrscratch1/aparga/ior/${numNodes}_fpr_MPIIO
        sleep 3
        srun -N $numNodes --ntasks=$nTasks /users/aparga/xr/bin/ior -C -Q $taskPerNode -k -E -a POSIX -F -v  -b 4G -s 16 -t 1M -D 30 -r -o /lustre/xrscratch1/aparga/ior/${numNodes}_fpr_posix
        sleep 3
        srun -N $numNodes --ntasks=$nTasks /users/aparga/xr/bin/ior -C -Q $taskPerNode -k -E -a MPIIO -F -v  -b 4G -s 16 -t 4M -D 30 -r -o /lustre/xrscratch1/aparga/ior/${numNodes}_fpr_MPIIO
        sleep 3


        taskPerNode=10
        nTasks=$(( taskPerNode * numNodes ))
        lfs setstripe -c 4 /lustre/xrscratch1/aparga/ior/${numNodes}_nto1_posix
        srun -N $numNodes --ntasks=$nTasks /users/aparga/xr/bin/ior -k -e -E -a POSIX -v  -b $size -s $segments -t 1M -D 180 -w -o /lustre/xrscratch1/aparga/ior/${numNodes}_nto1_posix
        sleep 3
        lfs setstripe -c 4 /lustre/xrscratch1/aparga/ior/${numNodes}_nto1_MPIIO
        srun -N $numNodes --ntasks=$nTasks /users/aparga/xr/bin/ior -k -e -E -a MPIIO -v  -b $size -s $segments -t 1M -D 180 -w -o /lustre/xrscratch1/aparga/ior/${numNodes}_nto1_MPIIO
        sleep 3

        srun -N $numNodes --ntasks=$nTasks /users/aparga/xr/bin/ior -C -Q $taskPerNode -k -E -a POSIX -v  -b $size -s $segments -t 1M -D 45 -r -o /lustre/xrscratch1/aparga/ior/${numNodes}_nto1_posix
        sleep 3
        srun -N $numNodes --ntasks=$nTasks /users/aparga/xr/bin/ior -C -Q $taskPerNode -k -E -a MPIIO -v  -b $size -s $segments -t 1M -D 45 -r -o /lustre/xrscratch1/aparga/ior/${numNodes}_nto1_MPIIO
        sleep 3
done