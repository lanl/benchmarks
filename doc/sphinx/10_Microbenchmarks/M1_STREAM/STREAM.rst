************************************
STREAM
************************************

Purpose
=======

The `STREAM <https://github.com/jeffhammond/STREAM>`_ benchmark is a widely used benchmark for measuring memory bandwidth. It measures the sustainable memory bandwidth (in MB/s) of a computer system by performing simple read and write operations on large arrays. This guide will walk you through the process of running the STREAM benchmark using MPI on a single node of an HPC system.

Prerequisites
=============

This benchmark requires MPI and a C/Fortran compiler

Building
========

Generic
=======

1. Visit the official STREAM website: https://www.cs.virginia.edu/stream/
2. Download the benchmark source code by clicking on the "Download STREAM benchmark" link.
3. Extract the downloaded archive to a directory of your choice.

Trinitite
=========

1. Open a terminal or command prompt and navigate to the directory where you extracted the STREAM benchmark source code.
2. Inside the directory, you will find four source code files: `stream.c`, `stream.h`, `mysecond.c`, and `README`.
3. Compile the benchmark by running the following command:

   .. code-block:: bash

      gcc -O3 -fopenmp -DSTREAM_ARRAY_SIZE=200000000 -o stream stream.c mysecond.c -lm

   This command will compile the benchmark with optimization level 3 (-O3), enable OpenMP support (-fopenmp), define the array size (-DSTREAM_ARRAY_SIZE=200000000), and link the necessary math library (-lm).

Running
=======

General
=======

Open a terminal or command prompt and navigate to the directory where you compiled the STREAM benchmark.
To run the benchmark using MPI, execute the following command:

   .. code-block:: bash

      mpirun -np <num_processes> ./stream

   Replace `<num_processes>` with the number of MPI processes you want to use. For example, if you want to use 4 MPI processes, the command will be:

   .. code-block:: bash

      mpirun -np 4 ./stream

   The benchmark will run and display the results for the sustainable memory bandwidth (in MB/s) for the four operations: Copy, Scale, Add, and Triad.

Trinitite
=========

Additional Considerations
=========================

- Ensure that you have a sufficient number of cores available on your Broadwell platform to achieve accurate benchmark results.
- You can modify the STREAM_ARRAY_SIZE value in the compilation step to change the array size used by the benchmark. Adjusting the array size can help accommodate the available memory on your system.

Conclusion
==========

The STREAM benchmark is a powerful tool for evaluating the memory bandwidth of a system. By following the steps outlined in this documentation, you can run the STREAM benchmark using MPI on a Broadwell platform. Remember to analyze the benchmark results carefully and consider other factors that may affect the performance of your system.
