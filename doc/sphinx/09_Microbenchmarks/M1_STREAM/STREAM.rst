******
STREAM
******

Purpose
=======

The `STREAM <https://github.com/jeffhammond/STREAM>`_ benchmark is a widely used benchmark for measuring memory bandwidth. It measures the sustainable memory bandwidth of a computer system by performing simple read and write operations on large arrays.

Characteristics
===============

STREAM is available in the benchmarks repository.

* `Benchmarks https://github.com/lanl/benchmarks/tree/main/microbenchmarks/stream>`_ 
.. * Github: `STREAM_github <https://github.com/jeffhammond/STREAM>`_ 
.. * Github: `STREAM_github <https://github.com/jeffhammond/STREAM>`_ 
.. * Official site: `STREAM_official <https://www.cs.virginia.edu/stream/>`_
.. * LANL Crossroads site: `STREAM_LANL <https://www.lanl.gov/projects/crossroads/_assets/docs/micro/stream-bench-crossroads-v1.0.0.tgz>`_

Problem
-------

There are four memory operations that the STREAM benchmark measures: Copy, Scale, Add, and Triad.

Copy - Copies data from one array to another:

.. math:: 

  \mathbf{b[i]} = \mathbf{a[i]}

Scale - Multiplies each array element by a constant, a daxpy operation.

.. math::

  \mathbf{b[i]} = \mathbf{q}*\mathbf{a[i]}

Add - Adds two arrays element-wise:

.. math::

  \mathbf{c[i]} = \mathbf{a[i]} + \mathbf{b[i]}

Triad - Multiply-add operation:

.. math::

  \mathbf{a[i]} = \mathbf{b[i]} + \mathbf{q}\times\mathbf{c[i]}

These operations stress memory and floating point pipelines.They test memory transfer speed, computation speed, and different combinations of these two components of overall performance performance.

Figure of Merit
---------------

The primary FOM is the MAX Triad rate (MB/s).

Run Rules
---------

The program must synchronize between each operation. For instance:

On a heterogeneous system, run stream for all computational devices. Where there is unified or heterogeneously addressable memory, also provide performance numbers for each device's access to available memory types.

For instance:
On a heterogenous node architecture with multi-core CPU with HBM2 memory and a GPU with HBM3 memory Stream performance should be reported for: CPU <-> HBM2, GPU <-> HBM3, CPU <-> HBM3, GPU <-> HBM2

Present CPU data as it scales with as a function of cores. 
It is acceptable to simply present the maximum bandwidth on GPUs/accelerators.
More descriptive statistics are acceptable and welcome.

Building
========

Adjustments to ``GOMP_CPU_AFFINITY`` may be necessary.

The ``STREAM_ARRAY_SIZE`` value is a critical parameter set at compile time and controls the size of the array used to measure bandwidth. STREAM requires different amounts of memory to run on different systems, depending on both the system cache size(s) and the granularity of the system timer.

You should adjust the value of ``STREAM_ARRAY_SIZE`` to meet ALL of the following criteria:

1. Each array must be at least 4 times the size of the available cache memory. In practice the minimum array size is about 3.8 times the cache size.
   1. Example 1: One Xeon E3 with 8 MB L3 cache ``STREAM_ARRAY_SIZE`` should be ``>= 4 million``, giving an array size of 30.5 MB and a total memory requirement of 91.5 MB.
   2. Example 2: Two Xeon E5's with 20 MB L3 cache each (using OpenMP) ``STREAM_ARRAY_SIZE`` should be ``>= 20 million``, giving an array size of 153 MB and a total memory requirement of 458 MB.
2. The size should be large enough so that the 'timing calibration' output by the program is at least 20 clock-ticks. For example, most versions of Windows have a 10 millisecond timer granularity. 20 "ticks" at 10 ms/tic is 200 milliseconds. If the chip is capable of 10 GB/s, it moves 2 GB in 200 msec. This means the each array must be at least 1 GB, or 128M elements.
3. The value ``24xSTREAM_ARRAY_SIZExRANKS_PER_NODE`` must be less than the amount of RAM on a node. STREAM creates 3 arrays of doubles; that is where 24 comes from. Each rank has 3 of these arrays.

Set ``STREAM_ARRAY_SIZE`` using the -D flag on your compile line.

The formula for ``STREAM_ARRAY_SIZE`` is:

:: 

 ARRAY_SIZE ~= 4 x (last_level_cache_size x num_sockets) / size_of_double = last_level_cache_size

This reduces to a number of elements equal to the size of the last level cache of a single socket in bytes, assuming a node has two sockets.
This is the minimum size unless other system attributes constrain it.

The array size only influences the capacity of STREAM to fully load the memory bus.
At capacity, the measured values should reach a steady state where increasing the value of ``STREAM_ARRAY_SIZE`` doesn't influence the measurement for a certain number of processors.

For Crossroads, the benchmark was build with ``STREAM_ARRAY_SIZE=40000000`` and ``NTIMES=20`` with optmizations and OpenMP enabled.

.. code-block:: bash
  
   make CC=`which mpicc` FF=`which mpifort` CFLAGS="-O2 -fopenmp -DSTREAM_ARRAY_SIZE=40000000 -DNTIMES=20" FFLAGS="-O2 -fopenmp -DSTREAM_ARRAY_SIZE=40000000 -DNTIMES=20"


Running
=======

.. code-block:: bash

  export OMP_NUM_THREADS=1
  srun -n <num_processes> --cpu-bind=core ./stream-mpi.exe

Replace `<num_processes>` with the number of MPI processes you want to use. For example, if you want to use 4 MPI processes, the command will be:

.. code-block:: bash

  export OMP_NUM_THREADS=1
  srun -n 4 --cpu-bind=core ./stream-mpi.exe

Example Results
===============

Results for STREAM are provided on the following systems:

* Crossroads (see :ref:`GlobalSystemATS3`)

Crossroads
----------

These results were obtained using the cce v15.0.1 compiler and cray-mpich v 8.1.25. 
Results using the intel-oneapi and intel-classic v2023.1.0 and the same cray-mpich were also collected; cce performed the best.

``STREAM_ARRAY_SIZE=40000000 NTIMES=20``

.. csv-table:: STREAM microbenchmark bandwidth measurement
   :file: stream-xrds_ats5cce-cray-mpich.csv
   :align: center
   :widths: 10, 10, 10
   :header-rows: 1

.. figure:: stream_cpu_ats3.png
   :align: center
   :scale: 50%
   :alt: STREAM microbenchmark bandwidth measurement
