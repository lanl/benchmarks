******
STREAM
******

Purpose
=======

The `STREAM <https://github.com/jeffhammond/STREAM>`_ benchmark is a widely used benchmark for measuring memory bandwidth. It measures the sustainable memory bandwidth of a computer system by performing simple read and write operations on large arrays.

Characteristics
===============

STREAM is available:
- Github: `STREAM <https://github.com/jeffhammond/STREAM>`_ 
- Official site: `STREAM <https://www.cs.virginia.edu/stream/>`_
- LANL Crossroads site: `STREAM <https://www.lanl.gov/projects/crossroads/_assets/docs/micro/stream-bench-crossroads-v1.0.0.tgz>`_

Problem
-------

There are four memory operations that the STREAM benchmark measures: Copy, Scale, Add, and Triad.

Copy - Copies data from one array to another:
b[i] = a[i]

Scale - Multiplies each array element by a constant, a daxpy operation.
b[i] = q*a[i]

Add - Adds two arrays element-wise:
c[i] = a[i] + b[i]

Triad - Multiply-add operation:
a[i] = b[i] + q*c[i]

These operations stress memory and floating point pipelines.They test memory transfer speed, computation speed, and different combinations of these two components of overall performance performance.

Figure of Merit
---------------

Building
========

Adjustments to GOMP_CPU_AFFINITY may also be necessary.

You can modify the STREAM_ARRAY_SIZE value in the compilation step to change the array size used by the benchmark. Adjusting the array size can help accommodate the available memory on your system.

RHEL Systems
------------

CrayOS Systems
--------------

Running
=======

.. code-block:: bash

  mpirun -np <num_processes> ./stream

Replace `<num_processes>` with the number of MPI processes you want to use. For example, if you want to use 4 MPI processes, the command will be:

.. code-block:: bash

  mpirun -np 4 ./stream

Input
-----

Independent Variables
---------------------

Dependent Variable(s)
---------------------

1. Maximum bandwidth while utilizing all hardware cores and threads. MAX_BW
2. A minimum number of cores and threads that achieves MAX_BW. MIN_CT 

Example Results
===============