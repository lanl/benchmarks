******
MDTEST
******

Purpose
=======

The intent of this benchmark is to measure the performance of file metadata operations on the platform storage.
MDtest is an MPI-based application for evaluating the metadata performance of a file system and has been designed to test parallel file systems.
It can be run on any type of POSIX-compliant file system but has been designed to test the performance of parallel file systems.

Characteristics
===============

MDTEST is available in the benchmarks repository.

* `Benchmarks https://github.com/lanl/benchmarks/tree/main/microbenchmarks/mdtest>`_ 
.. LANL Crossroads Site: `MDTEST <https://www.lanl.gov/projects/crossroads/_assets/docs/micro/mdtest-1.8.4-xroads_v1.0.0.tgz>`_

Problem
-------

MDtest measures the performance of various metadata operations using MPI to coordinate execution and collect the results.
In this case, the operations in question are file creation, stat, and removal.

Run Rules
---------

Observed benchmark performance shall be obtained from a storage system configured as closely as possible to the proposed platform storage. 
If the proposed solution includes multiple file access protocols (e.g., pNFS and NFS) or multiple tiers accessible by applications, benchmark results for mdtest shall be provided for each protocol and/or tier.

Performance projections are permissible if they are derived from a similar system that is considered an earlier generation of the proposed system.

Modifications to the benchmark application code are only permissible to enable correct compilation and execution on the target platform. 
Any modifications must be fully documented (e.g., as a diff or patch file) and reported with the benchmark results.

Building
========

After extracting the tar file, ensure that the MPI is loaded and that the relevant compiler wrappers, ``cc`` or ``mpicc``, are in ``$PATH``.

.. code-block:: bash

    cd microbenchmarks/mdtest
    make

Running
=======

The results for the three operations, create, stat, remove, should be obtained for three different file configurations:

1) ``2^20`` files in a single directory.
2) ``2^20`` files in separate directories, 1 per MPI process.
3) 1 file accessed by multiple MPI processes.

These configurations are launched as follows.

.. code-block:: bash

    # Shared Directory
    srun -n 64 ./mdtest -F -C -T -r -n 16384 -d /scratch/$USER -N 16
    # Unique Directories
    srun -n 64 ./mdtest -F -C -T -r -n 16384 -d /scratch/$USER -N 16 -u
    # One File Multi-Proc
    srun -n 64 ./mdtest -F -C -T -r -n 16384 -d /scratch/$USER -N 16 -S

The following command-line flags MUST be changed:

* ``-n`` - the number of files **each MPI process** should manipulate.  For a test run with 64 MPI processes, specifying ``-n 16384`` will produce the equired ``2^20`` files (``2^6`` MPI processes x ``2^14`` files each).  This parameter must be changed for each level of concurrency.
* ``-d /scratch`` - the **absolute path** to the directory in which this test should be run. 
* ``-N`` - MPI rank offset for each separate phase of the test.  This parameter must be equal to the number of MPI processes per node in use (e.g., ``-N 16`` for a test with 16 processes per node) to ensure that each test phase (read, stat, and delete) is performed on a different node.

The following command-line flags MUST NOT be changed or omitted:

* ``-F`` - only operate on files, not directories
* ``-C`` - perform file creation test
* ``-T`` - perform file stat test
* ``-r`` - perform file remove test

Example Results
===============

These nine tests: three operations, three file conditions should be performed under 4 different launch conditions, for a total of 36 results:

1) A single MPI process
2) The optimal number of MPI processes on a single compute node
3) The minimal number of MPI processes on multiple compute nodes that achieves the peak results for the proposed system.
4) The maximum possible MPI-level concurrency on the proposed system. This could mean:
   1) Using one MPI process per CPU core across the entire system.
   2) Using the maximum number of MPI processes possible if one MPI process per core will not be possible on the proposed architecture.
   3) Using more than ``2^20`` files if the system is capable of launching more than ``2^20`` MPI processes.

Crossroads
----------

.. csv-table:: MDTEST Microbenchmark Crossroads (MB/s)
   :file: ats3_mdtest.csv
   :align: left
   :widths: 10, 10, 10, 10, 10
   :header-rows: 1
   :stub-columns: 2

