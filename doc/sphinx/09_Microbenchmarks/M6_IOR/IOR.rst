***
IOR
***

Purpose
=======

IOR is used for testing performance of parallel file systems using various interfaces and access patterns at the POSIX and MPI-IO level.

Characteristics
===============

IOR is available in the benchmarks repository.

* LANL Benchmarks: `Benchmarks <https://github.com/lanl/benchmarks/tree/main/microbenchmarks/ior>`_ 
* Github: `IOR Public <https://github.com/hpc/ior>`_

.. - LANL Crossroads Site: `IOR <https://www.lanl.gov/projects/crossroads/_assets/docs/micro/ior-3.0.1-xroads_v1.0.0.tgz>`_

The github repo also contains mdtest.

Problem
-------

IOR measures parallel I/O performance at the POSIX and MPI-IO levels.
It writes and reads files, one per rank or shared between all ranks, on a parallel file system.
It should be run in lustre space.

Run Rules
---------

Modifications to the benchmark application code are only permissible to enable correct compilation and execution on the target platform. Any modifications must be fully documented (e.g., as a diff or patch file) and reported with the benchmark results.

Building
========

MPI, MPI-IO, and OpenMP are required in order to build and run the code. The
source code used for this benchmark is derived from IOR 3.0.1 and it is
included here. 

Ensure that the MPI compiler wrappers (e.g., `mpicc`) are in `$PATH`. Then create a build directory and an (optional) install directory.

.. code-block:: bash
    
    <BENCHMARK_PATH>/microbenchmarks/ior/configure --prefix=<INSTALL_DIR>
    make
    #make install
..

This will build both IOR with the POSIX and MPI-IO interfaces and create the
IOR executable at `src/ior`.

Running
=======

The ior tests can be run using the following command:

.. code-block:: bash

srun -n <nnodes> --ntasks-per node=<cores_per_node> <INSTALL_DIR>/bin/ior -f <BENCHMARK_PATH>/microbenchmarks/ior/inputs.xroads/<load_type>-<io_type>-<access_type>.ior
..

Where `load_type` is `load1` for sequential loads and `load2` for random loads, `io_type` is `posix` or `mpiio`, and `access_type` is `filepertask` and `sharedfile` for per task and shared accesses respectively.
There are six input decks in the `inputs.xroads` directory; each should be run on a single node and across the full system in parallel.

"*Note: Benchmark values for random loads are not presented here.*"

Input
-----

Sample production input files in `microbenchmarks/ior/inputs.xroads`

Running IOR does not require using the input files. All arguments can be given on the command line.

Example Results
===============

Results for IOR are provided on the following systems:

* Crossroads (see :ref:`GlobalSystemATS3`)

Crossroads
----------

Full system tests were run with 5000 nodes and 10 tasks per node.
Single node tests were run with 112 tasks per node.
This test was compiled with cce/16.0.0 and cray-mpich/8.1.26.

.. csv-table:: IOR benchmark (MB/s)
   :file: ats3_ior.csv
   :align: left
   :widths: 10, 10, 10, 10, 10
   :header-rows: 1
   :stub-columns: 1

