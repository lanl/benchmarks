***
IOR
***

Purpose
=======

IOR is used for testing performance of parallel file systems using various interfaces and access patterns at the POSIX and MPI-IO level.

Characteristics
===============

IOR is available:
- Github: `IOR Public <https://github.com/hpc/ior>`_
- LANL Crossroads Site: `IOR <https://www.lanl.gov/projects/crossroads/_assets/docs/micro/ior-3.0.1-xroads_v1.0.0.tgz>`_

The github repo also contains mdtest.

Problem
-------

Figure of Merit
---------------

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

    
    ${BENCHMARK_PATH}/microbenchmarks/ior/configure --prefix=<INSTALL_DIR>
    make
    #make install
..

This will build both IOR with the POSIX and MPI-IO interfaces and create the
IOR executable at `src/ior`.

Running
=======

Input
-----

Sample production input files in `microbenchmarks/ior/inputs.xroads`

Running IOR does not require using the input files. All arguments can be given on the command line.

Example Results
===============

