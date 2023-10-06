Crossroads/NERSC-9 mdtest Benchmark
================================================================================

I. Benchmark Description
--------------------------------------------------------------------------------
mdtest is designed to measure the performance of various metadata operations and
uses MPI to coordinate the operations and to collect the results.  All of the
general run rules for XRoads benchmarking apply.


II. Build Instructions
--------------------------------------------------------------------------------
MPI is required in order to build and run the code.  The source code used for
this benchmark is derived from mdtest 1.8.4 and it is included with this
benchmark specification.  More information about mdtest is available on 
http://mdtest.sourceforge.net.

After extracting the tar file, ensure that the MPI compiler wrappers (e.g.,
`mpicc`) are in `$PATH` and then

    cd mdtest-1.8.4-xroads
    make

This will build the mdtest executable, called `mdtest`.  It may be necessary to
specify the `CC`, `CFLAGS`, and `LDFLAGS` variables to ensure correct 
compilation of `mdtest`.  A simplified Makefile, `Makefile.XROADS`, is also
provided to this end, e.g.,

    make -f Makefile.XROADS CC=mpicc CFLAGS=-g

Either `make` or `make -f Makefile.XROADS` can be used to build the binary used
for this benchmark, but any additional `CFLAGS` or `LDFLAGS` required for
compilation must be reported with the benchmark results.


III. Run Rules
--------------------------------------------------------------------------------
The intent of this benchmark is to measure the performance of file metadata
operations on the platform storage.  

Observed benchmark performance shall be obtained from a storage system
configured as closely as possible to the proposed platform storage. If the
proposed solution includes multiple file access protocols (e.g., pNFS and NFS)
or multiple tiers accessible by applications, benchmark results for mdtest
shall be provided for each protocol and/or tier.

Performance projections are permissible if they are derived from a similar
system that is considered an earlier generation of the proposed system.

### Required Runs

This benchmark is intended to measure the capability of the storage subsystem
to create and delete files, and it contains features that minimize 
caching/buffering effects.  As such, the Offerer should not utilize
optimizations that cache/buffer file metadata or metadata operations in compute
node memory.

The Offeror shall run the following tests:

* creating, statting, and removing at least 1,048,576 files in a single directory.
* creating, statting, and removing at least 1,048,576 files in separate
  directories (one directory per MPI process)
* creating, statting, and removing one file by multiple MPI processes

Each of these tests must be run at the following levels of concurrency:

1. a single MPI process
2. the optimal number of MPI processes on a single compute node
3. the minimal number of MPI processes on multiple compute nodes that achieves
   the peak results for the proposed system
4. the maximum possible MPI-level concurrency on the proposed system.  This
   could mean
   * using one MPI process per CPU core across the entire system
   * using the maximum number of MPI processes possible if one MPI process per
     core will not be possible on the proposed architecture
   * using more than 1,048,576 files if the system is capable of launching
     more than 1,048,576 MPI processes

These tests are configured via command-line arguments, and the following
section provides guidance on passing the correct options to `mdtest` for each
test.

### Running mdtest

mdtest is executed as any other standard MPI application would be on the
proposed system (e.g., with `mpirun` or `srun`).  For the sake of the
following examples, `mpirun` is used.

**To run create, stat, and delete tests on files in a shared directory**, an
appropriate `mdtest` command-line invocation may look like

    mpirun -np 64 ./mdtest -F -C -T -r -n 16384 -d /scratch -N 16

The following command-line flags MUST be changed:

* `-n` - the number of files **each MPI process** should manipulate.  For a
  test run with 64 MPI processes, specifying `-n 16384` will produce the
  required 1048576 files (64 MPI processes x 16384).  This parameter must
  be changed for each level of concurrency.
* `-d /scratch` - the directory in which this test should be run.  **This
  must be an absolute path.**
* `-N` - MPI rank offset for each separate phase of the test.  This parameter
  must be equal to the number of MPI processes per node in use (e.g., `-N 16`
  for a test with 16 processes per node) to ensure that each test phase (read,
  stat, and delete) is performed on a different node.

The following command-line flags MUST NOT be changed or omitted:

* `-F` - only operate on files, not directories
* `-C` - perform file creation test
* `-T` - perform file stat test
* `-r` - perform file remove test

**To have each MPI process write files into a unique directory,** add the `-u`
option:

    mpirun -np 64 ./mdtest -F -C -T -r -n 16384 -d /scratch -N 16 -u

**To create, stat, and remove one file by multiple MPI processes,** add the `-S`
option:

    mpirun -np 64 ./mdtest -F -C -T -r -n 16384 -d /scratch -N 16 -S


IV. Permitted Modifications
--------------------------------------------------------------------------------

Modifications to the benchmark application code are only permissible to enable
correct compilation and execution on the target platform.  Any modifications
must be fully documented (e.g., as a diff or patch file) and reported with the
benchmark results.


V. Reporting Results
--------------------------------------------------------------------------------

mdtest will execute file creation, file statting, and file deletion tests for
each run.  The rate of file creating/statting/deleting are reported to stdout
at the conclusion of each test, and the following rates should be reported:

* `File creation`
* `File stat`
* `File removal`

The maximum values for these rates must be reported for all tests.  Reporting
the maximum creation rates from one run and the maximum deletion rates from a
different run is NOT valid.  File creation rate has slightly higher importance
for this test, so if the highest observed file creation rate came from a
different run than the highest observed deletion rate, report the results from
the run with the highest file creation rate.

### Benchmark Platform Description

The Offeror must provide a comprehensive description of the environment in which
each benchmark was run.  This must include:

* Client and server system configurations, including node and processor counts,
  processor models, memory size and speed, and OS (names and versions)
* Storage media and their configurations used for each tier of the storage
  subsystem
* Network fabric used to connect servers, clients, and storage, including
  network configuration settings and topology
* Client and server configuration settings including
    * Client and server sysctl settings
    * Driver options
    * Network interface options
    * File system configuration and mount options
* Compiler name and version, compiler options, and libraries used to build
  benchmarks

