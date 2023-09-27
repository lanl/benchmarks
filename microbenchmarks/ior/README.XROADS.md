Crossroads/NERSC-9 IOR Benchmark 
================================================================================

I. Benchmark Description
--------------------------------------------------------------------------------
IOR is designed to measure parallel I/O performance at both the POSIX and MPI-IO
level.  All of the general run rules for APEX benchmarking apply.


II. Build Instructions
--------------------------------------------------------------------------------
MPI, MPI-IO, and OpenMP are required in order to build and run the code. The
source code used for this benchmark is derived from IOR 3.0.1 and it is
included here.  More information about IOR is available on 
https://github.com/LLNL/ior.

After extracting the tar file, ensure that the MPI compiler wrappers (e.g.,
`mpicc`) are in `$PATH` and then

    cd ior-3.0.1-xroads
    ./configure
    make

This will build both IOR with the POSIX and MPI-IO interfaces and create the
IOR executable at `src/ior`.


III. Run Rules
--------------------------------------------------------------------------------
The intent of these benchmarks is to measure the performance of I/O operations
on the platform storage under three loads:

1. fully sequential, large-transaction reads and writes
2. fully random, small-transaction reads and writes

Observed benchmark performance shall be obtained from a storage system
configured as closely as possible to the proposed platform storage.  If the
proposed solution includes multiple file access protocols (e.g., pNFS and NFS)
or multiple tiers accessible by applications, benchmark results for IOR
shall be provided for each protocol and/or tier.

Performance projections are permissible if they are derived from a similar
system that is considered an earlier generation of the proposed system.

### Load 1: Sequential Loads

This benchmark is intended to measure the throughput of the storage subsystem
and contains features that minimize caching/buffering effects.  As such, the
Offerer should not utilize optimizations that cache or buffer the transferred
data in compute node memory.

The Offeror shall run the following tests:

* MPI/IO file per process (i.e., N processes operate on N files)
* MPI/IO shared file (i.e., N processes operate on 1 file)
* POSIX I/O file per process 
* POSIX I/O shared file 

Each of these four tests must be run at the following levels of concurrency:

1. a single compute node
2. 10% of the proposed system's compute nodes
3. 50% of the proposed system's compute nodes
4. sufficient compute nodes to achieve the maximum bandwidth results

These tests must be configured via a combination of input configuration files
and command line options.  Annotated input configuration files are provided in
the `inputs.xroads/` subdirectory and demonstrate how these tests can be defined
for the purposes of these benchmarks.

The Offeror MUST modify the following parameters for each benchmark test:

* `numTasks` - the number of MPI processes to use.  The Offeror may choose to
  run multiple MPI processes per compute node to achieve the highest bandwidth
  results.
* `segmentCount` - number of segments (blocks * numTasks) in a file.  This
  governs the size of the file(s) written/read, and the amount of data 
  written/read by each node must exceed 1.5 times the memory available for the
  file system's page cache (typically the entire node's RAM).
* `memoryPerNode` - size (in %) of each node's RAM to be filled before
  running the benchmark test.  This value must be no less than 80% of the total
  RAM available on each compute node and is intended to represent the memory
  footprint of a real application.

In addition, the Offeror MAY modify the following parameters for each test:

* `transferSize` - the size (in bytes) of a single data buffer to be transferred
   in a single I/O call.  The Offeror should find the transferSize that produces
   the highest bandwidth results and report this optimal transferSize.
   `blockSize` must always be equal to `transferSize`.
* `testFile` - path to data files to be read or written for this benchmark
* `hintsFileName` - path to MPI-IO hints file
* `collective` - MPI-IO collective vs. independent operation mode

As mentioned above, `segmentCount` must be set so that the total amount of
data written is greater than 1.5 times the amount of RAM on the compute nodes.
The total fileSize is given by

    fileSize = segmentCount * blockSize * numTasks

So for a 10-node test with an aggregate 640 GB of RAM, fileSize must be at
least 960 GB.  Assuming `blockSize=1MB` and `numTasks=240` (24 MPI processes
per node) is optimal, an appropriate `segmentCount` would be

    segmentCount = fileSize / ( blockSize * numTasks ) = 4096

Page caches on the storage subsystem's servers may still be used, but they must
be configured as they would be for the delivered Crossroads/NERSC-9 systems.

Providing an MPI-IO "hints" file for the MPI-IO runs, which IOR will look for in
the file specified by the `hintsFileName` keyword in the input file, is allowed.
Documentation on IOR's support for MPI-IO hints can be found in the "HOW DO I
USE HINTS?" section of the IOR User Guide (found in `doc/USER_GUIDE`).

### Load 2: Fully Random Loads

As with Load 1, this benchmark is intended to measure the throughput of the
storage subsystem and contains features that minimize caching/buffering effects.
As such, the Offerer should not utilize optimizations that cache or buffer the
transferred data in system memory.

The Offeror shall run the following tests:

* POSIX I/O file per process 
* POSIX I/O shared file 

Both of these tests must be run at the following levels of concurrency:

1. a single compute node
2. 10% of the proposed system's compute nodes
3. 20% of the proposed system's compute nodes

These tests must be configured via a combination of input configuration files
and command line options.  Annotated input configuration files are provided in
the `inputs.xroads/` subdirectory and demonstrate how these tests can be defined
for the purposes of these benchmarks.

The Offeror MUST modify the following parameters for each benchmark test:

* `numTasks` - the number of MPI processes to use.  The Offeror may choose to
  run multiple MPI processes per compute node to achieve the highest bandwidth
  results.
* `segmentCount` - number of segments (blocks * numTasks) in a file.  This
  governs the size of the file(s) written/read, and the amount of data 
  written/read by each node must exceed 1.5 times the memory available for the
  file system's page cache (typically the entire node's RAM).
* `memoryPerNode` - size (in %) of each node's RAM to be filled before
  running the benchmark test.  This value must be no less than 80% of the total
  RAM available on each compute node and is intended to represent the memory
  footprint of a real application.

In addition, the Offeror MAY modify the following parameter for each test:

* `testFile` - path to data files to be read or written for this benchmark

As with the other loads, `segmentCount` must be set so that the total amount
of data written is greater than 1.5 times the amount of RAM on the compute
nodes.  The total fileSize is given by

    fileSize = segmentCount * 4K * numTasks

So for a 10-node test with an aggregate 640 GB of RAM, fileSize must be at
least 960 GB.  Assuming `numTasks=240` (24 MPI processes per node), an
appropriate `segmentCount` would be

    segmentCount = fileSize / ( 4K * numTasks ) = 1048576

### Running IOR

IOR is executed as any other standard MPI application would be on the proposed
system.  For example,

    mpirun -np 64 ./ior -f ./load1-posix-filepertask.ior

or

    srun -n 64 ./ior -f ./load1-posix-filepertask.ior

will execute IOR with 64 processes and use the input configuration file called
`load1-posix-filepertask.ior`.

Example input configuration files for all required tests are supplied in the
`inputs.apex/` directory with additional annotations and details where
appropriate.


IV. Permitted Modifications
--------------------------------------------------------------------------------

Modifications to the benchmark application code are only permissible to enable
correct compilation and execution on the target platform.  Any modifications
must be fully documented (e.g., as a diff or patch file) and reported with the
benchmark results.


V. Reporting Results
--------------------------------------------------------------------------------

### Load 1: Sequential Loads

IOR will execute both read and write tests for each run.  The bandwidth
measurements to be reported are the `Max Write` and `Max Read` values (in 
units of `MB/s`) reported to stdout.  In addition, the concurrency (number of
compute nodes and number of MPI processes used) for each run must be stated.

The maximum write and read bandwidths for a single read-and-write test must be
reported for Load 1.  Reporting the maximum read bandwidth from one run and the
maximum write bandwidth from a different run is NOT valid.  Write bandwidth
has slightly higher importance for this test, so if the highest observed read
rate came from a different run than the highest observed write rate, report the
results from the run with the highest write rate.

### Load 2: Fully Random Loads

IOR will execute both read and write tests for each run.  The bandwidth
measurements to be reported are the `Max Write` and `Max Read` values (in 
units of `MB/s`) reported to stdout.

As with Load 1, the maximum write and read bandwidths for a single 
read-and-write test must be reported for Load 2.  Read bandwidth has slightly
higher importance for this test, so report results from the run with the highest
read rate.

### Benchmark Platform Description

In addition to maximum bandwidths, the Offeror must also provide a comprehensive
description of the environment in which each benchmark was run.  This must
include:

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

