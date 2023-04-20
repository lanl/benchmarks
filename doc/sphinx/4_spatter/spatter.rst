******
Spatter
******

This is the documentation for the ATS-5 Benchmark Spatter - Scatter/Gather kernels. 


Purpose
=======


Characteristics
===============

Problem
-------

Figure of Merit
---------------


Building
========

Accessing the benchmark, memory access patterns, and scaling scripts

.. code-block:: bash

   cd <path to benchmarks>
   git submodule update --init --recursive
   cd spatter

..

Set-up:

.. code-block:: bash

    git lfs pull
    tar -xvzf patterns/flag/static_2d/001.json.tar.gz
    tar -xvzf patterns/flag/static_2d/001.nonfp.json.tar.gz
    tar -xvzf patterns/flag/static_2d/001.fp.json.tar.gz
    tar -xvzf patterns/xrage/asteroid/spatter.json.tar.gz
    mv spatter.json patterns/xrage/asteroid/spatter.json

..

Create a Module Environment
Copy the current Darwin Module Environment File

For a base CPU Module file: 

.. code-block:: bash

    cp modules/darwin_skylake.mod modules/<name>.mod

..

For a base GPU Module file: 

.. code-block::

    cp modules/darwin_a100.mod modules/<name>.mod

..

Edit the modules required on your system

For CPU builds, you need CMake, an Intel Compiler, and MPI at a minimum For GPU builds (CUDA), you need CMake, nvcc, gcc, and MPI

For plotting (see scripts/plot_mpi.py), you will need a Python 3 installation with matplotlib and pandas

Editing the Configuration
Edit the configuration bash script (scripts/config.sh)

Change the HOMEDIR to the path to the root of this repository (absolute path). Change the MODULEFILE to your new module file (absolute path). Change threadlist and ranklist and sizelist as appropriate for your system. This sets the number of OpenMP threads or MPI ranks Spatter will scale through You may leave SPATTER unchanged unless you have another Spatter binary on your system. If so, you may update this variable to point to you Spatter binary. Otherwise, we will build Spatter in the next
step.


Build requirements:

Building Spatter on CPUs:

.. code-block:: bash
   
   cd ${HOMEDIR}
   bash scripts/build_omp_mpi_intel.sh

..

Building Spatter on NVIDIA GPUs:

.. code-block:: bash

    cd ${HOMEDIR}
    bash scripts/builds_cuda.sh

..


Running
=======

Running a Scaling Experiment
This will perform a weak scaling experiment

The scripts/scaling.sh script has the following options: 
* a: Application name 
* p: Problem name 
* f: Pattern name 
* n: Architecture/Partition 
* c: Core binding (optional, default: off) 
* g: Plotting/Post-processing (optional, default: on) 
* r: Toggle MPI scaling (optional, default: off) 
* t: Toggle OpenMP scaling (optional, default: off) 
* w: Toggle Weak/Strong Scaling (optional, default: off = strong scaling) 
* h: Print usage message

The Application name, Problem name, and Pattern name each correspond to subdirectories in this repository containing patterns stored as Spatter JSON input files. The JSON file of interest should be located at ///.json

The results of the weak scaling experiment will be stored in the spatter.weakscaling or spatter.strongscaling directory, within the /// subdirectory.

If MPI scaling is enabled, full bandwidth results will be stored in the mpi_r)1t.txt files. Additionally, the r subdirectories hold sorted bandwidth data for each rank from each pattern that was found in the Spatter JSON input file. These files will be labeled r/r_1t_<pattern_num>p.txt.

If OpenMP threading is turned on, full bandwidth results will be stored in the openmp_1r_t.txt files.

.. code-block:: bash

    cd spatter

    bash scripts/scaling.sh -a flag -p static_2d -f 001 -n A100 -r

..

Running Spatter Serially
Simply update the threadlist and ranklist variables in scripts/config.sh to the value of ( 1  )

.. code-block:: bash

    bash scripts/scaling.sh -a flag -p static_2d -f 001 -n A100 -r

..

CTS-1
------------

Power9+V100
------------

Verification of Results
=======================

References
==========

.. [Spatter] Jered Dominguez-Trujillo, Kevin Sheridan, Galen Shipman, 'Spatter', 2023. [Online]. Available: https://github.com/lanl/spatter. [Accessed: 19- Apr- 2023]
