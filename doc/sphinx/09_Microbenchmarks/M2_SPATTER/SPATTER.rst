*******
Spatter
*******

This is the documentation for the ATS-5 Benchmark Spatter - Scatter/Gather kernels. 


Purpose
=======

Micro-benchmark scatter/gather performance of current and future node-level architectures.

From the [Spatter]_ Benchmark Repository:

With this benchmark, we aim to characterize the performance of memory systems in a new way. We want to be able to make comparisons across architectures about how well data can be rearranged, and we want to be able to use benchmark results to predict the runtimes of sparse algorithms on these various architectures.

See [Spatter-Paper]_ for earlier details and results. Additionally, see [LANL-Memory-Wall]_ and [Spatter]_ for enhancements and additional features.

Characteristics
===============


Problem
-------

We have profiled the Flag and xRAGE applications to collect relevant gather/scatter patterns. These have been collected into a repository [LANL-Spatter]_, along with several utility scripts which allow us to perform weak scaling and strong scaling experiments with this data. The Flag patterns were obtained from a problem utilizing a 2D static mesh, while the xRAGE patterns were obtained from a 3D simulation of an asteroid impact.

Figure of Merit
---------------

The Figure of Merit is defined as the measured bandwidth in MB/s. This is measured for each rank to obtain the average bandwidth per rank. This is obtained by taking the total data movement divided by the runtime for the gather/scatter operation for each rank.

Building
========

Accessing the benchmark, memory access patterns, and scaling scripts [LANL-Spatter]

.. code-block:: bash

   cd <path to benchmarks>
   git submodule update --init --recursive
   cd microbenchmarks/spatter

..

Set-up:

The setup script will initialize your CPU configuration file (scripts/cpu_config.sh) with ATS-3 defaults and the GPU configurationo file (scripts/gpu_config.sh) with V100/A100 defaults, and will buid Spatter for CPU and GPU. See the Spatter documentation and other build scripts (scripts/build_cpu.sh and scripts/build_cuda.sh) for further instructions for building with different compilers or for GPUs.

The scripts/setup.sh scripts has the following options

* c: Toggle CPU Build (default: off)
* g: Toggle GPU Build (default: off)
* h: Print usage message

To setup and build for only the CPU, run the following:

.. code-block:: bash

   bash scripts/setup.sh -c

..


Or to build only for the GPU, run:

.. code-block:: bash

   bash scripts/setup.sh -g

..


To setup and build for both the CPU and GPU, run the following:

.. code-block:: bash

    bash scripts/setup.sh -c -g

..

This setup script performs the following:

#. Untars the Pattern JSON files located in the patterns directory

   * patterns/flag/static_2d/001.fp.json

   * patterns/flag/static_2d/001.nonfp.json

   * patterns/flag/static_2d/001.json

   * patterns/xrage/asteroid/spatter.json

#. Extracts patterns from patterns/xrage/asteroid/spatter.json to separate JSON files located at patterns/xrage/asteroid/spatter{1-9}.json

#. Generates default module files located in modules/cpu.mod and modules/gpu.mood

   * Contains generic module load statements for CPU and GPU dependencies

   * Assumes you are utilizing the module load system to configure environment. Change as needed (i.e. changes to PATH etc.) if you utilize a different system.

#. Populates the configuration file (scripts/cpu_config.sh) with reasonable defaults for a ATS-3 system

   * HOMEDIR is set to the directory this repository sits in

   * MODULEFILE is set to modules/cpu.mod

   * SPATTER is set to path of the Spatter CPU executable

   * ranklist is set to sweep from 1-112 ranks respectively for a ATS-3 type system

   * sizelist is set to reasonable defaults for strong scaling experiments (specifies the size of the pattern to truncate at)

   * count list is set to defaults of 1.

#. Populates the GPU configuration file (scripts/gpu_config.sh) with reasonable defaults for single-GPU throughput experiments on a V100 or A100 system

   * HOMEDIR is set to the directory this repository sits in

   * MODULEFILE is set to modules/gpu.mod

   * SPATTER is set to path of the Spatter GPU executable

   * ranklist is set to a constant of 1 for 8 different runs (8 single-GPU runs)

   * sizelist is set to reasonable defaults for strong scaling experiments (specifies the size of the pattern to truncate at)

   * countlist is set to reasonable defaults to control the number of gathers/scatters performed by an experiment. This is the parameter that is varied to perform throughput experiments.

#. Attempts to build Spatter on CPU with CMake, GCC, and MPI and on GPU with CMake and NVCC

    * You will need CMake, GCC, and MPI loaded into your environment (include them in your modules/cpu.mod if not already included)

    * You will need CMake, CUDA, and NVCC loaded into your environment for the GPU build (include them in your modules/gpu.mod)

Optional Manual Build
---------------------

In the case you need to build manually, the following scripts can be modified to build for CPU:

.. code-block:: bash

    bash scripts/build_cpu.sh

..

and to build for GPUs which support CUDA:

.. code-block:: bash

    bash scripts/build_cuda.sh

..

Further build documentation can be found here: [Spatter]_


Running
=======

Running a Scaling Experiment

This will perform a weak scaling experiment

The scripts/scaling.sh script has the following options (a scripts/mpirunscaling.sh script with identical options has been provided if required to use mpirun rather than srun): 

* a: Application name
* p: Problem name
* f: Pattern name
* n: User-defined run name (for saving results)
* c: Core binding (optional, default: off)
* g: Toggle GPU (optional, default: off)
* m: Toggle Atomics (optional, default: off)
* r: Toggle count parameter on pattern with countlist (default: off)
* s: Toggle pattern size limit (optional, default: off for weak scaling, will be overridden to on for strong scaling)
* t: Toggle throughput plot generation (optional, default: off)
* w: Toggle weak/strong scaling (optional, default: off = strong scaling)
* x: Toggle plotting/post-processing (optional, default: on)
* h: Print usage message

The Application name, Problem name, and Pattern name each correspond to subdirectories in this repository containing patterns stored as Spatter JSON input files.

All Figures use solid lines for Gathers and dashed lines for Scatters.

Crossroads
------------

These weak-scaling experiements were ran on 1, 2, 4, 8, 16, 32, 56, 64, 96, and 112 ranks with a single Crossroads node.

These experiments were ran with core-binding turned on and plotting enabled.

xRAGE Asteroid Spatter Pattern 5
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Weak-scaling experiment for the pattern in patterns/xrage/asteroid/spatter5.json. Results will be found in spatter.weakscaling/Crossroads/xrage/asteroid/spatter5/ and Figures will be found in figures/spatter.weakscaling/Crossroads/xrage/asteroid/spatter5

This pattern is a Gather with a length of 8,368,968 elements with a target vector length of 1,120,524.


.. code-block:: bash

   bash scripts/scaling.sh -a xrage -p asteroid -f spatter5 -n Crossroads -c -w

..

.. csv-table:: Spatter Weak Scaling Performance (MB/s per Rank) for xRAGE Spatter Pattern 5 on Crossroads
   :file: ats3_weak_average_asteroid_5.csv
   :align: center
   :widths: 5, 5
   :header-rows: 1

.. figure:: ats3_weak_average_asteroid_5.png
   :align: center
   :scale: 50%
   :alt: Spatter Weak Scaling Performance for xRAGE Spatter Pattern 5 on Crossroads

   Spatter Weak Scaling Performance for xRAGE Spatter Pattern 5 on Crossroads


xRAGE Asteroid Spatter Pattern 9
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Weak-scaling experiment for the pattern in patterns/xrage/asteroid/spatter5.json. Results will be found in spatter.weakscaling/Crossroads/xrage/asteroid/spatter9/ and Figures will be found in figures/spatter.weakscaling/Crossroads/xrage/asteroid/spatter9

This pattern is a Scatter with a length of 6,664,304 elements with a target vector length of 2,051,100.

.. code-block:: bash

   bash scripts/scaling.sh -a xrage -p asteroid -f spatter9 -n Crossroads -c -w

..

.. csv-table:: Spatter Weak Scaling Performance (MB/s per Rank) for xRAGE Spatter Pattern 9 on Crossroads
   :file: ats3_weak_average_asteroid_9.csv
   :align: center
   :widths: 5, 5
   :header-rows: 1

.. figure:: ats3_weak_average_asteroid_9.png
   :align: center
   :scale: 50%
   :alt: Spatter Weak Scaling Performance for xRAGE Spatter Pattern 9 on Crossroads

   Spatter Weak Scaling Performance for xRAGE Spatter Pattern 9 on Crossroads


H100
------------

Strong-Scaling throughput experiments with plotting enabled.


xRAGE Asteroid Spatter Pattern 5
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Throughput experiment for the pattern in patterns/xrage/asteroid/spatter5.json. Results will be found in spatter.strongscaling/H100/xrage/asteroid/spatter5/ and Figures will be found in figures/spatter.strongscaling/H100/xrage/asteroid/spatter5/

.. code-block:: bash

   bash scripts/scaling.sh -a xrage -p asteroid -f spatter5 -n H100 -g -s -r -t

..

.. csv-table:: Spatter Throughput (MB/s) on H100 xRAGE Asteroid Pattern 5
   :file: h100_throughput_asteroid_5.csv
   :align: center
   :widths: 5, 5
   :header-rows: 1

.. figure:: h100_throughput_asteroid_5.png
   :align: center
   :scale: 50%
   :alt: Spatter Throughput on H100 xRAGE Asteroid Pattern 5 on H100

   Spatter Throughput on H100 xRAGE Asteroid Pattern 5 on H100


xRAGE Asteroid Spatter Pattern 9
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Throughput experiment for the pattern in patterns/xrage/asteroid/spatter9.json. Results will be found in spatter.strongscaling/H100/xrage/asteroid/spatter9/ and Figures will be found in figures/spatter.strongscaling/H100/xrage/asteroid/spatter9/

Note that we need to enable atomics since this is a scatter pattern which overwrites the same location multiple times.

.. code-block:: bash

   bash scripts/scaling.sh -a xrage -p asteroid -f spatter9 -n H100 -g -s -r -t -m

..

.. csv-table:: Spatter Throughput (MB/s) on H100 xRAGE Asteroid Pattern 9
   :file: h100_throughput_asteroid_9.csv
   :align: center
   :widths: 5, 5, 5
   :header-rows: 1

.. figure:: h100_throughput_asteroid_9.png
   :align: center
   :scale: 50%
   :alt: Spatter Throughput on H100 xRAGE Asteroid Pattern 9 on H100

   Spatter Throughput on H100 xRAGE Asteroid Pattern 9 on H100


References
==========

.. [Spatter] Patrick Lavin, Jeffrey Young, Jered Dominguez-Trujillo, Agustin Vaca Valverde, Vincent Huang, James Wood, 'Spatter', 2023. [Online]. Available: https://github.com/hpcgarage/spatter
.. [Spatter-Paper] Lavin, P., Young, J., Vuduc, R., Riedy, J., Vose, A. and Ernst, D., Evaluating Gather and Scatter Performance on CPUs and GPUs. In The International Symposium on Memory Systems (pp. 209-222). September 2020.
.. [LANL-Spatter] Jered Dominguez-Trujillo, Kevin Sheridan, Galen Shipman, 'Spatter', 2023. [Online]. Available: https://github.com/lanl/spatter. [Accessed: 19- Apr- 2023]
.. [LANL-Memory-Wall] G. M. Shipman, J. Dominguez-Trujillo, K. Sheridan and S. Swaminarayan, "Assessing the Memory Wall in Complex Codes," 2022 IEEE/ACM Workshop on Memory Centric High Performance Computing (MCHPC), Dallas, TX, USA, 2022, pp. 30-35, doi: 10.1109/MCHPC56545.2022.00009.
