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
   cd spatter

..

Set-up:

The setup script will initialize your CPU configuration file (scripts/cpu_config.sh) with CTS-1 defaults and the GPU configurationo file (scripts/gpu_config.sh) with V100/A100 defaults, and will buid Spatter for CPU and GPU. See the Spatter documentation and other build scripts (scripts/build_cpu.sh and scripts/build_cuda.sh) for further instructions for building with different compilers or for GPUs.

The scripts/setup.sh scripts has the following options

* c: Toggle CPU Build (default: off)
* g: Toggle GPU Build (default: off)
* h: Print usage message

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

#. Generates default module files located in modules/cpu.mod and modules/gpu.mood

   * Contains generic module load statements for CPU and GPU dependencies

   * Assumes you are utilizing the module load system to configure environment. Change as needed (i.e. changes to PATH etc.) if you utilize a different system.

#. Populates the configuration file (scripts/cpu_config.sh) with reasonable defaults for a CTS-1 system

   * HOMEDIR is set to the directory this repository sits in

   * MODULEFILE is set to modules/cpu.mod

   * SPATTER is set to path of the Spatter CPU executable

   * ranklist is set to sweep from 1-36 threads/ranks respectively for a CTS-1 type system

   * boundarylist is set to reasonable defaults for scaling experiments (specifies the maximum value of a pattern index, limiting the size of the data array)

   * sizelist is set to reasonable defaults for strong scaling experiments (specifies the size of the pattern to truncate at)

#. Poopulates the GPU configuration file (scripts/gpu_config.sh) with reasonable defaults for single-GPU throughput experiments on a V100 or A100 system

   * HOMEDIR is set to the directory this repository sits in

   * MODULEFILE is set to modules/gpu.mod

   * SPATTER is set to path of the Spatter GPU executable

   * ranklist is set to a constant of 1 for 8 different runs (8 single-GPU runs)

   * boundarylist is set to reasonable defaults for scaling experiments (specifies the maximum value of a pattern index, limiting the size of the data array)

   * sizelist is set to reasonable defaults for strong scaling experiments (specifies the size of the pattern to truncate at)

   * countlist is set to reasonable defaults to control the number of gathers/scatters performed by an experiment. This is the parameter that is varied to perform throughput experiments.

#. Attempts to build Spatter on CPU with CMake, GCC, and MPI and on GPU with CMake and NVCC

    * You will need CMake, GCC, and MPI loaded into your environment (include them in your modules/cpu.mod if not already included)

    * You will need CMAke, CUDA, and NVCC loaded into your environment for the GPU build (include them in your modules/gpu.mod)

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
* b: Toggle boundary limit (option, default: off for weak scaling, will be overridden to on for strong scaling)
* c: Core binding (optional, default: off)
* g: Toggle GPU (optional, default: off)
* s: Toggle pattern size limit (optional, default: off for weak scaling, will be overridden to on for strong scaling)
* t: Toggle throughput plot generation (optional, default: off)
* w: Toggle weak/strong scaling (optional, default: off = strong scaling)
* x: Toggle plotting/post-processing (optional, default: on)
* h: Print usage message

The Application name, Problem name, and Pattern name each correspond to subdirectories in this repository containing patterns stored as Spatter JSON input files.

All Figures use solid lines for Gathers and dashed lines for Scatters.


CTS-1
------------


Flag Static 2D 001
~~~~~~~~~~~~~~~~~~

Weak-scaling experiment for the 8 patterns in patterns/flag/static_2d/001.json with core-binding turned on and plotting enabled. This experiment was ran at 1, 2, 4, 8, 16, 18, 32, and 36 ranks. Results will be found in spatter.weakscaling/CTS1/flag/static_2d/001/ and Figures will be found in figures/spatter.weakscaling/CTS1/flag/static_2d/001/

.. code-block:: bash

   bash scripts/scaling.sh -a flag -p static_2d -f 001 -n CTS1 -c -w

..

.. csv-table:: Spatter Weak Scaling Performance (MB/s per rank) on CTS-1 Flag Static 2D 001 Patterns
   :file: cts1_weak_average_001.csv
   :align: center
   :widths: 5, 8, 8, 8, 8, 8, 8, 8, 8
   :header-rows: 1

.. figure:: cts1_weak_average_001.png
   :align: center
   :scale: 50%
   :alt: Spatter Weak Scaling Performance on CTS-1 Flag Static 2D 001 Patterns

   Spatter Weak Scaling Performance on CTS-1 Flag Static 2D 001 Patterns


Flag Static 2D 001.FP
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   bash scripts/scaling.sh -a flag -p static_2d -f 001.fp -n CTS1 -c -w

..

.. csv-table:: Spatter Weak Scaling Performance (MB/s per rank) on CTS-1 Flag Static 2D 001 FP Patterns
   :file: cts1_weak_average_001fp.csv
   :align: center
   :widths: 5, 8, 8, 8, 8
   :header-rows: 1

.. figure:: cts1_weak_average_001fp.png
   :align: center
   :scale: 50%
   :alt: Spatter Weak Scaling Performance on CTS-1 Flag Static 2D 001 FP Patterns

   Spatter Weak Scaling Performance on CTS-1 Flag Static 2D 001 FP Patterns


Flag Static 2D 001.NONFP
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   bash scripts/scaling.sh -a flag -p static_2d -f 001.nonfp -n CTS1 -c -w

..

.. csv-table:: Spatter Weak Scaling Performance (MB/s per rank) on CTS-1 Flag Static 2D 001 Non-FP Patterns
   :file: cts1_weak_average_001nonfp.csv
   :align: center
   :widths: 5, 8, 8, 8, 8, 8, 8, 8, 8
   :header-rows: 1

.. figure:: cts1_weak_average_001nonfp.png
   :align: center
   :scale: 50%
   :alt: Spatter Weak Scaling Performance on CTS-1 Flag Static 2D 001 Non-FP Patterns

   Spatter Weak Scaling Performance on CTS-1 Flag Static 2D 001 Non-FP Patterns


xRAGE Asteroid
~~~~~~~~~~~~~~

Weak-scaling experiment for the x patterns in patterns/xrage/asteroid/spatter.json with core-binding turned on and plotting enabled. This experiment was ran at 1, 2, 4, 8, 16, and 18 ranks due to memory constraints. Results will be found in spatter.weakscaling/CTS1/xrage/asteroid/spatter/ and Figures will be found in figures/spatter.weakscaling/CTS1/xrage/asteroid/spatter/

First, modifying the ranklist in scripts/cpu_config.sh to the following:

.. code-block:: bash

   ranks=( 1 2 4 8 16 18 )

..

.. code-block:: bash

   bash scripts/scaling.sh -a xrage -p asteroid -f spatter -n CTS1 -c -w

..

.. csv-table:: Spatter Weak Scaling Performance (MB/s per rank) on CTS-1 xRAGE Asteroid Patterns
   :file: cts1_weak_average_asteroid.csv
   :align: center
   :widths: 5, 8, 8, 8, 8, 8, 8, 8, 8, 8
   :header-rows: 1

.. figure:: cts1_weak_average_asteroid.png
   :align: center
   :scale: 50%
   :alt: Spatter Weak Scaling Performance on CTS-1 xRAGE Asteroid Patterns

   Spatter Weak Scaling Performance on CTS-1 xRAGE Asteroid Patterns


Skylake
------------


Flag Static 2D 001
~~~~~~~~~~~~~~~~~~

Weak-scaling experiment for the 8 patterns in patterns/flag/static_2d/001.json with core-binding turned on and plotting enabled. This experiment was ran at 1, 2, 4, 8, 16, 22, 32, and 44 ranks. Results will be found in spatter.weakscaling/Skylake/flag/static_2d/001/ and Figures will be found in figures/spatter.weakscaling/Skylake/flag/static_2d/001/

First, modifying the ranklist in scripts/cpu_config.sh to the following:

.. code-block:: bash

   ranks=( 1 2 4 8 16 22 32 44 )

..


.. code-block:: bash

   bash scripts/scaling.sh -a flag -p static_2d -f 001 -n Skylake -c -w

..

.. csv-table:: Spatter Weak Scaling Performance (MB/s per rank) on Skylake Flag Static 2D 001 Patterns
   :file: skylake_weak_average_001.csv
   :align: center
   :widths: 5, 8, 8, 8, 8, 8, 8, 8, 8
   :header-rows: 1

.. figure:: skylake_weak_average_001.png
   :align: center
   :scale: 50%
   :alt: Spatter Weak Scaling Performance on Skylake Flag Static 2D 001 Patterns

   Spatter Weak Scaling Performance on Skylake Flag Static 2D 001 Patterns


Flag Static 2D 001.FP
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   bash scripts/scaling.sh -a flag -p static_2d -f 001.fp -n Skylake -c -w

..

.. csv-table:: Spatter Weak Scaling Performance (MB/s per rank) on Skylake Flag Static 2D 001 FP Patterns
   :file: skylake_weak_average_001fp.csv
   :align: center
   :widths: 5, 8, 8, 8, 8
   :header-rows: 1

.. figure:: skylake_weak_average_001fp.png
   :align: center
   :scale: 50%
   :alt: Spatter Weak Scaling Performance on Skylake Flag Static 2D 001 FP Patterns

   Spatter Weak Scaling Performance on Skylake Flag Static 2D 001 FP Patterns



Flag Static 2D 001.NONFP
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   bash scripts/scaling.sh -a flag -p static_2d -f 001.nonfp -n Skylake -c -w

..

.. csv-table:: Spatter Weak Scaling Performance (MB/s per rank) on Skylake Flag Static 2D 001 Non-FP Patterns
   :file: skylake_weak_average_001nonfp.csv
   :align: center
   :widths: 5, 8, 8, 8, 8, 8, 8, 8, 8
   :header-rows: 1

.. figure:: skylake_weak_average_001nonfp.png
   :align: center
   :scale: 50%
   :alt: Spatter Weak Scaling Performance on Skylake Flag Static 2D 001 Non-FP Patterns

   Spatter Weak Scaling Performance on Skylake Flag Static 2D 001 Non-FP Patterns


xRAGE Asteroid
~~~~~~~~~~~~~~

Weak-scaling experiment for the 9 patterns in patterns/xrage/asteroid/spatter.json with core-binding turned on and plotting enabled. This experiment was ran at 1, 2, 4, 8, 16, and 22 ranks due to memory constraints. Results will be found in spatter.weakscaling/Skylake/xrage/asteroid/spatter/ and Figures will be found in figures/spatter.weakscaling/Skylake/xrage/asteroid/spatter/

First, modifying the ranklist in scripts/cpu_config.sh to the following:

.. code-block:: bash

   ranks=( 1 2 4 8 16 22 )

..

.. code-block:: bash

   bash scripts/scaling.sh -a xrage -p asteroid -f spatter -n Skylake -c -w

..

.. csv-table:: Spatter Weak Scaling Performance (MB/s per rank) on Skylake xRAGE Asteroid Patterns
   :file: skylake_weak_average_asteroid.csv
   :align: center
   :widths: 5, 8, 8, 8, 8, 8, 8, 8, 8, 8
   :header-rows: 1

.. figure:: skylake_weak_average_asteroid.png
   :align: center
   :scale: 50%
   :alt: Spatter Weak Scaling Performance on Skylake xRAGE Asteroid Patterns

   Spatter Weak Scaling Performance on Skylake xRAGE Asteroid Patterns


V100
------------

Strong-Scaling throughput experiment with plotting enabled. Results will be found in spatter.strongscaling/V100/flag/static_2d/001 and Figures will be found in figures/spatter.strongscaling/V100/flag/static_2d/001.


Flag Static 2D 001
~~~~~~~~~~~~~~~~~~

Throughput experiment for the 8 patterns in patterns/flag/static_2d/001.json on a single GPU with plotting enabled. Results will be found in spatter.strongscaling/V100/flag/static_2d/001/ and Figures will be found in figures/spatter.strongscaling/V100/flag/static_2d/001/

.. code-block:: bash

   bash scripts/scaling.sh -a flag -p static_2d -f 001 -n V100 -g -t

..

.. csv-table:: Spatter Throughput (MB/s) on V100 Flag Static 2D 001 Patterns
   :file: v100_throughput_001.csv
   :align: center
   :widths: 5, 8, 8, 8, 8, 8, 8, 8, 8
   :header-rows: 1

.. figure:: v100_throughput_001.png
   :align: center
   :scale: 50%
   :alt: Spatter Throughput on V100 Flag Static 2D 001 Patterns

   Spatter Throughput on V100 Flag Static 2D 001 Patterns


Flag Static 2D 001.FP
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   bash scripts/scaling.sh -a flag -p static_2d -f 001.fp -n V100 -g -t

..

.. csv-table:: Spatter Throughput (MB/s) on V100 Flag Static 2D 001 FP Patterns
   :file: v100_throughput_001fp.csv
   :align: center
   :widths: 5, 8, 8, 8, 8
   :header-rows: 1

.. figure:: v100_throughput_001fp.png
   :align: center
   :scale: 50%
   :alt: Spatter Throughput on V100 Flag Static 2D 001 FP Patterns

   Spatter Throughput on V100 Flag Static 2D 001 FP Patterns



Flag Static 2D 001.NONFP
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   bash scripts/scaling.sh -a flag -p static_2d -f 001.nonfp -n V100 -g -t

..

.. csv-table:: Spatter Throughput (MB/s) on V100 Flag Static 2D 001 Non-FP Patterns
   :file: v100_throughput_001nonfp.csv
   :align: center
   :widths: 5, 8, 8, 8, 8, 8, 8, 8, 8
   :header-rows: 1

.. figure:: v100_throughput_001nonfp.png
   :align: center
   :scale: 50%
   :alt: Spatter Throughput on V100 Flag Static 2D 001 Non-FP Patterns

   Spatter Throughput on V100 Flag Static 2D 001 Non-FP Patterns


xRAGE Asteroid
~~~~~~~~~~~~~~

Throughput experiment for the 9 patterns in patterns/xrage/asteroid/spatter.json with plotting enabled. Results will be found in spatter.strongscaling/V100/xrage/asteroid/spatter/ and Figures will be found in figures/spatter.strongscaling/V100/xrage/asteroid/spatter/

.. code-block:: bash

   bash scripts/scaling.sh -a xrage -p asteroid -f spatter -n V100 -g -t

..

.. csv-table:: Spatter Throughput (MB/s) on V100 xRAGE Asteroid Patterns
   :file: v100_throughput_asteroid.csv
   :align: center
   :widths: 5, 8, 8, 8, 8, 8, 8, 8, 8, 8
   :header-rows: 1

.. figure:: v100_throughput_asteroid.png
   :align: center
   :scale: 50%
   :alt: Spatter Throughput on V100 xRAGE Asteroid Patterns

   Spatter Throughput on V100 xRAGE Asteroid Patterns



A100
------------

Strong-Scaling throughput experiment with plotting enabled. Results will be found in spatter.strongscaling/A100/flag/static_2d/001 and Figures will be found in figures/spatter.strongscaling/A100/flag/static_2d/001.

.. code-block:: bash

    cd spatter

    bash scripts/scaling.sh -a flag -p static_2d -f 001 -n A100 -g -t

..

Flag Static 2D 001
~~~~~~~~~~~~~~~~~~

Throughput experiment for the 8 patterns in patterns/flag/static_2d/001.json on a single GPU with plotting enabled. Results will be found in spatter.strongscaling/A100/flag/static_2d/001/ and Figures will be found in figures/spatter.strongscaling/A100/flag/static_2d/001/

.. code-block:: bash

   bash scripts/scaling.sh -a flag -p static_2d -f 001 -n A100 -g -t

..

.. csv-table:: Spatter Throughput (MB/s) on A100 Flag Static 2D 001 Patterns
   :file: a100_throughput_001.csv
   :align: center
   :widths: 5, 8, 8, 8, 8, 8, 8, 8, 8
   :header-rows: 1

.. figure:: a100_throughput_001.png
   :align: center
   :scale: 50%
   :alt: Spatter Throughput on A100 Flag Static 2D 001 Patterns

   Spatter Throughput ono A100 Flag Static 2D 001 Patterns


Flag Static 2D 001.FP
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   bash scripts/scaling.sh -a flag -p static_2d -f 001.fp -n A100 -g -t

..

.. csv-table:: Spatter Throughput (MB/s) on A100 Flag Static 2D 001 FP Patterns
   :file: a100_throughput_001fp.csv
   :align: center
   :widths: 5, 8, 8, 8, 8
   :header-rows: 1

.. figure:: a100_throughput_001fp.png
   :align: center
   :scale: 50%
   :alt: Spatter Throughput on A100 Flag Static 2D 001 FP Patterns

   Spatter Throughput on A100 Flag Static 2D 001 FP Patterns



Flag Static 2D 001.NONFP
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   bash scripts/scaling.sh -a flag -p static_2d -f 001.nonfp -n A100 -g -t

..

.. csv-table:: Spatter Throughput (MB/s) on A100 Flag Static 2D 001 Non-FP Patterns
   :file: a100_throughput_001nonfp.csv
   :align: center
   :widths: 5, 8, 8, 8, 8, 8, 8, 8, 8
   :header-rows: 1

.. figure:: a100_throughput_001nonfp.png
   :align: center
   :scale: 50%
   :alt: Spatter Throughput on A100 Flag Static 2D 001 Non-FP Patterns

   Spatter Throughput on A100 Flag Static 2D 001 Non-FP Patterns


xRAGE Asteroid
~~~~~~~~~~~~~~

Throughput experiment for the 9 patterns in patterns/xrage/asteroid/spatter.json with plotting enabled. Results will be found in spatter.strongscaling/A100/xrage/asteroid/spatter/ and Figures will be found in figures/spatter.strongscaling/A100/xrage/asteroid/spatter/

.. code-block:: bash

   bash scripts/scaling.sh -a xrage -p asteroid -f spatter -n A100 -g -t

..

.. csv-table:: Spatter Throughput (MB/s) on A100 xRAGE Asteroid Patterns
   :file: a100_throughput_asteroid.csv
   :align: center
   :widths: 5, 8, 8, 8, 8, 8, 8, 8, 8, 8
   :header-rows: 1

.. figure:: a100_throughput_asteroid.png
   :align: center
   :scale: 50%
   :alt: Spatter Throughput on A100 xRAGE Asteroid Patterns

   Spatter Throughput on A100 xRAGE Asteroid Patterns



References
==========

.. [Spatter] Patrick Lavin, Jeffrey Young, Jered Dominguez-Trujillo, Agustin Vaca Valverde, Vincent Huang, James Wood, 'Spatter', 2023. [Online]. Available: https://github.com/hpcgarage/spatter
.. [Spatter-Paper] Lavin, P., Young, J., Vuduc, R., Riedy, J., Vose, A. and Ernst, D., Evaluating Gather and Scatter Performance on CPUs and GPUs. In The International Symposium on Memory Systems (pp. 209-222). September 2020.
.. [LANL-Spatter] Jered Dominguez-Trujillo, Kevin Sheridan, Galen Shipman, 'Spatter', 2023. [Online]. Available: https://github.com/lanl/spatter. [Accessed: 19- Apr- 2023]
.. [LANL-Memory-Wall] G. M. Shipman, J. Dominguez-Trujillo, K. Sheridan and S. Swaminarayan, "Assessing the Memory Wall in Complex Codes," 2022 IEEE/ACM Workshop on Memory Centric High Performance Computing (MCHPC), Dallas, TX, USA, 2022, pp. 30-35, doi: 10.1109/MCHPC56545.2022.00009.
