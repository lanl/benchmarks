******************
Parthenon-VIBE
******************

This is the documentation for the ATS-5 Benchmark, Parthenon-VIBE.

Purpose
=======

The Parthenon-VIBE benchmark [Parthenon-VIBE]_ solves the Vector Inviscid Burgers' Equation on a block-AMR mesh.
This benchmark is configured to use three levels of mesh resolution and mesh blocks of size 16^3. This AMR configuration is meant to
mimic applications which require high resolution, the ability to capture sharp and dynamic interfaces, while balancing global memory footprint and the overhead of "ghost" cells.
This configuration should not be changed as it would violate the intent of the benchmark.

Problem
-------
The benchmark performance problem solves

.. math::
   \partial_t \mathbf{u} + \nabla\cdot\left(\frac{1}{2}\mathbf{u} \mathbf{u}\right) = 0

and evolves one or more passive scalar quantities :math:`q^i` according to

.. math::
   \partial_t q^i + \nabla \cdot \left( q^i \mathbf{u} \right) = 0


as well as computing an auxiliary quantity :math:`d` that resemebles a kinetic energy

.. math::
   d = \frac{1}{2} q^0 \mathbf{u}\cdot\mathbf{u}.

Parthenon-VIBE makes use of a Godunov-type finite volume scheme with options for slope-limited linear or WENO5 reconstruction, HLL fluxes, and second order Runge-Kutta time integration.


Figure of Merit
---------------

The Figure of Merit is defined as cell zone-cycles / wallsecond which is the number of AMR zones processed per second of execution time.


Building
========

Accessing the sources

* Clone the submodule from the benchmarks repository checkout

.. code-block:: bash

   cd <path to benchmarks>
   git submodule update --init --recursive
   cd parthenon

..


Build requirements:

* CMake 3.16 or greater
* C++17 compatible compiler
* Kokkos 3.6 or greater
* MPI

To build Parthenon on CPU, including this benchmark, with minimal external dependencies, start here:

.. code-block:: bash

   parthenon$ mkdir build && cd build
   build$ cmake -DPARTHENON_DISABLE_HDF5=ON  -DPARTHENON_ENABLE_PYTHON_MODULE_CHECK=OFF -DREGRESSION_GOLD_STANDARD_SYNC=OFF  -DPARTHENON_ENABLE_TESTING=OFF -DCMAKE_BUILD_TYPE=Release ../
   build$ make -j

..

On Crossroads the relevant modules for the results shown here are:

.. code-block:: bash

   intel-classic/2023.2.0 cray-mpich/8.1.25 

..

To build for execution on a single GPU, it should be sufficient to add flags similar to the CMake configuration line

.. code-block:: bash

   cmake -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_HOPPER90=ON

..

where `Kokkos_ARCH` should be set appropriately for the machine (see [here](https://kokkos.github.io/kokkos-core-wiki/keywords.html)).


Running
=======


The benchmark includes an input file ``_burgers.pin_`` that specifies the base (coarsest level) mesh size, the size of a mesh block, the number of levels, and a variety of other parameters that control the behavior of Parthenon and the benchmark problem configuration.


The executable `burgers-benchmark` will be built in `parthenon/build/benchmarks/burgers/` and can be run as, e.g.

Note that the 

.. code-block:: bash

   NX=128
   NXB=16
   NLIM=250
   NLVL=3
   mpirun -np 112 burgers-benchmark -i ../../../benchmarks/burgers/burgers.pin parthenon/mesh/nx{1,2,3}=${NX} parthenon/meshblock/nx{1,2,3}=${NXB} parthenon/time/nlim=${NLIM} parthenon/mesh/numlevel=${NLVL}"
   #srun -n 112 ... also works. Note that mpirun does not exist on HPE machines at LANL.
..

Varying the ``parthenon/mesh/nx*`` parameters will change the memory footprint. The memory footprint scales roughly as the product of ``parthenon/mesh/nx1``, ``parthen/mesh/nx2``, and ``parthenon/mesh/nx3``. The ``parthen/meshblock/nx*`` parameters select the granularity of refinement: the mesh is distributed accross MPI ranks and refined/de-refined in chunks of this size.
For this benchmark only the ``parthenon/mesh/nx*`` parameters may be changed.

``parthenon/mesh/nx1`` must be evenly divisible by ``parthenon/meshblock/nx1`` and the same for the other dimensions. Smaller meshblock sizes mean finer granularity and a problem that can be broken up accross more cores. However, each meshblock carries with it some overhead, so smaller meshblock sizes may hinder performance.

The results presented here use 128 and 160 for  memory footprints of approximate 40%, and 60%  respectively. These problem sizes are run with  8, 32, 56, 88, and 112 processes on a single node without threading.

Results from Parthenon are provided on the following systems:

* Crossroads (see :ref:`GlobalSystemATS3`)
* A Grace Hopper (Grace ARM CPU 72 cores with 120GB, H100 GPU with 96GB)

The mesh and meshblock size parameters are chosen to balance
realism/performance with memory footprint. For the following tests we
examine memory footprints of 20%, 40%, and 60%. Memory was measured
using the tool ``parse_spatter_top.py`` found in this repository. It
was independently verified with the [Kokkos Tools Memory High Water
Mark](https://github.com/kokkos/kokkos-tools/wiki/MemoryHighWater)
tool. Increasing the `parthenon/mesh/nx*` parameters will increase the
memory footprint.

Included with this repository under ``utils/parthenon`` is a ``do_strong_scaling_cpu.sh``
script, which takes one argument, specifying the desired memory
footprint on a system with 128GB system memory. Running it will generate a csv file
containing scaling numbers.

Also included with this respository under ``doc/sphinx/03_vibe/scaling/scripts/weak_scale_cpu_threads.sh``

Crossroads
-------------------


.. csv-table:: VIBE Throughput Performance on Crossroads using ~20% Memory
   :file: cpu_20.csv
   :align: center
   :widths: 10, 10, 10
   :header-rows: 1

.. figure:: ats3_20.png
   :align: center
   :scale: 50%
   :alt: VIBE Throughput Performance on Crossroads using ~20% Memory

   VIBE Throughput Performance on Crossroads using ~20% Memory

.. csv-table:: VIBE Throughput Performance on Crossroads using ~40% Memory
   :file: cpu_40.csv
   :align: center
   :widths: 10, 10, 10
   :header-rows: 1

.. figure:: ats3_40.png
   :align: center
   :scale: 50%
   :alt: VIBE Throughput Performance on Crossroads using ~40% Memory

   VIBE Throughput Performance on Crossroads using ~40% Memory

.. csv-table:: VIBE Throughput Performance on Crossroads using ~60% Memory
   :file: cpu_60.csv
   :align: center
   :widths: 10, 10, 10
   :header-rows: 1

.. figure:: ats3_60.png
   :align: center
   :scale: 50%
   :alt: VIBE Throughput Performance on Crossroads using ~60% memory

   VIBE Throughput Performance on Crossroads using ~60% memory

Nvidia Grace Hopper
------------------------

Throughput performance of Parthenon-VIBE on a 96 GB H100 is provided within the following table and figure.

.. csv-table:: VIBE Throughput Performance on H100
   :file: gpu.csv
   :align: center
   :widths: 10, 10
   :header-rows: 1

.. figure:: gpu.png
   :align: center
   :scale: 50%
   :alt: VIBE Throughput Performance on H100

   VIBE Throughput Performance on H100


Multi-node scaling on Crossroads
================================

The results of the scaling runs performed on Crossroads are presented below.
Parthenon was built with intel oneapi 2023.1.0 and cray-mpich 8.1.25.
These runs used between 4 and 4096 nodes with 8 ranks per node and 14 threads per rank (using Kokkos OpenMP) with the following problem sizes. 

.. code-block:: bash

NXs=(256 320 400 512 640 800 1024 1280 1616 2048 2576)
NODES=(4 8 16 32 64 128 256 512 1024 2048 4096)

..

Output files can be found in ``./docs/sphinx/03_vibe/scaling/output/``

.. figure:: ./scaling/weak-august.png
   :align: center
   :scale: 50%
   :alt: VIBE Weak scaling per node.

.. csv-table:: Multi Node Scaling Parthenon
   :file: ./scaling/weak-august.csv
   :align: center
   :widths: 10, 10, 10, 10
   :header-rows: 1

Timings were captured using Caliper and are presented below. 
Caliper files can be found in ``./doc/sphinx/03_vibe/scaling/plots/Caliper``

.. figure:: ./scaling/plots/parthenon-totaltime-line.png
   :align: center
   :scale: 50%
   :alt: VIBE time spent (exclusive) in each function/region.


.. figure:: ./scaling/plots/parthenon-totaltime-area.png
   :align: center
   :scale: 50%
   :alt: VIBE time spent (exclusive) in each function/region (Area plot).

.. figure:: ./scaling/plots/parthenon-pct.png
   :align: center
   :scale: 50%
   :alt: Percentage of VIBE time spent (exclusive) in each function/region.

Validation
==========

Parthenon-VIBE prints to a history file (default name ``burgers.hst``) a
time series of the sum of squares of evolved variables integrated over
volume for each octant of the domain, as well as the total number of
meshblocks in the simulation at that time. To compare these quantities
between runs, we provide the ``burgers_diff.py`` program in the
benchmark folder. This will diff two history files and report when the
relative difference is greater than some tolerance.

.. note::

   ``burgers.hst`` is **appended** to when the executable is re-run. So
   if you want to compare two different history files, rename the
   history file by changing either ``problem_id`` in the ``parthenon/job``
   block in the input deck (this can be done on the command line. When
   you start the program, add ``parthenon/job/problem_id=mynewname`` to
   the command line argument), or copy the old file to back it up.

To check that a modified calculation is still correct, run
``burgers_diff.py`` to compare a new run to the fiducial one at the
default tolerance. If no diffs are reported, the modified calculation
is correct.

References
==========

.. [Parthenon-VIBE] Jonah Miller, 'Parthenon', 2024. [Online]. Available: https://github.com/parthenon-hpc-lab/parthenon. [Accessed: 06- Feb- 2024]
