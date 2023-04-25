******
Parthenon-VIBE
******

This is the documentation for the ATS-5 Benchmark, Parthenon-VIBE. 
Purpose
=======

From their [site]_:

A benchmark that solves the Vector Inviscid Burgers' Equation on a block-AMR mesh.


===============

Problem
-------
The benchmark performance problem solves

.. math::
   \partial_t \mathbf{u} + \nabla\cdot\left(\frac{1}{2}\mathbf{u} \mathbf{u}\right) = 0

and evolves one or more passive scalar quantities :math:`q^i` according to

.. math:: 
   \partial_t q^i + \nabla \cdot \left( q^i \mathbf{u} \right) = 0


as well as computing an auxiliary quantity :math:`d`` that resemebles a kinetic energy

.. math:: 
   d = \frac{1}{2} q^0 \mathbf{u}\cdot\mathbf{u}.

Parthenon-VIBE makes use of a Godunov-type finite volume scheme with options for slope-limited linear or WENO5 reconstruction, HLL fluxes, and second order Runge-Kutta time integration.
Characteristics


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
   build$ export CXXFLAGS="-mavx2 -fno-math-errno -march=broadwell"
   build$ cmake -DPARTHENON_DISABLE_HDF5=ON -DPARTHENON_DISABLE_OPENMP=ON -DPARTHENON_ENABLE_PYTHON_MODULE_CHECK=OFF -DREGRESSION_GOLD_STANDARD_SYNC=OFF ../
   build$ make -j

.. 

On a CTS-1 machine the relevant modules are:

.. code-block:: bash
   
   intel-classic/2021.2.0 intel-mpi/2019.9.304 cmake/3.22.3

To build for execution on a single GPU, it should be sufficient to add the following flags to the CMake configuration line

.. code-block:: bash
   
   cmake -DPARTHENON_DISABLE_MPI=ON -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE80=ON

..

where `Kokkos_ARCH` should be set appropriately for the machine (see [here](https://kokkos.github.io/kokkos-core-wiki/keywords.html)).


Running
=======


The benchmark includes an input file ``_burgers.pin_`` that specifies the base (coarsest level) mesh size, the size of a mesh block, the number of levels, and a variety of other parameters that control the behavior of Parthenon and the benchmark problem configuration.


The executable `burgers-benchmark` will be built in `parthenon/build/benchmarks/burgers/` and can be run as, e.g.

.. code-block:: bash

   mpirun -n 36 burgers-benchmark -i ../../../benchmarks/burgers/burgers.pin parthenon/mesh/nx1=128 parthenon/mesh/nx2=128 parthenon/mesh/nx3=128 parthenon/meshblock/nx1=16 parthenon/meshblock/nx2=16 parthenon/meshblock/nx3=16 parthenon/nlim=250

..

Varying the ``parthenon/mesh/nx*`` parameters will change the memory footprint. The memory footprint scales roughly as the product of ``parthenon/mesh/nx1``, ``parthen/mesh/nx2``, and ``parthenon/mesh/nx3``. The ``parthen/meshblock/nx*`` parameters select the granularity of refinement: the mesh is distributed accross MPI ranks and refined/de-refined in chunks of this size. ``parthenon/mesh/nx1`` must be evenly divisible by ``parthenon/meshblock/nx1`` and the same for the other dimensions. Smaller meshblock sizes mean finer granularity and a problem that can be broken up accross more cores. However, each meshblock carries with it some overhead, so smaller meshblock sizes may hinder performance.

Results from Branson are provided on the following systems:

* Commodity Technology System 1 (CTS-1) with Intel Broadwell processors,
* An Nvidia A100 GPU hosted on an [Nvidia Arm HPC Developer Kit](https://developer.nvidia.com/arm-hpc-devkit)

CTS-1
--------

The mesh and meshblock size parameters are chosen to balance
realism/performance with memory footprint. For the following tests we
examine memory footprints of 20%, 40%, and 60%. Memory was measured
using the tool ``parse_spatter_top.py`` found in this repository. It
was independently verified with the [Kokkos Tools Memory High Water
Mark](https://github.com/kokkos/kokkos-tools/wiki/MemoryHighWater)
tool. Increasing the `parthenon/mesh/nx*` parameters will increase the
memory footprint.

Included with this repository is a ``do_strong_scaling_cpu.sh``
script, which takes one argument, specifying the desired memory
footprint on a CTS-1 system. Running it will generate a csv file
containing scaling numbers.

Strong scaling performance of Parthenon-VIBE with a 20% memory footprint on CTS-1 machines is provided within the following table and figure.

.. csv-table:: VIBE Strong Scaling Performance on CTS-1 20% Memory Footprint
   :file: cpu_20.csv
   :align: center
   :widths: 10, 10, 10
   :header-rows: 1

.. figure:: cpu_20.png
   :align: center
   :scale: 50%
   :alt: VIBE Strong Scaling Performance on CTS-1 20% Memory Footprint

Strong scaling performance of Parthenon-VIBE with a 40% memory footprint on CTS-1 machines is provided within the following table and figure.

.. csv-table:: VIBE Strong Scaling Performance on CTS-1 40% Memory Footprint
   :file: cpu_40.csv
   :align: center
   :widths: 10, 10, 10
   :header-rows: 1

.. figure:: cpu_40.png
   :align: center
   :scale: 50%
   :alt: VIBE Strong Scaling Performance on CTS-1 40% Memory Footprint

Strong scaling performance of Parthenon-VIBE with a 60% memory footprint on CTS-1 machines is provided within the following table and figure.

.. csv-table:: VIBE Strong Scaling Performance on CTS-1 60% Memory Footprint
   :file: cpu_60.csv
   :align: center
   :widths: 10, 10, 10
   :header-rows: 1

.. figure:: cpu_60.png
   :align: center
   :scale: 50%
   :alt: VIBE Strong Scaling Performance on CTS-1 60% Memory Footprint

A100
-----

Throughput performance of Parthenon-VIBE on a 40GB A100 is provided within the following table and figure.

.. csv-table:: VIBE Throughput Performance on A100
   :file: gpu.csv
   :align: center
   :widths: 10, 10
   :header-rows: 1

.. figure:: gpu.png
   :align: center
   :scale: 50%
   :alt: VIBE Throughput Performance on A100

Verification of Results
=======================

References
==========

.. [site]  'Parthenon', 2023. [Online]. Available: https://github.com/parthenon-hpc-lab/parthenon. [Accessed: 20- Mar- 2023]
