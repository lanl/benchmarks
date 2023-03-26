******
MiniEM
******

This is the documentation for the ATS-5 Benchmark MiniEM. The content herein was
created by the following authors (in alphabetical order).

- `Anthony M. Agelastos <mailto:amagela@sandia.gov>`_
- `David I. Collins <mailto:dcollin@sandia.gov>`_
- `Christian A. Glusa <mailto:caglusa@sandia.gov>`_
- `Douglas M. Pase <mailto:dmpase@sandia.gov>`_
- `Roger P. Pawlowski <mailto:rppawlo@sandia.gov>`_
- `Joel O. Stevenson <mailto:josteve@sandia.gov>`_

This material is based upon work supported by the Sandia National Laboratories
(SNL), a multimission laboratory managed and operated by National Technology and
Engineering Solutions of Sandia under the U.S. Department of Energy's National
Nuclear Security Administration under contract DE-NA0003525.


Purpose
=======

MiniEM solves a first order formulation of Maxwell's equations of
electromagnetics. MiniEM is the [Trilinos]_ proxy driver for the
electromagnetics sub-problem solved by EMPIRE and exercises the relevant
Trilinos components (i.e., Tpetra, Belos, MueLu, Ifpack2, Intrepid2, Panzer).


Characteristics
===============


Problem
-------

The [Maxwell-Large]_ problem given by the input deck "maxwell-large.xml"
describes a uniform mesh of a 3D box which makes it ideal for scaling studies.
The stock input file for this can be found within the Trilinos repository in the
aforementioned link.

Useful parameters from within this input deck are shown below.

.. code-block::

   <snip>
   23    <ParameterList name="Inline Mesh">
   <snip>
   28      <ParameterList name="Mesh Factory Parameter List">
   <snip>
   35        <Parameter name="X Elements" type="int" value="40" />
   36        <Parameter name="Y Elements" type="int" value="40" />
   37        <Parameter name="Z Elements" type="int" value="40" />

These parameters are described below.

``X Elements, Y Elements, Z Elements``
   This sets the size of the problem.


Figure of Merit
---------------

Each MiniEM simulation writes out a timer block to STDOUT. The relevant portion
of this block is in the below example.

.. code-block::

   Mini-EM: 678.6 [1] {min=678.581, max=678.613, std dev=0.00763025} <1, 3, 4, 3, 4, 6, 7, 10, 7, 3>
   |   Mini-EM: Total Time: 678.6 - 100% [1] {min=678.581, max=678.613, std dev=0.00763291} <1, 3, 4, 3, 4, 6, 7, 10, 7, 3>
   <snip>
   |   |   Mini-EM: timestepper: 656.961 - 96.8112% [1] {min=656.96, max=656.961, std dev=0.000231646} <2, 0, 0, 0, 0, 0, 1, 6, 19, 20>
   |   |   |   Mini-EM: Advance Time Step: 656.961 - 99.9999% [450] {min=656.96, max=656.961, std dev=0.000263652} <1, 0, 1, 0, 0, 0, 0, 5, 17, 24>

The quantity of interest (QOI) is "time steps per second," which can be computed
from the above table by extracting the number of timesteps (last line and within
brackets, is "450" in this example) and dividing by the total time in that
region (the first floating point number on the same line as the number of
timesteps, is "656.961" in this example).

The number of steps must be large enough so the timestepper time exceeds 600
(i.e., so it runs for at least 10 minutes). The figure of merit (FOM) is the QOI
for a simulation above the 10 minute mark.


System Information
==================

The platforms utilized for benchmarking activities are listed and described below.

* Commodity Technology System 1 (CTS-1) with Intel Cascade Lake processors,
  known as Manzano at SNL (see :ref:`MiniEMSystemCTS1`)
* Advanced Technology System 3 (ATS-3), also known as Crossroads (see
  :ref:`MiniEMSystemATS3`)
* Advanced Technology System 2 (ATS-2), also known as Sierra (see
  :ref:`MiniEMSystemATS2`)


.. _MiniEMSystemCTS3:

CTS-1/Manzano
-------------

.. note::
   The CTS-1/Manzano system is used as a placeholder for when ATS-3/Crossroads
   is available.

The Manzano HPC cluster has 1,488 compute nodes connected together by a
high-bandwidth, low-latency Intel OmniPath network where each compute node uses
two Intel Xeon Platinum 8268 (Cascade Lake) processors. Each processor has 24
cores, and each node has 48 physical cores and 96 virtual cores. Each core has a
base frequency of 2.9 GHz and a max frequency of 3.9 GHz. Cores support two
AVX512 SIMD units each, with peak floating-point performance (RPEAK) of 2.9 GHz
x 32 FLOP/clock x 48 cores = 4.45 TF/s. Measured DGEMM performance is just under
3.5 TF/s per node (78.5% efficiency).

Compute nodes are a Non-Uniform Memory Access (NUMA) design, with each processor
representing a separate NUMA domain. Each processor (domain) supports six
channels of 2,933 MT/s DDR4 memory. Total memory capacity is 4 GB/core, or 192
GB/node. Memory bandwidth for the node is 12 channels x 8 bytes / channel x
2.933 GT/s = 281.568 GB/s, and measured STREAM TRIAD throughput for local memory
access is approximately 215 GB/s (76% efficiency). Cache design uses three
levels of cache, with L1 using separate instruction and data caches, L2 unifying
instruction and data, and L3 being shared across all cores in the processor. The
cache size is 1.5 MB/core, 35.75 MB/processor, or 71.5 MB/node.


.. _MiniEMSystemATS3:

ATS-3/Crossroads
----------------

This system is not available yet but is slated to be the reference platform.


.. _MiniEMSystemATS2:

ATS-2/Sierra
------------

This system has a plethora of compute nodes that are made up of Power9
processors with four NVIDIA V100 GPUs. Please refer to [Sierra-LLNL]_ for more
detailed information.

A Sierra application and regression testbed system named Vortex, housed at SNL,
was used for benchmarking for convenience. Vortex has the same compute node
hardware as Sierra.


Building
========

Instructions are provided on how to build MiniEM for the following systems:

* Commodity Technology System 1 (CTS-1) with Intel Cascade Lake processors,
  known as Manzano at SNL (see :ref:`MiniEMBuildCTS1`)
* Advanced Technology System 2 (ATS-2), also known as Sierra (see
  :ref:`MiniEMBuildATS2`)

If submodules were cloned within this repository, then the source code to build
MiniEM is already present at the top level within the "trilinos" folder.


.. _MiniEMBuildCTS1:

CTS-1/Manzano
-------------

.. note::
   The CTS-1/Manzano system is used as a placeholder for when ATS-3/Crossroads
   is available.

Instructions for building on Manzano are provided below.

.. code-block:: bash

   module unload intel
   module unload openmpi-intel
   module use /apps/modules/modulefiles-apps/cde/v3/
   module load cde/v3/devpack/gcc-ompi
   mkdir build-trilinos
   pushd build-trilinos
   bash ../helper-scripts/configure_trilinos.sh
   make -j 16
   make install


.. _MiniEMBuildATS2:

ATS-2/Vortex
------------

Instructions for building on ATS-2 are provided below.

.. code-block:: bash

   export BASEPATH=${PWD}
   export LLNL_USE_OMPI_VARS=y
   export OMPI_CC=gcc
   export OMPI_CXX=${BASEPATH}/Trilinos/packages/kokkos/bin/nvcc_wrapper
   mkdir -p build-trilinos
   cd build-trilinos
   cp -p ../files-from-David_used/* .
   . ./load_matching_env.sh
   cmake -C vortex-cuda-opt-Volta70-static-rdc.cmake -D CMAKE_INSTALL_PREFIX=/projects/scs/josteve/projects/miniEM/vortex/build-trilinos/tpls/trilinos/miniem-shared-opt /projects/scs/josteve/projects/miniEM/vortex/Trilinos/
   cmake --build . -j 16
   cmake --install .


Running
=======

Instructions are provided on how to run MiniEM for the following systems:

* Commodity Technology System 1 (CTS-1) with Intel Cascade Lake processors,
  known as Manzano at SNL (see :ref:`MiniEMRunCTS1`)
* Advanced Technology System 2 (ATS-2), also known as Sierra (see
  :ref:`MiniEMRunATS2`)


.. _MiniEMRunCTS1:

CTS-1/Manzano
-------------

.. note::
   The CTS-1/Manzano system is used as a placeholder for when ATS-3/Crossroads
   is available.

An example of how to run the test case on Manzano with 450 time steps is
provided below.

.. code-block:: bash

   basepath=`pwd -P`
   installpath="build-trilinos/tpls/trilinos/miniem-shared-opt/example/PanzerMiniEM"
   exe=${basepath}/${installpath}/PanzerMiniEM_BlockPrec.exe

   module unload intel
   module unload openmpi-intel
   module use /apps/modules/modulefiles-apps/cde/v3/
   module load cde/v3/devpack/gcc-ompi

   export OMP_PLACES=threads
   export OMP_PROC_BIND=true
   export OMP_NUM_THREADS=1

   mpiexec \
       --np 48 \
       --bind-to socket \
       --map-by socket:span \
       "${exe}" \
           --stacked-timer --solver=MueLu-RefMaxwell \
           --numTimeSteps=450  --linAlgebra=Tpetra \
           --inputFile="${basepath}/maxwell-large.xml" \
           >"miniem-sim.out" 2>&1


.. _MiniEMRunATS2:

ATS-2/Vortex
------------

An example of how to run the test case with a single GPU on Sierra is provided
below.

.. code-block:: bash

   basepath=`pwd -P`
   installpath="build-trilinos/tpls/trilinos/miniem-shared-opt/example/PanzerMiniEM"
   exe=${basepath}/${installpath}/PanzerMiniEM_BlockPrec.exe

   # convenience script that loads appropriate modules
   pushd build-trilinos
   . ./load_matching_env.sh
   unset KOKKOS_NUM_DEVICES
   export TPETRA_ASSUME_CUDA_AWARE_MPI=1
   popd

   jsrun -M "-gpu -disable_gdr" \
       -n 1 -a 1 -c 1 -g 1 -d packed \
       "${exe}" \
           --stacked-timer --solver=MueLu-RefMaxwell \
           --numTimeSteps=450 --linAlgebra=Tpetra \
           --inputFile="{basepath}/maxwell-large.xml" \
           >"miniem-sim.out" 2>&1


Verification of Results
=======================

Results from MiniEM are provided on the following systems:

* Commodity Technology System 1 (CTS-1) with Intel Cascade Lake processors,
  known as Manzano at SNL (see :ref:`MiniEMResultsCTS1`)
* Advanced Technology System 2 (ATS-2), also known as Sierra (see
  :ref:`MiniEMResultsATS2`)


.. _MiniEMResultsCTS1:

CTS-1/Manzano
-------------

.. note::
   The CTS-1/Manzano system is used as a placeholder for when ATS-3/Crossroads
   is available.

Strong scaling performance of MiniEM on CTS-1/Manzano is provided within the
following table and figure.

.. csv-table:: MiniEM Strong Scaling Performance on Manzano
   :file: cts1.csv
   :align: center
   :widths: 10, 10, 10
   :header-rows: 1

.. image:: cts1.png
   :align: center
   :width: 512
   :alt: MiniEM Strong Scaling Performance on Manzano


.. _MiniEMResultsATS2:

ATS-2/Vortex
------------

Throughput performance of MiniEM on ATS-2/Vortex is provided within the
following table and figure.

.. csv-table:: MiniEM Throughput Performance on ATS-2/Vortex
   :file: ats2.csv
   :align: center
   :widths: 10, 10
   :header-rows: 1

.. image:: ats2.png
   :align: center
   :width: 512
   :alt: MiniEM Throughput Performance on ATS-2/Vortex


References
==========

.. [Trilinos] M. A. Heroux and R. A. Bartlett and V. E. Howle and R. J. Hoekstra
              and J. J. Hu and T. G. Kolda and R. B. Lehoucq and K. R. Long
              and R. P. Pawlowski and E. T. Phipps and A. G. Salinger and H. K.
              Thornquist and R. S. Tuminaro and J. M. Willenbring and A.
              Williams and K. S. Stanley, 'An Overview of the Trilinos Project',
              2005, ACM Trans. Math. Softw., Volume 31, No. 3, ISSN 0098-3500.
.. [Maxwell-Large] Trilinos developers, 'maxwell-large.xml', 2023. [Online]. Available: https://github.com/trilinos/Trilinos/blob/master/packages/panzer/mini-em/example/BlockPrec/maxwell-large.xml. [Accessed: 22- Feb- 2023]
.. [Sierra-LLNL] Lawrence Livermore National Laboratory, 'Sierra | HPC @ LLNL', 2023. [Online]. Available: https://hpc.llnl.gov/hardware/compute-platforms/sierra. [Accessed: 26- Mar- 2023]
