******
MiniEM
******

This is the documentation for the ATS-5 Benchmark MiniEM.


Purpose
=======

MiniEM solves a first order formulation of Maxwell's equations of
electromagnetics. MiniEM is the [Trilinos]_ proxy driver for the electromagnetics
sub-problem solved by EMPIRE and exercises the relevant Trilinos components
(i.e., Tpetra, Belos, MueLu, Ifpack2, Intrepid2, Panzer).

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
(i.e., so it runs for at least 10 minutes). The figure of merit (FOM) is the
QOI for a simulation above the 10 minute mark.


Building
========

Instructions are provided on how to build MiniEM for the following systems:

* Advanced Technology System 3 (ATS-3), also known as Crossroads (see :ref:`MiniEMBuildATS3`)
* Advanced Technology System 2 (ATS-2), also known as Sierra (see :ref:`MiniEMBuildATS2`)


.. _MiniEMBuildATS3:

CTS-1/Manzano (Intel Cascade Lake)
----------------------------------

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

* Advanced Technology System 3 (ATS-3), also known as Crossroads (see :ref:`MiniEMRunATS3`)
* Advanced Technology System 2 (ATS-2), also known as Sierra (see :ref:`MiniEMRunATS2`)


.. _MiniEMRunATS3:

CTS-1/Manzano (Intel Cascade Lake)
----------------------------------

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

* Advanced Technology System 3 (ATS-3), also known as Crossroads (see :ref:`MiniEMResultsATS3`)
* Advanced Technology System 2 (ATS-2), also known as Sierra (see :ref:`MiniEMResultsATS2`)


.. _MiniEMResultsATS3:

CTS-1/Manzano (Intel Cascade Lake)
----------------------------------

.. note::
   The CTS-1/Manzano system is used as a placeholder for when ATS-3/Crossroads
   is available.

Strong scaling performance of MiniEM is provided within the following table and
figure.

.. csv-table:: MiniEM Strong Scaling Performance on Manzano
   :file: cts1.csv
   :widths: 10, 10, 10
   :header-rows: 1

.. image:: cts1.png
   :width: 512
   :alt: MiniEM Strong Scaling Performance on Manzano


.. _MiniEMResultsATS2:

ATS-2/Vortex
------------

Throughput performance of MiniEM on ATS-2/Vortex (a small version of
ATS-2/Sierra) is provided within the following table and figure.

.. csv-table:: MiniEM Throughput Performance on ATS-2/Vortex
   :file: ats2.csv
   :widths: 10, 10
   :header-rows: 1

.. image:: ats2.png
   :width: 512
   :alt: MiniEM Throughput Performance on ATS-2/Vortex


References
==========

.. [Trilinos] M. A. Heroux and R. A. Bartlett and V. E. Howle and R. J. Hoekstra and J. J. Hu and T. G. Kolda and R. B. Lehoucq and K. R. Long and R. P. Pawlowski and E. T. Phipps and A. G. Salinger and H. K. Thornquist and R. S. Tuminaro and J. M. Willenbring and A. Williams and K. S. Stanley, 'An Overview of the Trilinos Project', 2005, ACM Trans. Math. Softw., Volume 31, No. 3, ISSN 0098-3500.

.. [Maxwell-Large] Trilinos developers, 'maxwell-large.xml', 2023. [Online]. Available: https://github.com/trilinos/Trilinos/blob/master/packages/panzer/mini-em/example/BlockPrec/maxwell-large.xml. [Accessed: 22- Feb- 2023]
