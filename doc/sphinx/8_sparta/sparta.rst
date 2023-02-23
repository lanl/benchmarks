******
SPARTA
******

This is the documentation for the ATS-5 Benchmark [SPARTA]_.


Purpose
=======

From their [site]_:

   SPARTA is an acronym for **S**\ tochastic **PA**\ rallel **R**\ arefied-gas
   **T**\ ime-accurate **A**\ nalyzer. SPARTA is a parallel DSMC or Direct
   Simulation Monte Carlo code for performing simulations of low-density gases
   in 2d or 3d. Particles advect through a hierarchical Cartesian grid that
   overlays the simulation box. The grid is used to group particles by grid cell
   for purposes of performing collisions and chemistry. Physical objects with
   triangulated surfaces can be embedded in the grid, creating cut and split
   grid cells. The grid is also used to efficiently find particle/surface
   collisions. SPARTA runs on single processors or in parallel using
   message-passing techniques and a spatial-decomposition of the simulation
   domain. The code is designed to be easy to modify or extend with new
   functionality.


Characteristics
===============

Be sure to insert a compelling problem description. Also discuss how this is
within the repository.


Building
========

Instructions are provided on how to build SPARTA for the following systems:

* Advanced Technology System 2 (ATS-2), also known as Sierra (see :ref:`BuildATS2`)
* Advanced Technology System 3 (ATS-3), also known as Crossroads (see :ref:`BuildATS3`)


.. _BuildATS2:

ATS-2/Sierra
------------

Instructions for building on Sierra are provided below.

.. code-block:: bash

   module load cuda/11.2.0
   module load gcc/8.3.1
   git clone https://github.com/sparta/sparta.git sparta
   pushd "sparta/src"
   make yes-kokkos
   make -j 64 vortex_kokkos
   ls -lh `pwd -P`/spa_vortex_kokkos
   popd


.. _BuildATS3:

ATS-3/Crossroads
----------------

Instructions for building on Crossroads are provided below.

.. code-block:: bash

   module unload intel
   module unload openmpi-intel
   module use /apps/modules/modulefiles-apps/cde/v3/
   module load cde/v3/devpack/intel-ompi
   module list
   git clone https://github.com/sparta/sparta.git sparta
   cp -a Makefile.manzano_kokkos "sparta/src/MAKE"
   pushd "sparta/src"
   make yes-kokkos
   make -j 16 manzano_kokkos
   ls -lh `pwd -P`/spa_manzano_kokkos
   popd


Running
=======

Verification of Results
=======================

References
==========

.. [SPARTA]  S. J. Plimpton and S. G. Moore and A. Borner and A. K. Stagg and T. P. Koehler and J. R. Torczynski and M. A. Gallis, 'Direct Simulation Monte Carlo on petaflop supercomputers and beyond', 2019, Physics of Fluids, 31, 086101.
.. [site] M. Gallis and S. Plimpton and S. Moore, 'SPARTA Direct Simulation Monte Carlo Simulator', 2023. [Online]. Available: https://sparta.github.io. [Accessed: 22- Feb- 2023]
