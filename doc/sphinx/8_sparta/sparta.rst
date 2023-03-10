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

Problem
-------

Be sure to insert a compelling problem description. Also discuss how this is
within the repository.

The primary input file that controls the simulation is "in.cylinder". An
excerpt from this input file that has its key parameters is provided below.

.. code-block::

   <snip>
    37 ###################################
    38 # Simulation initialization standards
    39 ###################################
    40 variable            ppc equal 34
   <snip>
   149 ###################################
   150 # Unsteady Output
   151 ###################################
   <snip>
   174 run                 1000

These parameters are described below.

``ppc``
   This sets the **p**\ articles **p**\ er **c**\ ell variable. This variable
   controls the size of the problem and, accordingly, the amount of memory it
   uses.

``run``
   This sets how many iterations it will run for, which also controls the wall
   time required for termination.


Figure of Merit
---------------

Each SPARTA simulation writes out a file named "log.sparta". At the end of this
simulation is a block that resembles the following example.

.. code-block::

   Step CPU Np Natt Ncoll Maxlevel
          0            0   446441        0        0        5
         50   0.95011643   446367     3671     2981        5
        100    2.1636236   446384     5096     4079        5
        150     3.459164   446330     5588     4380        5
        200    4.7954215   446424     5895     4606        5
        250    6.1550201   446373     6104     4720        5
        300    7.5329763   446354     6083     4669        5
        350    8.9225474   446391     6178     4775        5
        400    10.324853   446388     6380     4915        5
        450    11.736653   446369     6349     4769        5
        500    13.157484   446307     6470     4903        5
        550    14.587244   446341     6363     4751        5
        600    16.023752   446378     6457     4845        5
        650    17.468165   446372     6475     4829        5
        700    18.918792   446382     6514     4789        5
        750    20.375701   446378     6623     4842        5
        800    21.840051   446423     6550     4798        5
        850    23.309482   446431     6615     4876        5
        900    24.784149   446377     6676     4950        5
        950    26.263906   446406     6746     4862        5
       1000    27.748297   446377     6542     4847        5
   Loop time of 27.7483 on 1 procs for 1000 steps with 446377 particles



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

Instructions are provided on how to run SPARTA for the following systems:

* Advanced Technology System 2 (ATS-2), also known as Sierra (see :ref:`RunATS2`)
* Advanced Technology System 3 (ATS-3), also known as Crossroads (see :ref:`RunATS3`)


.. _RunATS2:

ATS-2/Sierra
------------

An example of how to run the test case with a single GPU on Sierra is provided
below.

.. code-block:: bash

   module load gcc/8.3.1
   module load cuda/11.2.0
   jsrun \
       -M "-gpu -disable_gdr" \
       -n 1 -a 1 -c 1 -g 1 -d packed \
       "sparta/src/spa_vortex_kokkos" -in "in.cylinder" \
       -k on g 1 -sf kk -pk kokkos reduction atomic \
       >"sparta.out" 2>&1


.. _RunATS3:

ATS-3/Crossroads
----------------

An example of how to run the test case on Crossroads is provided below.

.. code-block:: bash

   module unload intel
   module unload openmpi-intel
   module use /apps/modules/modulefiles-apps/cde/v3/
   module load cde/v3/devpack/intel-ompi
   mpiexec \
       --np ${num_procs} \
       --bind-to socket \
       --map-by socket:span \
       "sparta/src/spa_manzano_kokkos" -in "in.cylinder" \
       >"sparta.out" 2>&1



Verification of Results
=======================

References
==========

.. [SPARTA]  S. J. Plimpton and S. G. Moore and A. Borner and A. K. Stagg and T. P. Koehler and J. R. Torczynski and M. A. Gallis, 'Direct Simulation Monte Carlo on petaflop supercomputers and beyond', 2019, Physics of Fluids, 31, 086101.
.. [site] M. Gallis and S. Plimpton and S. Moore, 'SPARTA Direct Simulation Monte Carlo Simulator', 2023. [Online]. Available: https://sparta.github.io. [Accessed: 22- Feb- 2023]
