******
SPARTA
******

This is the documentation for the ATS-5 Benchmark [SPARTA]_. The content herein
was created by the following authors (in alphabetical order).

- `Anthony M. Agelastos <mailto:amagela@sandia.gov>`_
- `Michael A. Gallis <mailto:magalli@sandia.gov>`_
- `Stan Moore <mailto:stamoor@sandia.gov>`_
- `Joel O. Stevenson <mailto:josteve@sandia.gov>`_

This material is based upon work supported by the Sandia National Laboratories
(SNL), a multimission laboratory managed and operated by National Technology and
Engineering Solutions of Sandia under the U.S. Department of Energy's National
Nuclear Security Administration under contract DE-NA0003525. Content herein
considered unclassified with unlimited distribution under SAND2023-01070O.


Purpose
=======

Heavily pulled from their [site]_:

   SPARTA is an acronym for **S**\ tochastic **PA**\ rallel **R**\ arefied-gas
   **T**\ ime-accurate **A**\ nalyzer. SPARTA is a parallel Direct Simulation
   Monte Carlo (DSMC) code for performing simulations of low-density gases in
   2d or 3d. Particles advect through a hierarchical Cartesian grid that
   overlays the simulation box. The grid is used to group particles by grid
   cell for purposes of performing collisions and chemistry. Physical objects
   with triangulated surfaces can be embedded in the grid, creating cut and
   split grid cells. The grid is also used to efficiently find particle/surface
   collisions. SPARTA runs on single processors or in parallel using
   message-passing techniques and a spatial-decomposition of the simulation
   domain. The code is designed to be easy to modify or extend with new
   functionality. Running SPARTA and the input command syntax is very similar
   to the LAMMPS molecular dynamics code (but SPARTA and LAMMPS use different
   underlying algorithms).


Characteristics
===============

The goal is to utlize the specified version of SPARTA (see
:ref:`SPARTAApplicationVersion`) that runs the benchmark problem (see
:ref:`SPARTAProblem`) correctly (see :ref:`SPARTACorrectness` if changes are
made to SPARTA) for the SSI and SSNI problems (see :ref:`SPARTASSNISSI`) and
other single-node strong scaling benchmarking (see :ref:`SPARTAResults`).


.. _SPARTAApplicationVersion:

Application Version
-------------------

The target application version corresponds to the Git SHA that the SPARTA git
submodule at the root of this repository is set to, i.e., within ``sparta``.


.. _SPARTAProblem:

Problem
-------

This problem models 2D hypersonic flow of nitrogen over a circle with periodic
boundary conditions in the z dimension, which physically translates to 3D flow
over a cylinder of infinite length. Particles are continuously emitted from the
4 faces of the simulation box during the simulation, bounce off the circle, and
then exit. The hierarchical cartesian grid is statically adapted to 6 levels
around the circle. The memory array used to hold particles is reordered by grid
cell every 100 timesteps to improve data locality and cache access patterns.

This problem is present within the upstream SPARTA repository. The components of
this problem are listed below (paths given are within SPARTA repository). Each
of these files will need to be copied into a run directory for the simulation.

``examples/cylinder/in.cylinder``
   This is the primary input file that controls the simulation. Some parameters
   within this file may need to be changed depending upon what is being run
   (i.e., these parameters control how long this simulation runs for and how
   much memory it uses).

``examples/cylinder/circle_R0.5_P10000.surf``
   This is the mesh file and will remain unchanged.

``examples/cylinder/air.*``
   These three files (i.e., ``air.species``, ``air.tce``, and ``air.vss``)
   contain the composition and reactions inherent with the air. These files,
   like the mesh file, are not to be edited.

An excerpt from this input file that has its key parameters is
provided below.

.. code-block::
   :emphasize-lines: 6,11,17,23,25

   <snip>
   ###################################
   # Trajectory inputs
   ###################################
   <snip>
   variable            L equal 1.
   <snip>
   ###################################
   # Simulation initialization standards
   ###################################
   variable            ppc equal 55
   <snip>
   #####################################
   # Gas/Collision Model Specification #
   #####################################
   <snip>
   collide_modify      vremax 100 yes vibrate no rotate smooth nearcp yes 10
   <snip>
   ###################################
   # Output
   ###################################
   <snip>
   stats                100
   <snip>
   run                 4346

These parameters are described below.

``L``
   This corresponds to the **l**\ ength scale factor. This will scale the x and
   y dimensions of the problem, e.g., a doubling of this parameter will result
   in a domain that is 4x larger. This is used to weak scale a problem, e.g.,
   setting this to 32 would be sufficient to weak scale a single-node problem
   onto 1,024 nodes.

``ppc``
   This sets the **p**\ articles **p**\ er **c**\ ell variable. This variable
   controls the size of the problem and, accordingly, the amount of memory it
   uses.

``collide_modify``
   The official documentation for this value is `here
   <https://sparta.github.io/doc/collide_modify.html>`_. This resets
   the number of collisions and attempts to enable consistent work for
   each time step.

``stats``
   This sets the interval at which the output required to compute the
   :ref:`SPARTAFigureOfMerit` is generated. In general, it is good to select a
   value that will produce approx. 20 entries between the time range of
   interest. If it produces too much data, then it may slow down the simulaton.
   If it produces too little, then it may adversely impact the FOM calculations.

``run``
   This sets how many iterations it will run for, which also controls the wall
   time required for termination.

.. _SPARTAFigureOfMerit:

Figure of Merit
---------------

Each SPARTA simulation writes out a file named "log.sparta". At the end of this
simulation is a block that resembles the following example.

.. code-block::
   :emphasize-lines: 8-25

       Step          CPU        Np     Natt    Ncoll Maxlevel
          0            0 392868378        0        0        6
        100    18.246846 392868906       33       30        6
        200    35.395156 392868743      166      145        6
   <snip>
       1700    282.11911 392884637     3925     3295        6
       1800    298.63468 392886025     4177     3577        6
       1900    315.12601 392887614     4431     3799        6
       2000    331.67258 392888822     4700     4055        6
       2100    348.07854 392888778     4939     4268        6
       2200    364.41121 392890325     5191     4430        6
       2300    380.85177 392890502     5398     4619        6
       2400    397.32636 392891138     5625     4777        6
       2500    413.76181 392891420     5857     4979        6
       2600    430.15228 392892709     6077     5165        6
       2700    446.56604 392895923     6307     5396        6
       2800    463.05626 392897395     6564     5613        6
       2900    479.60999 392897644     6786     5777        6
       3000    495.90306 392899444     6942     5968        6
       3100    512.24813 392901339     7092     6034        6
       3200    528.69194 392903824     7322     6258        6
       3300    545.07902 392904150     7547     6427        6
       3400    561.46527 392905692     7758     6643        6
       3500    577.82469 392905983     8002     6826        6
       3600    594.21442 392906621     8142     6971        6
       3700    610.75031 392907947     8298     7110        6
       3800    627.17841 392909478     8541     7317        6
   <snip>
       4346    716.89228 392914687  1445860  1069859        6
   Loop time of 716.906 on 112 procs for 4346 steps with 392914687 particles

The quantity of interest (QOI) is "Mega particle steps per second," which can be
computed from the above table by multiplying the third column (no. of particles)
by the first (no. of steps), dividing the result by the second column (elapsed
time in seconds), and finally dividing by 1,000,000 (normalize). The number of
steps must be large enough so the times mentioned in the second column exceed
600 (i.e., so it runs for at least 10 minutes).

The Figure of Merit (**FOM**) is the harmonic mean of the QOI computed from the
times between 300 and 600 seconds and then divided by the number of nodes, i.e.,
"Mega particle steps per second per node." A Python script
(:download:`sparta_fom.py <sparta_fom.py>`) is included within the repository to
aid in computing this quantity. Pass it the ``-h`` command line argument to view
its help page for additional information.


.. _SPARTACorrectness:

Correctness
-----------

The aforementioned relevant block of output within "log.sparta" is replicated
below.

.. code-block::
   :emphasize-lines: 8-25

       Step          CPU        Np     Natt    Ncoll Maxlevel
          0            0 392868378        0        0        6
        100    18.246846 392868906       33       30        6
        200    35.395156 392868743      166      145        6
   <snip>
       1700    282.11911 392884637     3925     3295        6
       1800    298.63468 392886025     4177     3577        6
       1900    315.12601 392887614     4431     3799        6
       2000    331.67258 392888822     4700     4055        6
       2100    348.07854 392888778     4939     4268        6
       2200    364.41121 392890325     5191     4430        6
       2300    380.85177 392890502     5398     4619        6
       2400    397.32636 392891138     5625     4777        6
       2500    413.76181 392891420     5857     4979        6
       2600    430.15228 392892709     6077     5165        6
       2700    446.56604 392895923     6307     5396        6
       2800    463.05626 392897395     6564     5613        6
       2900    479.60999 392897644     6786     5777        6
       3000    495.90306 392899444     6942     5968        6
       3100    512.24813 392901339     7092     6034        6
       3200    528.69194 392903824     7322     6258        6
       3300    545.07902 392904150     7547     6427        6
       3400    561.46527 392905692     7758     6643        6
       3500    577.82469 392905983     8002     6826        6
       3600    594.21442 392906621     8142     6971        6
       3700    610.75031 392907947     8298     7110        6
       3800    627.17841 392909478     8541     7317        6
   <snip>
       4346    716.89228 392914687  1445860  1069859        6
   Loop time of 716.906 on 112 procs for 4346 steps with 392914687 particles

There are several columns of interest regarding correctness; these are listed below.

``Step``
   This is the step number and is the first column.

``CPU``
   This is the elapsed time and is the second column.

``Np``
   This is the number of particles and is the third column.

``Natt``
   This is the number of attempts and is the fourth column.

``Ncoll``
   This is the number of collisions and is the fifth column.

Assessing the correctness will involve comparing these quantities across
modified (henceforth denoted with "mod" subscript) and unmodified ("unmod"
subscript) SPARTA subject to the methodology below.

The **first** step is to adjust the ``run`` input file parameter so
that SPARTA\ :sub:`mod` has ``CPU`` output that exceeds 600 seconds
(per :ref:`SPARTAFigureOfMerit`). Also, adjust the ``stats`` parameter
to a value of 1 so fine-grained output is generated. Then, produce
output from SPARTA\ :sub:`unmod` with the same ``run`` and ``stats``
settings.

.. note::
   The example above is generating output every 100 time steps, which
   is also what the value of ``collide_modify`` is set to. This has
   the side effect of having low attempt and collision values since it
   is outputting on the reset step. The final value shown at a time
   step of 4,346 has values that are more inline with the actual
   problem. This is why output, for this correctness step, needs to
   occur at each time step.

The **second** step is to compute the absolute differences between modified and
unmodified SPARTA for ``Np``, ``Natt``, and ``Ncoll`` for each row, *i*, whose
``Step`` is relevant for the FOM for SPARTA\ :sub:`mod`,

.. math::
   \Delta \texttt{Np}_i &= | \texttt{Np}_{\textrm{mod},i}-\texttt{Np}_{\textrm{unmod},i} | \\
   \Delta \texttt{Natt}_i &= | \texttt{Natt}_{\textrm{mod},i}-\texttt{Natt}_{\textrm{unmod},i} | \\
   \Delta \texttt{Ncoll}_i &= | \texttt{Ncoll}_{\textrm{mod},i}-\texttt{Ncoll}_{\textrm{unmod},i} |

where

* *i* is each line whose ``CPU`` time is between 300 and 600 seconds for SPARTA\ :sub:`mod`

The **third** step is to compute the arithmetic mean of each of the
aforementioned quantities over the *n* rows,

.. math::
   \mu _{\Delta \texttt{Np}} &= \frac{\sum_{i} \Delta \texttt{Np}_i}{n} \\
   \mu _{\Delta \texttt{Natt}} &= \frac{\sum_{i} \Delta \texttt{Natt}_i}{n} \\
   \mu _{\Delta \texttt{Ncoll}} &= \frac{\sum_{i} \Delta \texttt{Ncoll}_i}{n}

where

.. math::
   n = \sum_{i} 1

The **fourth** step is to compute the arithmetic mean of the *n* matching rows
of the unmodified SPARTA,

.. math::
   \mu _{\texttt{Np},\textrm{unmod}} &= \frac{\sum_{i} \texttt{Np}_{\textrm{unmod},i}}{n} \\
   \mu _{\texttt{Natt},\textrm{unmod}} &= \frac{\sum_{i} \texttt{Natt}_{\textrm{unmod},i}}{n} \\
   \mu _{\texttt{Ncoll},\textrm{unmod}} &= \frac{\sum_{i} \texttt{Ncoll}_{\textrm{unmod},i}}{n}

The **fifth** step is to normalize the differences with the baseline values to
create the error ratios,

.. math::
   \varepsilon _{\texttt{Np}} &= \frac{\mu _{\Delta \texttt{Np}}}{\mu _{\texttt{Np},\textrm{unmod}}} \\
   \varepsilon _{\texttt{Natt}} &= \frac{\mu _{\Delta \texttt{Natt}}}{\mu _{\texttt{Natt},\textrm{unmod}}} \\
   \varepsilon _{\texttt{Ncoll}} &= \frac{\mu _{\Delta \texttt{Ncoll}}}{\mu _{\texttt{Ncoll},\textrm{unmod}}}

The **sixth** and final step is to check over all of the error ratios and if any
of them exceed 25%, then the modifications are not approved without discussing
them with this benchmark's authors. This is the same criteria that SPARTA uses
for its own testing. The success criteria are:

.. math::
   \varepsilon _{\texttt{Np}} &\le 25\% \\
   \varepsilon _{\texttt{Natt}} &\le 25\% \\
   \varepsilon _{\texttt{Ncoll}} &\le 25\%


.. _SPARTASSNISSI:

SSNI & SSI
----------

The SSNI will focus on the problem with 35 particles per cell running at 100%
node utilization.

.. note::
   The SSI problem is being finalized and will be documented herein soon.


System Information
==================

The platforms utilized for benchmarking activities are listed and described below.

* Advanced Technology System 3 (ATS-3), also known as Crossroads (see
  :ref:`GlobalSystemATS3`)


Building
========

If Git Submodules were cloned within this repository, then the source code to
build the appropriate version of SPARTA is already present at the top level
within the "sparta" folder. Instructions are provided on how to build SPARTA for
the following systems:

* Generic (see :ref:`BuildGeneric`)
* Advanced Technology System 3 (ATS-3), also known as Crossroads (see
  :ref:`BuildATS3`)


.. _BuildGeneric:

Generic
-------

Refer to SPARTA's [build]_ documentation for generic instructions.


.. _BuildATS3:

Crossroads
----------

Instructions for building on Crossroads are provided below. These instructions
assume this repository has been cloned and that the current working directory is
at the top level of this repository. This is tested with Intel's 2023 developer
tools release. The script discussed below is :download:`build-crossroads.sh
<build-crossroads.sh>`.

.. code-block:: bash

   cd doc/sphinx/08_sparta
   ./build-crossroads.sh


Running
=======

Instructions are provided on how to run SPARTA for the following systems:

* Advanced Technology System 3 (ATS-3), also known as Crossroads (see
  :ref:`RunATS3`)


.. _RunATS3:

Crossroads
----------

Instructions for performing the simulations on Crossroads are provided below.
There are two scripts that facilitate running several single-node strong-scaling
ensembles.

:download:`run-crossroads-mapcpu.sh <run-crossroads-mapcpu.sh>`
   This script successively executes SPARTA on a single node for the same set of
   input parameters; there are many environment variables that can be set to
   control what it runs.

:download:`sbatch-crossroads-mapcpu.sh <sbatch-crossroads-mapcpu.sh>`
   This script runs the previous script for different numbers of MPI ranks,
   problem size, problem duration, and other parameters to yield several strong
   scaling trends.

:download:`scale-crossroads-mapcpu.sh <scale-crossroads-mapcpu.sh>`
   This script successively executes SPARTA on varying numbers of nodes for the
   same set of input parameters; there are many environment variables that can
   be set to control what it runs.

:download:`sbatch-crossroads-mapcpu.sh <sbatch-crossroads-mapcpu-scale.sh>`
   This script runs the previous script for different numbers of MPI ranks per
   node, problem size, problem duration, and other parameters to yield several
   strong scaling trends.


.. _SPARTAResults:

Verification of Results
=======================

Results from SPARTA are provided on the following systems:

* Advanced Technology System 3 (ATS-3), also known as Crossroads (see
  :ref:`ResultsATS3`)

  - As best practices for utilizing Crossroads are developed, its data may be
    updated.


.. _ResultsATS3:

Crossroads
----------

Strong single-node scaling throughput (i.e., fixed problem size being run on
different MPI rank counts on a single node) plots of SPARTA on Crossroads are
provided within the following subsections. The throughput corresponds to Mega
particle steps per second per node.

15 Particles per Cell
^^^^^^^^^^^^^^^^^^^^^

.. csv-table:: SPARTA Single Node Strong Scaling Throughput and Memory on Crossroads with ppc=15
   :file: ats3--15.csv
   :align: center
   :widths: 10, 10, 10, 10
   :header-rows: 1

.. figure:: ats3--15.png
   :align: center
   :scale: 50%
   :alt: SPARTA Single Node Strong Scaling Throughput on Crossroads with ppc=15

   SPARTA Single Node Strong Scaling Throughput on Crossroads with ppc=15

.. figure:: ats3mem--15.png
   :align: center
   :scale: 50%
   :alt: SPARTA Single Node Strong Scaling Memory on Crossroads with ppc=15

   SPARTA Single Node Strong Scaling Memory on Crossroads with ppc=15

35 Particles per Cell
^^^^^^^^^^^^^^^^^^^^^

.. csv-table:: SPARTA Single Node Strong Scaling Throughput and Memory on Crossroads with ppc=35
   :file: ats3--35.csv
   :align: center
   :widths: 10, 10, 10, 10
   :header-rows: 1

.. figure:: ats3--35.png
   :align: center
   :scale: 50%
   :alt: SPARTA Single Node Strong Scaling Throughput on Crossroads with ppc=35

   SPARTA Single Node Strong Scaling Throughput on Crossroads with ppc=35

.. figure:: ats3mem--35.png
   :align: center
   :scale: 50%
   :alt: SPARTA Single Node Strong Scaling Memory on Crossroads with ppc=35

   SPARTA Single Node Strong Scaling Memory on Crossroads with ppc=35

55 Particles per Cell
^^^^^^^^^^^^^^^^^^^^^

.. csv-table:: SPARTA Single Node Strong Scaling Throughput and Memory on Crossroads with ppc=55
   :file: ats3--55.csv
   :align: center
   :widths: 10, 10, 10, 10
   :header-rows: 1

.. figure:: ats3--55.png
   :align: center
   :scale: 50%
   :alt: SPARTA Single Node Strong Scaling Throughput on Crossroads with ppc=55

   SPARTA Single Node Strong Scaling Throughput on Crossroads with ppc=55

.. figure:: ats3mem--55.png
   :align: center
   :scale: 50%
   :alt: SPARTA Single Node Strong Scaling Memory on Crossroads with ppc=55

   SPARTA Single Node Strong Scaling Memory on Crossroads with ppc=55


References
==========

.. [SPARTA] S. J. Plimpton and S. G. Moore and A. Borner and A. K. Stagg
            and T. P. Koehler and J. R. Torczynski and M. A. Gallis, 'Direct
            Simulation Monte Carlo on petaflop supercomputers and beyond',
            2019, Physics of Fluids, 31, 086101.
.. [site] M. Gallis and S. Plimpton and S. Moore, 'SPARTA Direct Simulation
          Monte Carlo Simulator', 2023. [Online]. Available:
          https://sparta.github.io. [Accessed: 22- Feb- 2023]
.. [build] M. Gallis and S. Plimpton and S. Moore, 'SPARTA Documentation Getting
           Started', 2023. [Online]. Available:
           https://sparta.github.io/doc/Section_start.html#start_2. [Accessed:
           26- Mar- 2023]
