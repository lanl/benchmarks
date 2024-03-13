******
MiniEM
******

This is the documentation for the ATS-5 Benchmark MiniEM. The content herein was
created by the following authors (in alphabetical order).

- `Anthony M. Agelastos <mailto:amagela@sandia.gov>`_
- `Christian A. Glusa <mailto:caglusa@sandia.gov>`_
- `Roger P. Pawlowski <mailto:rppawlo@sandia.gov>`_
- `Joel O. Stevenson <mailto:josteve@sandia.gov>`_

This material is based upon work supported by the Sandia National Laboratories
(SNL), a multimission laboratory managed and operated by National Technology and
Engineering Solutions of Sandia under the U.S. Department of Energy's National
Nuclear Security Administration under contract DE-NA0003525. Content herein
considered unclassified with unlimited distribution under SAND2023-01069O.

Purpose
=======

MiniEM solves a first order formulation of Maxwell's equations of
electromagnetics. MiniEM is the [Trilinos]_ proxy driver for the
electromagnetics sub-problem solved by EMPIRE and exercises the relevant
Trilinos components (i.e., Tpetra, Belos, MueLu, Ifpack2, Intrepid2, Panzer).


Characteristics
===============


Application Version
-------------------

The target application version corresponds to the Git SHA that the Trilinos git
submodule at the root of this repository is set to, i.e., within ``trilinos``.


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
   This sets the size of the problem, which is the product of these 3
   quantities. These parameters are set to other values with the cases shown
   herein. These values should be identical for the calculations herein.


Figure of Merit
---------------

Each MiniEM simulation writes out a Figure of Merit (FOM) block to
STDOUT. The relevant portion of this block is in the below example.

.. code-block::

   =================================
   FOM Calculation
   =================================
     Number of cells = 4116000
     Time for Belos Linear Solve = 705.737 seconds
     Number of Time Steps (one linear solve per step) = 1541
     FOM ( num_cells * num_steps / solver_time / 1000) = 8987.42 k-cell-steps per second 
   =================================

The number of steps, specified with the ``--numTimeSteps`` command
line option, described below in :ref:`MiniEMRunATS3`), must be large
enough so the time for the Belos Linear Solve is greater than 600
seconds, i.e., so the solver runs for at least 10 minutes. The figure
of merit (FOM) is the bottom entry in this block, i.e., ``FOM (
num_cells * num_steps / solver_time / 1000)``.

It is desired to capture the FOM for varying problem sizes that
encompass utilizing 35% to 75% of available memory (when all PEs are
utilized). The ultimate goal is to maximize this throughput FOM while
utilizing at least 50% of available memory.


Correctness Check
-----------------

MiniEM also provides the [Maxwell-AnalyticSolution]_ problem given by
the input deck "maxwell-analyticSolution.xml". This will output
analytic error values (see below for an example) and will cause the
simulation to fail (and return a non-zero exit code) if it exceeds
appropriate thresholds. This should be used to verify the build of
MiniEM upon the system to assess both the used programming environment
and any changes made to the benchmark.

.. code-block::
   :emphasize-lines: 2

   The Belos solver "GMRES block system" of type ""Belos::BlockGmresSolMgr": {Flexible: true, Num Blocks: 10, Maximum Iterations: 10, Maximum Restarts: 20, Convergence Tolerance: 1e-08}" returned a solve status of "SOLVE_STATUS_CONVERGED" in 1 iterations with total CPU time of 0.0189103 sec
   L2 Error E maxwell - analyticSolution = 0.0566793

   * finished time step 6, t = 5e-09
   **************************************************

This case can be run simply by following the overall instructions in
:ref:`MiniEMRunning` and replacing the benchmark input file with
"maxwell-analyticSolution.xml". Example output of a failed case is
provided below (also note that this case exited with an exit code of
134).

.. code-block::

   what():  /path/to/trilinos/packages/panzer/mini-em/example/BlockPrec/main.cpp:690:

   Throw number = 1

   Throw test that evaluated to true: !( (std::sqrt(Thyra::get_ele(*g,0))) < (0.065) )

   Error, (std::sqrt(Thyra::get_ele(*g,0)) = 0.0819696) < (0.065 = 0.065)! FAILED!
   terminate called after throwing an instance of 'std::out_of_range'
   what():  /path/to/trilinos/packages/panzer/mini-em/example/BlockPrec/main.cpp:690:


System Information
==================

The platforms utilized for benchmarking activities are listed and described below.

* Crossroads (see :ref:`GlobalSystemATS3`)


Building
========

MiniEM is a part of Trilinos, so building Trilinos and its dependencies is
required. The [TrilinosBuild]_ documentation provides a lot of guidance.
Information to augment the official Trilinos documentation is provided below.

The following requirements are present for MiniEM.

* CMake version 3.23 or greater
* OpenMPI version 3.1 or greater
* Compilers ca. 2023

Detailed instructions are provided on how to build MiniEM for the
following systems:

* Advanced Technology System 3 (ATS-3), also known as Crossroads (see
  :ref:`MiniEMBuildATS3`)

If submodules were cloned within this repository, then the source code
to build MiniEM is already present at the top level within the
"trilinos" and "miniem_build" folders.


.. _MiniEMBuildATS3:

Crossroads
----------

Instructions for building on Crossroads are provided below. The
"miniem_build" folder contains the following items.

``build-crossroads.sh``
   This script carries out the build. All that should be needed is for
   the `spack.yaml` to be generated from `template.yaml` and then for
   this script to be executed.
``spack``
   This contains a specific checkout of Spack needed to build
   MiniEM. This will need to be patched; the patch is taken care of
   via ``build-crossroads.sh``.
``spack-fixes-v0.21.0.patch``
   This is the patch file needed to address issues within the Spack
   checkout.
``template.yaml``
   This file needs to be copied into ``spack.yaml`` and edited to
   contain the paths to the necessary items.


.. _MiniEMRunning:

Running
=======

Instructions are provided on how to run MiniEM for the following systems:

* Advanced Technology System 3 (ATS-3), also known as Crossroads (see
  :ref:`MiniEMRunATS3`)


.. _MiniEMRunATS3:

Crossroads
----------

An example of how to run the test case on Crossroads is provided
within the script (:download:`run-crossroads-mapcpu.sh
<run-crossroads-mapcpu.sh>`)


Verification of Results
=======================

Results from MiniEM are provided on the following systems:

* Advanced Technology System 3 (ATS-3), also known as Crossroads (see
  :ref:`MiniEMResultsATS3`)


.. _MiniEMResultsATS3:

Crossroads
----------

Strong scaling performance (i.e., fixed problem size being run on
different MPI rank counts) plots of MiniEM on Crossroads are provided
within the following subsections.

Problem Size 40 (18-43 GiB)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

This problem size corresponds to X, Y, and Z Element values set to 40
which results in an overall discretization that contains 768,000
cells.

.. csv-table:: MiniEM Strong Scaling Performance and Memory on Crossroads with 768k cells (18-43 GiB)
   :file: ats3-0768k.csv
   :align: center
   :widths: 10, 10, 10, 10, 10, 10, 10
   :header-rows: 1

.. figure:: ats3-0768k.png
   :align: center
   :scale: 50%
   :alt: MiniEM Strong Scaling Performance on Crossroads with 768k cells (18-43 GiB)

   MiniEM Strong Scaling Performance on Crossroads with 768k cells (18-43 GiB)

.. figure:: ats3-0768k-mem.png
   :align: center
   :scale: 50%
   :alt: MiniEM Strong Scaling Memory on Crossroads with 768k cells (18-43 GiB)

   MiniEM Strong Scaling Memory on Crossroads with 768k cells (18-43 GiB)

Problem Size 60 (57-84 GiB)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

This problem size corresponds to X, Y, and Z Element values set to 60
which results in an overall discretization that contains 2,592,000
cells.

.. csv-table:: MiniEM Strong Scaling Performance and Memory on Crossroads with 2,592k cells (57-84 GiB)
   :file: ats3-2592k.csv
   :align: center
   :widths: 10, 10, 10, 10, 10, 10, 10
   :header-rows: 1

.. figure:: ats3-2592k.png
   :align: center
   :scale: 50%
   :alt: MiniEM Strong Scaling Performance on Crossroads with 2,592k cells (57-84 GiB)

   MiniEM Strong Scaling Performance on Crossroads with 2,592k cells (57-84 GiB)

.. figure:: ats3-2592k-mem.png
   :align: center
   :scale: 50%
   :alt: MiniEM Strong Scaling Memory on Crossroads with 2,592k cells (57-84 GiB)

   MiniEM Strong Scaling Memory on Crossroads with 2,592k cells (57-84 GiB)


Problem Size 70 (89-118 GiB)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This problem size corresponds to X, Y, and Z Element values set to 70
which results in an overall discretization that contains 4,116,000
cells.

.. csv-table:: MiniEM Strong Scaling Performance and Memory on Crossroads with 4,116k cells (57-84 GiB)
   :file: ats3-4116k.csv
   :align: center
   :widths: 10, 10, 10, 10, 10, 10, 10
   :header-rows: 1

.. figure:: ats3-4116k.png
   :align: center
   :scale: 50%
   :alt: MiniEM Strong Scaling Performance on Crossroads with 4,116k cells (57-84 GiB)

   MiniEM Strong Scaling Performance on Crossroads with 4,116k cells (57-84 GiB)

.. figure:: ats3-4116k-mem.png
   :align: center
   :scale: 50%
   :alt: MiniEM Strong Scaling Memory on Crossroads with 4,116k cells (57-84 GiB)

   MiniEM Strong Scaling Memory on Crossroads with 4,116k cells (57-84 GiB)


References
==========

.. [Trilinos] M. A. Heroux and R. A. Bartlett and V. E. Howle and R. J. Hoekstra
              and J. J. Hu and T. G. Kolda and R. B. Lehoucq and K. R. Long
              and R. P. Pawlowski and E. T. Phipps and A. G. Salinger and H. K.
              Thornquist and R. S. Tuminaro and J. M. Willenbring and A.
              Williams and K. S. Stanley, 'An Overview of the Trilinos Project',
              2005, ACM Trans. Math. Softw., Volume 31, No. 3, ISSN 0098-3500.
.. [TrilinosBuild] R. A. Bartlett, 'Trilinos Configure, Build, Test, and Install
                   Reference Guide', 2023. [Online]. Available:
                   https://docs.trilinos.org/files/TrilinosBuildReference.html.
                   [Accessed: 26- Mar- 2023]
.. [Maxwell-Large] Trilinos developers, 'maxwell-large.xml', 2024. [Online]. Available: https://github.com/trilinos/Trilinos/blob/master/packages/panzer/mini-em/example/BlockPrec/maxwell-large.xml. [Accessed: 22- Feb- 2024]
.. [Maxwell-AnalyticSolution] Trilinos developers, 'maxwell-analyticSolution.xml', 2024. [Online]. Available: https://github.com/trilinos/Trilinos/blob/master/packages/panzer/mini-em/example/BlockPrec/maxwell-analyticSolution.xml. [Accessed: 22- Feb- 2024]
