******
UMT
******

This is the documentation for the ATS-5 Benchmark UMT - 3D unstructured mesh single node. 

Purpose
=======

UMT (Unstructured Mesh Transport) is an LLNL ASC proxy application (mini-app) that solves a thermal radiative transport equation using discrete ordinates (Sn). 
It utilizes an upstream corner balance method to compute the solution to the Boltzmann transport equation on unstructured spatial grids.

It is available at https://github.com/LLNL/UMT .

Characteristics
===============

Problem
-------

The benchmark problem is a single node sweep performance problem (SPP) on a 3D unstructured mesh. Two variants of interest exist:

- SPP 1, a configuration with a high number of unknowns per spatial element with 72 directions and 128 energy bins to solve per
  mesh cell.
- SPP 2, a configuration with a low number of unknowns per spatial element with 32 directions and 16 energy bins to solve per mesh
  cell.  SPP 2 is still a work in progress.


Figure of Merit
---------------

The Figure of Merit is defined as the number of unknowns solved per second, which is calculated by:

.. code-block::

   number of unknowns =  <# mesh cells> * <# sub-cell 'corner' elements per cell> * <# directions> * <number of energy bins>

The number of corners in a mesh cell is 8 for the 3D unstructured mesh problem. (For a 2D mesh problem it would be 4.)


Building
========

Accessing the Source
--------------------

UMT can be found on github and cloned via:

.. code-block::

   git clone https://github.com/LLNL/UMT.git


Build Requirements
------------------

* C/C++ compiler(s) with support for C++11 and Fortran compiler(s) with support for F2003.
* `CMake 3.18X <https://cmake.org/download/>`_
* `Conduit v0.8.9 (pending), or the develop branch as of 1/1/2024. <https://github.com/LLNL/conduit>`_
* `Spack <https://github.com/spack/spack>`_ (optional)

* MPI 3.0+

  * `OpenMPI 1.10+ <https://www.open-mpi.org/software/ompi/>`_
  * `mpich <http://www.mpich.org>`_
  * `mvapich2 <https://mvapich.cse.ohio-state.edu>`_

* For CPU threading support, a Fortran compiler that support OpenMP 4.5, and an MPI implementation that supports MPI_THREAD_MULTIPLE.
* For GPU support, a Fortran compiler will full support for OpenMP 4.5 target offloading.

Instructions for building the code can be found in the UMT github repo under
`BUILDING.md <https://github.com/LLNL/UMT/blob/master/BUILDING.md>`_

Running
=======

To run the test problems, select SPP 1 or SPP 2 using the -b command line switch.  Select the mesh size to generate by using
'-d x,y,z' where x,y,z is the number of tiles to produce in each cartesian axis.  Use '-B global' to specify that the size
is for the global mesh, which is suitable for strong scaling studies.  If performing a weak scaling study, you can
specify '-B local' to specify the size of the mesh per rank instead.

Benchmark problems should target roughly half the node memory (for CPUs) or half the device memory (for GPUs).

For example, to create a global mesh of size 20,20,20 tiles:

.. code-block::

   mpirun -n 1 test_driver -B global -d 20,20,20 -b $num

where num = 1 for SPP 1 or num = 2 for SPP 2.

Example FOM Results 
===================

Results from UMT are provided on the following systems:

* Crossroads (see :ref:`GlobalSystemATS3`)
* Sierra (see :ref:`GlobalSystemATS2`)

Strong scaling data for SPP 1 on Crossroads is shown in the table and figure below

.. csv-table:: Strong scaling of SPP 1 on Crossroads
   :file: spp1_strong_scaling_cts2_abridged.csv
   :align: center
   :widths: 8, 10, 10
   :header-rows: 1
		 
.. figure:: spp1_strong_scaling_cts2.png
   :alt: Strong scaling of SPP 1 on Crossroads
   :align: center
   :scale: 50%

   Strong scaling of SPP 1 on Crossroads

.. csv-table:: SPP #2 on CTS-2
   :file: spp2_strong_scaling_cts2_abridged.csv
   :align: center
   :widths: 8, 10, 10
   :header-rows: 1
		 
.. figure:: spp2_strong_scaling_cts2.png
   :alt: CPU Strong Scaling (Fixed problem size, SPP #2)
   :align: center
   :scale: 50%
	   
   Strong scaling of SPP 2 on CTS-2

Throughput study of SPP 1 performance on Sierra, single GPU, as a function of problem size:

.. note::
   Performance data for SPP 1 coming soon.
.. todo csv-table:: Throughput for SPP 1 on Sierra
   :file: spp1_throughput_V100.csv
   :align: center
   :widths: 10, 10
   :header-rows: 1

.. todo figure:: spp1_throughput_V100.png
   :alt: Throughput of SPP 1 on Sierra
   :align: center

   Throughput of SPP 1 on Sierra

.. note::
   Performance data for SPP 2 coming soon.

.. todo csv-table:: SPP 2 throughput on Power9 and V100
   :file: umtsp2_throughput_gpu.csv
   :align: center
   :widths: 10, 10, 10
   :header-rows: 1

.. todo figure:: umtsp2-throughput_gpu.png
   :alt: SPP 2 GPU throughput as a function of  problem size
   :align: center


Verification of Results
=======================

UMT will perform a verification step at the end of the benchmark problem and print out a PASS or FAIL.

Example output:

.. code-block::

   RESULT CHECK PASSED: Energy check (this is relative to total energy) 1.26316e-15 within tolerance of +/- 1e-09; check './UMTSPP1.csv' for tally details

Additional diagnostic data on this energy check, as well as throughput and memory use, is provided in a UMTSPP#.csv file that
UMT writes out at run end.

References
==========
