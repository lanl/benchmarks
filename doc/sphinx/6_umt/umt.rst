******
UMT
******

This is the documentation for the ATS-5 Benchmark UMT - 3D unstructured mesh single node. 


Purpose
=======


UMT (Unstructured Mesh Transport) is an LLNL ASC proxy application
(mini-app) that solves a thermal radiative transport equation using
discrete ordinates (Sn). It utilizes an upstream corner balance method
to compute the solution to the Boltzmann transport equation on
unstructured spatial grids.

It is available at https://github.com/LLNL/UMT .

Characteristics
===============

Problem
-------

The benchmark problem is a single node sweep performance problem (SPP)
on a 3D unstructured mesh. Two variants of interest exist:

- SPP 1, a configuration with a high number of unknowns per spatial
  element with 72 directions and 128 energy bins to solve per mesh
  cell.
- SPP 2, a configuration with a low number of unknowns per spatial
  element with 32 directions and 16 energy bins to solve per mesh
  cell.


Figure of Merit
---------------

The Figure of Merit is defined as the number of unknowns solved per
second. The number of unknowns solved by UMT is defined as:

.. code-block::

number of unknowns =  <# mesh cells> * <# sub-cell 'corner' elements per cell> *
                      <# directions> * <number of energy bins>

The number of corners in a mesh cell is 8 for the 3D unstructured mesh
problem. (For a 2D mesh problem it would be 4.)


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
* `Spack <https://github.com/spack/spack>`_ (optional)

* MPI 3.0+

  * `OpenMPI 1.10+ <https://www.open-mpi.org/software/ompi/>`_
  * `mpich <http://www.mpich.org>`_
  * `mvapich2 <https://mvapich.cse.ohio-state.edu>`_

If OpenMP threading is used, the MPI implementation must support MPI_THREAD_MULTIPLE.

Instructions for building the code can be found in the UMT github repo
under `BUILDING.md
<https://github.com/LLNL/UMT/blob/master/BUILDING.md>`_

Generating the problem input
============================

For strong scaling on a CPU the memory footprint of UMT should be
between 45%-55% of the computational device's main memory. Python
scripts in the github repo /benchmarks directory are provided to
assist in generating a series of runs with UMT.

Example of creating a mesh sized to use 128GB of memory (50% of a
test node with 256GB). Will refine the mesh once, splitting each mesh
cell edge into 27 edges and produce a mesh called 'refined_mesh.mesh'.


TODO - The 'unstructBox3D.mesh' will be added to UMT repo, so running
'makeUnstructuredBox' line will be removed.

.. code-block::
		
   makeUnstructuredBox 
   mpirun -n 1 test_driver -i unstructBox3D.mesh -c 0 -r 1 -R 27 -o .


Running
=======

To run test problem, select SPP 1 or SPP 2 using the -b command line switch.  For example,

.. code-block::

   mpirun -n 1 test_driver -c 1 -b $num -i ./refined_mesh.mesh

where num = 1 for SPP 1 or num = 2 for SPP 2.



Example FOM Results 
===================

Strong scaling data for SPP 1 on CTS-2 (Sapphire Rapids) is shown in the table and figure below

.. csv-table:: Strong scaling of SPP 1 on CTS-2
   :file: spp1_strong_scaling_cts2_abridged.csv
   :align: center
   :widths: 8, 10, 10
   :header-rows: 1
		 
.. figure:: spp1_strong_scaling_cts2.png
   :alt: Strong scaling of SPP 1 on CTS-2
   :align: center
   :scale: 50%

   Strong scaling of SPP 1 on CTS-2

.. todo csv-table:: SPP #2 on CTS-2
   :file: spp2_strong_scaling_cts2_abridged.csv
   :align: center
   :widths: 8, 10, 10
   :header-rows: 1
		 
.. todo figure:: spp2_strong_scaling_cts2.png
   :alt: CPU Strong Scaling (Fixed problem size, SPP #2)
   :align: center
   :scale: 50%
	   
   Strong scaling of SPP 2 on CTS-2

Throughput study of SPP 1 performance on Power9/V100, single GPU, as a function of
problem size:

.. TODO - add runtime in this table??
.. csv-table:: Throughput for SPP 1 on Power9 and V100
   :file: spp1_throughput_V100.csv
   :align: center
   :widths: 10, 10
   :header-rows: 1

.. figure:: spp1_throughput_V100.png
   :alt: Throughput of SPP 1 on Power9 and V100
   :align: center

Throughput of SPP 1 on Power9 and V100

.. TODO - add runtime in this table??
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

Correctness on the SPP 1 and SPP 2 problems are checked by verifying
that the amount of radiation energy in the problem is within
tolerance. The test driver will automatically check this value at the
end of the run and output if the test is a pass or fail.

References
==========
