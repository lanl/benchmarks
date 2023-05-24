******
UMT
******

This is the documentation for the ATS-5 Benchmark UMT - 3D unstructured mesh single node. 


Purpose
=======

From their [site]_:

UMT (Unstructured Mesh Transport) is an LLNL ASC proxy application (mini-app) that solves a thermal radiative transport equation using discrete ordinates (Sn).  It utilizes an upstream corner balance method to compute the solution to the Boltzmann transport equation on unstructured spatial grids.

It is available at https://github.com/LLNL/UMT .

Characteristics
===============

Problem
-------
The benchmark performance problem is a single node problem on a 3D unstructured mesh.
- UMT SPP #1, a configuration with a high number of unknowns per spatial element with 72 directions and 128 energy bins to solve per mesh cell.

A future variant is still a work in progress:
- UMT SPP #2, a configuration with a low number of unknowns per spatial element with 32 directions and 16 energy bins to solve per mesh cell.


Figure of Merit
---------------
The Figure of Merit is defined as the number of unknowns solved per second.

The number of unknowns solved by UMT is defined as:

.. code-block:: bash
   number of unknowns =  <# mesh cells> * <# sub-cell 'corner' elements per cell> * <# directions> * <number of energy bins>
..

The number of corners in a mesh cell is 8 for the 3D unstructured mesh problem.  (For a 2D mesh problem it would be 4.)

Building
========

Accessing the source

* UMT can be found on github and cloned via:

.. code-block:: bash
   git clone https://github.com/LLNL/UMT.git
..


Build requirements:

* C/C++ compiler(s) with support for C++11 and Fortran compiler(s) with support for F2003.
* `CMake 3.18X <https://cmake.org/download/>`_
* `Spack <https://github.com/spack/spack>`_ (optional)

* MPI 3.0+

  * `OpenMPI 1.10+ <https://www.open-mpi.org/software/ompi/>`_
  * `mpich <http://www.mpich.org>`_
  * `mvapich2 https://mvapich.cse.ohio-state.edu>`_

If OpenMP threading is used, the MPI implementation must support MPI_THREAD_MULTIPLE.

Instructions for building the code can be found in the UMT github repo under `BUILDING.md <https://github.com/LLNL/UMT/blob/master/BUILDING.md>`_

Generating the problem input
============================

For strong scaling on a CPU the memory footprint of UMT should be between 45%-55% of the computational device's main memory.  A python script is available in the UMT github preop /benchmarks directory that can calculate a correct mesh size to use for your memory needs.

Example of creating a mesh sized to use 128GB of memory ( 50% of a test node with 256GB ).  Will refine the mesh once, splitting each mesh cell edge into 27 edges and produce a mesh called 'umt_spp1.mesh'.

.. code-block:: bash
   makeUnstructuredBox -r 27 -o umt_spp1.mesh
..


Running
=======

* To run the included UMTSP1 3D test problem:

.. code-block:: bash
  mpirun -n 1 test_driver -b 1 ./umt_spp1.mesh
..


Example FOM Results 
===================

# TODO - Look into combining both UMTSP1 and UMTSP2 on same gnuplot graph?
Strong scaling of UMT on CTS-2 (Sapphire Rapids) for Sweep Problem #1 (UMTSP #1):

.. csv-table:: UMT SP #1 on CTS-2
   :file: umtsp1_strong_scaling_cpu_abridged.csv
   :align: center
   :widths: 10, 10, 10
   :header-rows: 1
		 
.. figure:: umtsp1_strong_scaling_cpu.png
   :alt: CPU Strong Scaling (Fixed problem size, UMT SP #1)
   :align: center
   :scale: 50%

   CPU Strong Scaling on CTS-2

.. csv-table:: UMT SP #2 on CTS-2
   :file: umtsp2_strong_scaling_cpu_abridged.csv
   :align: center
   :widths: 10, 10, 10
   :header-rows: 1
		 
.. figure:: umtsp2-strong_scaling_cpu.png
   :alt: CPU Strong Scaling (Fixed problem size, UMT SP #2)
   :align: center
   :scale: 50%
	   
   CPU Strong Scaling on CTS-2

Throughput study of UMT on Power9/V100, single GPU, as a function of problem size for Sweep Problem #1 (UMTSP #1):
# TODO - add runtime in this table??
.. csv-table:: UMT SP #2 throughput on Power9 and V100
   :file: umtsp1_throughput_gpu.csv
   :align: center
   :widths: 10, 10, 10
   :header-rows: 1

# TODO - need to either update the gnuplot script to not expect 'ideal' column, or add the ideal.		 
.. figure:: plots/umtsp1-throughput_gpu.png
   :alt: UMT SP #1 GPU throughput as a function of problem size.
   :align: center

Throughput study of UMT on Power9/V100, single GPU, as a function of problem size for Sweep Problem #2 (UMTSP #2):
# TODO - add runtime in this table??
.. csv-table:: UMT SP #2 throughput on Power9 and V100
   :file: umtsp2_throughput_gpu.csv
   :align: center
   :widths: 10, 10, 10
   :header-rows: 1

# TODO - need to either update the gnuplot script to not expect 'ideal' column, or add the ideal.
.. figure:: plots/umtsp2-throughput_gpu.png
   :alt: UMT SP #2 GPU throughput as a function of  problem size
   :align: center

Verification of Results
=======================

Correctness on the UMTSP#1 and UMTSP#2 problems are checked by verifying that the amount of radiation energy in the problem is within tolerance.  The test driver will automatically check this value at the end of the run and output if the test is a pass or fail.

References
==========
