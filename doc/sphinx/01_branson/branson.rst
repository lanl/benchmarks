*******
Branson
*******

This is the documentation for the ATS-5 Benchmark Branson - 3D hohlraum single node. 
 


Purpose
=======

From their [Branson]_:

Branson is not an acronym.

Branson is a proxy application for parallel Monte Carlo transport. 
It contains a particle passing method for domain decomposition. 

   

Characteristics
===============

Problem
-------
The benchmark performance problem is a single node 3D hohlraum problem that is meant to be run with a 30 group build of Branson. 
It is in replicated mode which means there is very little MPI communication (end of cycle reductions).

Figure of Merit
---------------
The Figure of Merit is defined as particles/second and is obtained by dividing the number of particles in the problem divided by the `Total transport` value in the output. Future versions will output this number directly.


Building
========

Accessing the sources

* Clone the submodule from the benchmarks repository checkout 

.. code-block:: bash

   cd <path to benchmarks>
   git submodule update --init --recursive
   cd branson
 
..


Build requirements:

* C/C++ compiler(s) with support for C11 and C++14.
* `CMake 3.9X <https://cmake.org/download/>`_

* MPI 3.0+

  * `OpenMPI 1.10+ <https://www.open-mpi.org/software/ompi/>`_
  * `mpich <http://www.mpich.org>`_

* There is only one CMake user option right now: ``CMAKE_BUILD_TYPE`` which can be  
  set on the command line with ``-DCMAKE_BUILD_TYPE=<Debug|Release>`` and the
  default is Release.
* If cmake has trouble finding your installed TPLs, you can try
  
 * appending their locations to ``CMAKE_PREFIX_PATH``,
 * try running ``ccmake .`` from the build directory and changing the values of
    build system variables related to TPL locations.

* If building a CUDA enabled version of Branson use the ``CUDADIR`` environment variable to specify your CUDA directory. 

.. code-block:: bash

   export CXX=`which g++`
   cd <path/to/branson> 
   mkdir build 
   cd build 
   cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=<install-location> <path/to/branson/src>
   make -j

.. 

Testing the build:

.. code-block:: bash

   cd $build_dir
   ctest -j 32

.. 


Running
=======

* The ``inputs`` folder contains the 3D hohlraum
 3D hohlraums and should be run with a 30 group build of Branson (see Special builds section above).
* The ``3D_hohlraum_single_node.xml`` problem is meant to be run on a full node. 
 It is run with:

.. code-block:: bash

   mpirun -n <procs_on_node> <install-location/BRANSON> <path/to/branson/inputs/3D_hohlaum_single_node.xml>

..

For strong scaling on a CPU, Branson should be run with three different problem sizes such that the memory 
footprint at the smallest process count per node is approximately: 4 to 5%, 8 to 10%, and 20 to 22%; during step 2 of the simulation. 
Memory footprint is the sum of all Branson processes resident set size (or equivalent) on the node. 
This can be obtained on a CPU system using the following (while the application is in step 2): 

.. code-block:: bash

   ps -C BRANSON -o euser,c,pid,ppid,cmd,%cpu,%mem,rss --sort=-rss
   
   ps -C BRANSON -o rss | awk '{sum+=$1;} END{print sum/1024/1024;}'
.. 




For throughput curves on a GPU the memory footprint of Branson must vary between ~5% and ~60% in increments of at most 5% of the computational device's main memory.
The memory footprint can be controlled by editing "photons" in the input file. 


Results from Branson are provided on the following systems:

* Crossroads (see :ref:`GlobalSystemATS3`)
* Sierra (see :ref:`GlobalSystemATS2`)

Crossroads
------------
Strong scaling performance of Crossroads 66M Particles is provided within the following table and
figure.

.. csv-table:: Branson Strong Scaling Performance on Crossroads 10M particles
   :file: cpu_10M.csv
   :align: center
   :widths: 10, 10, 10, 10, 10
   :header-rows: 1

.. figure:: cpu_10M.png
   :align: center
   :scale: 50%
   :alt: Branson Strong Scaling Performance on Crossroads 10M particles

Branson Strong Scaling Performance on Crossroads 10M particles   

Strong scaling performance of Branson Crossroads 66M  Particles is provided within the following table and
figure.

.. csv-table:: Branson Strong Scaling Performance on Crossroads 66M  particles
   :file: cpu_66M.csv
   :align: center
   :widths: 10, 10, 10, 10, 10
   :header-rows: 1

.. figure:: cpu_66M.png
   :align: center
   :scale: 50%
   :alt: Branson Strong Scaling Performance on Crossroads 66M particles

Branson Strong Scaling Performance on Crossroads 66M  particles  

Strong scaling performance of Branson Crossroads 200M Particles is provided within the following table and
figure.

.. csv-table:: Branson Strong Scaling Performance on Crossroads 200M particles
   :file: cpu_200M.csv
   :align: center
   :widths: 10, 10, 10, 10, 10
   :header-rows: 1

.. figure:: cpu_200M.png
   :align: center
   :scale: 50%
   :alt: Branson Strong Scaling Performance on Crossroads 200M particles

Branson Strong Scaling Performance on Crossroads 200M particles  

Sierra
------------

Throughput performance of Branson on Sierra is provided within the
following table and figure.

.. csv-table:: Branson Throughput Performance on Sierra
   :file: gpu.csv
   :align: center
   :widths: 10, 10
   :header-rows: 1

.. figure:: gpu.png
   :align: center
   :scale: 50%
   :alt: Branson Throughput Performance on Sierra
Branson Throughput Performance on Sierra


Verification of Results
=======================

References
==========

.. [Branson] Alex R. Long, 'Branson', 2023. [Online]. Available: https://github.com/lanl/branson. [Accessed: 22- Feb- 2023]