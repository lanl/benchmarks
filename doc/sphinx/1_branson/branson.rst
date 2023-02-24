******
Branson
******

This is the documentation for the ATS-5 Benchmark.


Purpose
=======

From their [site]_:

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
The Figure of Merit is defined as particles/second and is obtained by dividing the number of particles in the problem divided by the total runtime. 


Building
========

Accessing the sources

* Download a tarball from Github, or
* Fork and clone the git repository.

.. code-block:: bash

   # after forking the repository
   git clone git@github.com:[github-username]/branson
   cd branson
   git remote add upstream git@github.com:lanl/branson
   git checkout -b develop upstream/develop

..

Installing Branson:

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

   EXPORT CXX=`which g++`
   cd $build_dir
   cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=<install-location> ${branson_source_dir}/src
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

   mpirun -n <procs_on_node> <path/to/branson> 3D_hohlaum_single_node.xml

..


Example FOM Results 
===================

.. table:: CTS-1 problem size of 10 million particles
   :align: center

   +--------+----------+---------------------+
   |cores   | runtime  |     throughput      |
   |        |          |                     | 
   |        |          |  (particles/second) |
   +========+==========+=====================+
   |1       | 1201.9   |      8.32E+03       |
   +--------+----------+---------------------+
   |2       | 616.6    |      1.62E+04       |
   +--------+----------+---------------------+
   |4       | 317.7    |      3.15E+04       |
   +--------+----------+---------------------+
   |8       | 174.3    |      5.74E+04       |
   +--------+----------+---------------------+
   |16      | 96.6     |      1.04E+05       |
   +--------+----------+---------------------+
   |32      | 49.1     |      2.04E+05       |
   +--------+----------+---------------------+
   |36      | 43.6     |      2.29E+05       |
   +--------+----------+---------------------+

.. figure:: plots/cpu-strong.png
   :alt: CPU Strong Scaling (Fixed problem size, 10M Particles)
   :caption: CPU Strong Scaling (Fixed problem size, 10M Particles)
   :align: center

Verification of Results
=======================

References
==========

.. [site] Alex R. Long, 'Branson', 2023. [Online]. Available: https://github.com/lanl/branson. [Accessed: 22- Feb- 2023]
