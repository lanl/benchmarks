******
Sandia Microbenchmarks - Message rate
******

This is the documentation for the ATS-5 Benchmark SMB Message Rate - A multi-node MPI point-to-point benchmark 


Purpose
=======


Characteristics
===============

Problem
-------

The SMB implements four different communication patterns; single direction, pair based, pre-posted, and all-start. Each of these is a variation of behavior in a given communication pattern. For simplicity we're limiting this evaluation to pre-posted. 

    -p <num>     Number of peers used in communication
    -i <num>     Number of iterations per test
    -m <num>     Number of messages per peer per iteration
    -s <size>    Number of bytes per message
    -c <size>    Cache size in bytes
    -n <ppn>     Number of procs per node
    -o           Format output to be machine readable


Figure of Merit
---------------
As this is meant to represent a variety of applicaiton behaviors there isn't a single figure of merrit we can identify. However, we can identify a subset of input parameters to test. 
For the purposes of this test the figure of merit is the message rate of pre-posted across different message sizes, and a number of peer count.

Building
========

Accessing the sources

* Clone the submodule from the benchmarks repository checkout 

.. code-block:: bash

   cd <path to benchmarks>
   git submodule update --init --recursive
   cd SMB/src/msgrate/
 
..


Build requirements:

* C/C++ compiler(s) with support for C11 and C++14.

* MPI 3.0+

  * `OpenMPI 1.10+ <https://www.open-mpi.org/software/ompi/>`_
  * `mpich <http://www.mpich.org>`_


.. code-block:: bash

   cd <path/to/smb> 
   make -j

.. 

Testing the build:

.. code-block:: bash

    mpirun -n 8 msgrate -n 1

.. 

You should see output simlar to the following (note, because you're presumably testing on a single node at this point, the -n parameter need to be set to 1. While this is erronous from a performance standpoint, the SMB tries to ensure all communication is done across the network, and thus can't be run on a single node. 

.. code-block:: bash

    job size:   8
    npeers:     6
    niters:     4096
    nmsgs:      128
    nbytes:     8
    cache size: 8388608
    ppn:        1
    single direction: 2578047.02
    pair-based: 4343577.14
      pre-post: 1889840.49
     all-start: 2398236.06

..



Running
=======

We have two tests using SMB message rate, that we will describe here. The first is a based on a 2D 9-point stensil code and the second is a 3D 27-point stensil. 
Each of these needs to be run for various message sizes and scales to test the performance of the entire system.


We define some system specific variables for these tests.

* PPN - the number of processes per node.
* CACHE - 2x the size of the largest cache size (note: we use 2x here to be thourough)


* 9 point stencil

.. code-block:: bash

   for i in {0..24}; do mpirun msgrate -n $PPN -p 8 -c $CACHE -s $((2**i)) -o; done

..


* 27 point stencil

.. code-block:: bash

   for i in {0..24}; do mpirun msgrate -n $PPN -p 26 -c $CACHE -s $((2**i)) -o; done

..



Results from SMB are provided on the following systems:

* Commodity Technology System 1 (CTS-1) with Intel Broadwell processors,
* IBM Power9 with Nvidia V100 GPU, 

CTS-1
------------
Strong scaling performance of Branson CTS-1 66M Particles is provided within the following table and
figure.

.. csv-table:: Branson Strong Scaling Performance on CTS-1 66M particles
   :file: cpu_66M.csv
   :align: center
   :widths: 10, 10, 10
   :header-rows: 1

.. figure:: cpu_66M.png
   :align: center
   :scale: 50%
   :alt: Branson Strong Scaling Performance on CTS-1 66M particles

Branson Strong Scaling Performance on CTS-1 66M particles   

Strong scaling performance of Branson CTS-1 133M Particles is provided within the following table and
figure.

.. csv-table:: Branson Strong Scaling Performance on CTS-1 133M particles
   :file: cpu_133M.csv
   :align: center
   :widths: 10, 10, 10
   :header-rows: 1

.. figure:: cpu_133M.png
   :align: center
   :scale: 50%
   :alt: Branson Strong Scaling Performance on CTS-1 133M particles

Branson Strong Scaling Performance on CTS-1 133M particles  

Strong scaling performance of Branson CTS-1 200M Particles is provided within the following table and
figure.

.. csv-table:: Branson Strong Scaling Performance on CTS-1 200M particles
   :file: cpu_200M.csv
   :align: center
   :widths: 10, 10, 10
   :header-rows: 1

.. figure:: cpu_200M.png
   :align: center
   :scale: 50%
   :alt: Branson Strong Scaling Performance on CTS-1 200M particles

Branson Strong Scaling Performance on CTS-1 200M particles  

Power9+V100
------------

Throughput performance of Branson on Power9+V100 is provided within the
following table and figure.

.. csv-table:: Branson Throughput Performance on Power9+V100
   :file: gpu.csv
   :align: center
   :widths: 10, 10
   :header-rows: 1

.. figure:: gpu.png
   :align: center
   :scale: 50%
   :alt: Branson Throughput Performance on Power9+V100
Branson Throughput Performance on Power9+V100


Verification of Results
=======================

References
==========

.. [site] Alex R. Long, 'Branson', 2023. [Online]. Available: https://github.com/lanl/branson. [Accessed: 22- Feb- 2023]
