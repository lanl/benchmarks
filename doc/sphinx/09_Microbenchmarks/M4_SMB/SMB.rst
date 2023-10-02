*************************************
Sandia Microbenchmarks - Message rate
*************************************

This is the documentation for the ATS-5 Benchmark SMB Message Rate - A multi-node MPI point-to-point benchmark 


Purpose
=======

Sandia Microbenchmarks - Message Rate is a realistic messaging benchmark designed to emulate real application behavior. In particular there are a few things that set this benchmark apart; 1. It uses a peer count to emulate different communication patterns. 2. It clears the cache between each iteration to get realistic performance numbers. 3. It test different program behaviors, such at pre-posting receives, to evaluate performance under different scenarios. 


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


Verification of Results
=======================

References
==========

