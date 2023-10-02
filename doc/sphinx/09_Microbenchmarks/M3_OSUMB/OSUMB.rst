*******************
OSU Microbenchmarks
*******************

Purpose
=======

Characteristics
===============

- Official site: `OSUMB <https://mvapich.cse.ohio-state.edu/download/mvapich/osu-micro-benchmarks-7.2.tar.gz>`_

Problem
-------

The OSU benchmarks are a suite of microbenchmarks designed to measure network characteristics on HPC systems.

Run Rules
---------

Building
========

On GPU enabled systems add these flags to the following configure lines: 

.. code-block:: bash

    --enable-cuda
    --with-cuda-include=/path/to/cuda/include
    --with-cuda-libpath=/path/to/cuda/lib

Build and install the benchmarks.

.. code-block:: bash

    ./configure --prefix=$INSTALL_DIR
    make -j 
    make -j install


Before configuring make sure your CXX and CC environment variables are set to an 
MPI compiler or wrapper. 
On most systems this will look like:

.. code-block:: bash

    export CC=mpicc CXX=mpicxx

On systems with vendor provided wrappers it may look different. 
For example, on HPE-Cray systems:

.. code-block:: bash

    export CC=cc CXX=CC
    
Running
=======

.. csv-table:: OSU Microbenchmark Tests
   :file: OSU_req.csv
   :align: center
   :widths: 10, 10, 10, 10, 10
   :header-rows: 1

Example Results
===============

