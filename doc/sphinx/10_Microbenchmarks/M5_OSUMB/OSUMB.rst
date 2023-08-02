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

Figure of Merit
---------------

Building
========

On GPU enabled systems add these flags to the following configure lines: 

.. code-block:: bash

    --enable-cuda
    --with-cuda-include=/path/to/cuda/include
    --with-cuda-libpath=/path/to/cuda/lib

.. code-block:: bash

    ./configure --prefix=$INSTALL_DIR
    make -j 
    make -j install

RHEL Systems
------------

.. code-block:: bash

    export CC=mpicc CXX=mpicxx

CrayOS Systems
--------------

.. code-block:: bash

    exportCC=cc CXX=CC
    
Running
=======

Input
-----

Independent Variables
---------------------

Dependent Variable(s)
---------------------

Example Results
===============

