*****
DGEMM
*****

Purpose
=======

The DGEMM benchmark measures the sustained floating-point rate of a single node.

Characteristics
===============

- LANL Crossroads Site: `DGEMM <https://www.lanl.gov/projects/crossroads/_assets/docs/micro/mtdgemm-crossroads-v1.0.0.tgz>`_

Problem
-------

.. math::
    \mathbf{C} = \alpha*\mathbf{A}*\mathbf{B} + \beta*\mathbf{C}

Where :math:`A B C` are square :math:`NxN` vectors and :math:`\alpha` and :math:`\beta` are scalars. This operation is repeated :math:`R` times.

Figure of Merit
---------------

The Gigaflops per second rate reported at the end of the run

GFLOP/s rate:         <FOM> GF/s

Run Rules
---------

          
* Vendors are permitted to change the source code in the region marked in the source.
* Optimized BLAS/DGEMM routines are permitted (and encouraged) to demonstrate the highest performance.
* Vendors may modify the Makefile(s) as required

Building
========

Makefiles are provided for the intel and gcc compilers. Before building, load the compiler and blas libraries into the PATH and LD_LIBRARY_PATH. 

.. code-block:: 

    cd src
    patch -p1 < ../dgemm_omp_fixes.patch
    make

If using a different compiler, copy and modify the simple makefiles to apply the appropriate flags.

If using a different blas library than mkl or openblas, modify the C source file to use the correct header and dgemm command.

Running
=======

DGEMM uses OpenMP but does not use MPI.

Set the number of OpenMP threads before running.

.. code-block:: bash
    export OPENBLAS_NUM_THREADS = <nthreads>
    export OMP_NUM_THREADS = <nthreads>

.. code-block:: bash
    ./mt-dgemm <N> <R> <alpha> <beta>

These values default to: :math:`N=256, R=8, \alpha=1.0, \beta=1.0`

These inputs are subject to the conditions :math:`N>128, R>4`.

These are positional arguments, so, for instance, R cannot be set without setting N.

Example Results
===============

