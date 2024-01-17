************
Introduction
************

This is the documentation for the **ATS-5 Benchmarks**. 

Assuring that real applications perform efficiently on ATS-5 is key to their success. 
A suite of benchmarks  have been developed for Request For Proposal (RFP) response evaluation and system acceptance. 
These codes are representative of the workloads of the NNSA laboratories. 

The benchmarks contained within this site represent a pre-RFP draft state. Over the next few months the 
benchmarks will change somewhat. While we expect most of the changes will be additions and modifications it is possible that we will remove 
benchmarks prior to RFP. 

To use these benchmarks please refer to the ATS-5 benchmarks repository `ATS-5 repo <https://github.com/lanl/benchmarks>`_

The benchmarks will, eventually, be generated atop Crossroads as the reference
system (see :ref:`ReferenceCrossroads` for more information).

Benchmark Changes from Crossroads
=================================

The key differences from Crossroads benchmarks and ATS-5 benchmarks are as summarized below: 

.. list-table::

 * - **Crossroads**
   - **ATS-5**
   - **Notes**
 * - Few GPU-ready benchmarks
   - | All proxy benchmarks have 
     | GPU implementations. 
   - 
 * - | System level performance metric: 
     | Scalable System Improvement 
     | geometric mean of app FOMs.
     | Use of single node benchmarks 
     | for RFP.
   - | Multi-node benchmarking for 
     | system acceptance based on 
     | RFP benchmarks, negotiated 
     | with vendor as part of SOW. 
   - | Attempting to limit multi-node 
     | benchmarking for RFP
     | to communication (MPI), and 
     | IO (IOR). Expect responses to 
     | include multiple node 
     | configurations and ability to 
     | compose them to meet our needs 
     | in a codesign partnership.  
     | Will use scaled single node 
     | improvement to assess proposals 
     | (along with other factors) and 
     | SSI for acceptance. 
 * - | Mini-Apps + full scale apps 
     | some of which were export 
     | controlled.
   - | Mini-apps only - all open 
     | source. 
   - 
 * - No Machine Learning. 
   - | ML training and inference 
     | included. 
   - | Focuses on material science
     | workloads of relevance. 



Benchmark Overview 
==================

.. list-table::

 * - **Benchmark**
   - **Description**
   - **Language**
   - **Parallelism** 
 * - Branson
   - Implicit Monte Carlo transport
   - C++
   - MPI + Cuda/HIP
 * - AMG2023
   - | AMG solver of sparse matrices 
     | using Hypre 
   - C 
   - | MPI+CUDA/HIP/SYCL
     | OpenMP on CPU
 * - MiniEM
   - Electro-Magnetics solver
   - C++
   - MPI+Kokkos
 * - MLMD
   - | ML Training of interatomic 
     | potential model using HIPYNN 
     | on VASP Simulation data. 
     | ML inference using LAMMPS, 
     | Kokkos, and HIPYNN trained 
     | interatomic potential model.
   - Python, C++, C
   - MPI+Cuda/HIP
 * - Parthenon-VIBE
   - | Block structured AMR proxy using 
     | the Parthenon framework.
   - C++
   - MPI+Kokkos
 * - Sparta
   - Direct Simulation Monte Carlo
   - C++
   - MPI+Kokkos
 * - UMT
   - Deterministic (Sn) transport
   - Fortran
   - | MPI+OpenMP and 
     | OpenMP Offload



Microbenchmark Overview
=======================

.. list-table::

 * - **Benchmark**
   - **Description**
   - **Language**
   - **Parallelism** 
   - **Multi-node**
 * - Stream
   - Streaming memory bandwidth test
   - C/Fortran
   - OpenMP 
   - No
 * - Spatter
   - | Sparse memory bandwidth test
     | driven by application memory 
     | access patterns. 
   - C++
   - | MPI+OpenMP/
     | CUDA/OpenCL 
   - No
 * - | OSU MPI + 
     | Sandia SMB 
     | message rate
   - MPI Performance Benchmarks
   - C++
   - MPI
   - Yes 
 * - DGEMM
   - | Single node floating-point 
     | performance on matrix multiply. 
   - C/Fortran
   - Various
   - No
 * - IOR
   - | Performance testing of parallel 
     | file system using various
     | interfaces and access patterns. 
   - C
   - MPI
   - Yes
 * - mdtest
   - | Metadata benchmark that performs 
     | open/stat/close operations on 
     | files and directories. 
   - C
   - MPI
   - Yes


.. _GlobalRunRules:

Run Rules Synopsis
==================

Single node benchmarks will require respondent to provide estimates on

* strong scaling for CPU architectures. 

* throughput curves for GPU architectures. 

* estimates must be provided for each compute node type (including options).

* Problem size must be changed to meet % of memory requirements. 

* Respondent shall provide CPU strong scaling and GPU throughput results on current generation representative architectures.
  If no representative architecture exists respondent can provide modeled / projected CPU strong scaling and GPU throughput results. 
  respondent may provide both results on current generation representative architectures and modeled / projected architectures. 

* For SSNI projections respondent shall use the specific problem size(s) specified for SSNI.  

Source code modification categories: 

* Baseline: “out-of-the-box” performance

  * Code modifications not permitted 

  * Compiler options can be modified, library substitutions permitted, problem decomposition may be changed 

* Ported: “alternative baseline for new architectures” 
  
  * Limited source-code modifications are permitted to port and tune for the target architecture using directives or commonly used interfaces. 

* Optimized: "speed of light"
  
  * Aggressive code changes that enhance performance are permitted.

  * Algorithms fundamental to the program may not be replaced. 

  * The modified code must still pass validation tests. 

  * Optimizations will be reviewed by subject matter experts for applicability to the larger application portfolio and other goals such as performance portability and programmer productivity. 


Required results: 

 * A **baseline** or **ported** result is required for each benchmark. If baseline cannot be obtained, ported results may be provided. 

Optional results: 

 * **Ported** results may be provided in addition to the baseline if minor code changes enable substantial performance gain. 

 * **Optimized** results to showcase system capabilities. 

Scaled Single Node Improvement
==============================
One element of evaluation will focus on scaled single node improvement (SSNI). SSNI is defined as follows: 

Given two platforms using one as a reference (Crossroads), SSNI is defined as a weighted geometric mean using the following equation. 

.. math::

   SSNI = N(\prod_{i=1}^{M}(S_i)^{w_i})^\frac{1}{\sum_{i=1}^{M}{W_i}}


Where: 

*	N = Number of nodes on ATS-5 system / Number of nodes on reference system (Crossroads),

*	M = total number of Benchmarks,

*	S = application speedup; Figure of Merit on ATS-5 system / Figure of Merit on reference system (Crossroads); S must be greater than 1, 

*	w = weighting factor. 



.. _GlobalSSNIWeightsSizes:

SSNI Weights and SSNI problem sizes
===================================


.. list-table::

 * - **SSNI Benchmark**
   - **SSNI Weight**
   - **SSNI Problem size - % device memory**
 * - Branson
   - TBD
   - 30
 * - AMG2023 Problem 1 Setup
   - TBD
   - 20
 * - AMG2023 Problem 2 Setup
   - TBD
   - 20
 * - AMG2023 Problem 1 Solve
   - TBD
   - 20
 * - AMG2023 Problem 2 Solve
   - TBD
   - 20
 * - MiniEM
   - TBD
   - TBD
 * - MLMD Training
   - TBD
   - N/A 
 * - MLMD Simulation
   - TBD
   - 60
 * - Parthenon-VIBE
   - TBD
   - 40 
 * - Sparta
   - TBD
   - TBD
 * - UMT
   - TBD
   - TBD


System Information
==================

The baseline platform for the ATS-5 procurement is the ATS-3 system (described below). 
GPU performance is provided on the ATS-2 system and in some cases other GPU based systems 
and is for information only, these are not to be used as baselines. 
In most cases the performance numbers provided herein were collected on smaller scale 
testbed systems that are the same architecture as that of ATS-3 and ATS-2 systems. 

* Advanced Technology System 3 (ATS-3), also known as Crossroads (see :ref:`GlobalSystemATS3`)
* Advanced Technology System 2 (ATS-2), also known as Sierra (see :ref:`GlobalSystemATS2`)


.. _GlobalSystemATS3:

ATS-3/Crossroads
----------------

This system has over 6,140 compute nodes that are made up of two Intel(R) Xeon(R) Max 9480 CPUs 
interconnected with HPE Slingshot 11 interconnect. 

.. _GlobalSystemATS2:

ATS-2/Sierra
------------

This system has 4,284  compute nodes that are made up of two Power9
CPUs with four NVIDIA V100 GPUs. Please refer to [Sierra-LLNL]_ for more
detailed information.



Approvals
=========

- LA-UR-23-22084 Approved for public release; distribution is unlimited.
- Content from Sandia National Laboratories considered unclassified with
  unlimited distribution under SAND2023-12176O, SAND2023-01069O, and
  SAND2023-01070O.


