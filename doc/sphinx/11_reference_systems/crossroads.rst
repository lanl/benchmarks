.. _ReferenceCrossroads:

**********
Crossroads
**********

The Crossroads (see [ACESCrossroads]_) reference system is the third Advanced
Technology System (ATS-3) in the Advanced Simulation and Computing (ASC)
Program. Each compute node has dual sockets with each sporting an Intel Xeon
Sapphire Rapids (SPR) CPU Max 9480 processor configured with Sub-NUMA Clustering
4 (SNC-4) affinity. This provides 8 NUMA Domains across the node (4 per socket).
Each NUMA Domain has 14 physical cores and 28 virtual cores, which totals 112
physical and 224 virtual cores across the compute node. Each processor has a
base clock frequency of 1.9 GHz with a Max Turbo Frequency of 3.50 GHz. SPR
delivers ultra-wide (512-bit) vector operations capabilities with up to 2 Fused
Multiply Add (FMA) instructions with Intel Advanced Vector Extensions 512
(AVX-512). The total node-level memory, including cache, quantities are listed
below.

- **High-Bandwidth Memory**: 128 GiB
- **L1d cache**: 5.3 MiB (112 instances)
- **L1i cache**: 3.5 MiB (112 instances)
- **L2 cache**: 224 MiB (112 instances)
- **L3 cache**: 225 MiB (2 instances)

Refer to Intel's Ark page (see [IntelArk]_) for more information.


References
==========

.. [ACESCrossroads] ACES, 'Crossroads', 2023. [Online]. Available:
                    https://www.lanl.gov/projects/crossroads/. [Accessed: 18-
                    Sep- 2023]
.. [IntelArk] Intel, 'Intel Xeon CPU Max 9480 Processor', 2023. [Online].
              Available:
              https://www.intel.com/content/www/us/en/products/sku/232592/intel-xeon-cpu-max-9480-processor-112-5m-cache-1-90-ghz/specifications.html.
              [Accessed: 18- Sep- 2023]
