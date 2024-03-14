******
MLMD
******

This is the documentation for the ATS-5 MLMD Benchmark for HIPPYNN [HIPPYNN]_ driven kokkos-Lammps Molecular Dynamics. 

Purpose
=======

To benchmark performance of Lammps [Lammps]_ driven Molecular Dynamics. The problem configured in this test is a small Ag model built using the data included in the Allegro paper (DOI: https://doi.org/10.1038/s41467-023-36329-y)   

Characteristics
===============

Problem
-------
This is a set of simulations on 1,022,400 silver atoms at a 5fs time step. 

This benchmark is comparable to the Allegro Ag simulation: https://www.nature.com/articles/s41467-023-36329-y

This is a general purpose proxy for many MD simulations. 

This test will evaluate Pytorch performance, GPU performance, and MPI performance. 

Figure of Merit
---------------
Their are two figures of merit for this benchmark. 
The first is the Average Epoch time of the training task. 
The second is the throughput of the MD simulations, which is reported by Lammps as 'Katom-step/s' or 'Matom-step/s'. 
Note that there is no explicit inference FOM as inference is conducted inline with the simulation and is implicitly a 
component of the simulation throughput FOM. 

Building
========


Building the Lammps Python interface environment is somewhat challenging. Below is an outline of the process used to get the environment working on Chicoma. Also, in the benchmarks/kokkos_lammps_hippynn/benchmark-env.yml file is a list of the packages installed in the test environment. Most of these will not affect performance, but the pytorch (1.11.0) and cuda (11.2) versions should be kept the same. 

Building on Chicoma
-------------------

.. code-block::

   #Load modules:
   module switch PrgEnv-cray PrgEnv-gnu
   module load cuda
   module load cpe-cuda
   module load cray-libsci
   module load cray-fftw
   module load python
   module load cmake
   module load cudatoolkit
   
   #Create virtual python environment
   virtenvpath =virt <Set Path> 
   conda create --prefix=${virtenvpath} python=3.10
   source activate ${virtenvpath}
   conda install pytorch-gpu cudatoolkit=11.6 cupy -c pytorch -c nvidia
   conda install matplotlib h5py tqdm python-graphviz cython numba scipy ase -c conda-forge
   
   #Install HIPPYNN
   git clone git@github.com:lanl/hippynn.git
   cd hippynn
   git fetch
   git checkout f8ed7390beb8261c8eec75580c683f5121226b30
   pip install --no-deps -e .
   
   #Install Lammps: 
   git clone git@github.com:bnebgen-LANL/lammps-kokkos-mliap.git
   git checkout lammps-kokkos-mliap
   mkdir build
   cd build
   export CMAKE_PREFIX_PATH="${FFTW_ROOT}" 
   cmake ../cmake 
     -DCMAKE_BUILD_TYPE=RelWithDebInfo \
     -DCMAKE_VERBOSE_MAKEFILE=ON \
     -DLAMMPS_EXCEPTIONS=ON \
     -DBUILD_SHARED_LIBS=ON \
     -DBUILD_MPI=ON \
     -DKokkos_ENABLE_OPENMP=ON \
     -DKokkos_ENABLE_CUDA=ON \
     -DKokkos_ARCH_ZEN2=ON \
     -DPKG_KOKKOS=ON \
     -DCMAKE_CXX_STANDARD=17 \
     -DPKG_MANYBODY=ON \
     -DPKG_MOLECULE=ON \
     -DPKG_KSPACE=ON \
     -DPKG_REPLICA=ON \
     -DPKG_ASPHERE=ON \
     -DPKG_RIGID=ON \
     -DPKG_MPIIO=ON \
     -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
     -DPKG_ML-SNAP=on \
     -DPKG_ML-IAP=on \
     -DPKG_PYTHON=on \
     -DMLIAP_ENABLE_PYTHON=on
   make -j 12
   make install-python

.. Building on nv-devkit
.. -------------------------
.. Building on nv-devkit builds the python environment through spack, since conda building is not available. 

.. .. code-block::

..    gcc_ver=11.2.0
..    gcc_openblas=8.4.0
..    module load gcc/$gcc_ver
..    git clone https://github.com/spack/spack.git
..    source spack/share/spack/setup-env.sh
   
..    spack compiler find
   
..    module load gcc/$gcc_openblas
   
..    spack compiler find
   
..    module load gcc/$gcc_ver
   
..    spack install py-torch%gcc@$gcc_ver cuda=True cuda_arch=80 mkldnn=False ^py-numpy@1.22.4 ^openblas%gcc@$gcc_openblas
..    spack install py-cupy%gcc@$gcc_ver ^nccl cuda_arch=80 ^py-numpy@1.22.4
..    spack install py-numba%gcc@$gcc_ver ^py-numpy@1.22.4 ^openblas%gcc@$gcc_openblas
..    spack install py-scipy%gcc@$gcc_ver ^py-numpy@1.22.4 ^openblas%gcc@$gcc_openblas
..    spack install py-matplotlib%gcc@$gcc_ver  ^py-numpy@1.22.4 ^openblas%gcc@$gcc_openblas
..    spack install py-h5py%gcc@$gcc_ver ^py-numpy@1.22.4 ^openblas%gcc@$gcc_openblas
   
..    spack load py-torch py-cupy py-numba py-numpy py-scipy py-matplotlib py-h5py
   
..    #Install HIPPYNN
..    git clone git@github.com:lanl/hippynn.git
..    cd hippynn
..    git fetch
..    git checkout f8ed7390beb8261c8eec75580c683f5121226b30
..    pip install -e --no-deps ./
   
..    #Build Lammps instructions
..    git clone git@github.com:bnebgen-LANL/lammps-kokkos-mliap --branch v1.0.0
..    cd  lammps-kokkos-mliap
..    mkdir build
..    cd build
..    cmake ../cmake \
..     -DCMAKE_VERBOSE_MAKEFILE=ON \
..     -DLAMMPS_EXCEPTIONS=ON \
..     -DBUILD_SHARED_LIBS=ON \
..     -DBUILD_MPI=ON \
..     -DKokkos_ARCH_AMPERE90=ON \
..     -DKokkos_ENABLE_CUDA=ON \
..     -DCMAKE_CXX_STANDARD=17 \
..     -DPKG_KOKKOS=ON \
..     -DPKG_MANYBODY=ON \
..     -DPKG_MOLECULE=ON \
..     -DPKG_KSPACE=ON \
..     -DPKG_REPLICA=ON \
..     -DPKG_ASPHERE=ON \
..     -DPKG_RIGID=ON \
..     -DPKG_MPIIO=ON \
..     -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
..     -DPKG_ML-SNAP=on \
..     -DPKG_ML-IAP=on \
..     -DPKG_PYTHON=on \
..     -DMLIAP_ENABLE_PYTHON=on \
   
..    make -j 12
..    make install-python



Building on Crossroads
----------------------

.. code-block::

   module load intel-mkl
   module load cray-fftw
   module load python/3.10-anaconda-2023.03

   mkdir $HOME/mlmd-env
   virtenv=$HOME/mlmd-env
   conda create --prefix=${virtenv} python=3.10 
   
   source activate ${virtenv}
   conda install pytorch 
   conda install matplotlib h5py tqdm python-graphviz cython numba scipy ase -c conda-forge

   cd $HOME 
   git clone git@github.com:lanl/hippynn.git 
   cd hippynn/
   git fetch 
   git checkout f8ed7390beb8261c8eec75580c683f5121226b30
   pip install --no-deps -e .
   git clone git@github.com:bnebgen-LANL/lammps-kokkos-mliap.git
   cd  lammps-kokkos-mliap
   mkdir build
   cd build/
   export CMAKE_PREFIX_PATH="${FFTW_ROOT}"
   export CXX=`which icpx`
   export CC=`which icx` 
   cmake ../cmake  -DCMAKE_BUILD_TYPE=Release   -DCMAKE_VERBOSE_MAKEFILE=ON   -DLAMMPS_EXCEPTIONS=ON   -DBUILD_SHARED_LIBS=ON   -DBUILD_MPI=ON   -DKokkos_ENABLE_OPENMP=ON   -DKokkos_ENABLE_CUDA=OFF   -DKokkos_ARCH_SPR=ON   -DPKG_KOKKOS=ON   -DCMAKE_CXX_STANDARD=17   -DPKG_MANYBODY=ON   -DPKG_MOLECULE=ON   -DPKG_KSPACE=ON   -DPKG_REPLICA=ON   -DPKG_ASPHERE=ON   -DPKG_RIGID=ON   -DPKG_MPIIO=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON   -DPKG_ML-SNAP=on   -DPKG_ML-IAP=on   -DPKG_PYTHON=on 
   make -j 12
   make install-python


Running
=======

Once the software is downloaded, compiled and the environment configured, go to the benchmarks/kokkos_lammps_hippynn directory. The exports.bash file will need to be modified to first configure the environment that was constructed in the previous step. This usually consists of "module load" and "source activate <python environment>" commands.Additionally the ${lmpexec} environment variable will need to be set to the absolute path to your lammps executable, compiled in the previous step.

External Files
--------------
The data used to train the network is located here: https://doi.org/10.24435/materialscloud:fr-ts , in particular, Ag_warm_nospin.xyz.

Download the file and put it into the benchmarks/kokkos_lammps_hippynn directory.

Model Training
--------------
Train a network using ``python train_model.py``. This will read the dataset downloaded above and train a network to it.
The process can take quite some time. This will write several files to disk. The final errors of
the model are captured in ``model_results.txt``. An example is shown here::

                        train         valid          test
    -----------------------------------------------------
    EpA-RMSE :        0.46335       0.49286       0.45089
    EpA-MAE  :        0.36372        0.4036       0.36639
    EpA-RSQ  :        0.99893       0.99888       0.99884
    ForceRMSE:         21.255         21.74        20.967
    ForceMAE :         16.759        17.145        16.591
    ForceRsq :         0.9992       0.99916       0.99922
    T-Hier   :     0.00086736    0.00089796    0.00087634
    L2Reg    :         193.15        193.15        193.15
    Loss-Err :       0.046285       0.04785      0.045731
    Loss-Reg :      0.0010605     0.0010911     0.0010695
    Loss     :       0.047346      0.048941        0.0468
    -----------------------------------------------------

The numbers will vary from run to run due random seeds and the non-deterministic nature of multi-threaded / data parallel execution. However you should find that the Energy Per Atom mean absolute error "EpA-MAE" for test is below 0.40 (meV/atom). The test Force MAE "Force MAE" should be below 18 (meV/Angstrom).

The training script will also output the initial box file ``ag_box.data`` as well as an file used to run the resulting potential with LAMMPS, ``hippynn_lammps_model.pt``. Several other files for the training run are put in a directory, ``model_files``.

The "Figure of Merit" for the training task is printed near the end of the ``model_files/model_results.txt`` and is lead with the line "FOM Average Epoch time:" This is the average time to compute an epoch over the training proceedure

Following this process, benchmarks can be run.

Running the Benchmark
----------------------

Two run scripts are provided for reference. Run_Strong_CPU.bash which was used for running on Crossroads and Run_Throughput_GPU.bash which was used for running on Chicoma. 

Finally, the figures of merrit values can be extracted and plotted with the "Benchmark-Plotting.py" script. This will execute even if not all benchmarks are complete. 

Results 
=======

Results from MLMD are provided on the following systems:

* Crossroads (see :ref:`GlobalSystemATS3`)
* Chicoma:  Each node contains 1 AMD EPYC 7713 processor (64 cores), 256 GB CPU memory, and 4 Nvidia A100 GPUs with 40 GB GPU Memory.  

.. Two quantities are extracted from the MD simulations to evaluate performance, though they are directly correlated. The throughput (grad/s) should be viewed as the figure of merit, though ns/day is more useful for users who wish to know the physical processes they can simulate. Thus both are reported here. 

Training HIPNN Model
--------------------
For the training task, only a single FOM needs to be reported, the average epoch time found in the ``model_results.txt`` file. 

* On Chicoma using a single GPU - 1 / FOM Average Epoch time:  1/0.24648178 = 4.05709
* On Crossroads using a single node - 1 / FOM Average Epoch time:   1/2.63468153 = .37956426022925681317

Simulation+Inference 
--------------------
Throughput performance of MLMD Simulation+Inference is provided within the
following figures and tables.


MLMD strong scaling on Crossroads: 4,544 atoms 

.. csv-table::  MLMD strong scaling on Crossroads 4,544 atoms 
   :file: cpu_4k.csv
   :align: center
   :widths: 10, 10, 10
   :header-rows: 1


.. figure:: cpu_4k.png
   :align: center
   :scale: 50%
   :alt: MLMD strong scaling on Crossroads: 4,544 atoms 
   
   MLMD strong scaling on Crossroads: 4,544 atoms. 


MD strong scaling on Crossroads: 18,176 atoms 

.. csv-table::  MLMD strong scaling on Crossroads 18,176 atoms 
   :file: cpu_18k.csv
   :align: center
   :widths: 10, 10, 10
   :header-rows: 1


.. figure:: cpu_18k.png
   :align: center
   :scale: 50%
   :alt: MLMD strong scaling on Crossroads: 18,176 atoms 
   
   MLMD strong scaling on Crossroads: 18,176 atoms  


Single GPU Throughput Scaling on Chicoma
----------------------------------------

Throughput performance of MLMD Simulation+Inference is provided within the
following table and figure.

.. csv-table::  MLMD throughput performance on Chicaoma
   :file: gpu.csv
   :align: center
   :widths: 10, 10
   :header-rows: 1


.. figure:: gpu.png
   :align: center
   :scale: 50%
   :alt: MLMD throughput performance on Chicaoma

   MLMD throughput performance on Chicaoma 


References
==========

.. [HIPPYNN] Nicolas Lubbers, "HIPPYNN" 2021. [Online]. Available: https://github.com/lanl/hippynn. [Accessed: 6- Mar- 2023]
.. [Lammps] Axel Kohlmeyer et. Al, "Lammps". [Online]. Available: https://github.com/lammps/lammps. [Accessed: 6- Mar- 2023]
