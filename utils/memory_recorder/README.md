 # Memory Recorder

This code implements a memory recorder struct that can be used to track the memory usage of a program. 
The struct provides methods to read the meminfo file for each numa node, summarize the maximum RSS (resident set size) usage, and write the results to files.

## Build and installation

This package `REQUIRES` mpi.
This package only works on LINUX.
This package cannot be built with optimization.

This package uses a simple makefile.
By default it installs into prefix `${HOME}/.local` using subfolders `bin`, `lib`, and `include`.
Set `PREFIX` on command line to override this behavior.
Set `CXX` in your environment or on the command line to point at the correct C++ compiler (C++11 required).

A convenience script `install.sh` is provided for building on LANL HPC systems.
This script will build the package in two subdirectories `${MACHINE_NAME}/${COMPILER_NAME}_${MPI_NAME}` under the install prefix. 
The install prefix is set to `${HOME}/proj/installs` by default, set the `PRE_PREFIX` variable on the command line to override this behavior.
Using this method, the `utils/memory_recorder.tcl` modulefile provided here will load the build corresponding to the currently loaded compiler/mpi combination if it exists.
To use the modulefile, copy it to `${HOME}/privatemodules` and load the `use.own` module before loading `memory_recorder.tcl`.
Adjust the `softdir` variable on ine 23 of `memory_recorder.tcl` if setting a custom `PRE_PREFIX`.

## How to use the code

To use the code, you will need to build the program, include the `memory_recorder.h` header file in your program and link the libmemory_recorder.so library. You can then create an instance of the `MemoryRecorder` struct and call the methods to record the memory usage.

The following code snippet shows how to use the struct:

```cxx
#include "memory_recorder.h"

int main() {
  // Create an instance of the MemoryRecorder struct
  MemoryRecorder recorder;

  // Read the meminfo file
  recorder.read_meminfo("NoSpaceStringDescribingCodeLocation");

  // Write the results
  recorder.write_meminfo();
  recorder.write_rss();

  return 0;
}
```

## Memrecorder library.

1. The `MemoryRecorder` struct is defined in the `memory_recorder.h` header file. The struct contains the following static methods:
    * `getRamSize()`: This method returns the total amount of RAM on the system.
    * `getMaxRSS()`: This method returns the maximum RSS usage of the current process.
2. The following are member 
    * `summarizeMaxRSS()`: This method summarizes the maximum RSS usage for all processes on the system. This method is called by `write_rss()` and doesn't need to be called explitly by the user.
    * `read_meminfo(std::string const &loc)`: This method reads the meminfo files and stores the results as a `std::vector` value in a map with key `loc`.
    * `write_meminfo()`: This method writes the results of `read_meminfo()`, the raw Free Memory and percent Free Memory values, for each node to CSV files:    `pctmeminfo_{rel_node_number}_{node_name}.csv`, `meminfo_{rel_node_number}_{node_name}.csv`. If no `read_meminfo()` calls have been performed in the struct instance, this method takes no action.
    * `write_rss()`: This method writes the maximum RSS usage for each process and node to files `rss_{rel_node_number}_{node_name}.meminfo`, and writes the total RSSMax of the program to stdout. Calling this method alone before `MPI_Finalize` will gather and write all the info.

### Summary

To collect the meminfo values at any given point in the code, call `read_meminfo("LOC")` with some string that describes that point in the code.
In order 
To write the RSSMax values, call `write_rss()`. No other method calls are necessary to perform this operation.

## Bin files

### Test binary `memrecord`

Built from `main.cpp`

3. The `read_meminfo("LOC")` methods read the meminfo file and store the results in a map with keys ("LOC") "PostMalloc", "PostFill", and "PostFree" at respective points in the code.
1. Takes 1 argument: `int`, the number of elements in each array. Default is `2**26`, resulting in 3 `256 MiB` arrays.
2. The `main()` function creates an instance of the `MemoryRecorder` struct and calls the methods to record the memory usage.
4. Two arrays are filled with the current time as a float; the other is filled with the sum of those two. This prevents the arrays from being optimized out. Then the arrays are freed.
5. The `write_meminfo()` method writes the data gathered by  `read_meminfo()` to csv files.
6. The `write_rss()` method summarizes the maximum RSS usage for all processes on the system and writes RSSMax info and summary data to `*.meminfo` files in cwd.

### Collection utility `mem_collect`

Copied from `collect_results.py`

* Requires `pandas`.
* Takes two arguments: 
  1. -i, --input_dir: Directory with all the output files from libmemory_recorder. Default: current working directory.
  2. -o, --output_dir: Directory to copy all libmemory_recorder output files into, and create summary files in `summary` subdir.
* Summarizes rss and meminfo data with pandas.

### Summary files
* `rssfractions.csv` contains the RSSMax as a fraction of total memory for each node. It has columns: `NodeNum,NodeName,RamFraction`.
* `mem(raw|pct)_*.csv` transforms the `[pct]meminfo*` files (per node files), csvs output by libmemory_record to contain all nodes with one file per code_location.
* `minloc_mempct_*.csv` shows a sorted version of whatever code location has the lowest average free memory across all nodes. 
* `mempct_minline` contains the low water mark for percent free memory for all meminfo reads.

## Utilities

* `memory_recorder.tcl` is a modulefile, suitable for LANL machines, to load the memory_recorder library into the proper paths (users still need to point -I at the include folder in their build systems).
* `basic_sbatch.sh` is a simple sbatch script that runs the `memrecord` test binary with several array sizes and collects the results. See script for user input options.