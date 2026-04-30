/*
 * Copyright (C) 2002-2023 the Network-Based Computing Laboratory
 * (NBCL), The Ohio State University.
 *
 * Contact: Dr. D. K. Panda (panda@cse.ohio-state.edu)
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level OMB directory.
 */

#include "osu_util_mpi.h"
#include <nccl.h>

extern cudaStream_t nccl_stream;
extern ncclComm_t nccl_comm;

#if 0
#define CUDA_CHECK(cmd)                                                        \
    do {                                                                       \
        int e = cmd;                                                           \
        if (e != cudaSuccess) {                                                \
            fprintf(stderr, "Failed: Cuda Error %s: %d '%s'\n", __FILE__,      \
                    __LINE__, cudaGetErrorString(e));                          \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)
#endif

#define NCCL_CHECK(cmd)                                                        \
    do {                                                                       \
        ncclResult_t r = cmd;                                                  \
        if (r != ncclSuccess) {                                                \
            fprintf(stderr, "Failed: NCCL Error %s: %d '%s'\n", __FILE__,      \
                    __LINE__, ncclGetErrorString(cmd));                        \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define CUDA_STREAM_SYNCHRONIZE(_nccl_stream)                                  \
    do {                                                                       \
        cudaError_t err = cudaErrorNotReady;                                   \
        int flag;                                                              \
        while (err == cudaErrorNotReady) {                                     \
            err = cudaStreamQuery(_nccl_stream);                               \
            MPI_CHECK(MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD,  \
                                 &flag, MPI_STATUS_IGNORE));                   \
        }                                                                      \
        CUDA_CHECK(err);                                                       \
    } while (0)

#define IS_ACCELERATOR_CUDA()                                                  \
    do {                                                                       \
        if (PO_OKAY == po_ret && CUDA != options.accel) {                      \
            fprintf(stderr, "Error: OSU NCCL benchmarks expect the "           \
                            "accelerator to be cuda.\nSet '-d cuda' for "      \
                            "collectives, or 'D D' for pt2pt benchmarks "      \
                            "and try again.\n\n");                             \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Print Function
#define PRINT_ERROR(FMT, args...)                                              \
    do {                                                                       \
        fprintf(stderr, FMT, ##args);                                          \
    } while (0)

void create_nccl_comm(int nRanks, int rank);
void allocate_nccl_stream();
void destroy_nccl_comm();
void deallocate_nccl_stream();
