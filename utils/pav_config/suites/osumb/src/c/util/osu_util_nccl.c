/*
 * Copyright (C) 2002-2023 the Network-Based Computing Laboratory
 * (NBCL), The Ohio State University.
 *
 * Contact: Dr. D. K. Panda (panda@cse.ohio-state.edu)
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level directory.
 */

#include "osu_util_nccl.h"

cudaStream_t nccl_stream = 0;
ncclComm_t nccl_comm = NULL;

void create_nccl_comm(int nRanks, int rank)
{
    ncclUniqueId nccl_commId;
    if (rank == 0) {
        /* Generates an Id to be used in ncclCommInitRank. */
        ncclGetUniqueId(&nccl_commId);
    }
    /*
     * ncclGetUniqueId should be called once when creating a
     * communicator and the Id should be distributed to all
     * ranks in the communicator before calling ncclCommInitRank
     */
    MPI_CHECK(MPI_Bcast((void *)&nccl_commId, sizeof(nccl_commId), MPI_BYTE, 0,
                        MPI_COMM_WORLD));
    /* Create a new NCCL communicator */
    NCCL_CHECK(ncclCommInitRank(&nccl_comm, nRanks, nccl_commId, rank));
}

void destroy_nccl_comm() { NCCL_CHECK(ncclCommDestroy(nccl_comm)); }

void allocate_nccl_stream()
{
    CUDA_CHECK(cudaStreamCreateWithFlags(&nccl_stream, cudaStreamNonBlocking));
}

void deallocate_nccl_stream() { CUDA_CHECK(cudaStreamDestroy(nccl_stream)); }
