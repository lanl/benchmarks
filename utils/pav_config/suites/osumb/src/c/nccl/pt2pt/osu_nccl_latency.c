#define BENCHMARK "OSU NCCL%s Latency Test"
/*
 * Copyright (C) 2002-2023 the Network-Based Computing Laboratory
 * (NBCL), The Ohio State University.
 *
 * Contact: Dr. D. K. Panda (panda@cse.ohio-state.edu)
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level OMB directory.
 */
#include <osu_util_nccl.h>

int main(int argc, char *argv[])
{
    int myid, numprocs, i;
    int size;
    MPI_Status reqstat;
    char *send_buf, *recv_buf;
    double t_start = 0.0, t_end = 0.0;
    int po_ret = 0;
    options.bench = PT2PT;
    options.subtype = LAT;

    set_header(HEADER);
    set_benchmark_name("osu_nccl_latency");

    po_ret = process_options(argc, argv);

    if (options.accel != CUDA || options.src != 'D' || options.dst != 'D') {
        fprintf(
            stderr,
            "Warning: Host buffer was set for one of the processes. NCCL "
            "does not support host buffers. Implicitly converting to device "
            "buffer (D D).\n\n");
        options.accel = CUDA;
        options.src = 'D';
        options.dst = 'D';
    }

    if (init_accel()) {
        fprintf(stderr, "Error initializing device\n");
        exit(EXIT_FAILURE);
    }

    MPI_CHECK(MPI_Init(&argc, &argv));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &numprocs));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myid));

    if (0 == myid) {
        switch (po_ret) {
            case PO_CUDA_NOT_AVAIL:
                fprintf(stderr, "CUDA support not enabled.  Please recompile "
                                "benchmark with CUDA support.\n");
                break;
            case PO_OPENACC_NOT_AVAIL:
                fprintf(stderr, "OPENACC support not enabled.  Please "
                                "recompile benchmark with OPENACC support.\n");
                break;
            case PO_BAD_USAGE:
                print_bad_usage_message(myid);
                break;
            case PO_HELP_MESSAGE:
                print_help_message(myid);
                break;
            case PO_VERSION_MESSAGE:
                print_version_message(myid);
                MPI_CHECK(MPI_Finalize());
                exit(EXIT_SUCCESS);
            case PO_OKAY:
                break;
        }
    }

    switch (po_ret) {
        case PO_CUDA_NOT_AVAIL:
        case PO_OPENACC_NOT_AVAIL:
        case PO_BAD_USAGE:
            MPI_CHECK(MPI_Finalize());
            exit(EXIT_FAILURE);
        case PO_HELP_MESSAGE:
        case PO_VERSION_MESSAGE:
            MPI_CHECK(MPI_Finalize());
            exit(EXIT_SUCCESS);
        case PO_OKAY:
            break;
    }

    if (numprocs != 2) {
        if (myid == 0) {
            fprintf(stderr, "This test requires exactly two processes\n");
        }

        MPI_CHECK(MPI_Finalize());
        exit(EXIT_FAILURE);
    }

    allocate_nccl_stream();
    create_nccl_comm(numprocs, myid);

    if (allocate_memory_pt2pt(&send_buf, &recv_buf, myid)) {
        /* Error allocating memory */
        MPI_CHECK(MPI_Finalize());
        exit(EXIT_FAILURE);
    }

    print_header(myid, LAT);

    /* Latency test */
    for (size = options.min_message_size; size <= options.max_message_size;
         size = (size ? size * 2 : 1)) {
        set_buffer_pt2pt(send_buf, myid, options.accel, 'a', size);
        set_buffer_pt2pt(recv_buf, myid, options.accel, 'b', size);

        if (size > LARGE_MESSAGE_SIZE) {
            options.iterations = options.iterations_large;
            options.skip = options.skip_large;
        }

        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

        if (myid == 0) {
            for (i = 0; i < options.iterations + options.skip; i++) {
                if (i == options.skip) {
                    t_start = MPI_Wtime();
                }

                NCCL_CHECK(ncclSend(send_buf, size, ncclChar, 1, nccl_comm,
                                    nccl_stream));
                CUDA_STREAM_SYNCHRONIZE(nccl_stream);
                NCCL_CHECK(ncclRecv(recv_buf, size, ncclChar, 1, nccl_comm,
                                    nccl_stream));
                CUDA_STREAM_SYNCHRONIZE(nccl_stream);
            }

            t_end = MPI_Wtime();
        }

        else if (myid == 1) {
            for (i = 0; i < options.iterations + options.skip; i++) {
                NCCL_CHECK(ncclRecv(recv_buf, size, ncclChar, 0, nccl_comm,
                                    nccl_stream));
                CUDA_STREAM_SYNCHRONIZE(nccl_stream);
                NCCL_CHECK(ncclSend(send_buf, size, ncclChar, 0, nccl_comm,
                                    nccl_stream));
                CUDA_STREAM_SYNCHRONIZE(nccl_stream);
            }
        }

        if (myid == 0) {
            double latency =
                (t_end - t_start) * 1e6 / (2.0 * options.iterations);

            fprintf(stdout, "%-*d%*.*f\n", 10, size, FIELD_WIDTH,
                    FLOAT_PRECISION, latency);
            fflush(stdout);
        }
    }

    free_memory(send_buf, recv_buf, myid);
    deallocate_nccl_stream();
    destroy_nccl_comm();
    MPI_CHECK(MPI_Finalize());

    if (NONE != options.accel) {
        if (cleanup_accel()) {
            fprintf(stderr, "Error cleaning up device\n");
            exit(EXIT_FAILURE);
        }
    }

    return EXIT_SUCCESS;
}
