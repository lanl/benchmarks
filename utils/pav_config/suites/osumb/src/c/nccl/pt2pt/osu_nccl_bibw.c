#define BENCHMARK "OSU NCCL%s Bi-Directional Bandwidth Test"
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
    int myid, numprocs, i, j;
    int size;
    char *s_buf, *r_buf;
    double t_start = 0.0, t_end = 0.0, t = 0.0;
    int window_size = 64;
    int po_ret = 0;
    int peer;
    options.bench = PT2PT;
    options.subtype = BW;

    set_header(HEADER);
    set_benchmark_name("osu_nccl_bibw");

    po_ret = process_options(argc, argv);
    window_size = options.window_size;

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

    peer = 0 == myid ? 1 : 0;

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

    if (allocate_memory_pt2pt(&s_buf, &r_buf, myid)) {
        /* Error allocating memory */
        MPI_CHECK(MPI_Finalize());
        exit(EXIT_FAILURE);
    }

    print_header(myid, BW);

    /* Bi-Directional Bandwidth test */
    for (size = options.min_message_size; size <= options.max_message_size;
         size *= 2) {
        /* touch the data */
        set_buffer_pt2pt(s_buf, myid, options.accel, 'a', size);
        set_buffer_pt2pt(r_buf, myid, options.accel, 'b', size);

        if (size > LARGE_MESSAGE_SIZE) {
            options.iterations = options.iterations_large;
            options.skip = options.skip_large;
        }

        for (i = 0; i < options.iterations + options.skip; i++) {
            if (myid == 0) {
                if (i == options.skip) {
                    t_start = MPI_Wtime();
                }
            }
            ncclGroupStart();
            for (j = 0; j < window_size; j++) {
                NCCL_CHECK(ncclSend(s_buf, size, ncclChar, peer, nccl_comm,
                                    nccl_stream));
                NCCL_CHECK(ncclRecv(r_buf, size, ncclChar, peer, nccl_comm,
                                    nccl_stream));
            }
            ncclGroupEnd();
            CUDA_STREAM_SYNCHRONIZE(nccl_stream);
        }

        if (myid == 0) {
            t_end = MPI_Wtime();
            t = t_end - t_start;
            double tmp = size / 1e6 * options.iterations * window_size * 2;

            fprintf(stdout, "%-*d%*.*f\n", 10, size, FIELD_WIDTH,
                    FLOAT_PRECISION, tmp / t);
            fflush(stdout);
        }
    }

    free_memory(s_buf, r_buf, myid);
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
