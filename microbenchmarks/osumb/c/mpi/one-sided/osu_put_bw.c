#define BENCHMARK "OSU MPI_Put%s Bandwidth Test"
/*
 * Copyright (C) 2003-2023 the Network-Based Computing Laboratory
 * (NBCL), The Ohio State University.
 *
 * Contact: Dr. D. K. Panda (panda@cse.ohio-state.edu)
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level OMB directory.
 */

#include <osu_util_mpi.h>

double t_start = 0.0, t_end = 0.0;
char *sbuf = NULL, *win_base = NULL;
omb_graph_options_t omb_graph_op;

void print_bw(int, int, double);
void run_put_with_lock(int, enum WINDOW);
void run_put_with_fence(int, enum WINDOW);
void run_put_with_pscw(int, enum WINDOW);
#if MPI_VERSION >= 3
void run_put_with_lock_all(int, enum WINDOW);
void run_put_with_flush(int, enum WINDOW);
void run_put_with_flush_local(int, enum WINDOW);
#endif

int main(int argc, char *argv[])
{
    int rank, nprocs;
    int po_ret = PO_OKAY;

#if MPI_VERSION >= 3
    options.win = WIN_ALLOCATE;
    options.sync = FLUSH;
#else
    options.win = WIN_CREATE;
    options.sync = LOCK;
#endif

    options.bench = ONE_SIDED;
    options.subtype = BW;
    options.synctype = ALL_SYNC;
    MPI_Datatype mpi_type_list[OMB_NUM_DATATYPES];

    set_header(HEADER);
    set_benchmark_name("osu_put_bw");

    po_ret = process_options(argc, argv);
    omb_populate_mpi_type_list(mpi_type_list);
    if (options.validate) {
        OMB_ERROR_EXIT("Benchmark does not support validation");
    }

    if (PO_OKAY == po_ret && NONE != options.accel) {
        if (init_accel()) {
            fprintf(stderr, "Error initializing device\n");
            exit(EXIT_FAILURE);
        }
    }

    MPI_CHECK(MPI_Init(&argc, &argv));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &nprocs));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    if (0 == rank) {
        if (options.omb_dtype_itr > 1 || mpi_type_list[0] != MPI_CHAR) {
            fprintf(stderr, "Benchmark supports only MPI_CHAR. Continuing with "
                            "MPI_CHAR.\n");
            fflush(stderr);
        }
    }

    if (0 == rank) {
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
                print_bad_usage_message(rank);
            case PO_HELP_MESSAGE:
                usage_one_sided("osu_put_bw");
                break;
            case PO_VERSION_MESSAGE:
                print_version_message(rank);
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

    if (nprocs != 2) {
        if (rank == 0) {
            fprintf(stderr, "This test requires exactly two processes\n");
        }

        MPI_CHECK(MPI_Finalize());

        return EXIT_FAILURE;
    }

    print_header_one_sided(rank, options.win, options.sync, MPI_CHAR);

    switch (options.sync) {
        case LOCK:
            run_put_with_lock(rank, options.win);
            break;
        case PSCW:
            run_put_with_pscw(rank, options.win);
            break;
        case FENCE:
            run_put_with_fence(rank, options.win);
            break;
#if MPI_VERSION >= 3
        case LOCK_ALL:
            run_put_with_lock_all(rank, options.win);
            break;
        case FLUSH_LOCAL:
            run_put_with_flush_local(rank, options.win);
            break;
        default:
            run_put_with_flush(rank, options.win);
            break;
#endif
    }

    MPI_CHECK(MPI_Finalize());

    if (NONE != options.accel) {
        if (cleanup_accel()) {
            fprintf(stderr, "Error cleaning up device\n");
            exit(EXIT_FAILURE);
        }
    }

    return EXIT_SUCCESS;
}

void print_bw(int rank, int size, double t)
{
    if (rank == 0) {
        double tmp = size / 1e6 * options.iterations * options.window_size;

        fprintf(stdout, "%-*d%*.*f\n", 10, size, FIELD_WIDTH, FLOAT_PRECISION,
                tmp / t);
        fflush(stdout);
    }
}

#if MPI_VERSION >= 3
/*Run PUT with flush local */
void run_put_with_flush_local(int rank, enum WINDOW type)
{
    double t = 0.0;
    int size, i, j;
    double t_graph_start = 0.0, t_graph_end = 0.0;
    int papi_eventset = OMB_PAPI_NULL;
    omb_graph_data_t *omb_graph_data = NULL;
    MPI_Aint disp = 0;
    MPI_Win win;

    int window_size = options.window_size;
    omb_papi_init(&papi_eventset);
    for (size = options.min_message_size; size <= options.max_message_size;
         size = size * 2) {
        allocate_memory_one_sided(rank, &sbuf, &win_base, size * window_size,
                                  type, &win);

        if (type == WIN_DYNAMIC) {
            disp = disp_remote;
        }

        if (size > LARGE_MESSAGE_SIZE) {
            options.iterations = options.iterations_large;
            options.skip = options.skip_large;
        }
        omb_graph_allocate_and_get_data_buffer(&omb_graph_data, &omb_graph_op,
                                               size, options.iterations);
        if (rank == 0) {
            MPI_CHECK(MPI_Win_lock(MPI_LOCK_SHARED, 1, 0, win));
            for (i = 0; i < options.skip + options.iterations; i++) {
                if (i == options.skip) {
                    omb_papi_start(&papi_eventset);
                    t_start = MPI_Wtime();
                }
                if (i >= options.skip) {
                    t_graph_start = MPI_Wtime();
                }
                for (j = 0; j < window_size; j++) {
                    MPI_CHECK(MPI_Put(sbuf + (j * size), size, MPI_CHAR, 1,
                                      disp + (j * size), size, MPI_CHAR, win));
                }
                MPI_CHECK(MPI_Win_flush_local(1, win));
                if (i >= options.skip) {
                    t_graph_end = MPI_Wtime();
                    if (options.graph) {
                        omb_graph_data->data[i - options.skip] =
                            (size / 1e6) * options.window_size /
                            (t_graph_end - t_graph_start);
                    }
                }
            }
            t_end = MPI_Wtime();
            MPI_CHECK(MPI_Win_unlock(1, win));
            t = t_end - t_start;
        }

        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

        omb_papi_stop_and_print(&papi_eventset, size);
        print_bw(rank, size, t);
        if (options.graph && 0 == rank) {
            omb_graph_data->avg =
                (size / 1e6 * options.iterations * options.window_size) / t;
        }
        if (options.graph) {
            omb_graph_plot(&omb_graph_op, benchmark_name);
        }
        free_memory_one_sided(sbuf, win_base, type, win, rank);
    }
    omb_graph_combined_plot(&omb_graph_op, benchmark_name);
    omb_graph_free_data_buffers(&omb_graph_op);
    omb_papi_free(&papi_eventset);
}

/*Run PUT with flush */
void run_put_with_flush(int rank, enum WINDOW type)
{
    double t = 0.0;
    int size, i, j;
    double t_graph_start = 0.0, t_graph_end = 0.0;
    int papi_eventset = OMB_PAPI_NULL;
    omb_graph_data_t *omb_graph_data = NULL;
    MPI_Aint disp = 0;
    MPI_Win win;

    int window_size = options.window_size;
    omb_papi_init(&papi_eventset);
    for (size = options.min_message_size; size <= options.max_message_size;
         size = size * 2) {
        allocate_memory_one_sided(rank, &sbuf, &win_base, size * window_size,
                                  type, &win);

        if (type == WIN_DYNAMIC) {
            disp = disp_remote;
        }

        if (size > LARGE_MESSAGE_SIZE) {
            options.iterations = options.iterations_large;
            options.skip = options.skip_large;
        }

        omb_graph_allocate_and_get_data_buffer(&omb_graph_data, &omb_graph_op,
                                               size, options.iterations);
        if (rank == 0) {
            MPI_CHECK(MPI_Win_lock(MPI_LOCK_SHARED, 1, 0, win));
            for (i = 0; i < options.skip + options.iterations; i++) {
                if (i == options.skip) {
                    omb_papi_start(&papi_eventset);
                    t_start = MPI_Wtime();
                }
                if (i >= options.skip) {
                    t_graph_start = MPI_Wtime();
                }
                for (j = 0; j < window_size; j++) {
                    MPI_CHECK(MPI_Put(sbuf + (j * size), size, MPI_CHAR, 1,
                                      disp + (j * size), size, MPI_CHAR, win));
                }
                MPI_CHECK(MPI_Win_flush(1, win));
                if (i >= options.skip) {
                    t_graph_end = MPI_Wtime();
                    if (options.graph) {
                        omb_graph_data->data[i - options.skip] =
                            (size / 1e6) * options.window_size /
                            (t_graph_end - t_graph_start);
                    }
                }
            }
            t_end = MPI_Wtime();
            MPI_CHECK(MPI_Win_unlock(1, win));
            t = t_end - t_start;
        }

        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

        omb_papi_stop_and_print(&papi_eventset, size);
        print_bw(rank, size, t);
        if (options.graph && 0 == rank) {
            omb_graph_data->avg =
                (size / 1e6 * options.iterations * options.window_size) / t;
        }
        if (options.graph) {
            omb_graph_plot(&omb_graph_op, benchmark_name);
        }
        free_memory_one_sided(sbuf, win_base, type, win, rank);
    }
    omb_graph_combined_plot(&omb_graph_op, benchmark_name);
    omb_graph_free_data_buffers(&omb_graph_op);
    omb_papi_free(&papi_eventset);
}

/*Run PUT with Lock_all/unlock_all */
void run_put_with_lock_all(int rank, enum WINDOW type)
{
    double t = 0.0;
    int size, i, j;
    double t_graph_start = 0.0, t_graph_end = 0.0;
    omb_graph_data_t *omb_graph_data = NULL;
    int papi_eventset = OMB_PAPI_NULL;
    MPI_Aint disp = 0;
    MPI_Win win;

    int window_size = options.window_size;
    omb_papi_init(&papi_eventset);
    for (size = options.min_message_size; size <= options.max_message_size;
         size = size * 2) {
        allocate_memory_one_sided(rank, &sbuf, &win_base, size * window_size,
                                  type, &win);

        if (type == WIN_DYNAMIC) {
            disp = disp_remote;
        }

        if (size > LARGE_MESSAGE_SIZE) {
            options.iterations = options.iterations_large;
            options.skip = options.skip_large;
        }

        omb_graph_allocate_and_get_data_buffer(&omb_graph_data, &omb_graph_op,
                                               size, options.iterations);
        if (rank == 0) {
            for (i = 0; i < options.skip + options.iterations; i++) {
                if (i == options.skip) {
                    omb_papi_start(&papi_eventset);
                    t_start = MPI_Wtime();
                }
                if (i >= options.skip) {
                    t_graph_start = MPI_Wtime();
                }
                MPI_CHECK(MPI_Win_lock_all(0, win));
                for (j = 0; j < window_size; j++) {
                    MPI_CHECK(MPI_Put(sbuf + (j * size), size, MPI_CHAR, 1,
                                      disp + (j * size), size, MPI_CHAR, win));
                }
                MPI_CHECK(MPI_Win_unlock_all(win));
                if (i >= options.skip) {
                    t_graph_end = MPI_Wtime();
                    if (options.graph) {
                        omb_graph_data->data[i - options.skip] =
                            (size / 1e6) * options.window_size /
                            (t_graph_end - t_graph_start);
                    }
                }
            }
            t_end = MPI_Wtime();
            t = t_end - t_start;
        }

        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

        omb_papi_stop_and_print(&papi_eventset, size);
        print_bw(rank, size, t);
        if (options.graph && 0 == rank) {
            omb_graph_data->avg =
                (size / 1e6 * options.iterations * options.window_size) / t;
        }
        if (options.graph) {
            omb_graph_plot(&omb_graph_op, benchmark_name);
        }
        free_memory_one_sided(sbuf, win_base, type, win, rank);
    }
    omb_graph_combined_plot(&omb_graph_op, benchmark_name);
    omb_graph_free_data_buffers(&omb_graph_op);
    omb_papi_free(&papi_eventset);
}
#endif

/*Run PUT with Lock/unlock */
void run_put_with_lock(int rank, enum WINDOW type)
{
    double t = 0.0;
    int size, i, j;
    double t_graph_start = 0.0, t_graph_end = 0.0;
    int papi_eventset = OMB_PAPI_NULL;
    omb_graph_data_t *omb_graph_data = NULL;
    MPI_Aint disp = 0;
    MPI_Win win;

    int window_size = options.window_size;
    omb_papi_init(&papi_eventset);
    for (size = options.min_message_size; size <= options.max_message_size;
         size = size * 2) {
        allocate_memory_one_sided(rank, &sbuf, &win_base, size * window_size,
                                  type, &win);

#if MPI_VERSION >= 3
        if (type == WIN_DYNAMIC) {
            disp = disp_remote;
        }
#endif

        if (size > LARGE_MESSAGE_SIZE) {
            options.iterations = options.iterations_large;
            options.skip = options.skip_large;
        }
        omb_graph_allocate_and_get_data_buffer(&omb_graph_data, &omb_graph_op,
                                               size, options.iterations);
        if (rank == 0) {
            for (i = 0; i < options.skip + options.iterations; i++) {
                if (i == options.skip) {
                    omb_papi_start(&papi_eventset);
                    t_start = MPI_Wtime();
                }
                if (i >= options.skip) {
                    t_graph_start = MPI_Wtime();
                }
                MPI_CHECK(MPI_Win_lock(MPI_LOCK_SHARED, 1, 0, win));
                for (j = 0; j < window_size; j++) {
                    MPI_CHECK(MPI_Put(sbuf + (j * size), size, MPI_CHAR, 1,
                                      disp + (j * size), size, MPI_CHAR, win));
                }
                MPI_CHECK(MPI_Win_unlock(1, win));
                if (i >= options.skip) {
                    t_graph_end = MPI_Wtime();
                    if (options.graph) {
                        omb_graph_data->data[i - options.skip] =
                            (size / 1e6) * options.window_size /
                            (t_graph_end - t_graph_start);
                    }
                }
            }
            t_end = MPI_Wtime();
            t = t_end - t_start;
        }

        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

        omb_papi_stop_and_print(&papi_eventset, size);
        print_bw(rank, size, t);
        if (options.graph && 0 == rank) {
            omb_graph_data->avg =
                (size / 1e6 * options.iterations * options.window_size) / t;
        }
        if (options.graph) {
            omb_graph_plot(&omb_graph_op, benchmark_name);
        }
        free_memory_one_sided(sbuf, win_base, type, win, rank);
    }
    omb_graph_combined_plot(&omb_graph_op, benchmark_name);
    omb_graph_free_data_buffers(&omb_graph_op);
    omb_papi_free(&papi_eventset);
}

/*Run PUT with Fence */
void run_put_with_fence(int rank, enum WINDOW type)
{
    double t = 0.0;
    int size, i, j;
    double t_graph_start = 0.0, t_graph_end = 0.0;
    int papi_eventset = OMB_PAPI_NULL;
    omb_graph_data_t *omb_graph_data = NULL;
    MPI_Aint disp = 0;
    MPI_Win win;

    int window_size = options.window_size;
    omb_papi_init(&papi_eventset);
    for (size = options.min_message_size; size <= options.max_message_size;
         size = size * 2) {
        allocate_memory_one_sided(rank, &sbuf, &win_base, size * window_size,
                                  type, &win);

#if MPI_VERSION >= 3
        if (type == WIN_DYNAMIC) {
            disp = disp_remote;
        }
#endif

        if (size > LARGE_MESSAGE_SIZE) {
            options.iterations = options.iterations_large;
            options.skip = options.skip_large;
        }

        omb_graph_allocate_and_get_data_buffer(&omb_graph_data, &omb_graph_op,
                                               size, options.iterations);
        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

        if (rank == 0) {
            for (i = 0; i < options.skip + options.iterations; i++) {
                if (i == options.skip) {
                    omb_papi_start(&papi_eventset);
                    t_start = MPI_Wtime();
                }
                if (i >= options.skip) {
                    t_graph_start = MPI_Wtime();
                }
                MPI_CHECK(MPI_Win_fence(0, win));
                for (j = 0; j < window_size; j++) {
                    MPI_CHECK(MPI_Put(sbuf + (j * size), size, MPI_CHAR, 1,
                                      disp + (j * size), size, MPI_CHAR, win));
                }
                MPI_CHECK(MPI_Win_fence(0, win));
                if (i >= options.skip) {
                    t_graph_end = MPI_Wtime();
                    if (options.graph) {
                        omb_graph_data->data[i - options.skip] =
                            (size / 1e6) * options.window_size /
                            (t_graph_end - t_graph_start);
                    }
                }
            }
            t_end = MPI_Wtime();
            t = t_end - t_start;
        } else {
            for (i = 0; i < options.skip + options.iterations; i++) {
                if (i == options.skip) {
                    omb_papi_start(&papi_eventset);
                }
                MPI_CHECK(MPI_Win_fence(0, win));
                MPI_CHECK(MPI_Win_fence(0, win));
            }
        }

        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

        omb_papi_stop_and_print(&papi_eventset, size);
        print_bw(rank, size, t);
        if (options.graph && 0 == rank) {
            omb_graph_data->avg =
                (size / 1e6 * options.iterations * options.window_size) / t;
        }
        if (options.graph) {
            omb_graph_plot(&omb_graph_op, benchmark_name);
        }
        free_memory_one_sided(sbuf, win_base, type, win, rank);
    }
    omb_graph_combined_plot(&omb_graph_op, benchmark_name);
    omb_graph_free_data_buffers(&omb_graph_op);
    omb_papi_free(&papi_eventset);
}

/*Run PUT with Post/Start/Complete/Wait */
void run_put_with_pscw(int rank, enum WINDOW type)
{
    double t = 0.0;
    int destrank, size, i, j;
    double t_graph_start = 0.0, t_graph_end = 0.0;
    int papi_eventset = OMB_PAPI_NULL;
    omb_graph_data_t *omb_graph_data = NULL;
    MPI_Aint disp = 0;
    MPI_Win win;
    MPI_Group comm_group, group;

    MPI_CHECK(MPI_Comm_group(MPI_COMM_WORLD, &comm_group));

    int window_size = options.window_size;
    omb_papi_init(&papi_eventset);
    for (size = options.min_message_size; size <= options.max_message_size;
         size = size * 2) {
        allocate_memory_one_sided(rank, &sbuf, &win_base, size * window_size,
                                  type, &win);

#if MPI_VERSION >= 3
        if (type == WIN_DYNAMIC) {
            disp = disp_remote;
        }
#endif

        if (size > LARGE_MESSAGE_SIZE) {
            options.iterations = options.iterations_large;
            options.skip = options.skip_large;
        }

        omb_graph_allocate_and_get_data_buffer(&omb_graph_data, &omb_graph_op,
                                               size, options.iterations);
        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

        if (rank == 0) {
            destrank = 1;
            MPI_CHECK(MPI_Group_incl(comm_group, 1, &destrank, &group));

            for (i = 0; i < options.skip + options.iterations; i++) {
                MPI_CHECK(MPI_Win_start(group, 0, win));
                if (i == options.skip) {
                    omb_papi_start(&papi_eventset);
                    t_start = MPI_Wtime();
                }
                if (i >= options.skip) {
                    t_graph_start = MPI_Wtime();
                }
                for (j = 0; j < window_size; j++) {
                    MPI_CHECK(MPI_Put(sbuf + j * size, size, MPI_CHAR, 1,
                                      disp + (j * size), size, MPI_CHAR, win));
                }
                MPI_CHECK(MPI_Win_complete(win));
                if (i >= options.skip) {
                    t_graph_end = MPI_Wtime();
                    if (options.graph) {
                        omb_graph_data->data[i - options.skip] =
                            (size / 1e6) * options.window_size /
                            (t_graph_end - t_graph_start);
                    }
                }
            }
            t_end = MPI_Wtime();
            t = t_end - t_start;
        } else {
            destrank = 0;
            MPI_CHECK(MPI_Group_incl(comm_group, 1, &destrank, &group));
            for (i = 0; i < options.skip + options.iterations; i++) {
                if (i == options.skip) {
                    omb_papi_start(&papi_eventset);
                }
                MPI_CHECK(MPI_Win_post(group, 0, win));
                MPI_CHECK(MPI_Win_wait(win));
            }
        }

        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

        omb_papi_stop_and_print(&papi_eventset, size);
        print_bw(rank, size, t);
        if (options.graph && 0 == rank) {
            omb_graph_data->avg =
                (size / 1e6 * options.iterations * options.window_size) / t;
        }
        if (options.graph) {
            omb_graph_plot(&omb_graph_op, benchmark_name);
        }
        MPI_CHECK(MPI_Group_free(&group));

        free_memory_one_sided(sbuf, win_base, type, win, rank);
    }
    omb_graph_combined_plot(&omb_graph_op, benchmark_name);
    omb_graph_free_data_buffers(&omb_graph_op);
    omb_papi_free(&papi_eventset);
    MPI_CHECK(MPI_Group_free(&comm_group));
}
/* vi: set sw=4 sts=4 tw=80: */
