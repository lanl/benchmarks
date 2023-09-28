#define BENCHMARK "OSU MPI_Get_accumulate latency Test"
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
char *sbuf = NULL, *rbuf = NULL, *cbuf = NULL;
MPI_Aint sdisp_remote;
MPI_Aint sdisp_local;
omb_graph_options_t omb_graph_op;

void allocate_memory_get_acc_lat(int, char *, int, enum WINDOW, MPI_Win *win);
void print_header_get_acc_lat(int, enum WINDOW, enum SYNC);
void print_latency_get_acc_lat(int, int);
void run_get_acc_with_lock(int, enum WINDOW);
void run_get_acc_with_fence(int, enum WINDOW);
void run_get_acc_with_lock_all(int, enum WINDOW);
void run_get_acc_with_flush(int, enum WINDOW);
void run_get_acc_with_flush_local(int, enum WINDOW);
void run_get_acc_with_pscw(int, enum WINDOW);

int main(int argc, char *argv[])
{
    int rank, nprocs;
    int page_size;
    int po_ret = PO_OKAY;
    size_t size;
    options.bench = ONE_SIDED;
    options.subtype = LAT;
    MPI_Datatype mpi_type_list[OMB_NUM_DATATYPES];

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

    if (nprocs != 2) {
        if (rank == 0) {
            fprintf(stderr, "This test requires exactly two processes\n");
        }
        MPI_CHECK(MPI_Finalize());
        return EXIT_FAILURE;
    }

    set_header(HEADER);
    set_benchmark_name("osu_get_acc_latency");

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

    switch (po_ret) {
        case PO_BAD_USAGE:
            print_help_message_get_acc_lat(rank);
            MPI_CHECK(MPI_Finalize());
            return EXIT_FAILURE;
        case PO_HELP_MESSAGE:
            print_help_message_get_acc_lat(rank);
            MPI_CHECK(MPI_Finalize());
            return EXIT_SUCCESS;
        case PO_VERSION_MESSAGE:
            print_version_message(rank);
            MPI_CHECK(MPI_Finalize());
            exit(EXIT_SUCCESS);
        case PO_OKAY:
            break;
    }

    page_size = getpagesize();
    assert(page_size <= MAX_ALIGNMENT);
    size = options.max_message_size;
    CHECK(posix_memalign((void **)&sbuf, page_size, size));
    memset(sbuf, 0, size);
    if (options.win != WIN_ALLOCATE) {
        CHECK(posix_memalign((void **)&rbuf, page_size, size));
        memset(rbuf, 0, size);
    }
    CHECK(posix_memalign((void **)&cbuf, page_size, size));
    memset(cbuf, 0, size);

    print_header_get_acc_lat(rank, options.win, options.sync);

    switch (options.sync) {
        case LOCK:
            run_get_acc_with_lock(rank, options.win);
            break;
        case LOCK_ALL:
            run_get_acc_with_lock_all(rank, options.win);
            break;
        case PSCW:
            run_get_acc_with_pscw(rank, options.win);
            break;
        case FLUSH_LOCAL:
            run_get_acc_with_flush_local(rank, options.win);
            break;
        case FENCE:
            run_get_acc_with_fence(rank, options.win);
            break;
        default:
            run_get_acc_with_flush(rank, options.win);
            break;
    }

    MPI_CHECK(MPI_Finalize());

    free(sbuf);
    if (options.win != WIN_ALLOCATE) {
        free(rbuf);
    }
    free(cbuf);

    return EXIT_SUCCESS;
}

/*Run Get_accumulate with flush */
void run_get_acc_with_flush(int rank, enum WINDOW type)
{
    int size, i;
    double t_graph_start = 0.0, t_graph_end = 0.0;
    int papi_eventset = OMB_PAPI_NULL;
    omb_graph_data_t *omb_graph_data = NULL;
    MPI_Aint disp = 0;
    MPI_Win win;

    omb_papi_init(&papi_eventset);
    for (size = options.min_message_size; size <= options.max_message_size;
         size = (size ? size * 2 : size + 1)) {
        allocate_memory_get_acc_lat(rank, rbuf, size, type, &win);

        if (type == WIN_DYNAMIC) {
            disp = sdisp_remote;
        }

        if (size > LARGE_MESSAGE_SIZE) {
            options.iterations = options.iterations_large;
            options.skip = options.skip_large;
        }

        omb_graph_allocate_and_get_data_buffer(&omb_graph_data, &omb_graph_op,
                                               size, options.iterations);
        if (rank == 0) {
            MPI_CHECK(MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 1, 0, win));
            for (i = 0; i < options.skip + options.iterations; i++) {
                if (i == options.skip) {
                    omb_papi_start(&papi_eventset);
                    t_start = MPI_Wtime();
                }
                if (i >= options.skip) {
                    t_graph_start = MPI_Wtime();
                }
                MPI_CHECK(MPI_Get_accumulate(sbuf, size, MPI_CHAR, cbuf, size,
                                             MPI_CHAR, 1, disp, size, MPI_CHAR,
                                             MPI_SUM, win));
                MPI_CHECK(MPI_Win_flush(1, win));
                if (i >= options.skip) {
                    t_graph_end = MPI_Wtime();
                    if (options.graph) {
                        omb_graph_data->data[i - options.skip] =
                            (t_graph_end - t_graph_start) * 1.0e6;
                    }
                }
            }
            t_end = MPI_Wtime();
            MPI_CHECK(MPI_Win_unlock(1, win));
        }

        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
        omb_papi_stop_and_print(&papi_eventset, size);
        print_latency_get_acc_lat(rank, size);
        if (options.graph && 0 == rank) {
            omb_graph_data->avg =
                (t_end - t_start) * 1.0e6 / options.iterations;
        }
        if (options.graph) {
            omb_graph_plot(&omb_graph_op, benchmark_name);
        }
        MPI_Win_free(&win);
    }
    omb_graph_combined_plot(&omb_graph_op, benchmark_name);
    omb_graph_free_data_buffers(&omb_graph_op);
    omb_papi_free(&papi_eventset);
}

/*Run Get_accumulate with flush local*/
void run_get_acc_with_flush_local(int rank, enum WINDOW type)
{
    int size, i;
    double t_graph_start = 0.0, t_graph_end = 0.0;
    int papi_eventset = OMB_PAPI_NULL;
    omb_graph_data_t *omb_graph_data = NULL;
    MPI_Aint disp = 0;
    MPI_Win win;

    omb_papi_init(&papi_eventset);
    for (size = options.min_message_size; size <= options.max_message_size;
         size = (size ? size * 2 : size + 1)) {
        allocate_memory_get_acc_lat(rank, rbuf, size, type, &win);

        if (type == WIN_DYNAMIC) {
            disp = sdisp_remote;
        }

        if (size > LARGE_MESSAGE_SIZE) {
            options.iterations = options.iterations_large;
            options.skip = options.skip_large;
        }

        omb_graph_allocate_and_get_data_buffer(&omb_graph_data, &omb_graph_op,
                                               size, options.iterations);
        if (rank == 0) {
            MPI_CHECK(MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 1, 0, win));
            for (i = 0; i < options.skip + options.iterations; i++) {
                if (i == options.skip) {
                    omb_papi_start(&papi_eventset);
                    t_start = MPI_Wtime();
                }
                if (i >= options.skip) {
                    t_graph_start = MPI_Wtime();
                }
                MPI_CHECK(MPI_Get_accumulate(sbuf, size, MPI_CHAR, cbuf, size,
                                             MPI_CHAR, 1, disp, size, MPI_CHAR,
                                             MPI_SUM, win));
                MPI_CHECK(MPI_Win_flush_local(1, win));
                if (i >= options.skip) {
                    t_graph_end = MPI_Wtime();
                    if (options.graph) {
                        omb_graph_data->data[i - options.skip] =
                            (t_graph_end - t_graph_start) * 1.0e6;
                    }
                }
            }
            t_end = MPI_Wtime();
            MPI_CHECK(MPI_Win_unlock(1, win));
        }

        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
        print_latency_get_acc_lat(rank, size);
        omb_papi_stop_and_print(&papi_eventset, size);
        if (options.graph && 0 == rank) {
            omb_graph_data->avg =
                (t_end - t_start) * 1.0e6 / options.iterations;
        }
        if (options.graph) {
            omb_graph_plot(&omb_graph_op, benchmark_name);
        }
        MPI_Win_free(&win);
    }
    omb_graph_combined_plot(&omb_graph_op, benchmark_name);
    omb_graph_free_data_buffers(&omb_graph_op);
    omb_papi_free(&papi_eventset);
}

/*Run Get_accumulate with Lock_all/unlock_all */
void run_get_acc_with_lock_all(int rank, enum WINDOW type)
{
    int size, i;
    double t_graph_start = 0.0, t_graph_end = 0.0;
    int papi_eventset = OMB_PAPI_NULL;
    omb_graph_data_t *omb_graph_data = NULL;
    MPI_Aint disp = 0;
    MPI_Win win;

    omb_papi_init(&papi_eventset);
    for (size = options.min_message_size; size <= options.max_message_size;
         size = (size ? size * 2 : size + 1)) {
        allocate_memory_get_acc_lat(rank, rbuf, size, type, &win);

        if (type == WIN_DYNAMIC) {
            disp = sdisp_remote;
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
                MPI_CHECK(MPI_Get_accumulate(sbuf, size, MPI_CHAR, cbuf, size,
                                             MPI_CHAR, 1, disp, size, MPI_CHAR,
                                             MPI_SUM, win));
                MPI_CHECK(MPI_Win_unlock_all(win));
                if (i >= options.skip) {
                    t_graph_end = MPI_Wtime();
                    if (options.graph) {
                        omb_graph_data->data[i - options.skip] =
                            (t_graph_end - t_graph_start) * 1.0e6;
                    }
                }
            }
            t_end = MPI_Wtime();
        }

        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

        omb_papi_stop_and_print(&papi_eventset, size);
        print_latency_get_acc_lat(rank, size);
        if (options.graph && 0 == rank) {
            omb_graph_data->avg =
                (t_end - t_start) * 1.0e6 / options.iterations;
        }
        if (options.graph) {
            omb_graph_plot(&omb_graph_op, benchmark_name);
        }
        MPI_Win_free(&win);
    }
    omb_graph_combined_plot(&omb_graph_op, benchmark_name);
    omb_graph_free_data_buffers(&omb_graph_op);
    omb_papi_free(&papi_eventset);
}

/*Run Get_accumulate with Lock/unlock */
void run_get_acc_with_lock(int rank, enum WINDOW type)
{
    int size, i;
    double t_graph_start = 0.0, t_graph_end = 0.0;
    int papi_eventset = OMB_PAPI_NULL;
    omb_graph_data_t *omb_graph_data = NULL;
    MPI_Aint disp = 0;
    MPI_Win win;

    omb_papi_init(&papi_eventset);
    for (size = options.min_message_size; size <= options.max_message_size;
         size = (size ? size * 2 : size + 1)) {
        allocate_memory_get_acc_lat(rank, rbuf, size, type, &win);

        if (type == WIN_DYNAMIC) {
            disp = sdisp_remote;
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
                MPI_CHECK(MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 1, 0, win));
                MPI_CHECK(MPI_Get_accumulate(sbuf, size, MPI_CHAR, cbuf, size,
                                             MPI_CHAR, 1, disp, size, MPI_CHAR,
                                             MPI_SUM, win));
                MPI_CHECK(MPI_Win_unlock(1, win));
                if (i >= options.skip) {
                    t_graph_end = MPI_Wtime();
                    if (options.graph) {
                        omb_graph_data->data[i - options.skip] =
                            (t_graph_end - t_graph_start) * 1.0e6;
                    }
                }
            }
            t_end = MPI_Wtime();
        }

        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

        omb_papi_stop_and_print(&papi_eventset, size);
        print_latency_get_acc_lat(rank, size);
        if (options.graph && 0 == rank) {
            omb_graph_data->avg =
                (t_end - t_start) * 1.0e6 / options.iterations;
        }
        if (options.graph) {
            omb_graph_plot(&omb_graph_op, benchmark_name);
        }
        MPI_Win_free(&win);
    }
    omb_graph_combined_plot(&omb_graph_op, benchmark_name);
    omb_graph_free_data_buffers(&omb_graph_op);
    omb_papi_free(&papi_eventset);
}

/*Run Get_accumulate with Fence */
void run_get_acc_with_fence(int rank, enum WINDOW type)
{
    int size, i;
    double t_graph_start = 0.0, t_graph_end = 0.0;
    int papi_eventset = OMB_PAPI_NULL;
    omb_graph_data_t *omb_graph_data = NULL;
    MPI_Aint disp = 0;
    MPI_Win win;

    omb_papi_init(&papi_eventset);
    for (size = options.min_message_size; size <= options.max_message_size;
         size = (size ? size * 2 : size + 1)) {
        allocate_memory_get_acc_lat(rank, rbuf, size, type, &win);

        if (type == WIN_DYNAMIC) {
            disp = sdisp_remote;
        }

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
                MPI_CHECK(MPI_Get_accumulate(sbuf, size, MPI_CHAR, cbuf, size,
                                             MPI_CHAR, 1, disp, size, MPI_CHAR,
                                             MPI_SUM, win));
                MPI_CHECK(MPI_Win_fence(0, win));
                MPI_CHECK(MPI_Win_fence(0, win));
                if (i >= options.skip) {
                    t_graph_end = MPI_Wtime();
                    if (options.graph) {
                        omb_graph_data->data[i - options.skip] =
                            (t_graph_end - t_graph_start) * 1.0e6 / 2.0;
                    }
                }
            }
            t_end = MPI_Wtime();
        } else {
            for (i = 0; i < options.skip + options.iterations; i++) {
                if (i == options.skip) {
                    omb_papi_start(&papi_eventset);
                }
                MPI_CHECK(MPI_Win_fence(0, win));
                MPI_CHECK(MPI_Win_fence(0, win));
                MPI_CHECK(MPI_Get_accumulate(sbuf, size, MPI_CHAR, cbuf, size,
                                             MPI_CHAR, 0, disp, size, MPI_CHAR,
                                             MPI_SUM, win));
                MPI_CHECK(MPI_Win_fence(0, win));
            }
        }

        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

        omb_papi_stop_and_print(&papi_eventset, size);
        if (rank == 0) {
            fprintf(stdout, "%-*d%*.*f\n", 10, size, FIELD_WIDTH,
                    FLOAT_PRECISION,
                    (t_end - t_start) * 1.0e6 / options.iterations / 2);
            fflush(stdout);
            if (options.graph && 0 == rank) {
                omb_graph_data->avg =
                    (t_end - t_start) * 1.0e6 / options.iterations / 2;
            }
            if (options.graph) {
                omb_graph_plot(&omb_graph_op, benchmark_name);
            }
        }

        MPI_Win_free(&win);
    }
    omb_graph_combined_plot(&omb_graph_op, benchmark_name);
    omb_graph_free_data_buffers(&omb_graph_op);
    omb_papi_free(&papi_eventset);
}

/*Run GET with Post/Start/Complete/Wait */
void run_get_acc_with_pscw(int rank, enum WINDOW type)
{
    int destrank, size, i;
    double t_graph_start = 0.0, t_graph_end = 0.0;
    int papi_eventset = OMB_PAPI_NULL;
    omb_graph_data_t *omb_graph_data = NULL;
    MPI_Aint disp = 0;
    MPI_Win win;

    MPI_Group comm_group, group;
    MPI_CHECK(MPI_Comm_group(MPI_COMM_WORLD, &comm_group));

    omb_papi_init(&papi_eventset);
    for (size = options.min_message_size; size <= options.max_message_size;
         size = (size ? size * 2 : 1)) {
        allocate_memory_get_acc_lat(rank, rbuf, size, type, &win);

        if (type == WIN_DYNAMIC) {
            disp = sdisp_remote;
        }

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
            MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

            for (i = 0; i < options.skip + options.iterations; i++) {
                MPI_CHECK(MPI_Win_start(group, 0, win));
                if (i == options.skip) {
                    omb_papi_start(&papi_eventset);
                    t_start = MPI_Wtime();
                }
                if (i >= options.skip) {
                    t_graph_start = MPI_Wtime();
                }
                MPI_CHECK(MPI_Get_accumulate(sbuf, size, MPI_CHAR, cbuf, size,
                                             MPI_CHAR, 1, disp, size, MPI_CHAR,
                                             MPI_SUM, win));
                MPI_CHECK(MPI_Win_complete(win));
                MPI_CHECK(MPI_Win_post(group, 0, win));
                MPI_CHECK(MPI_Win_wait(win));
                if (i >= options.skip) {
                    t_graph_end = MPI_Wtime();
                    if (options.graph) {
                        omb_graph_data->data[i - options.skip] =
                            (t_graph_end - t_graph_start) * 1.0e6 / 2.0;
                    }
                }
            }

            t_end = MPI_Wtime();
        } else {
            /* rank=1 */
            destrank = 0;

            MPI_CHECK(MPI_Group_incl(comm_group, 1, &destrank, &group));
            MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

            for (i = 0; i < options.skip + options.iterations; i++) {
                if (i == options.skip) {
                    omb_papi_start(&papi_eventset);
                }
                MPI_CHECK(MPI_Win_post(group, 0, win));
                MPI_CHECK(MPI_Win_wait(win));
                MPI_CHECK(MPI_Win_start(group, 0, win));
                MPI_CHECK(MPI_Get_accumulate(sbuf, size, MPI_CHAR, cbuf, size,
                                             MPI_CHAR, 0, disp, size, MPI_CHAR,
                                             MPI_SUM, win));
                MPI_CHECK(MPI_Win_complete(win));
            }
        }

        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

        omb_papi_stop_and_print(&papi_eventset, size);
        if (rank == 0) {
            fprintf(stdout, "%-*d%*.*f\n", 10, size, FIELD_WIDTH,
                    FLOAT_PRECISION,
                    (t_end - t_start) * 1.0e6 / options.iterations / 2);
            fflush(stdout);
            if (options.graph && 0 == rank) {
                omb_graph_data->avg =
                    (t_end - t_start) * 1.0e6 / options.iterations / 2;
            }
            if (options.graph) {
                omb_graph_plot(&omb_graph_op, benchmark_name);
            }
        }

        MPI_CHECK(MPI_Group_free(&group));

        MPI_Win_free(&win);
    }
    omb_graph_combined_plot(&omb_graph_op, benchmark_name);
    omb_graph_free_data_buffers(&omb_graph_op);
    omb_papi_free(&papi_eventset);

    MPI_CHECK(MPI_Group_free(&comm_group));
}

void allocate_memory_get_acc_lat(int rank, char *rbuf, int size,
                                 enum WINDOW type, MPI_Win *win)
{
    MPI_Status reqstat;

    switch (type) {
        case WIN_DYNAMIC:
            MPI_CHECK(
                MPI_Win_create_dynamic(MPI_INFO_NULL, MPI_COMM_WORLD, win));
            MPI_CHECK(MPI_Win_attach(*win, (void *)rbuf, size));
            MPI_CHECK(MPI_Get_address(rbuf, &sdisp_local));
            if (rank == 0) {
                MPI_CHECK(
                    MPI_Send(&sdisp_local, 1, MPI_AINT, 1, 1, MPI_COMM_WORLD));
                MPI_CHECK(MPI_Recv(&sdisp_remote, 1, MPI_AINT, 1, 1,
                                   MPI_COMM_WORLD, &reqstat));
            } else {
                MPI_CHECK(MPI_Recv(&sdisp_remote, 1, MPI_AINT, 0, 1,
                                   MPI_COMM_WORLD, &reqstat));
                MPI_CHECK(
                    MPI_Send(&sdisp_local, 1, MPI_AINT, 0, 1, MPI_COMM_WORLD));
            }
            break;
        case WIN_CREATE:
            MPI_CHECK(MPI_Win_create(rbuf, size, 1, MPI_INFO_NULL,
                                     MPI_COMM_WORLD, win));
            break;
        default:
            MPI_CHECK(MPI_Win_allocate(size, 1, MPI_INFO_NULL, MPI_COMM_WORLD,
                                       (void *)&rbuf, win));
            break;
    }
}

void print_header_get_acc_lat(int rank, enum WINDOW win, enum SYNC sync)
{
    if (rank == 0) {
        fprintf(stdout, HEADER);
        fprintf(stdout, "# Window creation: %s\n", win_info[win]);
        fprintf(stdout, "# Synchronization: %s\n", sync_info[sync]);
        fprintf(stdout, "%-*s%*s\n", 10, "# Size", FIELD_WIDTH, "Latency (us)");
        fflush(stdout);
    }
}

void print_latency_get_acc_lat(int rank, int size)
{
    if (rank == 0) {
        fprintf(stdout, "%-*d%*.*f\n", 10, size, FIELD_WIDTH, FLOAT_PRECISION,
                (t_end - t_start) * 1.0e6 / options.iterations);
        fflush(stdout);
    }
}
/* vi: set sw=4 sts=4 tw=80: */
