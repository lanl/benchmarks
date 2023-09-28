#define BENCHMARK "OSU MPI_Fetch_and_op%s latency Test"
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
#include <errno.h>

double t_start = 0.0, t_end = 0.0;
uint64_t *sbuf = NULL, *tbuf = NULL, *win_base = NULL;
omb_graph_options_t omb_graph_op;
int validation_error_flag = 0;
MPI_Datatype mpi_type_list[OMB_NUM_DATATYPES];

void print_latency(int, int, float);
void run_fop_with_lock(int rank, enum WINDOW win_type, MPI_Datatype data_type,
                       MPI_Op op);
void run_fop_with_fence(int rank, enum WINDOW win_type, MPI_Datatype data_type,
                        MPI_Op op);
void run_fop_with_lock_all(int rank, enum WINDOW win_type,
                           MPI_Datatype data_type, MPI_Op op);
void run_fop_with_flush(int rank, enum WINDOW win_type, MPI_Datatype data_type,
                        MPI_Op op);
void run_fop_with_flush_local(int rank, enum WINDOW win_type,
                              MPI_Datatype data_type, MPI_Op op);
void run_fop_with_pscw(int rank, enum WINDOW win_type, MPI_Datatype data_type,
                       MPI_Op op);

int main(int argc, char *argv[])
{
    int rank, nprocs;
    int po_ret = PO_OKAY;
    int ntypes = 0;
    MPI_Op op = MPI_SUM;
    int jdata_type = 0;
    int jrank_print = 0;
    options.win = WIN_ALLOCATE;
    options.sync = FLUSH;

    options.bench = ONE_SIDED;
    options.subtype = LAT;
    options.synctype = ALL_SYNC;
    options.show_validation = 1;

    set_header(HEADER);
    set_benchmark_name("osu_fop_latency");

    po_ret = process_options(argc, argv);
    omb_populate_mpi_type_list(mpi_type_list);
    ntypes = options.omb_dtype_itr;

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
                usage_one_sided("osu_fop_latency");
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

    for (jdata_type = 0; jdata_type < ntypes; jdata_type++) {
        print_header_one_sided(rank, options.win, options.sync,
                               mpi_type_list[jdata_type]);

        switch (options.sync) {
            case LOCK:
                run_fop_with_lock(rank, options.win, mpi_type_list[jdata_type],
                                  op);
                break;
            case LOCK_ALL:
                run_fop_with_lock_all(rank, options.win,
                                      mpi_type_list[jdata_type], op);
                break;
            case PSCW:
                run_fop_with_pscw(rank, options.win, mpi_type_list[jdata_type],
                                  op);
                break;
            case FENCE:
                run_fop_with_fence(rank, options.win, mpi_type_list[jdata_type],
                                   op);
                break;
            case FLUSH_LOCAL:
                run_fop_with_flush_local(rank, options.win,
                                         mpi_type_list[jdata_type], op);
                break;
            default:
                run_fop_with_flush(rank, options.win, mpi_type_list[jdata_type],
                                   op);
                break;
        }
    }

    if (options.validate) {
        for (jrank_print = 0; jrank_print < 2; jrank_print++) {
            if (jrank_print == rank) {
                printf("-------------------------------------------\n");
                printf("Atomic Data Validation results for Rank=%d:\n", rank);
                atomic_data_validation_print_summary();
                printf("-------------------------------------------\n");
            }
            MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
        }
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

void print_latency(int rank, int size, float latency_factor)
{
    char *validation_string;
    if (rank != 0)
        return;
    if (options.validate) {
        if (2 & validation_error_flag)
            validation_string = "skipped";
        else if (1 & validation_error_flag)
            validation_string = "failed";
        else
            validation_string = "passed";

        fprintf(stdout, "%-*d%*.*f%*s\n", 10, size, FIELD_WIDTH,
                FLOAT_PRECISION,
                (t_end - t_start) * 1.0e6 * latency_factor / options.iterations,
                FIELD_WIDTH, validation_string);
        fflush(stdout);
        validation_error_flag = 0;
        return;
    } else {
        fprintf(stdout, "%-*d%*.*f\n", 10, size, FIELD_WIDTH, FLOAT_PRECISION,
                (t_end - t_start) * 1.0e6 * latency_factor /
                    options.iterations);
        fflush(stdout);
        return;
    }
}

/*Run FOP with flush local*/
void run_fop_with_flush_local(int rank, enum WINDOW win_type,
                              MPI_Datatype data_type, MPI_Op op)
{
    double t_graph_start = 0.0, t_graph_end = 0.0;
    omb_graph_data_t *omb_graph_data = NULL;
    int papi_eventset = OMB_PAPI_NULL;
    int i, jrank, dtype_size;
    MPI_Win win;
    MPI_Aint disp = 0;

    omb_graph_op.number_of_graphs = 0;
    omb_graph_allocate_and_get_data_buffer(&omb_graph_data, &omb_graph_op, 8,
                                           options.iterations);
    omb_papi_init(&papi_eventset);
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    MPI_CHECK(MPI_Type_size(data_type, &dtype_size));

    allocate_atomic_memory(rank, (char **)&sbuf, (char **)&tbuf, NULL,
                           (char **)&win_base, options.max_message_size,
                           win_type, &win);

    if (options.validate) {
        atomic_data_validation_setup(data_type, rank, win_base,
                                     options.max_message_size);
        atomic_data_validation_setup(data_type, rank, sbuf,
                                     options.max_message_size);
        atomic_data_validation_setup(data_type, rank, tbuf,
                                     options.max_message_size);
    }
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    if (rank == 0) {
        if (win_type == WIN_DYNAMIC) {
            disp = disp_remote;
        }
        MPI_CHECK(MPI_Win_lock(MPI_LOCK_SHARED, 1, 0, win));
        for (i = 0; i < options.skip + options.iterations; i++) {
            if (i == options.skip) {
                omb_papi_start(&papi_eventset);
                t_start = MPI_Wtime();
            }
            if (i >= options.skip) {
                t_graph_start = MPI_Wtime();
            }
            MPI_CHECK(
                MPI_Fetch_and_op(sbuf, tbuf, data_type, 1, disp, op, win));
            MPI_CHECK(MPI_Win_flush_local(1, win));
            if (i >= options.skip) {
                t_graph_end = MPI_Wtime();
                if (options.graph) {
                    omb_graph_data->data[i - options.skip] =
                        (t_graph_end - t_graph_start) * 1.0e6;
                }
            }
            if (i == 0 && options.validate) {
                MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
                atomic_data_validation_check(data_type, op, rank, win_base,
                                             tbuf, options.max_message_size, 0,
                                             1, &validation_error_flag);
            }
        }
        t_end = MPI_Wtime();
        MPI_CHECK(MPI_Win_unlock(1, win));
    } else if (options.validate) {
        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
        atomic_data_validation_check(data_type, op, rank, win_base, tbuf,
                                     options.max_message_size, 1, 0,
                                     &validation_error_flag);
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    omb_papi_stop_and_print(&papi_eventset, dtype_size);
    print_latency(rank, dtype_size, 1);
    if (options.graph && 0 == rank) {
        omb_graph_data->avg = (t_end - t_start) * 1.0e6 / options.iterations;
    }
    omb_graph_plot(&omb_graph_op, benchmark_name);
    omb_graph_free_data_buffers(&omb_graph_op);
    omb_papi_free(&papi_eventset);
    free_atomic_memory(sbuf, win_base, tbuf, NULL, win_type, win, rank);
}

/*Run FOP with flush */
void run_fop_with_flush(int rank, enum WINDOW win_type, MPI_Datatype data_type,
                        MPI_Op op)
{
    double t_graph_start = 0.0, t_graph_end = 0.0;
    omb_graph_data_t *omb_graph_data = NULL;
    int papi_eventset = OMB_PAPI_NULL;
    int i, dtype_size;
    MPI_Aint disp = 0;
    MPI_Win win;

    omb_graph_op.number_of_graphs = 0;
    omb_graph_allocate_and_get_data_buffer(&omb_graph_data, &omb_graph_op, 8,
                                           options.iterations);
    omb_papi_init(&papi_eventset);
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    MPI_CHECK(MPI_Type_size(data_type, &dtype_size));

    allocate_atomic_memory(rank, (char **)&sbuf, (char **)&tbuf, NULL,
                           (char **)&win_base, options.max_message_size,
                           win_type, &win);

    if (options.validate) {
        atomic_data_validation_setup(data_type, rank, win_base,
                                     options.max_message_size);
        atomic_data_validation_setup(data_type, rank, sbuf,
                                     options.max_message_size);
        atomic_data_validation_setup(data_type, rank, tbuf,
                                     options.max_message_size);
    }
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    if (rank == 0) {
        if (win_type == WIN_DYNAMIC) {
            disp = disp_remote;
        }
        MPI_CHECK(MPI_Win_lock(MPI_LOCK_SHARED, 1, 0, win));
        for (i = 0; i < options.skip + options.iterations; i++) {
            if (i == options.skip) {
                omb_papi_start(&papi_eventset);
                t_start = MPI_Wtime();
            }
            if (i >= options.skip) {
                t_graph_start = MPI_Wtime();
            }
            MPI_CHECK(
                MPI_Fetch_and_op(sbuf, tbuf, data_type, 1, disp, op, win));
            MPI_CHECK(MPI_Win_flush(1, win));
            if (i >= options.skip) {
                t_graph_end = MPI_Wtime();
                if (options.graph) {
                    omb_graph_data->data[i - options.skip] =
                        (t_graph_end - t_graph_start) * 1.0e6;
                }
            }
            if (i == 0 && options.validate) {
                atomic_data_validation_check(data_type, op, rank, win_base,
                                             tbuf, options.max_message_size, 0,
                                             1, &validation_error_flag);
                MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
            }
        }
        t_end = MPI_Wtime();
        MPI_CHECK(MPI_Win_unlock(1, win));
    } else if (options.validate) {
        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
        atomic_data_validation_check(data_type, op, rank, win_base, tbuf,
                                     options.max_message_size, 1, 0,
                                     &validation_error_flag);
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    omb_papi_stop_and_print(&papi_eventset, dtype_size);
    print_latency(rank, dtype_size, 1);
    if (options.graph && 0 == rank) {
        omb_graph_data->avg = (t_end - t_start) * 1.0e6 / options.iterations;
    }
    omb_graph_plot(&omb_graph_op, benchmark_name);
    omb_graph_free_data_buffers(&omb_graph_op);
    omb_papi_free(&papi_eventset);
    free_atomic_memory(sbuf, win_base, tbuf, NULL, win_type, win, rank);
}

/*Run FOP with Lock_all/unlock_all */
void run_fop_with_lock_all(int rank, enum WINDOW win_type,
                           MPI_Datatype data_type, MPI_Op op)
{
    double t_graph_start = 0.0, t_graph_end = 0.0;
    omb_graph_data_t *omb_graph_data = NULL;
    int papi_eventset = OMB_PAPI_NULL;
    int i, dtype_size;
    MPI_Aint disp = 0;
    MPI_Win win;

    MPI_CHECK(MPI_Type_size(data_type, &dtype_size));

    omb_graph_op.number_of_graphs = 0;
    omb_graph_allocate_and_get_data_buffer(&omb_graph_data, &omb_graph_op, 8,
                                           options.iterations);
    omb_papi_init(&papi_eventset);
    allocate_atomic_memory(rank, (char **)&sbuf, (char **)&tbuf, NULL,
                           (char **)&win_base, options.max_message_size,
                           win_type, &win);

    if (options.validate) {
        atomic_data_validation_setup(data_type, rank, sbuf,
                                     options.max_message_size);
        atomic_data_validation_setup(data_type, rank, tbuf,
                                     options.max_message_size);
        atomic_data_validation_setup(data_type, rank, win_base,
                                     options.max_message_size);
    }
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    if (rank == 0) {
        if (win_type == WIN_DYNAMIC) {
            disp = disp_remote;
        }

        for (i = 0; i < options.skip + options.iterations; i++) {
            if (i == options.skip) {
                omb_papi_start(&papi_eventset);
                t_start = MPI_Wtime();
            }
            if (i >= options.skip) {
                t_graph_start = MPI_Wtime();
            }
            MPI_CHECK(MPI_Win_lock_all(0, win));
            MPI_CHECK(
                MPI_Fetch_and_op(sbuf, tbuf, data_type, 1, disp, op, win));
            MPI_CHECK(MPI_Win_unlock_all(win));
            if (i >= options.skip) {
                t_graph_end = MPI_Wtime();
                if (options.graph) {
                    omb_graph_data->data[i - options.skip] =
                        (t_graph_end - t_graph_start) * 1.0e6;
                }
            }
            if (i == 0 && options.validate) {
                MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
                atomic_data_validation_check(data_type, op, rank, win_base,
                                             tbuf, options.max_message_size, 0,
                                             1, &validation_error_flag);
            }
        }
        t_end = MPI_Wtime();
    } else if (options.validate) {
        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
        atomic_data_validation_check(data_type, op, rank, win_base, tbuf,
                                     options.max_message_size, 1, 0,
                                     &validation_error_flag);
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    omb_papi_stop_and_print(&papi_eventset, dtype_size);
    print_latency(rank, dtype_size, 1);
    if (options.graph && 0 == rank) {
        omb_graph_data->avg = (t_end - t_start) * 1.0e6 / options.iterations;
    }
    omb_graph_plot(&omb_graph_op, benchmark_name);
    omb_graph_free_data_buffers(&omb_graph_op);
    omb_papi_free(&papi_eventset);
    free_atomic_memory(sbuf, win_base, tbuf, NULL, win_type, win, rank);
}

/*Run FOP with Lock/unlock */
void run_fop_with_lock(int rank, enum WINDOW win_type, MPI_Datatype data_type,
                       MPI_Op op)
{
    int i, dtype_size;
    double t_graph_start = 0.0, t_graph_end = 0.0;
    omb_graph_data_t *omb_graph_data = NULL;
    int papi_eventset = OMB_PAPI_NULL;
    MPI_Aint disp = 0;
    MPI_Win win;

    MPI_CHECK(MPI_Type_size(data_type, &dtype_size));

    omb_graph_op.number_of_graphs = 0;
    omb_graph_allocate_and_get_data_buffer(&omb_graph_data, &omb_graph_op, 8,
                                           options.iterations);
    omb_papi_init(&papi_eventset);
    allocate_atomic_memory(rank, (char **)&sbuf, (char **)&tbuf, NULL,
                           (char **)&win_base, options.max_message_size,
                           win_type, &win);

    if (options.validate) {
        atomic_data_validation_setup(data_type, rank, sbuf,
                                     options.max_message_size);
        atomic_data_validation_setup(data_type, rank, tbuf,
                                     options.max_message_size);
        atomic_data_validation_setup(data_type, rank, win_base,
                                     options.max_message_size);
    }
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    if (rank == 0) {
        if (win_type == WIN_DYNAMIC) {
            disp = disp_remote;
        }

        for (i = 0; i < options.skip + options.iterations; i++) {
            if (i == options.skip) {
                omb_papi_start(&papi_eventset);
                t_start = MPI_Wtime();
            }
            if (i >= options.skip) {
                t_graph_start = MPI_Wtime();
            }
            MPI_CHECK(MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 1, 0, win));
            MPI_CHECK(
                MPI_Fetch_and_op(sbuf, tbuf, data_type, 1, disp, op, win));
            MPI_CHECK(MPI_Win_unlock(1, win));
            if (i >= options.skip) {
                t_graph_end = MPI_Wtime();
                if (options.graph) {
                    omb_graph_data->data[i - options.skip] =
                        (t_graph_end - t_graph_start) * 1.0e6;
                }
            }
            if (i == 0 && options.validate) {
                MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
                atomic_data_validation_check(data_type, op, rank, win_base,
                                             tbuf, options.max_message_size, 0,
                                             1, &validation_error_flag);
            }
        }
        t_end = MPI_Wtime();
    } else if (options.validate) {
        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
        atomic_data_validation_check(data_type, op, rank, win_base, tbuf,
                                     options.max_message_size, 1, 0,
                                     &validation_error_flag);
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    omb_papi_stop_and_print(&papi_eventset, dtype_size);
    print_latency(rank, dtype_size, 1);
    if (options.graph && 0 == rank) {
        omb_graph_data->avg = (t_end - t_start) * 1.0e6 / options.iterations;
    }
    omb_graph_plot(&omb_graph_op, benchmark_name);
    omb_graph_free_data_buffers(&omb_graph_op);
    omb_papi_free(&papi_eventset);
    free_atomic_memory(sbuf, win_base, tbuf, NULL, win_type, win, rank);
}

/*Run FOP with Fence */
void run_fop_with_fence(int rank, enum WINDOW win_type, MPI_Datatype data_type,
                        MPI_Op op)
{
    double t_graph_start = 0.0, t_graph_end = 0.0;
    omb_graph_data_t *omb_graph_data = NULL;
    int papi_eventset = OMB_PAPI_NULL;
    int i, dtype_size;
    MPI_Aint disp = 0;
    MPI_Win win;

    MPI_CHECK(MPI_Type_size(data_type, &dtype_size));

    allocate_atomic_memory(rank, (char **)&sbuf, (char **)&tbuf, NULL,
                           (char **)&win_base, options.max_message_size,
                           win_type, &win);

    if (win_type == WIN_DYNAMIC) {
        disp = disp_remote;
    }
    omb_graph_op.number_of_graphs = 0;
    omb_graph_allocate_and_get_data_buffer(&omb_graph_data, &omb_graph_op, 8,
                                           options.iterations);
    omb_papi_init(&papi_eventset);
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    if (rank == 0) {
        for (i = 0; i < options.skip + options.iterations; i++) {
            if (i == options.skip) {
                omb_papi_start(&papi_eventset);
                t_start = MPI_Wtime();
            }
            if (i >= options.skip) {
                if (options.validate) {
                    atomic_data_validation_setup(data_type, rank, sbuf,
                                                 options.max_message_size);
                    atomic_data_validation_setup(data_type, rank, tbuf,
                                                 options.max_message_size);
                    atomic_data_validation_setup(data_type, rank, win_base,
                                                 options.max_message_size);
                }
                t_graph_start = MPI_Wtime();
            }
            MPI_CHECK(MPI_Win_fence(0, win));
            MPI_CHECK(
                MPI_Fetch_and_op(sbuf, tbuf, data_type, 1, disp, op, win));
            MPI_CHECK(MPI_Win_fence(0, win));
            MPI_CHECK(MPI_Win_fence(0, win));
            if (i >= options.skip) {
                t_graph_end = MPI_Wtime();
                if (options.graph) {
                    omb_graph_data->data[i - options.skip] =
                        (t_graph_end - t_graph_start) * 1.0e6;
                }
                if (options.validate) {
                    atomic_data_validation_check(data_type, op, rank, win_base,
                                                 tbuf, options.max_message_size,
                                                 1, 1, &validation_error_flag);
                }
            }
        }
        t_end = MPI_Wtime();
    } else {
        for (i = 0; i < options.skip + options.iterations; i++) {
            if (i == options.skip) {
                omb_papi_start(&papi_eventset);
            }
            if (i >= options.skip && options.validate) {
                atomic_data_validation_setup(data_type, rank, sbuf,
                                             options.max_message_size);
                atomic_data_validation_setup(data_type, rank, tbuf,
                                             options.max_message_size);
                atomic_data_validation_setup(data_type, rank, win_base,
                                             options.max_message_size);
            }
            MPI_CHECK(MPI_Win_fence(0, win));
            MPI_CHECK(MPI_Win_fence(0, win));
            MPI_CHECK(
                MPI_Fetch_and_op(sbuf, tbuf, data_type, 0, disp, op, win));
            MPI_CHECK(MPI_Win_fence(0, win));
            if (i >= options.skip && options.validate) {
                atomic_data_validation_check(data_type, op, rank, win_base,
                                             tbuf, options.max_message_size, 1,
                                             1, &validation_error_flag);
            }
        }
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    omb_papi_stop_and_print(&papi_eventset, dtype_size);
    print_latency(rank, dtype_size, 0.5);
    if (options.graph && 0 == rank) {
        omb_graph_data->avg =
            (t_end - t_start) * 1.0e6 / options.iterations / 2;
    }
    omb_graph_plot(&omb_graph_op, benchmark_name);
    omb_graph_free_data_buffers(&omb_graph_op);

    omb_papi_free(&papi_eventset);
    free_atomic_memory(sbuf, win_base, tbuf, NULL, win_type, win, rank);
}

/*Run FOP with Post/Start/Complete/Wait */
void run_fop_with_pscw(int rank, enum WINDOW win_type, MPI_Datatype data_type,
                       MPI_Op op)
{
    double t_graph_start = 0.0, t_graph_end = 0.0;
    omb_graph_data_t *omb_graph_data = NULL;
    int papi_eventset = OMB_PAPI_NULL;
    int destrank, i, dtype_size;
    MPI_Aint disp = 0;
    MPI_Win win;

    MPI_Group comm_group, group;
    MPI_CHECK(MPI_Comm_group(MPI_COMM_WORLD, &comm_group));
    MPI_CHECK(MPI_Type_size(data_type, &dtype_size));

    omb_graph_op.number_of_graphs = 0;
    omb_graph_allocate_and_get_data_buffer(&omb_graph_data, &omb_graph_op, 8,
                                           options.iterations);
    omb_papi_init(&papi_eventset);

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    allocate_atomic_memory(rank, (char **)&sbuf, (char **)&tbuf, NULL,
                           (char **)&win_base, options.max_message_size,
                           win_type, &win);

    if (win_type == WIN_DYNAMIC) {
        disp = disp_remote;
    }

    if (rank == 0) {
        destrank = 1;
        MPI_CHECK(MPI_Group_incl(comm_group, 1, &destrank, &group));

        for (i = 0; i < options.skip + options.iterations; i++) {
            if (i >= options.skip) {
                atomic_data_validation_setup(data_type, rank, sbuf,
                                             options.max_message_size);
                atomic_data_validation_setup(data_type, rank, tbuf,
                                             options.max_message_size);
                atomic_data_validation_setup(data_type, rank, win_base,
                                             options.max_message_size);
                MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
            }

            MPI_CHECK(MPI_Win_start(group, 0, win));

            if (i == options.skip) {
                omb_papi_start(&papi_eventset);
                t_start = MPI_Wtime();
            }

            if (i >= options.skip) {
                t_graph_start = MPI_Wtime();
            }
            MPI_CHECK(
                MPI_Fetch_and_op(sbuf, tbuf, data_type, 1, disp, op, win));
            MPI_CHECK(MPI_Win_complete(win));
            MPI_CHECK(MPI_Win_post(group, 0, win));
            MPI_CHECK(MPI_Win_wait(win));
            if (i >= options.skip) {
                t_graph_end = MPI_Wtime();
                if (options.graph) {
                    omb_graph_data->data[i - options.skip] =
                        (t_graph_end - t_graph_start) * 1.0e6;
                }
                if (options.validate) {
                    atomic_data_validation_check(data_type, op, rank, win_base,
                                                 tbuf, options.max_message_size,
                                                 1, 1, &validation_error_flag);
                }
            }
        }

        t_end = MPI_Wtime();
    } else {
        /* rank=1 */
        destrank = 0;

        MPI_CHECK(MPI_Group_incl(comm_group, 1, &destrank, &group));

        for (i = 0; i < options.skip + options.iterations; i++) {
            if (i >= options.skip) {
                atomic_data_validation_setup(data_type, rank, sbuf,
                                             options.max_message_size);
                atomic_data_validation_setup(data_type, rank, tbuf,
                                             options.max_message_size);
                atomic_data_validation_setup(data_type, rank, win_base,
                                             options.max_message_size);
                MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
            }

            if (i == options.skip) {
                omb_papi_start(&papi_eventset);
            }
            MPI_CHECK(MPI_Win_post(group, 0, win));
            MPI_CHECK(MPI_Win_wait(win));
            MPI_CHECK(MPI_Win_start(group, 0, win));
            MPI_CHECK(
                MPI_Fetch_and_op(sbuf, tbuf, data_type, 0, disp, op, win));
            MPI_CHECK(MPI_Win_complete(win));
            if (i >= options.skip && options.validate) {
                atomic_data_validation_check(data_type, op, rank, win_base,
                                             tbuf, options.max_message_size, 1,
                                             1, &validation_error_flag);
            }
        }
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    omb_papi_stop_and_print(&papi_eventset, dtype_size);
    print_latency(rank, dtype_size, 0.5);
    if (options.graph && 0 == rank) {
        omb_graph_data->avg =
            (t_end - t_start) * 1.0e6 / options.iterations / 2;
    }

    omb_papi_free(&papi_eventset);
    MPI_CHECK(MPI_Group_free(&group));
    MPI_CHECK(MPI_Group_free(&comm_group));

    free_atomic_memory(sbuf, win_base, tbuf, NULL, win_type, win, rank);
}
/* vi: set sw=4 sts=4 tw=80: */
