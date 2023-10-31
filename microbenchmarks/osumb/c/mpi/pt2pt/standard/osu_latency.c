#define BENCHMARK "OSU MPI%s Latency Test"
/*
 * Copyright (C) 2002-2023 the Network-Based Computing Laboratory
 * (NBCL), The Ohio State University.
 *
 * Contact: Dr. D. K. Panda (panda@cse.ohio-state.edu)
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level OMB directory.
 */
#include <osu_util_mpi.h>

#ifdef _ENABLE_CUDA_KERNEL_
double measure_kernel_lo(char *, int);
void touch_managed_src(char *, int);
void touch_managed_dst(char *, int);
#endif /* #ifdef _ENABLE_CUDA_KERNEL_ */
double calculate_total(double, double, double);

int main(int argc, char *argv[])
{
    int myid, numprocs, i, j;
    int size;
    MPI_Status reqstat;
    omb_graph_options_t omb_graph_options;
    omb_graph_data_t *omb_graph_data = NULL;
    char *s_buf, *r_buf;
    double t_start = 0.0, t_end = 0.0, t_lo = 0.0, t_total = 0.0;
    int po_ret = 0;
    int errors = 0;
    size_t num_elements = 0;
    MPI_Datatype omb_curr_datatype = MPI_CHAR;
    size_t omb_ddt_transmit_size = 0;
    int mpi_type_itr = 0, mpi_type_size = 0, mpi_type_name_length = 0;
    char mpi_type_name_str[OMB_DATATYPE_STR_MAX_LEN];
    MPI_Datatype mpi_type_list[OMB_NUM_DATATYPES];
    int papi_eventset = OMB_PAPI_NULL;

    options.bench = PT2PT;
    options.subtype = LAT;

    set_header(HEADER);
    set_benchmark_name("osu_latency");

    po_ret = process_options(argc, argv);
    omb_populate_mpi_type_list(mpi_type_list);

    if (PO_OKAY == po_ret && NONE != options.accel) {
        if (init_accel()) {
            fprintf(stderr, "Error initializing device\n");
            exit(EXIT_FAILURE);
        }
    }

    MPI_CHECK(MPI_Init(&argc, &argv));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &numprocs));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myid));

    omb_graph_options_init(&omb_graph_options);
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

    if (options.buf_num == SINGLE) {
        if (allocate_memory_pt2pt(&s_buf, &r_buf, myid)) {
            /* Error allocating memory */
            MPI_CHECK(MPI_Finalize());
            exit(EXIT_FAILURE);
        }
    }

    print_header(myid, LAT);
    omb_papi_init(&papi_eventset);

    /* Latency test */
    for (mpi_type_itr = 0; mpi_type_itr < options.omb_dtype_itr;
         mpi_type_itr++) {
        MPI_CHECK(MPI_Type_size(mpi_type_list[mpi_type_itr], &mpi_type_size));
        MPI_CHECK(MPI_Type_get_name(mpi_type_list[mpi_type_itr],
                                    mpi_type_name_str, &mpi_type_name_length));
        omb_curr_datatype = mpi_type_list[mpi_type_itr];
        if (0 == myid) {
            fprintf(stdout, "# Datatype: %s.\n", mpi_type_name_str);
        }
        fflush(stdout);
        if (1 <= mpi_type_itr) {
            print_only_header(myid);
        }
        for (size = options.min_message_size; size <= options.max_message_size;
             size = (size ? size * 2 : 1)) {
            num_elements = size / mpi_type_size;
            if (0 == num_elements) {
                continue;
            }
            if (options.buf_num == MULTIPLE) {
                if (allocate_memory_pt2pt_size(&s_buf, &r_buf, myid, size)) {
                    /* Error allocating memory */
                    MPI_CHECK(MPI_Finalize());
                    exit(EXIT_FAILURE);
                }
            }

            omb_ddt_transmit_size =
                omb_ddt_assign(&omb_curr_datatype, mpi_type_list[mpi_type_itr],
                               num_elements) *
                mpi_type_size;
            num_elements = omb_ddt_get_size(num_elements);
            set_buffer_pt2pt(s_buf, myid, options.accel, 'a', size);
            set_buffer_pt2pt(r_buf, myid, options.accel, 'b', size);

            if (size > LARGE_MESSAGE_SIZE) {
                options.iterations = options.iterations_large;
                options.skip = options.skip_large;
            }

#ifdef _ENABLE_CUDA_KERNEL_
        if ((options.src == 'M' && options.MMsrc == 'D') ||
            (options.dst == 'M' && options.MMdst == 'D')) {
            t_lo = measure_kernel_lo(s_buf, size);
        }
#endif /* #ifdef _ENABLE_CUDA_KERNEL_ */

        omb_graph_allocate_and_get_data_buffer(
            &omb_graph_data, &omb_graph_options, size, options.iterations);
        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
        t_total = 0.0;

        for (i = 0; i < options.iterations + options.skip; i++) {
            if (i == options.skip) {
                omb_papi_start(&papi_eventset);
            }
            if (options.validate) {
                set_buffer_validation(s_buf, r_buf, size, options.accel, i,
                                      omb_curr_datatype);
                MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
            }
            if (myid == 0) {
                for (j = 0; j <= options.warmup_validation; j++) {
                    if (i >= options.skip && j == options.warmup_validation) {
                        t_start = MPI_Wtime();
                    }
#ifdef _ENABLE_CUDA_KERNEL_
                    if (options.src == 'M') {
                        touch_managed_src(s_buf, size);
                    }
#endif /* #ifdef _ENABLE_CUDA_KERNEL_ */
                    MPI_CHECK(MPI_Send(s_buf, num_elements, omb_curr_datatype,
                                       1, 1, MPI_COMM_WORLD));
                    MPI_CHECK(MPI_Recv(r_buf, num_elements, omb_curr_datatype,
                                       1, 1, MPI_COMM_WORLD, &reqstat));
#ifdef _ENABLE_CUDA_KERNEL_
                    if (options.src == 'M') {
                        touch_managed_src(r_buf, size);
                    }
#endif /* #ifdef _ENABLE_CUDA_KERNEL_ */
                    if (i >= options.skip && j == options.warmup_validation) {
                        t_end = MPI_Wtime();
                        t_total += calculate_total(t_start, t_end, t_lo);
                        if (options.graph) {
                            omb_graph_data->data[i - options.skip] =
                                calculate_total(t_start, t_end, t_lo) * 1e6 /
                                2.0;
                        }
                    }
                }
                if (options.validate) {
                    int errors_recv = 0;
                    MPI_CHECK(MPI_Recv(&errors_recv, 1, MPI_INT, 1, 2,
                                       MPI_COMM_WORLD, &reqstat));
                    errors += errors_recv;
                }
            } else if (myid == 1) {
                for (j = 0; j <= options.warmup_validation; j++) {
#ifdef _ENABLE_CUDA_KERNEL_
                    if (options.dst == 'M') {
                        touch_managed_dst(s_buf, size);
                    }
#endif /* #ifdef _ENABLE_CUDA_KERNEL_ */
                    MPI_CHECK(MPI_Recv(r_buf, num_elements, omb_curr_datatype,
                                       0, 1, MPI_COMM_WORLD, &reqstat));
#ifdef _ENABLE_CUDA_KERNEL_
                    if (options.dst == 'M') {
                        touch_managed_dst(r_buf, size);
                    }
#endif /* #ifdef _ENABLE_CUDA_KERNEL_ */
                    MPI_CHECK(MPI_Send(s_buf, num_elements, omb_curr_datatype,
                                       0, 1, MPI_COMM_WORLD));
                }
                if (options.validate) {
                    errors = validate_data(r_buf, size, 1, options.accel, i,
                                           omb_curr_datatype);
                    MPI_CHECK(
                        MPI_Send(&errors, 1, MPI_INT, 0, 2, MPI_COMM_WORLD));
                }
            }
        }

        omb_papi_stop_and_print(&papi_eventset, size);

        if (myid == 0) {
            double latency = (t_total * 1e6) / (2.0 * options.iterations);
            fprintf(stdout, "%-*d", 10, size);
            if (options.validate) {
                fprintf(stdout, "%*.*f%*s", FIELD_WIDTH, FLOAT_PRECISION,
                        latency, FIELD_WIDTH, VALIDATION_STATUS(errors));
            } else {
                fprintf(stdout, "%*.*f", FIELD_WIDTH, FLOAT_PRECISION, latency);
            }
            if (options.omb_enable_ddt) {
                fprintf(stdout, "%*zu", FIELD_WIDTH, omb_ddt_transmit_size);
            }
            fprintf(stdout, "\n");
            fflush(stdout);
            if (options.graph && 0 == myid) {
                omb_graph_data->avg = latency;
            }
        }
        omb_ddt_free(&omb_curr_datatype);
        if (options.buf_num == MULTIPLE) {
            free_memory(s_buf, r_buf, myid);
        }

        if (options.validate) {
            MPI_CHECK(MPI_Bcast(&errors, 1, MPI_INT, 0, MPI_COMM_WORLD));
            if (0 != errors) {
                break;
            }
        }
        }
    }
    if (options.graph) {
        omb_graph_plot(&omb_graph_options, benchmark_name);
    }
    omb_graph_combined_plot(&omb_graph_options, benchmark_name);
    omb_graph_free_data_buffers(&omb_graph_options);
    omb_papi_free(&papi_eventset);
    if (options.buf_num == SINGLE) {
        free_memory(s_buf, r_buf, myid);
    }

    MPI_CHECK(MPI_Finalize());

    if (NONE != options.accel) {
        if (cleanup_accel()) {
            fprintf(stderr, "Error cleaning up device\n");
            exit(EXIT_FAILURE);
        }
    }

    if (errors != 0 && options.validate && myid == 0) {
        fprintf(stdout,
                "DATA VALIDATION ERROR: %s exited with status %d on"
                " message size %d.\n",
                argv[0], EXIT_FAILURE, size);
        exit(EXIT_FAILURE);
    }
    return EXIT_SUCCESS;
}

#ifdef _ENABLE_CUDA_KERNEL_
double measure_kernel_lo(char *buf, int size)
{
    int i;
    double t_lo = 0.0, t_start, t_end;

    for (i = 0; i < 10; i++) {
        launch_empty_kernel(buf, size);
    }

    for (i = 0; i < 1000; i++) {
        t_start = MPI_Wtime();
        launch_empty_kernel(buf, size);
        synchronize_stream();
        t_end = MPI_Wtime();
        t_lo = t_lo + (t_end - t_start);
    }

    t_lo = t_lo / 1000;
    return t_lo;
}

void touch_managed_src(char *buf, int size)
{
    if (options.src == 'M') {
        if (options.MMsrc == 'D') {
            touch_managed(buf, size);
            synchronize_stream();
        } else if ((options.MMsrc == 'H') && size > PREFETCH_THRESHOLD) {
            prefetch_data(buf, size, cudaCpuDeviceId);
            synchronize_stream();
        } else {
            if (!options.validate) {
                memset(buf, 'c', size);
            }
        }
    }
}

void touch_managed_dst(char *buf, int size)
{
    if (options.dst == 'M') {
        if (options.MMdst == 'D') {
            touch_managed(buf, size);
            synchronize_stream();
        } else if ((options.MMdst == 'H') && size > PREFETCH_THRESHOLD) {
            prefetch_data(buf, size, -1);
            synchronize_stream();
        } else {
            if (!options.validate) {
                memset(buf, 'c', size);
            }
        }
    }
}
#endif /* #ifdef _ENABLE_CUDA_KERNEL_ */

double calculate_total(double t_start, double t_end, double t_lo)
{
    double t_total;

    if ((options.src == 'M' && options.MMsrc == 'D') &&
        (options.dst == 'M' && options.MMdst == 'D')) {
        t_total = (t_end - t_start) - (2 * t_lo);
    } else if ((options.src == 'M' && options.MMsrc == 'D') ||
               (options.dst == 'M' && options.MMdst == 'D')) {
        t_total = (t_end - t_start) - t_lo;
    } else {
        t_total = (t_end - t_start);
    }

    return t_total;
}
