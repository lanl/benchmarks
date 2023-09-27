#define BENCHMARK "OSU MPI%s Multi-process Latency Test"
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

void communicate(int myid);

int errors = 0;

int main(int argc, char *argv[])
{
    int numprocs = 0, myid = 0;
    int num_processes_sender = 0;
    int i = 0;
    int po_ret = 0;
    int is_child = 0;

    pid_t sr_processes[MAX_NUM_PROCESSES];

    options.bench = PT2PT;
    options.subtype = LAT_MP;

    set_header(HEADER);
    set_benchmark_name("osu_latency_mp");

    po_ret = process_options(argc, argv);

    if (PO_OKAY == po_ret && NONE != options.accel) {
        if (init_accel()) {
            fprintf(stderr, "Error initializing device\n");
            exit(EXIT_FAILURE);
        }
    }

    MPI_CHECK(MPI_Init(&argc, &argv));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &numprocs));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myid));

    if (0 == myid) {
        switch (po_ret) {
            case PO_CUDA_NOT_AVAIL:
                fprintf(stderr, "CUDA support not available.\n");
                break;
            case PO_OPENACC_NOT_AVAIL:
                fprintf(stderr, "OPENACC support not available.\n");
                break;
            case PO_HELP_MESSAGE:
                print_help_message(myid);
                break;
            case PO_BAD_USAGE:
                print_bad_usage_message(myid);
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

    if (options.sender_processes != -1) {
        num_processes_sender = options.sender_processes;
    }
    if (myid == 0) {
        fprintf(stdout, "# Number of forked processes in sender: %d\n",
                num_processes_sender);
        fprintf(stdout, "# Number of forked processes in receiver: %d\n",
                options.num_processes);

        print_header(myid, LAT_MP);
        fflush(stdout);

        for (i = 0; i < num_processes_sender; i++) {
            sr_processes[i] = fork();
            if (sr_processes[i] == 0) {
                is_child = 1;
                break;
            }
        }

        if (is_child == 0) {
            communicate(myid);
        } else {
            sleep(CHILD_SLEEP_SECONDS);
        }
    } else {
        for (i = 0; i < options.num_threads; i++) {
            sr_processes[i] = fork();
            if (sr_processes[i] == 0) {
                is_child = 1;
                break;
            }
        }
        if (is_child == 0) {
            communicate(myid);
        } else {
            sleep(CHILD_SLEEP_SECONDS);
        }
    }

    if (is_child == 0) {
        MPI_CHECK(MPI_Finalize());
        if (NONE != options.accel) {
            if (cleanup_accel()) {
                fprintf(stderr, "Error cleaning up device\n");
                exit(EXIT_FAILURE);
            }
        }
    }

    return EXIT_SUCCESS;
}

void communicate(int myid)
{
    /* Latency test */
    double t_start = 0.0, t_end = 0.0, t_total = 0.0;
    int size = 0, i = 0, j;
    char *s_buf, *r_buf;
    MPI_Status reqstat;
    int local_errors = 0;
    omb_graph_options_t omb_graph_options;
    omb_graph_data_t *omb_graph_data = NULL;
    size_t num_elements = 0;
    MPI_Datatype omb_curr_datatype = MPI_CHAR;
    size_t omb_ddt_transmit_size = 0;
    int mpi_type_itr = 0, mpi_type_size = 0, mpi_type_name_length = 0;
    char mpi_type_name_str[OMB_DATATYPE_STR_MAX_LEN];
    MPI_Datatype mpi_type_list[OMB_NUM_DATATYPES];
    int papi_eventset = OMB_PAPI_NULL;

    omb_populate_mpi_type_list(mpi_type_list);
    if (allocate_memory_pt2pt(&s_buf, &r_buf, myid)) {
        /* Error allocating memory */
        MPI_CHECK(MPI_Finalize());
        exit(EXIT_FAILURE);
    }
    omb_graph_options_init(&omb_graph_options);
    omb_papi_init(&papi_eventset);
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

            omb_graph_allocate_and_get_data_buffer(
                &omb_graph_data, &omb_graph_options, size, options.iterations);
            MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
            if (myid == 0) {
                t_total = 0.0;
                for (i = 0; i < options.iterations + options.skip; i++) {
                    if (i == options.skip) {
                        omb_papi_start(&papi_eventset);
                    }
                    if (options.validate) {
                        set_buffer_validation(s_buf, r_buf, size, options.accel,
                                              i, omb_curr_datatype);
                    }
                    for (j = 0; j <= options.warmup_validation; j++) {
                        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
                        if (i >= options.skip &&
                            j == options.warmup_validation) {
                            t_start = MPI_Wtime();
                        }
                        MPI_CHECK(MPI_Send(s_buf, num_elements,
                                           omb_curr_datatype, 1, 1,
                                           MPI_COMM_WORLD));
                        MPI_CHECK(MPI_Recv(r_buf, num_elements,
                                           omb_curr_datatype, 1, 1,
                                           MPI_COMM_WORLD, &reqstat));
                        if (i >= options.skip &&
                            j == options.warmup_validation) {
                            t_end = MPI_Wtime();
                            t_total += (t_end - t_start);
                            if (options.graph) {
                                omb_graph_data->data[i - options.skip] =
                                    (t_end - t_start) * 1e6 / 2.0;
                            }
                        }
                    }
                    if (options.validate) {
                        local_errors +=
                            validate_data(r_buf, size, 1, options.accel, i,
                                          omb_curr_datatype);
                    }
                }

            } else if (myid == 1) {
                for (i = 0; i < options.iterations + options.skip; i++) {
                    if (i == options.skip) {
                        omb_papi_start(&papi_eventset);
                    }
                    if (options.validate) {
                        set_buffer_validation(s_buf, r_buf, size, options.accel,
                                              i, omb_curr_datatype);
                    }
                    for (j = 0; j <= options.warmup_validation; j++) {
                        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
                        MPI_CHECK(MPI_Recv(r_buf, num_elements,
                                           omb_curr_datatype, 0, 1,
                                           MPI_COMM_WORLD, &reqstat));
                        MPI_CHECK(MPI_Send(s_buf, num_elements,
                                           omb_curr_datatype, 0, 1,
                                           MPI_COMM_WORLD));
                    }
                    if (options.validate) {
                        local_errors +=
                            validate_data(r_buf, size, 1, options.accel, i,
                                          omb_curr_datatype);
                    }
                }
            }

            if (options.validate) {
                MPI_CHECK(MPI_Allreduce(&local_errors, &errors, 1, MPI_INT,
                                        MPI_SUM, MPI_COMM_WORLD));
            }

            omb_papi_stop_and_print(&papi_eventset, size);
            if (myid == 0) {
                double latency = t_total * 1e6 / (2.0 * options.iterations);
                fprintf(stdout, "%-*d", 10, size);
                if (options.validate) {
                    fprintf(stdout, "%*.*f%*s", FIELD_WIDTH, FLOAT_PRECISION,
                            latency, FIELD_WIDTH, VALIDATION_STATUS(errors));
                } else {
                    fprintf(stdout, "%*.*f", 10, FLOAT_PRECISION, latency);
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
            if (options.validate) {
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
    free_memory(s_buf, r_buf, myid);
    if (0 != errors && options.validate && 0 == myid) {
        fprintf(stdout,
                "DATA VALIDATION ERROR: %s exited with status %d on"
                " message size %d.\n",
                "osu_latency_mp", EXIT_FAILURE, size);
        exit(EXIT_FAILURE);
    }
}

/* vi: set sw=4 sts=4 tw=80: */
