#define BENCHMARK "OSU MPI%s Multi-threaded Latency Test"
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

pthread_mutex_t finished_size_mutex;
pthread_cond_t finished_size_cond;
pthread_mutex_t finished_size_sender_mutex;
pthread_cond_t finished_size_sender_cond;

pthread_barrier_t sender_barrier;

double t_start = 0, t_end = 0, t_total = 0;

int finished_size = 0;
int finished_size_sender = 0;
int errors_reduced = 0, local_errors = 0;

int num_threads_sender = 1;
typedef struct thread_tag {
    int id;
} thread_tag_t;

void *send_thread(void *arg);
void *recv_thread(void *arg);

int main(int argc, char *argv[])
{
    int numprocs = 0, provided = 0, myid = 0, err = 0;
    int i = 0;
    int po_ret = 0;
    pthread_t sr_threads[MAX_NUM_THREADS];
    thread_tag_t tags[MAX_NUM_THREADS];

    pthread_mutex_init(&finished_size_mutex, NULL);
    pthread_cond_init(&finished_size_cond, NULL);
    pthread_mutex_init(&finished_size_sender_mutex, NULL);
    pthread_cond_init(&finished_size_sender_cond, NULL);

    options.bench = PT2PT;
    options.subtype = LAT_MT;

    set_header(HEADER);
    set_benchmark_name("osu_latency_mt");

    po_ret = process_options(argc, argv);

    if (PO_OKAY == po_ret && NONE != options.accel) {
        if (init_accel()) {
            fprintf(stderr, "Error initializing device\n");
            exit(EXIT_FAILURE);
        }
    }

    err = MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    if (err != MPI_SUCCESS) {
        MPI_CHECK(MPI_Abort(MPI_COMM_WORLD, 1));
    }

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

        return EXIT_FAILURE;
    }

    if (options.validate && options.num_threads != options.sender_thread) {
        if (myid == 0) {
            fprintf(stderr,
                    "Number of sender and receiver threads must be same"
                    " when validation is enabled. Use option -t to set\n");
        }

        MPI_CHECK(MPI_Finalize());

        return EXIT_FAILURE;
    }

    /* Check to make sure we actually have a thread-safe
     * implementation
     */

    finished_size = 1;
    finished_size_sender = 1;

    if (provided != MPI_THREAD_MULTIPLE) {
        if (myid == 0) {
            fprintf(stderr,
                    "MPI_Init_thread must return MPI_THREAD_MULTIPLE!\n");
        }

        MPI_CHECK(MPI_Finalize());

        return EXIT_FAILURE;
    }

    if (options.sender_thread != -1) {
        num_threads_sender = options.sender_thread;
    }

    pthread_barrier_init(&sender_barrier, NULL, num_threads_sender);

    if (myid == 0) {
        printf("# Number of Sender threads: %d \n# Number of Receiver threads: "
               "%d\n",
               num_threads_sender, options.num_threads);

        print_header(myid, LAT_MT);

        for (i = 0; i < num_threads_sender; i++) {
            tags[i].id = i;
            pthread_create(&sr_threads[i], NULL, send_thread, &tags[i]);
        }
        for (i = 0; i < num_threads_sender; i++) {
            pthread_join(sr_threads[i], NULL);
        }
    } else {
        for (i = 0; i < options.num_threads; i++) {
            tags[i].id = i;
            pthread_create(&sr_threads[i], NULL, recv_thread, &tags[i]);
        }

        for (i = 0; i < options.num_threads; i++) {
            pthread_join(sr_threads[i], NULL);
        }
    }

    MPI_CHECK(MPI_Finalize());

    return EXIT_SUCCESS;
}

void *recv_thread(void *arg)
{
    int size = 0, i = 0, val = 0, j;
    int iter = 0;
    int myid = 0;
    char *ret = NULL;
    char *s_buf, *r_buf;
    thread_tag_t *thread_id;
    size_t num_elements = 0;
    MPI_Datatype omb_curr_datatype = MPI_CHAR;
    size_t omb_ddt_transmit_size = 0;
    int mpi_type_itr = 0, mpi_type_size = 0, mpi_type_name_length = 0;
    char mpi_type_name_str[OMB_DATATYPE_STR_MAX_LEN];
    MPI_Datatype mpi_type_list[OMB_NUM_DATATYPES];

    thread_id = (thread_tag_t *)arg;
    val = thread_id->id;

    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myid));
    omb_populate_mpi_type_list(mpi_type_list);

    if (NONE != options.accel && init_accel()) {
        fprintf(stderr, "Error initializing device\n");
        exit(EXIT_FAILURE);
    }

    if (allocate_memory_pt2pt(&s_buf, &r_buf, myid)) {
        /* Error allocating memory */
        fprintf(stderr, "Error allocating memory on Rank %d, thread ID %d\n",
                myid, thread_id->id);
        *ret = '1';
        return ret;
    }

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
        for (size = options.min_message_size, iter = 0;
             size <= options.max_message_size; size = (size ? size * 2 : 1)) {
            num_elements = size / mpi_type_size;
            if (0 == num_elements) {
                continue;
            }
            omb_ddt_transmit_size =
                omb_ddt_assign(&omb_curr_datatype, mpi_type_list[mpi_type_itr],
                               num_elements) *
                mpi_type_size;
            num_elements = omb_ddt_get_size(num_elements);
            pthread_mutex_lock(&finished_size_mutex);

            if (finished_size == options.num_threads) {
                MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

                finished_size = 1;

                pthread_mutex_unlock(&finished_size_mutex);
                pthread_cond_broadcast(&finished_size_cond);
            }

            else {
                finished_size++;

                pthread_cond_wait(&finished_size_cond, &finished_size_mutex);
                pthread_mutex_unlock(&finished_size_mutex);
            }

            if (size > LARGE_MESSAGE_SIZE) {
                options.iterations = options.iterations_large;
                options.skip = options.skip_large;
            }

            /* touch the data */
            set_buffer_pt2pt(s_buf, myid, options.accel, 'a', size);
            set_buffer_pt2pt(r_buf, myid, options.accel, 'b', size);

            if (options.validate) {
                errors_reduced = 0;
            }

            for (i = val; i < (options.iterations + options.skip);
                 i += options.num_threads) {
                if (options.validate) {
                    set_buffer_validation(s_buf, r_buf, size, options.accel,
                                          (i - val), omb_curr_datatype);
                }
                for (j = 0; j <= options.warmup_validation; j++) {
                    if (options.sender_thread > 1) {
                        MPI_Recv(r_buf, num_elements, omb_curr_datatype, 0, i,
                                 MPI_COMM_WORLD, &reqstat[val]);
                        MPI_Send(s_buf, num_elements, omb_curr_datatype, 0, i,
                                 MPI_COMM_WORLD);
                    } else {
                        MPI_Recv(r_buf, num_elements, omb_curr_datatype, 0, 1,
                                 MPI_COMM_WORLD, &reqstat[val]);
                        MPI_Send(s_buf, num_elements, omb_curr_datatype, 0, 2,
                                 MPI_COMM_WORLD);
                    }
                }
                if (options.validate) {
                    local_errors += validate_data(r_buf, size, 1, options.accel,
                                                  (i - val), omb_curr_datatype);
                }
            }

            omb_ddt_free(&omb_curr_datatype);
            iter++;
            if (options.validate) {
                MPI_CHECK(MPI_Allreduce(&local_errors, &errors_reduced, 1,
                                        MPI_INT, MPI_SUM, MPI_COMM_WORLD));
                if (errors_reduced != 0) {
                    break;
                }
            }
        }
    }
    free_memory(s_buf, r_buf, myid);

    sleep(1);

    return 0;
}

void *send_thread(void *arg)
{
    int size = 0, i = 0, val = 0, iter = 0, j;
    int myid = 0;
    char *s_buf, *r_buf;
    double latency = 0;
    thread_tag_t *thread_id = (thread_tag_t *)arg;
    char *ret = NULL;
    int flag_print = 0;
    MPI_Datatype omb_curr_datatype = MPI_CHAR;
    size_t num_elements = 0;
    size_t omb_ddt_transmit_size = 0;
    int mpi_type_itr = 0, mpi_type_size = 0, mpi_type_name_length = 0;
    char mpi_type_name_str[OMB_DATATYPE_STR_MAX_LEN];
    MPI_Datatype mpi_type_list[OMB_NUM_DATATYPES];
    omb_graph_options_t omb_graph_options;
    omb_graph_data_t *omb_graph_data = NULL;

    val = thread_id->id;

    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myid));
    omb_populate_mpi_type_list(mpi_type_list);

    if (NONE != options.accel && init_accel()) {
        fprintf(stderr, "Error initializing device\n");
        exit(EXIT_FAILURE);
    }

    if (allocate_memory_pt2pt(&s_buf, &r_buf, myid)) {
        /* Error allocating memory */
        fprintf(stderr, "Error allocating memory on Rank %d, thread ID %d\n",
                myid, thread_id->id);
        *ret = '1';
        return ret;
    }
    omb_graph_options_init(&omb_graph_options);
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
        for (size = options.min_message_size, iter = 0;
             size <= options.max_message_size; size = (size ? size * 2 : 1)) {
            num_elements = size / mpi_type_size;
            if (0 == num_elements) {
                continue;
            }
            omb_ddt_transmit_size =
                omb_ddt_assign(&omb_curr_datatype, mpi_type_list[mpi_type_itr],
                               num_elements) *
                mpi_type_size;
            num_elements = omb_ddt_get_size(num_elements);
            pthread_mutex_lock(&finished_size_sender_mutex);

            if (finished_size_sender == num_threads_sender) {
                MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

                finished_size_sender = 1;

                pthread_mutex_unlock(&finished_size_sender_mutex);
                pthread_cond_broadcast(&finished_size_sender_cond);
            } else {
                finished_size_sender++;

                pthread_cond_wait(&finished_size_sender_cond,
                                  &finished_size_sender_mutex);
                pthread_mutex_unlock(&finished_size_sender_mutex);
            }

            if (size > LARGE_MESSAGE_SIZE) {
                options.iterations = options.iterations_large;
                options.skip = options.skip_large;
            }

            omb_graph_allocate_and_get_data_buffer(
                &omb_graph_data, &omb_graph_options, size, options.iterations);
            /* touch the data */
            set_buffer_pt2pt(s_buf, myid, options.accel, 'a', size);
            set_buffer_pt2pt(r_buf, myid, options.accel, 'b', size);

            if (options.validate) {
                errors_reduced = 0;
            }

            flag_print = 0;
            t_total = 0.0;
            for (i = val; i < options.iterations + options.skip;
                 i += num_threads_sender) {
                if (options.validate) {
                    set_buffer_validation(s_buf, r_buf, size, options.accel,
                                          (i - val), omb_curr_datatype);
                }

                for (j = 0; j <= options.warmup_validation; j++) {
                    if (i == options.skip) {
                        flag_print = 1;
                    }

                    if (i >= options.skip && j == options.warmup_validation) {
                        t_start = MPI_Wtime();
                    }

                    if (options.sender_thread > 1) {
                        MPI_CHECK(MPI_Send(s_buf, num_elements,
                                           omb_curr_datatype, 1, i,
                                           MPI_COMM_WORLD));
                        MPI_CHECK(MPI_Recv(r_buf, num_elements,
                                           omb_curr_datatype, 1, i,
                                           MPI_COMM_WORLD, &reqstat[val]));
                    } else {
                        MPI_CHECK(MPI_Send(s_buf, num_elements,
                                           omb_curr_datatype, 1, 1,
                                           MPI_COMM_WORLD));
                        MPI_CHECK(MPI_Recv(r_buf, num_elements,
                                           omb_curr_datatype, 1, 2,
                                           MPI_COMM_WORLD, &reqstat[val]));
                    }

                    if (i >= options.skip && j == options.warmup_validation) {
                        t_end = MPI_Wtime();
                        t_total += (t_end - t_start);
                        if (options.graph) {
                            omb_graph_data->data[i - options.skip] =
                                (t_end - t_start) * 1.0e6 / 2.0;
                        }
                    }
                }
                if (options.validate) {
                    local_errors += validate_data(r_buf, size, 1, options.accel,
                                                  (i - val), omb_curr_datatype);
                }
            }

            if (options.validate) {
                MPI_CHECK(MPI_Allreduce(&local_errors, &errors_reduced, 1,
                                        MPI_INT, MPI_SUM, MPI_COMM_WORLD));
            }

            pthread_barrier_wait(&sender_barrier);
            if (flag_print == 1) {
                latency = (t_total)*1.0e6 /
                          (2.0 * options.iterations / num_threads_sender);
                fprintf(stdout, "%-*d", 10, size);
                if (options.validate) {
                    fprintf(stdout, "%*.*f%*s", FIELD_WIDTH, FLOAT_PRECISION,
                            latency, FIELD_WIDTH,
                            VALIDATION_STATUS(errors_reduced));
                } else {
                    fprintf(stdout, "%*.*f", FIELD_WIDTH, FLOAT_PRECISION,
                            latency);
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
            iter++;
            if (options.validate && errors_reduced != 0) {
                break;
            }
        }
    }
    if (options.graph) {
        omb_graph_plot(&omb_graph_options, benchmark_name);
    }
    omb_graph_combined_plot(&omb_graph_options, benchmark_name);
    omb_graph_free_data_buffers(&omb_graph_options);

    free_memory(s_buf, r_buf, myid);

    if (0 != errors_reduced && options.validate && 0 == myid &&
        1 == flag_print) {
        fprintf(stdout,
                "DATA VALIDATION ERROR: %s exited with status %d on"
                " message size %d.\n",
                "osu_latency_mt", EXIT_FAILURE, size);
        exit(EXIT_FAILURE);
    }
    return 0;
}

/* vi: set sw=4 sts=4 tw=80: */
