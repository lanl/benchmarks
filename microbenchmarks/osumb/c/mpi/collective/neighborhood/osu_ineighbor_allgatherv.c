#define BENCHMARK "OSU MPI%s Non-blocking Neighbor Allgatherv Latency Test"
/*
 * Copyright (C) 2023-2024 the Network-Based Computing Laboratory
 * (NBCL), The Ohio State University.
 *
 * Contact: Dr. D. K. Panda (panda@cse.ohio-state.edu)
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level OMB directory.
 *
 */

#include <osu_util_mpi.h>

int main(int argc, char *argv[])
{
    int i = 0, j = 0, rank = 0, size = 0, disp = 0;
    int numprocs = 0;
    double latency = 0.0, t_start = 0.0, t_stop = 0.0;
    double tcomp = 0.0, tcomp_total = 0.0, latency_in_secs = 0.0;
    double test_time = 0.0, test_total = 0.0;
    double timer = 0.0;
    int errors = 0, local_errors = 0;
    double wait_time = 0.0, init_time = 0.0;
    double init_total = 0.0, wait_total = 0.0;
    double avg_time = 0.0;
    int *rdispls = NULL, *recvcounts = NULL;
    omb_graph_options_t omb_graph_options;
    omb_graph_data_t *omb_graph_data = NULL;
    int papi_eventset = OMB_PAPI_NULL;
    int errors_temp = 0;
    options.bench = COLLECTIVE;
    options.subtype = NBC_NHBR_GATHER;
    size_t num_elements = 0;
    MPI_Datatype omb_curr_datatype = MPI_CHAR;
    size_t omb_ddt_transmit_size = 0;
    int mpi_type_itr = 0, mpi_type_size = 0, mpi_type_name_length = 0;
    char mpi_type_name_str[OMB_DATATYPE_STR_MAX_LEN];
    MPI_Datatype mpi_type_list[OMB_NUM_DATATYPES];
    char *sendbuf = NULL;
    char *recvbuf = NULL;
    int po_ret = 0;
    size_t bufsize = 0;
    double t_gca = 0, t_gca_total = 0;
    MPI_Request request;
    MPI_Status status;
    MPI_Comm comm_dist_graph;
    int *indegree = NULL, *outdegree = NULL;
    int *sources = NULL, *sourceweights = NULL;
    int *destinations = NULL, *destweights = NULL;
    int reorder = 0;

    set_header(HEADER);
    set_benchmark_name("osu_ineighbor_allgatherv");
    po_ret = process_options(argc, argv);
    omb_populate_mpi_type_list(mpi_type_list);
    if (PO_OKAY == po_ret && NONE != options.accel) {
        if (init_accel()) {
            fprintf(stderr, "Error initializing device\n");
            exit(EXIT_FAILURE);
        }
    }
    MPI_CHECK(MPI_Init(&argc, &argv));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &numprocs));
    omb_graph_options_init(&omb_graph_options);
    switch (po_ret) {
        case PO_BAD_USAGE:
            print_bad_usage_message(rank);
            MPI_CHECK(MPI_Finalize());
            exit(EXIT_FAILURE);
        case PO_HELP_MESSAGE:
            print_help_message(rank);
            MPI_CHECK(MPI_Finalize());
            exit(EXIT_SUCCESS);
        case PO_VERSION_MESSAGE:
            print_version_message(rank);
            MPI_CHECK(MPI_Finalize());
            exit(EXIT_SUCCESS);
        case PO_OKAY:
            break;
    }
    if (numprocs < 2) {
        if (0 == rank) {
            fprintf(stderr, "This test requires at least two processes\n");
        }
        MPI_CHECK(MPI_Finalize());
        exit(EXIT_FAILURE);
    }
    check_mem_limit(numprocs);
    if (allocate_memory_coll((void **)&recvcounts, numprocs * sizeof(int),
                             NONE)) {
        fprintf(stderr, "Could Not Allocate Memory [rank %d]\n", rank);
        MPI_CHECK(MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE));
    }
    if (allocate_memory_coll((void **)&rdispls, numprocs * sizeof(int), NONE)) {
        fprintf(stderr, "Could Not Allocate Memory [rank %d]\n", rank);
        MPI_CHECK(MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE));
    }
    if (allocate_memory_coll((void **)&sendbuf, options.max_message_size,
                             options.accel)) {
        fprintf(stderr, "Could Not Allocate Memory [rank %d]\n", rank);
        MPI_CHECK(MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE));
    }
    set_buffer(sendbuf, options.accel, 1, options.max_message_size);
    bufsize = options.max_message_size * numprocs;
    if (allocate_memory_coll((void **)&recvbuf, bufsize, options.accel)) {
        fprintf(stderr, "Could Not Allocate Memory [rank %d]\n", rank);
        MPI_CHECK(MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE));
    }
    set_buffer(recvbuf, options.accel, 0, bufsize);
    omb_neighborhood_create(MPI_COMM_WORLD, &indegree, &sources, &sourceweights,
                            &outdegree, &destinations, &destweights);
    t_gca = MPI_Wtime();
    MPI_CHECK(MPI_Dist_graph_create_adjacent(
        MPI_COMM_WORLD, *indegree, sources, sourceweights, *outdegree,
        destinations, destweights, MPI_INFO_NULL, reorder, &comm_dist_graph));
    t_gca = MPI_Wtime() - t_gca;
    MPI_CHECK(MPI_Reduce(&t_gca, &t_gca_total, 1, MPI_DOUBLE, MPI_SUM, 0,
                         MPI_COMM_WORLD));
    if (0 == rank) {
        fprintf(stdout, "Time took to create topology graph:%.*f us.\n",
                FLOAT_PRECISION, t_gca_total * 1e6 / numprocs);
    }
    print_preamble_nbc(rank);
    omb_papi_init(&papi_eventset);
    for (mpi_type_itr = 0; mpi_type_itr < options.omb_dtype_itr;
         mpi_type_itr++) {
        MPI_CHECK(MPI_Type_size(mpi_type_list[mpi_type_itr], &mpi_type_size));
        MPI_CHECK(MPI_Type_get_name(mpi_type_list[mpi_type_itr],
                                    mpi_type_name_str, &mpi_type_name_length));
        omb_curr_datatype = mpi_type_list[mpi_type_itr];
        OMB_MPI_RUN_AT_RANK_ZERO(
            fprintf(stdout, "# Datatype: %s.\n", mpi_type_name_str));
        fflush(stdout);
        print_only_header_nbc(rank);
        for (size = options.min_message_size; size <= options.max_message_size;
             size *= 2) {
            num_elements = size / mpi_type_size;
            if (0 == num_elements) {
                continue;
            }
            if (size > LARGE_MESSAGE_SIZE) {
                options.skip = options.skip_large;
                options.iterations = options.iterations_large;
            }
            omb_graph_allocate_and_get_data_buffer(
                &omb_graph_data, &omb_graph_options, size, options.iterations);
            timer = 0.0;
            omb_ddt_transmit_size =
                omb_ddt_assign(&omb_curr_datatype, mpi_type_list[mpi_type_itr],
                               num_elements) *
                mpi_type_size;
            num_elements = omb_ddt_get_size(num_elements);
            disp = 0;
            for (i = 0; i < numprocs; i++) {
                recvcounts[i] = num_elements;
                rdispls[i] = disp;
                disp += num_elements;
            }
            MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
            for (i = 0; i < options.iterations + options.skip; i++) {
                if (i == options.skip) {
                    omb_papi_start(&papi_eventset);
                }
                if (options.validate) {
                    set_buffer_nhbr_validation(sendbuf, recvbuf, *indegree,
                                               sources, *outdegree, destinations,
                                               size, options.accel, i,
                                               omb_curr_datatype);
                    for (j = 0; j < options.warmup_validation; j++) {
                        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
                        MPI_CHECK(MPI_Ineighbor_allgatherv(
                            sendbuf, num_elements, omb_curr_datatype, recvbuf,
                            recvcounts, rdispls, omb_curr_datatype,
                            comm_dist_graph, &request));
                        MPI_CHECK(MPI_Wait(&request, &status));
                    }
                    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
                }
                t_start = MPI_Wtime();
                MPI_CHECK(MPI_Ineighbor_allgatherv(
                    sendbuf, num_elements, omb_curr_datatype, recvbuf,
                    recvcounts, rdispls, omb_curr_datatype, comm_dist_graph,
                    &request));
                MPI_CHECK(MPI_Wait(&request, &status));
                t_stop = MPI_Wtime();
                MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
                if (options.validate) {
                    local_errors += omb_validate_neighbour_col(
                        comm_dist_graph, recvbuf, *indegree, *outdegree, size,
                        options.accel, i, omb_curr_datatype);
                }
                if (i >= options.skip) {
                    timer += t_stop - t_start;
                }
            }
            MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
            omb_papi_stop_and_print(&papi_eventset, size);
            if (options.validate) {
                MPI_CHECK(MPI_Allreduce(&local_errors, &errors, 1, MPI_INT,
                                        MPI_SUM, MPI_COMM_WORLD));
            }
            latency = (timer * 1e6) / options.iterations;
            /* Comm. latency in seconds, fed to dummy_compute */
            latency_in_secs = timer / options.iterations;
            init_arrays(latency_in_secs);
            disp = 0;
            for (i = 0; i < numprocs; i++) {
                recvcounts[i] = num_elements;
                rdispls[i] = disp;
                disp += num_elements;
            }
            MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

            timer = 0.0;
            tcomp_total = 0;
            tcomp = 0;
            init_total = 0.0;
            wait_total = 0.0;
            test_time = 0.0, test_total = 0.0;

            for (i = 0; i < options.iterations + options.skip; i++) {
                if (options.validate) {
                    set_buffer_nhbr_validation(sendbuf, recvbuf, *indegree,
                                               sources, *outdegree, destinations,
                                               size, options.accel, i,
                                               omb_curr_datatype);
                    for (j = 0; j < options.warmup_validation; j++) {
                        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
                        MPI_CHECK(MPI_Ineighbor_allgatherv(
                            sendbuf, num_elements, omb_curr_datatype, recvbuf,
                            recvcounts, rdispls, omb_curr_datatype,
                            comm_dist_graph, &request));
                        MPI_CHECK(MPI_Wait(&request, &status));
                    }
                    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
                }
                t_start = MPI_Wtime();
                init_time = MPI_Wtime();
                MPI_CHECK(MPI_Ineighbor_allgatherv(
                    sendbuf, num_elements, omb_curr_datatype, recvbuf,
                    recvcounts, rdispls, omb_curr_datatype, comm_dist_graph,
                    &request));
                init_time = MPI_Wtime() - init_time;

                tcomp = MPI_Wtime();
                test_time = dummy_compute(latency_in_secs, &request);
                tcomp = MPI_Wtime() - tcomp;

                wait_time = MPI_Wtime();
                MPI_CHECK(MPI_Wait(&request, &status));
                wait_time = MPI_Wtime() - wait_time;

                t_stop = MPI_Wtime();
                MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

                if (options.validate) {
                    local_errors += omb_validate_neighbour_col(
                        comm_dist_graph, recvbuf, *indegree, *outdegree, size,
                        options.accel, i, omb_curr_datatype);
                }
                if (i >= options.skip) {
                    timer += t_stop - t_start;
                    tcomp_total += tcomp;
                    wait_total += wait_time;
                    test_total += test_time;
                    init_total += init_time;
                    if (options.graph && 0 == rank) {
                        omb_graph_data->data[i - options.skip] =
                            (t_stop - t_start) * 1e6;
                    }
                }
            }
            MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
            if (options.validate) {
                MPI_CHECK(MPI_Allreduce(&local_errors, &errors_temp, 1, MPI_INT,
                                        MPI_SUM, MPI_COMM_WORLD));
                errors += errors_temp;
            }
            avg_time = calculate_and_print_stats(
                rank, size, numprocs, timer, latency, test_total, tcomp_total,
                wait_total, init_total, errors);
            if (options.graph && 0 == rank) {
                omb_graph_data->avg = avg_time;
            }
            omb_ddt_append_stats(omb_ddt_transmit_size);
            omb_ddt_free(&omb_curr_datatype);
            if (0 != errors) {
                break;
            }
        }
    }
    if (0 == rank && options.graph) {
        omb_graph_plot(&omb_graph_options, benchmark_name);
    }
    omb_graph_combined_plot(&omb_graph_options, benchmark_name);
    omb_graph_free_data_buffers(&omb_graph_options);
    omb_papi_free(&papi_eventset);

    free_buffer(rdispls, NONE);
    free_buffer(recvcounts, NONE);
    free_buffer(sendbuf, options.accel);
    free_buffer(recvbuf, options.accel);
    free(sources);
    free(sourceweights);
    free(destinations);
    free(destweights);
    free(indegree);
    free(outdegree);
    MPI_CHECK(MPI_Comm_free(&comm_dist_graph));
    MPI_CHECK(MPI_Finalize());

    if (NONE != options.accel) {
        if (cleanup_accel()) {
            fprintf(stderr, "Error cleaning up device\n");
            exit(EXIT_FAILURE);
        }
    }
    if (0 != errors && options.validate && 0 == rank) {
        fprintf(stdout,
                "DATA VALIDATION ERROR: %s exited with status %d on"
                " message size %d.\n",
                argv[0], EXIT_FAILURE, size);
        exit(EXIT_FAILURE);
    }
    return EXIT_SUCCESS;
}
/* vi: set sw=4 sts=4 tw=80: */
