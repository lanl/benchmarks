#define BENCHMARK "OSU MPI%s Barrier Latency Test"
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

int main(int argc, char *argv[])
{
    int i = 0, rank;
    int numprocs;
    double avg_time = 0.0, max_time = 0.0, min_time = 0.0;
    double latency = 0.0, t_start = 0.0, t_stop = 0.0;
    double timer = 0.0;
    int po_ret;
    omb_graph_options_t omb_graph_options;
    omb_graph_data_t *omb_graph_data = NULL;
    int papi_eventset = OMB_PAPI_NULL;
    options.bench = COLLECTIVE;
    options.subtype = BARRIER;

    set_header(HEADER);
    set_benchmark_name("osu_barrier");
    po_ret = process_options(argc, argv);

    if (PO_OKAY == po_ret && NONE != options.accel) {
        if (init_accel()) {
            fprintf(stderr, "Error initializing device\n");
            exit(EXIT_FAILURE);
        }
    }

    options.show_size = 0;

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
        if (rank == 0) {
            fprintf(stderr, "This test requires at least two processes\n");
        }

        MPI_CHECK(MPI_Finalize());

        return EXIT_FAILURE;
    }

    omb_graph_options.number_of_graphs = 0;
    omb_graph_allocate_and_get_data_buffer(&omb_graph_data, &omb_graph_options,
                                           1, options.iterations);
    print_preamble(rank);
    omb_papi_init(&papi_eventset);

    timer = 0.0;

    for (i = 0; i < options.iterations + options.skip; i++) {
        if (i == options.skip) {
            omb_papi_start(&papi_eventset);
        }
        t_start = MPI_Wtime();
        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
        t_stop = MPI_Wtime();

        if (i >= options.skip) {
            timer += t_stop - t_start;
            if (options.graph && 0 == rank) {
                omb_graph_data->data[i - options.skip] =
                    (t_stop - t_start) * 1e6;
            }
        }
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    omb_papi_stop_and_print(&papi_eventset, 0);

    latency = (timer * 1e6) / options.iterations;

    MPI_CHECK(MPI_Reduce(&latency, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0,
                         MPI_COMM_WORLD));
    MPI_CHECK(MPI_Reduce(&latency, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0,
                         MPI_COMM_WORLD));
    MPI_CHECK(MPI_Reduce(&latency, &avg_time, 1, MPI_DOUBLE, MPI_SUM, 0,
                         MPI_COMM_WORLD));
    avg_time = avg_time / numprocs;

    print_stats(rank, 0, avg_time, min_time, max_time);
    if (0 == rank && options.graph) {
        omb_graph_data->avg = avg_time;
        omb_graph_plot(&omb_graph_options, benchmark_name);
        omb_graph_combined_plot(&omb_graph_options, benchmark_name);
        omb_graph_free_data_buffers(&omb_graph_options);
    }
    omb_papi_free(&papi_eventset);
    MPI_CHECK(MPI_Finalize());

    return EXIT_SUCCESS;
}

/* vi: set sw=4 sts=4 tw=80: */
