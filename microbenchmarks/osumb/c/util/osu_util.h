/*
 * Copyright (C) 2002-2023 the Network-Based Computing Laboratory
 * (NBCL), The Ohio State University.
 *
 * Contact: Dr. D. K. Panda (panda@cse.ohio-state.edu)
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level OMB directory.
 */
#ifndef OSU_UTIL_H
#define OSU_UTIL_H

#ifndef OSU_COLL_H
#define OSU_COLL_H 1
#endif

#ifndef OSU_PT2PT_H
#define OSU_PT2PT_H 1
#endif

#include <assert.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <getopt.h>
#include <pthread.h>
#include <inttypes.h>
#include <sys/time.h>
#include <limits.h>
#include <sys/types.h>
#include <ctype.h>

#ifdef _ENABLE_PAPI_
#include <papi.h>
#endif

#ifdef _ENABLE_CUDA_
#include "cuda.h"
#include "cuda_runtime.h"
#endif

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

#ifdef _ENABLE_OPENACC_
#define OPENACC_ENABLED 1
#include <openacc.h>
#else
#define OPENACC_ENABLED 0
#endif

#ifdef _ENABLE_CUDA_
#define CUDA_ENABLED 1
#else
#define CUDA_ENABLED 0
#endif

#ifdef _ENABLE_CUDA_KERNEL_
#define CUDA_KERNEL_ENABLED 1
#else
#define CUDA_KERNEL_ENABLED 0
#endif

#ifdef _ENABLE_NCCL_
#define NCCL_ENABLED 1
#else
#define NCCL_ENABLED 0
#endif

#ifdef _ENABLE_ROCM_
#define ROCM_ENABLED 1
#include "hip/hip_runtime.h"
#else
#define ROCM_ENABLED 0
#endif

#ifndef BENCHMARK
#define BENCHMARK "MPI%s BENCHMARK NAME UNSET"
#endif

#ifdef PACKAGE_VERSION
#define HEADER "# " BENCHMARK " v" PACKAGE_VERSION "\n"
#else
#define HEADER "# " BENCHMARK "\n"
#endif

#ifndef FIELD_WIDTH
#define FIELD_WIDTH 20
#endif

#ifndef FLOAT_PRECISION
#define FLOAT_PRECISION 2
#endif

#define OMB_ERROR_EXIT(msg)                                                    \
    do {                                                                       \
        fprintf(stderr, "[%s:%d] Error: '%s'\n", __FILE__, __LINE__, msg);     \
        fflush(stderr);                                                        \
        exit(EXIT_FAILURE);                                                    \
    } while (0)

#define OMB_CHECK_NULL_AND_EXIT(var, msg)                                      \
    do {                                                                       \
        if (NULL == var) {                                                     \
            fprintf(stderr, "[%s:%d] Failed with message '%s'\n", __FILE__,    \
                    __LINE__, msg);                                            \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define CHECK(stmt)                                                            \
    do {                                                                       \
        int errno = (stmt);                                                    \
        if (0 != errno) {                                                      \
            fprintf(stderr, "[%s:%d] function call failed with %d \n",         \
                    __FILE__, __LINE__, errno);                                \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
        assert(0 == errno);                                                    \
    } while (0)

#if defined(_ENABLE_CUDA_)
#define CUDA_CHECK(stmt)                                                       \
    do {                                                                       \
        int errno = (stmt);                                                    \
        if (0 != errno) {                                                      \
            fprintf(stderr, "[%s:%d] CUDA call '%s' failed with %d: %s \n",    \
                    __FILE__, __LINE__, #stmt, errno,                          \
                    cudaGetErrorString(errno));                                \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
        assert(cudaSuccess == errno);                                          \
    } while (0)
#endif

#if defined(_ENABLE_ROCM_)
#define ROCM_CHECK(stmt)                                                       \
    do {                                                                       \
        hipError_t errno = (stmt);                                             \
        if (0 != errno) {                                                      \
            fprintf(stderr, "[%s:%d] ROCM call '%s' failed with %d: %s \n",    \
                    __FILE__, __LINE__, #stmt, errno,                          \
                    hipGetErrorString(errno));                                 \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
        assert(hipSuccess == errno);                                           \
    } while (0)
#endif

#define TIME() getMicrosecondTimeStamp()
double getMicrosecondTimeStamp();

void print_header_coll(int rank, int full) __attribute__((unused));
void print_header_nbc(int rank, int full);
void print_data(int rank, int full, int size, double avg_time, double min_time,
                double max_time, int iterations) __attribute__((unused));
void print_data_nbc(int rank, int full, int size, double ovrl, double cpu,
                    double comm, double wait, double init, int iterations);

void allocate_host_arrays();

double calculate_and_print_stats(int rank, int size, int numprocs, double timer,
                                 double latency, double test_time,
                                 double cpu_time, double wait_time,
                                 double init_time, int errors);

enum mpi_req { MAX_REQ_NUM = 1000 };

#define OMB_LONG_OPTIONS_ARRAY_SIZE     25
#define BW_LOOP_SMALL                   100
#define BW_SKIP_SMALL                   10
#define BW_LOOP_LARGE                   20
#define BW_SKIP_LARGE                   2
#define LAT_LOOP_SMALL                  10000
#define LAT_SKIP_SMALL                  100
#define LAT_LOOP_LARGE                  1000
#define LAT_SKIP_LARGE                  10
#define COLL_LOOP_SMALL                 1000
#define COLL_SKIP_SMALL                 100
#define COLL_LOOP_LARGE                 100
#define COLL_SKIP_LARGE                 10
#define OSHM_LOOP_SMALL                 1000
#define OSHM_LOOP_LARGE                 100
#define OSHM_SKIP_SMALL                 200
#define OSHM_SKIP_LARGE                 10
#define OSHM_LOOP_SMALL_MR              500
#define OSHM_LOOP_LARGE_MR              50
#define OSHM_LOOP_ATOMIC                500
#define VALIDATION_SKIP_DEFAULT         5
#define VALIDATION_SKIP_MAX             10
#define OMB_DDT_STRIDE_DEFAULT          8
#define OMB_DDT_BLOCK_LENGTH_DEFAULT 4
#define OMB_FILE_PATH_MAX_LENGTH        1024
#define OMB_NHBRHD_FILE_PATH_MAX_LENGTH OMB_FILE_PATH_MAX_LENGTH
#define OMB_DDT_FILE_PATH_MAX_LENGTH OMB_FILE_PATH_MAX_LENGTH
#define MAX_MESSAGE_SIZE                (1 << 22)
#define MAX_MSG_SIZE_PT2PT              (1 << 20)
#define MAX_MSG_SIZE_COLL               (1 << 20)
#define MIN_MESSAGE_SIZE                1
#define LARGE_MESSAGE_SIZE              8192
#define MAX_ALIGNMENT                   65536
#define MAX_MEM_LIMIT                   (512 * 1024 * 1024)
#define MAX_MEM_LOWER_LIMIT             (1 * 1024 * 1024)
#define WINDOW_SIZE_LARGE               64
#define MYBUFSIZE                       MAX_MESSAGE_SIZE
#define ONESBUFSIZE                     ((MAX_MESSAGE_SIZE * WINDOW_SIZE_LARGE) + MAX_ALIGNMENT)
#define MESSAGE_ALIGNMENT               64
#define MESSAGE_ALIGNMENT_MR            (1 << 12)
#define OMB_NUM_DATATYPES               3
#define OMB_DATATYPE_STR_MAX_LEN        128

enum po_ret_type {
    PO_CUDA_NOT_AVAIL,
    PO_OPENACC_NOT_AVAIL,
    PO_BAD_USAGE,
    PO_HELP_MESSAGE,
    PO_VERSION_MESSAGE,
    PO_OKAY,
};

enum accel_type { NONE, CUDA, OPENACC, MANAGED, ROCM };

enum target_type { CPU, GPU, BOTH };

enum benchmark_type { COLLECTIVE, PT2PT, ONE_SIDED, MBW_MR, OSHM, UPC, UPCXX };

enum test_subtype {
    BW,
    LAT,
    LAT_MT,
    LAT_MP,
    NBC,
    BARRIER,
    ALLTOALL,
    GATHER,
    REDUCE_SCATTER,
    NHBR_GATHER,
    NHBR_ALLTOALL,
    NBC_REDUCE_SCATTER,
    NBC_ALLTOALL,
    NBC_GATHER,
    NBC_NHBR_GATHER,
    NBC_NHBR_ALLTOALL,
    NBC_REDUCE,
    NBC_SCATTER,
    NBC_BCAST,
    SCATTER,
    REDUCE,
    BCAST
};

enum test_synctype { ALL_SYNC, ACTIVE_SYNC };

enum WINDOW {
    WIN_CREATE = 0,
#if MPI_VERSION >= 3
    WIN_ALLOCATE,
    WIN_DYNAMIC
#endif
};

/* Synchronization */
enum SYNC {
    LOCK = 0,
    PSCW,
    FENCE,
#if MPI_VERSION >= 3
    FLUSH,
    FLUSH_LOCAL,
    LOCK_ALL,
#endif
};

enum buffer_num { SINGLE, MULTIPLE };

/*ddt types*/
enum omb_ddt_types_t { OMB_DDT_CONTIGUOUS, OMB_DDT_VECTOR, OMB_DDT_INDEXED };

/*ddt type parameters*/
typedef struct omb_ddt_type_parameters {
    size_t block_length;
    size_t stride;
    char filepath[OMB_DDT_FILE_PATH_MAX_LENGTH];
} omb_ddt_type_parameters_t;

/*Neighbourhood topology types*/
enum omb_nhbrhd_types_t { OMB_NHBRHD_TYPE_CART, OMB_NHBRHD_TYPE_GRAPH };

/*Neighbourhood type parameters*/
typedef struct omb_nhbrhd_type_parameters {
    int dim;
    int rad;
    char filepath[OMB_NHBRHD_FILE_PATH_MAX_LENGTH];
} omb_nhbrhd_type_parameters_t;

/*variables*/
extern char const *win_info[20];
extern char const *sync_info[20];

enum omb_dtypes_t { OMB_DTYPE_NULL, OMB_CHAR, OMB_INT, OMB_FLOAT };

struct options_t {
    enum accel_type accel;
    enum target_type target;
    int show_size;
    int show_full;
    int show_validation;
    size_t min_message_size;
    size_t max_message_size;
    size_t iterations;
    size_t iterations_large;
    size_t max_mem_limit;
    size_t skip;
    size_t skip_large;
    size_t warmup_validation;
    size_t window_size_large;
    int num_probes;
    int device_array_size;

    enum benchmark_type bench;
    enum test_subtype subtype;
    enum test_synctype synctype;

    char src;
    char dst;

    char MMsrc;
    char MMdst;

    int num_threads;
    int sender_thread;
    int num_processes;
    int sender_processes;
    char managedSend;
    char managedRecv;
    enum WINDOW win;
    enum SYNC sync;

    int window_size;
    int window_varied;
    int print_rate;
    int pairs;
    int validate;
    enum buffer_num buf_num;
    int graph;
    int graph_output_term;
    int graph_output_png;
    int graph_output_pdf;
    int omb_enable_ddt;
    enum omb_ddt_types_t ddt_type;
    omb_ddt_type_parameters_t ddt_type_parameters;
    enum omb_nhbrhd_types_t nhbrhd_type;
    omb_nhbrhd_type_parameters_t nhbrhd_type_parameters;
    int omb_dtype_itr;
    enum omb_dtypes_t omb_dtype_list[OMB_NUM_DATATYPES];
    int papi_enabled;
};

struct bad_usage_t {
    char const *message;
    char const *optarg;
    int opt;
};

extern char const *benchmark_header;
extern char const *benchmark_name;
extern int accel_enabled;
extern struct options_t options;
extern struct bad_usage_t bad_usage;

/*
 * Option Processing
 */
void omb_process_long_options(struct option *long_options,
                              const char *optstring);
extern int process_one_sided_options(int opt, char *arg);
int process_options(int argc, char *argv[]);
int omb_ddt_process_options(char *optarg, struct bad_usage_t *bad_usage);
int omb_nhbrhd_process_options(char *optarg, struct bad_usage_t *bad_usage);
int setAccel(char);

/*
 * Set Benchmark Properties
 */
void set_header(const char *header);
void set_benchmark_name(const char *name);
void enable_accel_support(void);

#define DEF_NUM_THREADS 2
#define MIN_NUM_THREADS 1
#define MAX_NUM_THREADS 128

#define DEF_NUM_PROCESSES   2
#define MIN_NUM_PROCESSES   1
#define MAX_NUM_PROCESSES   128
#define CHILD_SLEEP_SECONDS 2

#define WINDOW_SIZES                                                           \
    {                                                                          \
        1, 2, 4, 8, 16, 32, 64, 128                                            \
    }
#define WINDOW_SIZES_COUNT (8)

void wtime(double *t);

#endif
