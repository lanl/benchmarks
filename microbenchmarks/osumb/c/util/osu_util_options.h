/*
 * Copyright (C) 2023-2024 the Network-Based Computing Laboratory
 * (NBCL), The Ohio State University.
 *
 * Contact: Dr. D. K. Panda (panda@cse.ohio-state.edu)
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level OMB directory.
 */
#ifndef OMB_UTIL_OP_H
#define OMB_UTIL_OP_H                 1
#define OMBOP_GET_NONACCEL_NAME(A, B) OMBOP__##A##__##B
#define OMBOP_GET_ACCEL_NAME(A, B)    OMBOP__ACCEL__##A##__##B
#define OMBOP_OPTSTR_BLK(bench, subtype)                                       \
    if (accel_enabled) {                                                       \
        optstring = OMBOP_GET_ACCEL_NAME(bench, subtype);                      \
    } else {                                                                   \
        optstring = OMBOP_GET_NONACCEL_NAME(bench, subtype);                   \
    }
#define OMBOP_OPTSTR_CUDA_BLK(bench, subtype)                                  \
    if (accel_enabled) {                                                       \
        optstring = (CUDA_KERNEL_ENABLED) ?                                    \
                        OMBOP_GET_ACCEL_NAME(bench, subtype) "r:" :            \
                        OMBOP_GET_ACCEL_NAME(bench, subtype);                  \
    } else {                                                                   \
        optstring = OMBOP_GET_NONACCEL_NAME(bench, subtype);                   \
    }
#define OMBOP_LONG_OPTIONS_ALL                                                 \
    {                                                                          \
        {"help", no_argument, 0, 'h'},                                         \
        {"version", no_argument, 0, 'v'},                                      \
        {"full", no_argument, 0, 'f'},                                         \
        {"message-size", required_argument, 0, 'm'},                           \
        {"window-size", required_argument, 0, 'W'},                            \
        {"num-test-calls", required_argument, 0, 't'},                         \
        {"iterations", required_argument, 0, 'i'},                             \
        {"warmup", required_argument, 0, 'x'},                                 \
        {"array-size", required_argument, 0, 'a'},                             \
        {"sync-option", required_argument, 0, 's'},                            \
        {"win-options", required_argument, 0, 'w'},                            \
        {"mem-limit", required_argument, 0, 'M'},                              \
        {"accelerator", required_argument, 0, 'd'},                            \
        {"cuda-target", required_argument, 0, 'r'},                            \
        {"print-rate", required_argument, 0, 'R'},                             \
        {"num-pairs", required_argument, 0, 'p'},                              \
        {"vary-window", required_argument, 0, 'V'},                            \
        {"validation", no_argument, 0, 'c'},                                   \
        {"buffer-num", required_argument, 0, 'b'},                             \
        {"validation-warmup", required_argument, 0, 'u'},                      \
        {"graph", required_argument, 0, 'G'},                                  \
        {"papi", required_argument, 0, 'P'},                                   \
        {"ddt", required_argument, 0, 'D'},                                    \
        {"nhbr", required_argument, 0, 'N'},                                    \
        {"type", required_argument, 0, 'T'}                                    \
    }
/*OMBOP[__ACCEL]__<options.bench>__<options.subtype>*/
#define OMBOP__PT2PT__LAT                     "+:hvm:x:i:b:cu:G:D:P:T:"
#define OMBOP__ACCEL__PT2PT__LAT              "+:x:i:m:d:hvcu:G:D:T:"
#define OMBOP__PT2PT__BW                      "+:hvm:x:i:t:W:b:cu:G:D:P:T:"
#define OMBOP__ACCEL__PT2PT__BW               "+:x:i:t:m:d:W:hvb:cu:G:D:T:"
#define OMBOP__PT2PT__LAT_MT                  "+:hvm:x:i:t:cu:G:D:T:"
#define OMBOP__ACCEL__PT2PT__LAT_MT           OMBOP__ACCEL__PT2PT__LAT
#define OMBOP__PT2PT__LAT_MP                  "+:hvm:x:i:t:cu:G:D:P:T:"
#define OMBOP__ACCEL__PT2PT__LAT_MP           OMBOP__ACCEL__PT2PT__LAT
#define OMBOP__COLLECTIVE__ALLTOALL           "+:hvfm:i:x:M:a:cu:G:D:P:T:"
#define OMBOP__ACCEL__COLLECTIVE__ALLTOALL    "+:d:hvfm:i:x:M:a:cu:G:D:T:"
#define OMBOP__COLLECTIVE__GATHER             OMBOP__COLLECTIVE__ALLTOALL
#define OMBOP__ACCEL__COLLECTIVE__GATHER      OMBOP__ACCEL__COLLECTIVE__ALLTOALL
#define OMBOP__COLLECTIVE__SCATTER            OMBOP__COLLECTIVE__ALLTOALL
#define OMBOP__ACCEL__COLLECTIVE__SCATTER     OMBOP__ACCEL__COLLECTIVE__ALLTOALL
#define OMBOP__COLLECTIVE__BCAST              OMBOP__COLLECTIVE__ALLTOALL
#define OMBOP__ACCEL__COLLECTIVE__BCAST       OMBOP__ACCEL__COLLECTIVE__ALLTOALL
#define OMBOP__COLLECTIVE__NHBR_GATHER        "+:hvfm:i:x:M:a:cu:N:G:D:P:T:"
#define OMBOP__ACCEL__COLLECTIVE__NHBR_GATHER "+:hvfm:i:x:M:a:cu:N:G:D:T:"
#define OMBOP__COLLECTIVE__NHBR_ALLTOALL      OMBOP__COLLECTIVE__NHBR_GATHER
#define OMBOP__ACCEL__COLLECTIVE__NHBR_ALLTOALL                                \
    OMBOP__ACCEL__COLLECTIVE__NHBR_GATHER
#define OMBOP__COLLECTIVE__BARRIER               "+:hvfm:i:x:M:a:u:G:P:"
#define OMBOP__ACCEL__COLLECTIVE__BARRIER        "+:d:hvfm:i:x:M:a:u:G:"
#define OMBOP__COLLECTIVE__LAT                   "+:hvfm:i:x:M:a:cu:G:P:T:"
#define OMBOP__ACCEL__COLLECTIVE__LAT            "+:d:hvfm:i:x:M:a:cu:G:T:"
#define OMBOP__COLLECTIVE__REDUCE                OMBOP__COLLECTIVE__LAT
#define OMBOP__ACCEL__COLLECTIVE__REDUCE         OMBOP__ACCEL__COLLECTIVE__LAT
#define OMBOP__COLLECTIVE__REDUCE_SCATTER        OMBOP__COLLECTIVE__LAT
#define OMBOP__ACCEL__COLLECTIVE__REDUCE_SCATTER OMBOP__ACCEL__COLLECTIVE__LAT
#define OMBOP__COLLECTIVE__NBC                   "+:hvfm:i:x:M:t:a:G:P:"
#define OMBOP__ACCEL__COLLECTIVE__NBC            "+:d:hvfm:i:x:M:t:a:G:"
#define OMBOP__COLLECTIVE__NBC_GATHER            "+:hvfm:i:x:M:t:a:cu:G:D:P:T:"
#define OMBOP__ACCEL__COLLECTIVE__NBC_GATHER     "+:d:hvfm:i:x:M:t:a:cu:G:D:T:"
#define OMBOP__COLLECTIVE__NBC_ALLTOALL          OMBOP__COLLECTIVE__NBC_GATHER
#define OMBOP__ACCEL__COLLECTIVE__NBC_ALLTOALL                                 \
    OMBOP__ACCEL__COLLECTIVE__NBC_GATHER
#define OMBOP__COLLECTIVE__NBC_SCATTER OMBOP__COLLECTIVE__NBC_GATHER
#define OMBOP__ACCEL__COLLECTIVE__NBC_SCATTER                                  \
    OMBOP__ACCEL__COLLECTIVE__NBC_GATHER
#define OMBOP__COLLECTIVE__NBC_BCAST          OMBOP__COLLECTIVE__NBC_GATHER
#define OMBOP__ACCEL__COLLECTIVE__NBC_BCAST   OMBOP__ACCEL__COLLECTIVE__NBC_GATHER
#define OMBOP__COLLECTIVE__NBC_REDUCE         "+:hvfm:i:x:M:t:a:cu:G:P:T:";
#define OMBOP__ACCEL__COLLECTIVE__NBC_REDUCE  "+:d:hvfm:i:x:M:t:a:cu:G:T:"
#define OMBOP__COLLECTIVE__NBC_REDUCE_SCATTER OMBOP__COLLECTIVE__NBC_REDUCE
#define OMBOP__ACCEL__COLLECTIVE__NBC_REDUCE_SCATTER                           \
    OMBOP__ACCEL__COLLECTIVE__NBC_REDUCE
#define OMBOP__COLLECTIVE__NBC_NHBR_GATHER        "+:hvfm:i:x:M:t:a:cu:N:G:D:P:T:"
#define OMBOP__ACCEL__COLLECTIVE__NBC_NHBR_GATHER "+:hvfm:i:x:M:t:a:cu:N:G:D:T:"
#define OMBOP__COLLECTIVE__NBC_NHBR_ALLTOALL      OMBOP__COLLECTIVE__NBC_NHBR_GATHER
#define OMBOP__ACCEL__COLLECTIVE__NBC_NHBR_ALLTOALL                            \
    OMBOP__ACCEL__COLLECTIVE__NBC_NHBR_GATHER
#define OMBOP__ONE_SIDED__BW         "+:w:s:hvm:x:i:W:G:P:"
#define OMBOP__ACCEL__ONE_SIDED__BW  "+:w:s:hvm:d:x:i:W:G:"
#define OMBOP__ONE_SIDED__LAT        "+:w:s:hvm:x:i:G:P:"
#define OMBOP__ACCEL__ONE_SIDED__LAT "+:w:s:hvm:d:x:i:G:"
#define OMBOP__MBW_MR                "p:W:R:x:i:m:Vhvb:cu:G:D:P:T:"
#define OMBOP__ACCEL__MBW_MR         "p:W:R:x:i:m:d:Vhvb:cu:G:D:T:"
#define OMBOP__OSHM                  ":hvfm:i:M:";
#define OMBOP__UPC                   OMBOP__OSHM
#define OMBOP__UPCXX                 OMBOP__OSHM
#endif
