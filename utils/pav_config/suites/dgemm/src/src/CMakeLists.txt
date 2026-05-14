################################################################################
# NOTES ON COMPILATION
################################################################################

# Load the modules for each libary and the environment variables will be set
# correctly (load cmake, compiler, blas library)

# Enbable debug mode by passing -DCMAKE_BUILD_TYPE=Debug to CMake, default is
# Release
# If AVAILABLE: PLEASE ADD PACKAGE CMAKE DIR TO CMAKE_PREFIX_PATH.
#   Required for NVPL.

cmake_minimum_required(VERSION 3.11)

project(DGEMM
VERSION 1.0
DESCRIPTION "DGEMM tests the performance of BLAS libraries"
LANGUAGES C)

site_name( SITENAME )

if ( NOT DEFINED BLAS_NAME )
  message( SEND_ERROR "BLAS NAME MUST BE SPECIFIED: cblas, mkl, essl, cublas, libsci, libsci_acc, nvpl, or raw")
endif()

string( TOUPPER ${BLAS_NAME} BLAS_NAME_UPPER )

if ( NOT CMAKE_BUILD_TYPE )
  set(CMAKE_BUILD_TYPE "Release" )
endif()

set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DUSE_${BLAS_NAME_UPPER}")

set( CMAKE_C_FLAGS_DEBUG "-Wall -O0 -g" )
set( CMAKE_VERBOSE_MAKEFILE "TRUE" )
if (CMAKE_C_COMPILER_ID STREQUAL "GNU" )
    set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
    set( CMAKE_C_FLAGS_RELEASE "-ffast-math -ftree-vectorizer-verbose=3 -O3 -funroll-loops -fno-var-tracking-assignments")
elseif( CMAKE_C_COMPILER_ID MATCHES "Intel" )
    set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -openmp")
    set( CMAKE_C_FLAGS_RELEASE "-O3 -fp-speculation=fast -fp-model=precise -qno-opt-dynamic-align")
elseif( CMAKE_C_COMPILER_ID STREQUAL "Clang" )
    set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
    set( CMAKE_C_FLAGS_RELEASE "-O3")
endif()

if (CMAKE_BUILD_TYPE  STREQUAL "Release" )
  set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_RELEASE}")
elseif (CMAKE_BUILD_TYPE  STREQUAL "Debug" )
  set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_DEBUG}")
endif()

# Summary of user-selectable build options
message( "\nBuild Summary:\n")
message( STATUS "Machine name     : ${SITENAME}")
message( STATUS "CMAKE_BUILD_TYPE : ${CMAKE_BUILD_TYPE}")
message( STATUS "Compiler         : ${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER}")
message( STATUS "BLAS             : ${BLAS_NAME}")
message( STATUS "BLAS_ROOT        : ${BLAS_ROOT}")
message( STATUS "-----------------------------------")
message( STATUS "Compiler Flags (All)    : ${CMAKE_C_FLAGS}")
message( STATUS "Compiler Flags (Debug)  : ${CMAKE_C_FLAGS_DEBUG}")
message( STATUS "Compiler Flags (Release): ${CMAKE_C_FLAGS_RELEASE}")
message("\n")

add_executable(dgemm mt-dgemm.c)
set( LIB_EXT "lib" )

if ( DEFINED BLAS_ROOT )
  if ( ${BLAS_NAME} MATCHES "cublas" )
    set( LIB_EXT "lib64" )
    target_link_libraries( dgemm LINK_PUBLIC "-lcublas;-lcudart")
  elseif ( ${BLAS_NAME} STREQUAL "nvpl" )
  # FIND_PKG METHOD
    # find_package(nvpl REQUIRED COMPONENTS blas)
    # target_link_libraries( dgemm PRIVATE "nvpl::blas_ilp64_omp") #libnvpl_blas_ilp64_gomp.so
    # set( BLAS_LIB_DIR "${nvpl_blas_LIBRARY_DIR}" )
    # set( BLAS_INCLUDE_DIR "${nvpl_blas_INCLUDE_DIR}")
  # NVPL_HOME METHOD
    target_link_libraries( dgemm LINK_PUBLIC "nvpl_blas_ilp64_gomp") #libnvpl_blas_ilp64_gomp.so
  elseif ( ${BLAS_NAME} STREQUAL "mkl" )
    if( CMAKE_C_COMPILER_ID MATCHES "Intel" )
      set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -qmkl=parallel")
    else()
      target_link_libraries( dgemm LINK_PUBLIC "-lmkl_core;-lmkl_gnu_thread;-lmkl_gf_ilp64" )
    endif()
  elseif ( ${BLAS_NAME} STREQUAL "cblas" )
    target_link_libraries( dgemm LINK_PUBLIC "openblas")
  elseif ( ${BLAS_NAME} STREQUAL "essl" )
    target_link_libraries( dgemm LINK_PUBLIC "essl")
  endif()
endif()

message( STATUS "BLAS PATH: ${BLAS_ROOT}/${LIB_EXT}")

if ( NOT DEFINED BLAS_LIB_DIR )
  set( BLAS_LIB_DIR "${BLAS_ROOT}/${LIB_EXT}" )
endif()
if ( NOT DEFINED BLAS_INCLUDE_DIR )
  set( BLAS_INCLUDE_DIR "${BLAS_ROOT}/include" )
endif()

target_link_directories( dgemm PRIVATE ${BLAS_LIB_DIR} )
target_include_directories( dgemm PRIVATE ${BLAS_INCLUDE_DIR} )

## Original library finding code:
# if ( ${BLAS_NAME} STREQUAL "mkl" )
#   set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -qmkl=parallel")
# elseif ( ${BLAS_NAME} STREQUAL "cblas" )
#   target_link_libraries( dgemm LINK_PUBLIC "openblas")
# elseif ( ${BLAS_NAME} STREQUAL "essl" )
#   target_link_libraries( dgemm LINK_PUBLIC "essl")
# elseif ( ${BLAS_NAME} STREQUAL "libsci" )
#   string( TOLOWER $ENV{PE_ENV} BLAS_COMPILER )
#   set( BLAS_COMPILER $ENV{PE_ENV} )
#   target_link_libraries( dgemm LINK_PUBLIC "sci_${BLAS_COMPILER}" )
# elseif ( ${BLAS_NAME} STREQUAL "cublas" OR ${BLAS_NAME} STREQUAL "cublasxt" )
# elseif ( ${BLAS_NAME} MATCHES "cublas" )
#   target_link_libraries( dgemm LINK_PUBLIC "-lcuda;-lcublas;-lcudart")
# endif()


