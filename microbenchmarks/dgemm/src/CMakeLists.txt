################################################################################
# NOTES ON COMPILATION
################################################################################

# Load the modules for each libary and the environment variables will be set
# correctly (load cmake, compiler, blas library)

# Enbable debug mode by passing -DCMAKE_BUILD_TYPE=Debug to CMake, default is
# Release

cmake_minimum_required(VERSION 3.11)

project(DGEMM
VERSION 1.0
DESCRIPTION "DGEMM tests the performance of BLAS libraries"
LANGUAGES C)

site_name( SITENAME )

if ( NOT DEFINED BLAS_NAME )
  message( SEND_ERROR "BLAS NAME MUST BE SPECIFIED: cblas, mkl, essl, cublas, libsci or raw")
endif()

string( TOUPPER ${BLAS_NAME} BLAS_NAME_UPPER )

if ( NOT DEFINED CMAKE_BUILD_TYPE )
  set(CMAKE_BUILD_TYPE "Release" )
endif()

set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DUSE_${BLAS_NAME_UPPER}")

set( CMAKE_C_FLAGS_DEBUG "-Wall -O0 -g" )
set( CMAKE_VERBOSE_MAKEFILE "TRUE" )
if (CMAKE_C_COMPILER_ID STREQUAL "GNU" )
    set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
    set( CMAKE_C_FLAGS_RELEASE "-ffast-math -mavx2 -ftree-vectorizer-verbose=3 -O3 -funroll-loops -fno-var-tracking-assignments")
elseif( CMAKE_C_COMPILER_ID MATCHES "Intel" )
    set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -openmp")
    set( CMAKE_C_FLAGS_RELEASE "-O3 -fp-speculation=fast -fp-model=precise -qno-opt-dynamic-align")
elseif( CMAKE_C_COMPILER_ID STREQUAL "Cray" )
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
message( STATUS "Machine name    : ${SITENAME}")
message( STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message( STATUS "Compiler        : ${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER}")
message( STATUS "BLAS            : ${BLAS_NAME}")
message( STATUS "-----------------------------------")
message( STATUS "Compiler Flags (All)    : ${CMAKE_C_FLAGS}")
message( STATUS "Compiler Flags (Debug)  : ${CMAKE_C_FLAGS_DEBUG}")
message( STATUS "Compiler Flags (Release): ${CMAKE_C_FLAGS_RELEASE}")
message("\n")

add_executable(dgemm mt-dgemm.c)

if ( DEFINED BLAS_ROOT )
  if ( ${BLAS_NAME} STREQUAL "cublas" )
    target_link_directories( dgemm PRIVATE "${BLAS_ROOT}/lib64")
  else()
    target_link_directories( dgemm PRIVATE "${BLAS_ROOT}/lib")
  endif()
  target_include_directories( dgemm PRIVATE "${BLAS_ROOT}/include")
endif()

if ( DEFINED BLAS_LIB_DIR )
  target_link_directories( dgemm PRIVATE ${BLAS_LIB_DIR} )
endif()

if ( DEFINED BLAS_INCLUDE_DIR )
  target_include_directories( dgemm PRIVATE ${BLAS_INCLUDE_DIR} )
endif()

if ( ${BLAS_NAME} STREQUAL "mkl" )
  set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -qmkl=parallel")
elseif ( ${BLAS_NAME} STREQUAL "cblas" )
  target_link_libraries( dgemm LINK_PUBLIC "openblas")
# elseif ( ${BLAS_NAME} STREQUAL "libsci" )
#   string( TOLOWER $ENV{PE_ENV} BLAS_COMPILER )
#   set( BLAS_COMPILER $ENV{PE_ENV} )
#   target_link_libraries( dgemm LINK_PUBLIC "sci_${BLAS_COMPILER}" )
# elseif ( ${BLAS_NAME} STREQUAL "cublas" OR ${BLAS_NAME} STREQUAL "cublasxt" )
elseif ( ${BLAS_NAME} MATCHES "cublas" )
  target_link_libraries( dgemm LINK_PUBLIC "-lcuda;-lcublas;-lcudart")
elseif ( ${BLAS_NAME} STREQUAL "essl" )
  target_link_libraries( dgemm LINK_PUBLIC "essl")
endif()


