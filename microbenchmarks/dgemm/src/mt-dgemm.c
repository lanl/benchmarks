
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define BLAS_LIB "nolib"

#ifdef USE_MKL
#include "mkl.h"
#define BLAS_LIB "mkl"
#endif

#ifdef USE_CUBLAS
#include <cublas_v2.h>
#include <cuda_runtime.h>
#define BLAS_LIB "cublas"
#endif

#ifdef USE_CUBLASXT
#include <cublasXt.h>
#include <cuda_runtime.h>
#define BLAS_LIB "cublasXt"
#endif

#ifdef USE_LIBSCI
#include <cblas.h>
#define BLAS_LIB "libsci"
#endif

#ifdef USE_LIBSCI_ACC
#include <libsci_acc.h>
#define BLAS_LIB "libsci_acc"
#endif


#ifdef USE_CBLAS
#include "cblas.h"
#define BLAS_LIB "cblas"
#endif

#ifdef USE_ESSL
#include "essl.h"
#define BLAS_LIB "essl"
#endif

#define DGEMM_RESTRICT __restrict__

// ------------------------------------------------------- //
// Function: get_seconds
//
// Vendor may modify this call to provide higher resolution
// timing if required
// ------------------------------------------------------- //
double get_seconds() {
	struct timeval now;
	gettimeofday(&now, NULL);

	const double seconds = (double) now.tv_sec;
	const double usec    = (double) now.tv_usec;

	return seconds + (usec * 1.0e-6);
}

// ------------------------------------------------------- //
// Function: main
//
// Modify only in permitted regions (see comments in the
// function)
// ------------------------------------------------------- //
int main(int argc, char* argv[]) {

	// ------------------------------------------------------- //
	// DO NOT CHANGE CODE BELOW
	// ------------------------------------------------------- //

	size_t N = 256;
	int repeats = 8;
	size_t block_size = 0;

    double alpha = 1.0;
    double beta  = 1.0;

	if(argc > 1) {
		N = atoi(argv[1]);
		printf("Matrix size input by command line: %zu\n", N);

		if(argc > 2) {
			repeats = atoi(argv[2]);

			if(repeats < 4) {
				fprintf(stderr, "Error: repeats must be at least 4, setting is: %d\n", repeats);
				exit(-1);
			}

			printf("Repeat multiply %d times.\n", repeats);

            if(argc > 3) {
                alpha = (double) atof(argv[3]);
                if(argc > 4) {
                    beta = (double) atof(argv[4]);
                    if(argc > 5) block_size = atoi(argv[5]);
                }
            }
		} else {
			printf("Repeat multiply defaulted to %d\n", repeats);
		}
	} else {
		printf("Matrix size defaulted to %zu\n", N);
	}

	if(N < 128) {
		printf("Error: N (%zu) is less than 128, the matrix is too small.\n", N);
		exit(-1);
	}
    
    const size_t matrixsize = sizeof(double) * N * N;
	if (block_size == 0) block_size = N/2;

    printf("Alpha =    %.2f\n", alpha);
    printf("Beta  =    %.2f\n", beta);
    printf("BlockSize  =    %zu\n", block_size);
	printf("Allocating Matrices...\n");

	double* DGEMM_RESTRICT matrixA = (double*) malloc(matrixsize);
	double* DGEMM_RESTRICT matrixB = (double*) malloc(matrixsize);
	double* DGEMM_RESTRICT matrixC = (double*) malloc(matrixsize);

	printf("Allocation complete, populating with values...");

	size_t i, j, k, r;
    double start, end, time_taken, time_section;

    start = get_seconds();
    #pragma omp parallel for private(i,j,k)
	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
            k=i*N + j;
			matrixA[k] = 2.0;
			matrixB[k] = 0.5;
			matrixC[k] = 1.0;
		}
	}

#if defined(USE_CUBLAS)
    // Create Cublas Handle
    cublasHandle_t handle;
    cublasCreate(&handle);
	printf("-- CUDA!!\nAllocating and transferring values...");
    double *dMatrixA, *dMatrixB, *dMatrixC;
    cudaMalloc((void **)&dMatrixA, matrixsize);
    cudaMalloc((void **)&dMatrixB, matrixsize);
    cudaMalloc((void **)&dMatrixC, matrixsize);

    cudaMemcpy(dMatrixA, matrixA, matrixsize, cudaMemcpyHostToDevice);
    cudaMemcpy(dMatrixB, matrixB, matrixsize, cudaMemcpyHostToDevice);
    cudaMemcpy(dMatrixC, matrixC, matrixsize, cudaMemcpyHostToDevice);
#endif

#ifdef USE_CUBLASXT
// Create CublasXt Handle and select all available devices.
// You don't want to use explicit device memory here because it needs
// to be distributed across all devices and cudaMalloc only assigns
// to the current device.
    int *devices = NULL;
    cublasXtHandle_t handle;
    int device_count, blockdim;
    cudaGetDeviceCount(&device_count);
    devices = (int *)malloc(sizeof(int) * device_count);
    cublasXtCreate(&handle);
    for (int i=0; i<device_count; i++) devices[i] = i;
    cublasXtDeviceSelect(handle, device_count, devices);
    cublasXtSetPinningMemMode(handle, CUBLASXT_PINNING_ENABLED);
    cublasXtSetBlockDim(handle, block_size);
    cublasXtGetBlockDim(handle, &blockdim);
    printf("CUBLASXT has block dim: %d\n", blockdim);
#endif

    end = get_seconds();
    time_section = (end - start);
    printf(" %g seconds\n", time_section);

	printf("Performing multiplication...\n");
	printf("Using Blas Type: %s\n", BLAS_LIB);
	printf("Iteration #:\n");

	start = get_seconds();

	// ------------------------------------------------------- //
	// VENDOR NOTIFICATION: START MODIFIABLE REGION
	//
	// Vendor is able to change the lines below to call optimized
	// DGEMM or other matrix multiplication routines. Do *NOT*
	// change any lines above this statement.
	// ------------------------------------------------------- //

	double sum = 0;

	// Repeat multiple times
	for(r = 0; r < repeats; r++) {
#if defined(USE_MKL) || defined(USE_CBLAS) || defined(USE_LIBSCI)
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            N, N, N, alpha, matrixA, N, matrixB, N, beta, matrixC, N);
#elif defined(USE_CUBLAS)
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, dMatrixA, N, dMatrixB, N,
                     &beta, dMatrixC, N);
        cudaDeviceSynchronize();
#elif defined(USE_CUBLASXT)
        cublasXtDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, matrixA, N, matrixB, N,
                     &beta, matrixC, N);
        cudaDeviceSynchronize();
#elif defined(USE_ESSL) || defined(USE_LIBSCI_ACC)
        dgemm('N', 'N',
            N, N, N, alpha, matrixA, N, matrixB, N, beta, matrixC, N);
#else
        #pragma omp parallel for private(sum, j, k)
		for(i = 0; i < N; i++) {
			for(j = 0; j < N; j++) {
				sum = 0;

				for(k = 0; k < N; k++) {
					sum += matrixA[i*N + k] * matrixB[k*N + j];
				}

				matrixC[i*N + j] = (alpha * sum) + (beta * matrixC[i*N + j]);
			}
		}
#endif
		if ( r%10 == 0 ) {
			printf("%zu, ", r);
			fflush(stdout); 
		}
	}
	printf("\n");

#if defined(USE_CUBLAS)
    cudaMemcpy(matrixC, dMatrixC, matrixsize, cudaMemcpyDeviceToHost);
#endif

	// ------------------------------------------------------- //
	// VENDOR NOTIFICATION: END MODIFIABLE REGION
	// ------------------------------------------------------- //

	// ------------------------------------------------------- //
	// DO NOT CHANGE CODE BELOW
	// ------------------------------------------------------- //

	end = get_seconds();
    time_taken = (end - start);

#ifdef USE_CUBLAS
    cublasDestroy(handle);
    cudaFree(dMatrixA);
    cudaFree(dMatrixB);
    cudaFree(dMatrixC);
    cudaDeviceSynchronize();
#endif

#ifdef USE_CUBLASXT
    cublasXtDestroy(handle);
    free(devices);
#endif

	printf("Calculating matrix check...");

	double final_sum = 0;
	double count     = 0;
    start = get_seconds();

	#pragma omp parallel for reduction(+:final_sum, count) private(i,j)
	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
			final_sum += matrixC[i*N + j];
			count += 1.0;
		}
	}

    end = get_seconds();
    time_section = (end - start);
    printf(" %g seconds\n", time_section);

	double matrix_memory = (3 * matrixsize);

	printf("\n");
	printf("===============================================================\n");

	printf("Final Sum is:         %f\n", (final_sum / (count * repeats)));
	printf("Memory for Matrices:  %.0f MB\n",
		(matrix_memory / (1024 * 1024)));

    double N_dbl = (double) N;

	printf("Multiply time:        %.6g seconds\n", time_taken);

	// O(N**3) elements each with one add and three multiplies
    	// (alpha, beta and A_i*B_i).
	double flops_computed = (N_dbl * N_dbl * 2.0 * (double)repeats)*(N_dbl+1.0);
    double total_time = ( flops_computed / time_taken) / 1.0e9;

	printf("FLOPs computed:       %.0g\n", flops_computed);
	printf("GFLOP/s rate:         %.8g GF/s\n", (total_time));

	printf("===============================================================\n");
	printf("\n");

	free(matrixA);
	free(matrixB);
	free(matrixC);

	return 0;
}
