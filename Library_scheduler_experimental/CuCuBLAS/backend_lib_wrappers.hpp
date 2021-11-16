///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief Some CUDA function calls with added error-checking
///
#ifndef LIBSC_BACKEND_H
#define LIBSC_BACKEND_H

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <pthread.h>

#define NUM_DEVICES 1

/* global variable declaration */
typedef struct globuf{
	short dev_id;
	void * gpu_mem_buf = NULL;
	long long gpu_mem_buf_sz = 0;
	long long dgemm_A_offset = -1, dgemm_B_offset = -1, dgemm_C_offset = -1;
}* BLAS3GPUBufPtr;

typedef struct gemm_backend_in{
	char TransA,  TransB;
	size_t M, N, K, ldA, ldB, ldC;
	double alpha, *A, *B, beta, *C;
	short dev_id;
}* gemm_backend_in_p;

#endif
