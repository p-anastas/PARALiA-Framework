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

//#define NUM_DEVICES 1

/* global variable declaration */
typedef struct globuf{
	short dev_id;
	void * gpu_mem_buf = NULL;
	long long gpu_mem_buf_sz = 0;
	long long gpu_mem_offset = 0;
}* DevBufPtr;

typedef struct gemm_backend_in{
	char TransA,  TransB;
	size_t M, N, K, ldA, ldB, ldC;
	double alpha, *A, *B, beta, *C;
	short dev_id;
}* gemm_backend_in_p;

void CoCoPeLiaSelectDevice(short dev_id);
void CoCoPeLiaDevGetMemInfo(long long* free_dev_mem, long long* max_dev_mem);
DevBufPtr CoCoPeLiaBufferInit(short dev_id);
void CoCoPeLiaRequestBuffer(short dev_id, long long size);
void* CoCoPeLiaAsignBuffer(short dev_id, long long size);
#endif
