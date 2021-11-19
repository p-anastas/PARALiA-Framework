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

typedef struct gemm_backend_in{
	char TransA,  TransB;
	int M, N, K, ldA, ldB, ldC;
	VALUE_TYPE alpha,beta;
	void **A, **B, **C;
	short dev_id;
}* gemm_backend_in_p;

void CoCoPeLiaSelectDevice(short dev_id);
void CoCoPeLiaDevGetMemInfo(long long* free_dev_mem, long long* max_dev_mem);

#endif
