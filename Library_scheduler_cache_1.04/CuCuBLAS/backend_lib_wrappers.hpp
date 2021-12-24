///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief Some CUDA function calls with added error-checking
///
#ifndef LIBSC_BACKEND_H
#define LIBSC_BACKEND_H


#include "unihelpers.hpp"

//#define NUM_DEVICES 1

typedef struct gemm_backend_in{
	char TransA,  TransB;
	int M, N, K, ldA, ldB, ldC;
	VALUE_TYPE alpha,beta;
	void **A, **B, **C;
	short dev_id;
}* gemm_backend_in_p;

#endif

void backend_init(short dev_id, CQueue_p h2d_q, CQueue_p d2h_q, CQueue_p exec_q);
void backend_run_operation(short dev_id, void* backend_data, const char* opname);
