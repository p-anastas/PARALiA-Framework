///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief BLAS lvl 3 wrappers for benchmarks.
///
#include <cassert>
#include <cublasXt.h>
#include <cblas.h>
#include <cuda_runtime.h>

#include "CoCoPeLia.hpp"
#include "unihelpers.hpp"
#include "backend_wrappers.hpp"

void cblas_dgemm_wrap_for_cublasXt(char* gpu_op_A, char* gpu_op_B, int* M, int* N, int* K, double* alpha, double* A, int* ldA, double* B, int* ldB, double* beta, double* C, int* ldC){
  CBLAS_TRANSPOSE cpu_op_A = CblasNoTrans, cpu_op_B = CblasNoTrans;    // CblasNoTrans, CblasTrans

    //fprintf(stderr, "%d %d %d %lf %d %d %lf %d\n",*M, *N, *K, *alpha, *ldA, *ldB, *beta, *ldC);

    if(*gpu_op_A == 'N') cpu_op_A = CblasNoTrans;
    else if(*gpu_op_A == 'T') cpu_op_A = CblasTrans;
    else error("cblas_dgemm_wrap -> Invalid CUBLAS_OP for A");
    if(*gpu_op_B == 'N') cpu_op_B = CblasNoTrans;
    else if(*gpu_op_B == 'T') cpu_op_B = CblasTrans;
    else error("cblas_dgemm_wrap -> Invalid CUBLAS_OP for B");

    cblas_dgemm(CblasColMajor, cpu_op_A, cpu_op_B, *M, *N, *K, *alpha, A, *ldA, B, *ldB, *beta, C, *ldC);
}

double cuBLASXtDgemmWrap(char TransA, char TransB, long int M, long int N, long int K, double alpha, double* A, long int ldA, double* B, long int ldB, double beta, double* C, long int ldC, long int T, double cpu_ratio, short dev_num, int dev_ids[] ){
	short lvl = 1;
	double total_t = csecond();
#ifdef DEBUG
	lprintf(lvl-1, "|-----> cuBLASXtDgemmWrap(%c,%c,%zu,%zu,%zu,%lf,A(%d),%zu,B(%d),%zu,%lf,C(%d),%zu)\n",
		TransA, TransB, M, N, K, alpha, CoCoGetPtrLoc(A), ldA,
		CoCoGetPtrLoc(B), ldB, beta, CoCoGetPtrLoc(C), ldC);
#endif

#ifdef TEST
	lprintf(lvl-1, "|-----> cuBLASXtDgemmWrap\n");
	double cpu_timer = csecond();
#endif

	cublasOperation_t gpu_op_A = OpCharToCublas(TransA), gpu_op_B = OpCharToCublas(TransB);
	cublasStatus_t stat;
	cublasXtHandle_t handle0;

	/// Required allocations for device
	assert(CUBLAS_STATUS_SUCCESS == cublasXtCreate(&handle0));
	assert(CUBLAS_STATUS_SUCCESS == cublasXtDeviceSelect(handle0, dev_num, dev_ids));
	assert(CUBLAS_STATUS_SUCCESS == cublasXtSetBlockDim(handle0, T));
	assert(CUBLAS_STATUS_SUCCESS == cublasXtSetCpuRoutine(handle0, CUBLASXT_GEMM, CUBLASXT_DOUBLE, (void*) &cblas_dgemm_wrap_for_cublasXt));
	assert(CUBLAS_STATUS_SUCCESS == cublasXtSetCpuRatio(handle0, CUBLASXT_GEMM, CUBLASXT_DOUBLE, cpu_ratio));
	assert(CUBLAS_STATUS_SUCCESS == cublasXtSetPinningMemMode(handle0, CUBLASXT_PINNING_ENABLED));
#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "cuBLASXt initialization/pinning -> t_init = %lf ms\n", cpu_timer*1000);
    	cpu_timer = csecond();
#endif
	assert(CUBLAS_STATUS_SUCCESS == cublasXtDgemm(handle0, gpu_op_A, gpu_op_B, M, N, K, &alpha, A, ldA, B, ldB, &beta, C, ldC));
	CoCoSyncCheckErr();
#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "cuBLASXt execution time -> t_kernel = %lf ms\n", cpu_timer*1000);
#endif

	/// Free local buffers
	cublasXtDestroy(handle0);
	CoCoSyncCheckErr();
	total_t = csecond() - total_t;
	return total_t;

}

void cblas_sgemm_wrap_for_cublasXt(char* gpu_op_A, char* gpu_op_B, int* M, int* N, int* K, float* alpha, float* A, int* ldA, float* B, int* ldB, float* beta, float* C, int* ldC){
  CBLAS_TRANSPOSE cpu_op_A = CblasNoTrans, cpu_op_B = CblasNoTrans;    // CblasNoTrans, CblasTrans

    //fprintf(stderr, "%d %d %d %lf %d %d %lf %d\n",*M, *N, *K, *alpha, *ldA, *ldB, *beta, *ldC);

    if(*gpu_op_A == 'N') cpu_op_A = CblasNoTrans;
    else if(*gpu_op_A == 'T') cpu_op_A = CblasTrans;
    else error("cblas_dgemm_wrap -> Invalid CUBLAS_OP for A");
    if(*gpu_op_B == 'N') cpu_op_B = CblasNoTrans;
    else if(*gpu_op_B == 'T') cpu_op_B = CblasTrans;
    else error("cblas_dgemm_wrap -> Invalid CUBLAS_OP for B");

cblas_sgemm(CblasColMajor, cpu_op_A, cpu_op_B, *M, *N, *K, *alpha, A, *ldA, B, *ldB, *beta, C, *ldC);
}

double cuBLASXtSgemmWrap(char TransA, char TransB, long int M, long int N, long int K, float alpha, float* A, long int ldA, float* B, long int ldB, float beta, float* C, long int ldC, long int T, double cpu_ratio, short dev_id){
	short lvl = 1;
	double total_t = csecond();
#ifdef DEBUG
	lprintf(lvl-1, "|-----> cuBLASXtSgemmWrap(%c,%c,%zu,%zu,%zu,%lf,A(%d),%zu,B(%d),%zu,%lf,C(%d),%zu)\n",
		TransA, TransB, M, N, K, alpha, CoCoGetPtrLoc(A), ldA,
		CoCoGetPtrLoc(B), ldB, beta, CoCoGetPtrLoc(C), ldC);
#endif

#ifdef TEST
	lprintf(lvl-1, "|-----> cuBLASXtSgemmWrap\n");
	double cpu_timer = csecond();
#endif

	cublasOperation_t gpu_op_A = OpCharToCublas(TransA), gpu_op_B = OpCharToCublas(TransB);
	cublasStatus_t stat;
	cublasXtHandle_t handle0;
	int device_ids[1] = {dev_id};

	// TODO: For now use only one device;
	int cur_id; cudaGetDevice(&cur_id);
	if ( cur_id != dev_id) printf("cuBLASXtSgemmWrap: Device change initiated(%d->%d)\n",cur_id, dev_id);
	cudaSetDevice(dev_id);

	/// Required allocations for device
	assert(CUBLAS_STATUS_SUCCESS == cublasXtCreate(&handle0));
	assert(CUBLAS_STATUS_SUCCESS == cublasXtDeviceSelect(handle0, 1, device_ids));
	assert(CUBLAS_STATUS_SUCCESS == cublasXtSetBlockDim(handle0, T));
	assert(CUBLAS_STATUS_SUCCESS == cublasXtSetCpuRoutine(handle0, CUBLASXT_GEMM, CUBLASXT_FLOAT, (void*) &cblas_sgemm_wrap_for_cublasXt));
	assert(CUBLAS_STATUS_SUCCESS == cublasXtSetCpuRatio(handle0, CUBLASXT_GEMM, CUBLASXT_DOUBLE, cpu_ratio));
	assert(CUBLAS_STATUS_SUCCESS == cublasXtSetPinningMemMode(handle0, CUBLASXT_PINNING_ENABLED));
#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "cuBLASXt initialization/pinning -> t_init = %lf ms\n", cpu_timer*1000);
    	cpu_timer = csecond();
#endif

	assert(CUBLAS_STATUS_SUCCESS == cublasXtSgemm(handle0, gpu_op_A, gpu_op_B, M, N, K, &alpha, A, ldA, B, ldB, &beta, C, ldC));
	CoCoSyncCheckErr();
#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "cuBLASXt execution time -> t_kernel = %lf ms\n", cpu_timer*1000);
#endif

	/// Free local buffers
	cublasXtDestroy(handle0);
	CoCoSyncCheckErr();
	total_t = csecond() - total_t;
	return total_t;

}
