///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The DGEMM CoCopeLia implementation.
///

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <pthread.h>
#include <cblas.h>

#include "backend_lib_wrappers.hpp"
#include "backend_wrappers.hpp"

cublasHandle_t handle[128] = {NULL};

void backend_init(short dev_id, CQueue_p h2d_q, CQueue_p d2h_q, CQueue_p exec_q){
  int dev_idc = -1;
  cudaError_t err = cudaGetDevice(&dev_idc);
  massert(dev_idc == dev_id,
    "backend_init: called on different device - actual(%d) vs called(%d)\n", dev_idc, dev_id);
  massert(CUBLAS_STATUS_SUCCESS == cublasCreate(&(handle[dev_id])), "cublasCreate failed\n");
  massert(CUBLAS_STATUS_SUCCESS == cublasSetStream(handle[dev_id],
    *(cudaStream_t*) (exec_q->cqueue_backend_ptr)), "cublasSetStream failed\n");
  return;
}

int CoCoPeLiaGetDevice(){
  int dev_id = -1;
  cudaError_t err = cudaGetDevice(&dev_id);
  massert(cudaSuccess == err,
    "CoCoPeLiaGetDevice: cudaGetDevice failed - %s\n", cudaGetErrorString(err));
  return dev_id;
}

void CoCoPeLiaSelectDevice(short dev_id){
  cudaSetDevice(dev_id);
  cudaError_t err = cudaSetDevice(dev_id);
  massert(cudaSuccess == err,
    "CoCoPeLiaSelectDevice: cudaSetDevice failed - %s\n", cudaGetErrorString(err));
}
void CoCoPeLiaDevGetMemInfo(long long* free_dev_mem, long long* max_dev_mem){
  size_t free_dev_mem_tmp, max_dev_mem_tmp;
    cudaError_t err = cudaMemGetInfo(&free_dev_mem_tmp, &max_dev_mem_tmp);
  	massert(cudaSuccess == err,
      "CoCoPeLiaDevGetMemInfo: cudaMemGetInfo failed - %s\n", cudaGetErrorString(err));
    *free_dev_mem = (long long) free_dev_mem_tmp;
    *max_dev_mem = (long long) max_dev_mem_tmp;
}

void backend_run_operation(short dev_id, void* backend_data, const char* opname){
  short lvl = 5;
  if (!strcmp(opname, "gemm")){
    gemm_backend_in_p ptr_ker_translate = (gemm_backend_in_p) backend_data;
  #ifdef DDEBUG
  	lprintf(lvl, "backend_run_operation: cublasDgemm(handle[%d], TransA = %c, TransB = %c, M = %d, N = %d, K = %d, alpha = %lf, A = %p, lda = %d, \n\
  	B = %p, ldb = %d, beta = %lf, C = %p, ldC = %d)\n", dev_id, ptr_ker_translate->TransA, ptr_ker_translate->TransB,
  		ptr_ker_translate->M, ptr_ker_translate->N, ptr_ker_translate->K, ptr_ker_translate->alpha, (VALUE_TYPE*) *ptr_ker_translate->A, ptr_ker_translate->ldA,
  		(VALUE_TYPE*) *ptr_ker_translate->B, ptr_ker_translate->ldB, ptr_ker_translate->beta, (VALUE_TYPE*) *ptr_ker_translate->C, ptr_ker_translate->ldC);
  #endif
  	massert(CUBLAS_STATUS_SUCCESS == cublasDgemm(handle[dev_id], OpCharToCublas(ptr_ker_translate->TransA), OpCharToCublas(ptr_ker_translate->TransB),
  		ptr_ker_translate->M, ptr_ker_translate->N, ptr_ker_translate->K, &ptr_ker_translate->alpha, (VALUE_TYPE*) *ptr_ker_translate->A, ptr_ker_translate->ldA,
  		(VALUE_TYPE*) *ptr_ker_translate->B, ptr_ker_translate->ldB, &ptr_ker_translate->beta, (VALUE_TYPE*) *ptr_ker_translate->C, ptr_ker_translate->ldC),
  		"backend_run_operation: cublasDgemm failed\n");
  }
  else error("backend_run_operation: unkown opname=%s\n", opname);
}
