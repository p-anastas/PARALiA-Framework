///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief Backened wrapped host functions for cuda queue firing -> void func (void*)
///

#include <cblas.h>

#include "backend_wrappers.hpp"

void CoCoQueueLock(void* wrapped_lock){
#ifdef ENABLE_MUTEX_LOCKING
  (*(std::mutex*)wrapped_lock).lock();
#else
  while(__sync_lock_test_and_set ((&(*((int*)wrapped_lock))), 1));
#endif
#ifdef DEBUG
  lprintf(6, "CoCoQueueLock(%p) ran succesfully.\n", wrapped_lock);
#endif
}

void CoCoQueueUnlock(void* wrapped_lock){
#ifdef ENABLE_MUTEX_LOCKING
	(*(std::mutex*)wrapped_lock).unlock();
#else
  //int* intptr = (int*) wrapped_lock;
  //*intptr = 0;
  __sync_lock_release((&(*((int*) wrapped_lock))));
#endif

#ifdef DEBUG
  lprintf(6, "CoCoQueueUnlock(%p) ran succesfully.\n", wrapped_lock);
#endif
}

void CoCoIncAsync(void* wrapped_ptr_int){
  Ptr_int_p unwrapped = (Ptr_int_p) wrapped_ptr_int;
  *(unwrapped->int_ptr)++;
#ifdef DEBUG
  lprintf(6, "CoCoIncAsync(%p, new_val=%d) ran succesfully.\n", unwrapped->int_ptr, *(unwrapped->int_ptr));
#endif
}

void CoCoDecAsync(void* wrapped_ptr_int){
  Ptr_int_p unwrapped = (Ptr_int_p) wrapped_ptr_int;
  (*(unwrapped->int_ptr))--;
#ifdef DEBUG
  lprintf(6, "CoCoDecAsync(%p, new_val=%d) ran succesfully.\n", unwrapped->int_ptr, *(unwrapped->int_ptr));
#endif
}

void CoCoSetInt(void* wrapped_ptr_and_val){
  Ptr_and_int_p unwrapped = (Ptr_and_int_p) wrapped_ptr_and_val;
  *(unwrapped->int_ptr) = unwrapped->val;
#ifdef DEBUG
  lprintf(6, "CoCoSetVal(%p, %d) ran succesfully.\n", unwrapped->int_ptr, unwrapped->val);
#endif
}

void CoCoSetPtr(void* wrapped_ptr_and_parent){
  Ptr_and_parent_p unwrapped = (Ptr_and_parent_p) wrapped_ptr_and_parent;
  void* prev_ptr = *(unwrapped->ptr_parent);
  *(unwrapped->ptr_parent) = unwrapped->ptr_val;
#ifdef DEBUG
  lprintf(6, "CoCoSetPtr(prev=%p, %p) ran succesfully.\n", prev_ptr, unwrapped->ptr_val);
#endif
}

void CoCoFreeAllocAsync(void* backend_data){
  free(backend_data);
}

void cblas_wrap_daxpy(void* backend_data){
  axpy_backend_in_p ptr_ker_translate = (axpy_backend_in_p) backend_data;
  cblas_daxpy(ptr_ker_translate->N, ptr_ker_translate->alpha,
    (double*) *ptr_ker_translate->x, ptr_ker_translate->incx, (double*)
    *ptr_ker_translate->y, ptr_ker_translate->incy);
}

void cblas_wrap_saxpy(void* backend_data){
  axpy_backend_in_p ptr_ker_translate = (axpy_backend_in_p) backend_data;
  cblas_saxpy(ptr_ker_translate->N, ptr_ker_translate->alpha,
    (float*) *ptr_ker_translate->x, ptr_ker_translate->incx, (float*)
    *ptr_ker_translate->y, ptr_ker_translate->incy);
}

void cblas_wrap_dgemm(void* backend_data){
  short lvl = 6;
  gemm_backend_in_p ptr_ker_translate = (gemm_backend_in_p) backend_data;
#ifdef DDEBUG
  if (ptr_ker_translate->dev_id != -1)
    warning("cblas_wrap_dgemm: Suspicious device %d instead of -1\n", ptr_ker_translate->dev_id);
#endif
#ifdef DDEBUG
  lprintf(lvl, "cblas_wrap_dgemm: cblas_dgemm(dev_id = %d, TransA = %c, TransB = %c,\
    M = %d, N = %d, K = %d, alpha = %lf, A = %p, lda = %d, \n\
    B = %p, ldb = %d, beta = %lf, C = %p, ldC = %d)\n",
    ptr_ker_translate->dev_id, ptr_ker_translate->TransA, ptr_ker_translate->TransB,
    ptr_ker_translate->M, ptr_ker_translate->N, ptr_ker_translate->K, ptr_ker_translate->alpha,
    (VALUE_TYPE*) *ptr_ker_translate->A, ptr_ker_translate->ldA,
    (VALUE_TYPE*) *ptr_ker_translate->B, ptr_ker_translate->ldB,
    ptr_ker_translate->beta, (VALUE_TYPE*) *ptr_ker_translate->C, ptr_ker_translate->ldC);
#endif
  cblas_dgemm(CblasColMajor,
    OpCharToCblas(ptr_ker_translate->TransA), OpCharToCblas(ptr_ker_translate->TransB),
    ptr_ker_translate->M, ptr_ker_translate->N, ptr_ker_translate->K, ptr_ker_translate->alpha,
    (VALUE_TYPE*) *ptr_ker_translate->A, ptr_ker_translate->ldA,
    (VALUE_TYPE*) *ptr_ker_translate->B, ptr_ker_translate->ldB,
    ptr_ker_translate->beta, (VALUE_TYPE*) *ptr_ker_translate->C, ptr_ker_translate->ldC);
}

void cblas_wrap_sgemm(void* backend_data){
error("cblas_wrap_sgemm: never let empty unimplimented wrapped functions, moron\n");
}

void cublas_wrap_daxpy(void* backend_data, void* queue_wrap_p){
  axpy_backend_in_p ptr_ker_translate = (axpy_backend_in_p) backend_data;
  CoCoPeLiaSelectDevice(ptr_ker_translate->dev_id);
#ifdef ENABLE_PARALLEL_BACKEND
  cublasHandle_t temp_handle = *((cublasHandle_t*)((CQueue_p)queue_wrap_p)->cqueue_backend_data
    [((CQueue_p)queue_wrap_p)->backend_ctr]);
#else
  cublasHandle_t temp_handle = *((cublasHandle_t*)((CQueue_p)queue_wrap_p)->cqueue_backend_data);
#endif
  massert(CUBLAS_STATUS_SUCCESS == cublasDaxpy(temp_handle,
    ptr_ker_translate->N, (double*) &ptr_ker_translate->alpha, (double*) *ptr_ker_translate->x,
    ptr_ker_translate->incx, (double*) *ptr_ker_translate->y, ptr_ker_translate->incy),
    "cublas_wrap_daxpy failed\n");
}

void cublas_wrap_dgemm(void* backend_data, void* queue_wrap_p){
  short lvl = 6;
  gemm_backend_in_p ptr_ker_translate = (gemm_backend_in_p) backend_data;
#ifdef DDEBUG
  int cur_dev_id = CoCoPeLiaGetDevice();
  if (ptr_ker_translate->dev_id != cur_dev_id)
    warning("cublas_wrap_dgemm: Changing device %d -> %d\n", cur_dev_id, ptr_ker_translate->dev_id);
#endif
  CoCoPeLiaSelectDevice(ptr_ker_translate->dev_id);
#ifdef DDEBUG
  lprintf(lvl, "cublas_wrap_dgemm: cublasDgemm(dev_id = %d, TransA = %c, TransB = %c,\
    M = %d, N = %d, K = %d, alpha = %lf, A = %p, lda = %d, \n\
    B = %p, ldb = %d, beta = %lf, C = %p, ldC = %d)\n",
    ptr_ker_translate->dev_id, ptr_ker_translate->TransA, ptr_ker_translate->TransB,
    ptr_ker_translate->M, ptr_ker_translate->N, ptr_ker_translate->K, ptr_ker_translate->alpha,
    (VALUE_TYPE*) *ptr_ker_translate->A, ptr_ker_translate->ldA,
    (VALUE_TYPE*) *ptr_ker_translate->B, ptr_ker_translate->ldB,
    ptr_ker_translate->beta, (VALUE_TYPE*) *ptr_ker_translate->C, ptr_ker_translate->ldC);
#endif
#ifdef ENABLE_PARALLEL_BACKEND
  cublasHandle_t temp_handle = *((cublasHandle_t*)((CQueue_p)queue_wrap_p)->cqueue_backend_data
    [((CQueue_p)queue_wrap_p)->backend_ctr]);
#else
  cublasHandle_t temp_handle = *((cublasHandle_t*)((CQueue_p)queue_wrap_p)->cqueue_backend_data);
#endif
  massert(CUBLAS_STATUS_SUCCESS == cublasDgemm(temp_handle,
    OpCharToCublas(ptr_ker_translate->TransA), OpCharToCublas(ptr_ker_translate->TransB),
    ptr_ker_translate->M, ptr_ker_translate->N, ptr_ker_translate->K, &ptr_ker_translate->alpha,
    (VALUE_TYPE*) *ptr_ker_translate->A, ptr_ker_translate->ldA,
    (VALUE_TYPE*) *ptr_ker_translate->B, ptr_ker_translate->ldB,
    &ptr_ker_translate->beta, (VALUE_TYPE*) *ptr_ker_translate->C, ptr_ker_translate->ldC),
    "cublas_wrap_dgemm: cublasDgemm failed\n");
}
