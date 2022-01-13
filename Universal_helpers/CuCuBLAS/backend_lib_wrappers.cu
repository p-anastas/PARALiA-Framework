///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The DGEMM CoCopeLia implementation.
///

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <pthread.h>
#include <cblas.h>

#include <thread>

#include "backend_wrappers.hpp"

cublasHandle_t handle[128] = {NULL};
CQueue_p reduce_queue = NULL;
int reduce_block_lock = 0;

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

void backend_free(short dev_id){
  massert(CUBLAS_STATUS_SUCCESS == cublasDestroy(handle[dev_id]), "cublasDestroy failed\n");
  return;
}

void backend_run_operation(void* backend_data, const char* opname){
  short lvl = 5;
  if (!strcmp(opname, "gemm")){
    gemm_backend_in_p ptr_ker_translate = (gemm_backend_in_p) backend_data;
#ifdef DDEBUG
    int cur_dev_id = CoCoPeLiaGetDevice();
    if (ptr_ker_translate->dev_id != cur_dev_id) warning("backend_run_operation: Changing device %d -> %d\n",
      cur_dev_id, ptr_ker_translate->dev_id);
#endif
    CoCoPeLiaSelectDevice(ptr_ker_translate->dev_id);
#ifdef DDEBUG
  	lprintf(lvl, "backend_run_operation: cublasDgemm(dev_id = %d, TransA = %c, TransB = %c, M = %d, N = %d, K = %d, alpha = %lf, A = %p, lda = %d, \n\
  	B = %p, ldb = %d, beta = %lf, C = %p, ldC = %d)\n", ptr_ker_translate->dev_id, ptr_ker_translate->TransA, ptr_ker_translate->TransB,
  		ptr_ker_translate->M, ptr_ker_translate->N, ptr_ker_translate->K, ptr_ker_translate->alpha, (VALUE_TYPE*) *ptr_ker_translate->A, ptr_ker_translate->ldA,
  		(VALUE_TYPE*) *ptr_ker_translate->B, ptr_ker_translate->ldB, ptr_ker_translate->beta, (VALUE_TYPE*) *ptr_ker_translate->C, ptr_ker_translate->ldC);
#endif
  massert(CUBLAS_STATUS_SUCCESS == cublasDgemm(handle[ptr_ker_translate->dev_id], OpCharToCublas(ptr_ker_translate->TransA), OpCharToCublas(ptr_ker_translate->TransB),
  		ptr_ker_translate->M, ptr_ker_translate->N, ptr_ker_translate->K, &ptr_ker_translate->alpha, (VALUE_TYPE*) *ptr_ker_translate->A, ptr_ker_translate->ldA,
  		(VALUE_TYPE*) *ptr_ker_translate->B, ptr_ker_translate->ldB, &ptr_ker_translate->beta, (VALUE_TYPE*) *ptr_ker_translate->C, ptr_ker_translate->ldC),
  		"backend_run_operation: cublasDgemm failed\n");
  }
  else if(!strcmp(opname, "axpy")){
    axpy_backend_in_p ptr_ker_translate = (axpy_backend_in_p) backend_data;
    CoCoPeLiaSelectDevice(ptr_ker_translate->dev_id);
    if (std::is_same<VALUE_TYPE, double>::value){
      if(ptr_ker_translate->dev_id == -1) cblas_daxpy(ptr_ker_translate->N, ptr_ker_translate->alpha,
          (double*) *ptr_ker_translate->x, ptr_ker_translate->incx, (double*) *ptr_ker_translate->y, ptr_ker_translate->incy);
      else if(ptr_ker_translate->dev_id >= 0) massert(CUBLAS_STATUS_SUCCESS == cublasDaxpy(handle[ptr_ker_translate->dev_id], ptr_ker_translate->N,
          (double*) &ptr_ker_translate->alpha, (double*) *ptr_ker_translate->x, ptr_ker_translate->incx, (double*) *ptr_ker_translate->y, ptr_ker_translate->incy),
  		     "backend_run_operation: cublasDaxpy failed\n");
      else error("backend_run_operation(axpy,double): Not implemented for dev_id = %d\n", ptr_ker_translate->dev_id);
    }
    else if (std::is_same<VALUE_TYPE, float>::value){
      if(ptr_ker_translate->dev_id == -1) cblas_saxpy(ptr_ker_translate->N, ptr_ker_translate->alpha,
          (float*) *ptr_ker_translate->x, ptr_ker_translate->incx, (float*) *ptr_ker_translate->y, ptr_ker_translate->incy);
      else if(ptr_ker_translate->dev_id >= 0) massert(CUBLAS_STATUS_SUCCESS == cublasSaxpy(handle[ptr_ker_translate->dev_id], ptr_ker_translate->N,
          (float*) &ptr_ker_translate->alpha, (float*) *ptr_ker_translate->x, ptr_ker_translate->incx, (float*) *ptr_ker_translate->y, ptr_ker_translate->incy),
  		     "backend_run_operation: cublasSaxpy failed\n");
      else error("backend_run_operation(axpy,float): Not implemented for dev_id = %d\n", ptr_ker_translate->dev_id);
    }
    else error("backend_run_operation(axpy): Not implemented for VALUETYPE\n");
  }
  else error("backend_run_operation: unkown opname=%s\n", opname);
}

template<typename VALUETYPE>
void CoCoAdd2D(VALUETYPE* dest, size_t ldest, VALUETYPE* src, size_t lsrc,
	size_t rows, size_t cols, short loc, CQueue_p add_queue){
	short lvl = 7;
#ifdef DDEBUG
	lprintf(lvl, "CoCoAdd2DAsync(dest = %p, ldest =%zu, src=%p, lsrc = %zu, rows = %zu, cols = %zu, loc = %d)\n",
		dest, ldest, src, lsrc, rows, cols, loc);
#endif
#ifdef TEST
	double cpu_timer = csecond();
#endif
  axpy_backend_in_p backend_axpy_wrapper = (axpy_backend_in_p) malloc(sizeof(struct axpy_backend_in));
  backend_axpy_wrapper->N = rows;
  backend_axpy_wrapper->incx = backend_axpy_wrapper->incy = 1;
  backend_axpy_wrapper->alpha = 1.0;
  backend_axpy_wrapper->dev_id = loc;
	for(int colidx = 0; colidx < cols; colidx++){
//#ifdef DDEBUG
//		lprintf(lvl, "colidx: %d x_offset = %d, y_offset = %d\n", colidx, colidx*lsrc, colidx*ldest);
//#endif
		VALUETYPE* x = &src[colidx*lsrc];
		VALUETYPE* y = &dest[colidx*ldest];
		backend_axpy_wrapper->x = (void**) &x;
		backend_axpy_wrapper->y = (void**) &y;
		backend_run_operation(backend_axpy_wrapper, "axpy");
  }
  CoCoSyncCheckErr();
  free(backend_axpy_wrapper);
#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
  lprintf(lvl, "CoCoAdd2D(loc=%d, cols=%d, rows=%d): t_reduce_add = %lf ms\n", loc, cols, rows, cpu_timer*1000);
#endif
}

template void CoCoAdd2D<double>(double* dest, size_t ldest, double* src, size_t lsrc,
	size_t rows, size_t cols, short loc, CQueue_p add_queue);


// Asunchronous Memcpy in internal buffer AND reduce to dest between two locations WITHOUT synchronous errorchecking. Use with caution.
void CoCoMemcpyReduce2D(void* reduce_buffer, void* dest, size_t ldest, void* src, size_t lsrc,
	size_t rows, size_t cols, short elemSize, short loc_dest, short loc_src, CQueue_p reduce_queue){
		short lvl = 5;
#ifdef DDEBUG
	lprintf(lvl, "CoCoMemcpyReduce2DAsync(buf = %p, dest=%p, ldest =%zu, src=%p, lsrc = %zu, rows = %zu, cols = %zu, elemsize = %d, loc_dest = %d, loc_src = %d)\n",
		reduce_buffer, dest, ldest, src, lsrc, rows, cols, elemSize, loc_dest, loc_src);
#endif
#ifdef TEST
	double cpu_timer = csecond();
#endif
	while(__sync_lock_test_and_set (&reduce_block_lock, 1)); // Naive global block lock.
	CoCoMemcpy2DAsync(reduce_buffer, lsrc, src, lsrc, rows, cols, elemSize, loc_dest, loc_src, reduce_queue);
	reduce_queue->sync_barrier();
	CoCoAdd2D<VALUE_TYPE>( (VALUE_TYPE*) dest, ldest, (VALUE_TYPE*) reduce_buffer, lsrc, rows, cols, loc_dest, reduce_queue);
	__sync_lock_release(&reduce_block_lock);
#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
  lprintf(lvl, "CoCoMemcpyReduce2D(loc_src=%d, loc_dest=%d, cols=%d, rows=%d): t_reduce_total = %lf ms\n", loc_src, loc_dest, cols, rows, cpu_timer*1000);
#endif
}
