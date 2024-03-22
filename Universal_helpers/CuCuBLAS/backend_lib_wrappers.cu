///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief Backend library wrappers for various cuda/cublas calls
///

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <pthread.h>
#include <cblas.h>

#include "backend_wrappers.hpp"

#define THREAD_POOL_SZ 128
pthread_t thread_pool[THREAD_POOL_SZ];
pthread_attr_t thread_pool_attr[THREAD_POOL_SZ];
int thread_ctr = 0, thread_lock = 0 ;

#ifdef MULTIDEVICE_REDUCTION_ENABLE
#ifdef ENABLE_MUTEX_LOCKING
#include <mutex> // std::mutex
std::mutex reduce_block_mutex[128*MAX_BUFFERING_L];// mutex for critical section
#else
int reduce_block_lock[128*MAX_BUFFERING_L] = {0};
#endif
#endif

void backend_init(short dev_id, CQueue_p h2d_q, CQueue_p d2h_q, CQueue_p exec_q){
  int dev_idc = -1;
  cudaError_t err = cudaGetDevice(&dev_idc);
  if(dev_idc != dev_id && dev_id != -1)
    warning("backend_init: called on different device - actual(%d) vs called(%d)\n", dev_idc, dev_id);
  return;
}

void backend_free(short dev_id){
  if (dev_id != -1){
    cudaSetDevice(dev_id);
    cudaDeviceSynchronize();
  }
  return;
}

void backend_run_operation(void* backend_data, const char* opname, CQueue_p run_queue){
  short lvl = 5;
  if (!strcmp(opname, "Dgemm")){
    gemm_backend_in<double>* ptr_ker_translate = (gemm_backend_in<double>*) backend_data;
    if(ptr_ker_translate->dev_id == -1) run_queue->add_host_func((void*)&cblas_wrap_dgemm, backend_data);
    else if(ptr_ker_translate->dev_id >= 0) cublas_wrap_dgemm(backend_data, run_queue);
    else error("backend_run_operation(dgemm): Not implemented for dev_id = %d\n", ptr_ker_translate->dev_id);
  }
  else if (!strcmp(opname, "Dgemv")){
  gemv_backend_in<double>* ptr_ker_translate = (gemv_backend_in<double>*) backend_data;
  if(ptr_ker_translate->dev_id == -1) run_queue->add_host_func((void*)&cblas_wrap_dgemv, backend_data);
  else if(ptr_ker_translate->dev_id >= 0) cublas_wrap_dgemv(backend_data, run_queue);
  else error("backend_run_operation(dgemv): Not implemented for dev_id = %d\n", ptr_ker_translate->dev_id);
  }
  else if(!strcmp(opname, "Sgemm")){
    gemm_backend_in<float>* ptr_ker_translate = (gemm_backend_in<float>*) backend_data;
    if(ptr_ker_translate->dev_id == -1) run_queue->add_host_func((void*)&cblas_wrap_sgemm, backend_data);
    else if(ptr_ker_translate->dev_id >= 0) cublas_wrap_sgemm(backend_data, run_queue);
    else error("backend_run_operation(sgemm): Not implemented for dev_id = %d\n", ptr_ker_translate->dev_id);
  }
  else if(!strcmp(opname, "Daxpy")){
    axpy_backend_in<double>* ptr_ker_translate = (axpy_backend_in<double>*) backend_data;
    if(ptr_ker_translate->dev_id == -1) run_queue->add_host_func((void*)&cblas_wrap_daxpy, backend_data);
    else if(ptr_ker_translate->dev_id >= 0) cublas_wrap_daxpy(backend_data, run_queue);
    else error("backend_run_operation(axpy,double): Not implemented for dev_id = %d\n", ptr_ker_translate->dev_id);
  }
  else if(!strcmp(opname, "Saxpy")){
      axpy_backend_in<float>* ptr_ker_translate = (axpy_backend_in<float>*) backend_data;
      //if(ptr_ker_translate->dev_id == -1) run_queue->add_host_func((void*)&cblas_wrap_saxpy, backend_data);
      //else if(ptr_ker_translate->dev_id >= 0) cublas_wrap_saxpy(backend_data, run_queue);
      //else
      error("backend_run_operation(axpy,float): Not implemented for dev_id = %d\n", ptr_ker_translate->dev_id);
  }
  else if(!strcmp(opname, "Ddot")){
    dot_backend_in<double>* ptr_ker_translate = (dot_backend_in<double>*) backend_data;
    if(ptr_ker_translate->dev_id == -1) run_queue->add_host_func((void*)&cblas_wrap_ddot, backend_data);
    else if(ptr_ker_translate->dev_id >= 0) cublas_wrap_ddot(backend_data, run_queue);
    else error("backend_run_operation(ddot): Not implemented for dev_id = %d\n", ptr_ker_translate->dev_id);
  }
  else if(!strcmp(opname, "Sdot")){
    dot_backend_in<float>* ptr_ker_translate = (dot_backend_in<float>*) backend_data;
    //if(ptr_ker_translate->dev_id == -1) run_queue->add_host_func((void*)&cblas_wrap_sdot, backend_data);
    //else if(ptr_ker_translate->dev_id >= 0) cublas_wrap_sdot(backend_data, run_queue);
    //else
    error("backend_run_operation(sdot): Not implemented for dev_id = %d\n", ptr_ker_translate->dev_id);
  }
  else error("backend_run_operation: unkown/not implemented opname=%s\n", opname);
}

void TransposeTranslate(char TransChar, CBLAS_TRANSPOSE* cblasFlag, cublasOperation_t* cuBLASFlag, long int* ldim, long int dim1, long int dim2){
	if (TransChar == 'N'){
 		*cblasFlag = CblasNoTrans;
 		*cuBLASFlag = CUBLAS_OP_N;
		*ldim = dim1;
	}
	else if (TransChar == 'T'){
 		*cblasFlag = CblasTrans;
 		*cuBLASFlag = CUBLAS_OP_T;
		*ldim = dim2;
	}
	else if (TransChar == 'C'){
 		*cblasFlag = CblasConjTrans;
 		*cuBLASFlag = CUBLAS_OP_C;
		*ldim = dim2;
	}
	else error("TransposeTranslate: %c is an invalid Trans flag", TransChar);
}

cublasOperation_t OpCblasToCublas(CBLAS_TRANSPOSE src)
{
	if(src == CblasNoTrans) return CUBLAS_OP_N;
	else if(src == CblasTrans) return CUBLAS_OP_T;
	else if(src == CblasConjTrans) return CUBLAS_OP_C;
	else error("OpCblasToCublas: Invalid Op\n");
}

cublasOperation_t OpCharToCublas(char src)
{
	if(src == 'N') return CUBLAS_OP_N;
	else if(src == 'T') return CUBLAS_OP_T;
	else if(src == 'C') return CUBLAS_OP_C;
	else error("OpCharToCublas: Invalid Op: %c\n", src);
}

CBLAS_TRANSPOSE OpCharToCblas(char src)
{
	if(src == 'N') return CblasNoTrans;
	else if(src == 'T') return CblasTrans;
	else if(src == 'C') return CblasConjTrans;
	else error("OpCharToCblas: Invalid Op: %c\n", src);
}

CBLAS_TRANSPOSE OpCublasToCblas(cublasOperation_t src)
{
	if(src == CUBLAS_OP_N) return CblasNoTrans;
	else if(src == CUBLAS_OP_T) return CblasTrans;
	else if(src == CUBLAS_OP_C) return CblasConjTrans;
	else error("OpCublasToCblas: Invalid Op\n");
}

char PrintCublasOp(cublasOperation_t src)
{

	if(src == CUBLAS_OP_N) return 'N';
	else if(src == CUBLAS_OP_T) return 'T';
	else if(src == CUBLAS_OP_C) return 'C';
	else error("PrintCublasOp: Invalid Op\n");
}

#ifdef MULTIDEVICE_REDUCTION_ENABLE

template<typename VALUETYPE>
void CoCoAdd2D(VALUETYPE* dest, long int ldest, VALUETYPE* src, long int lsrc,
	long int rows, long int cols, short loc, CQueue_p add_queue){
	short lvl = 7;
#ifdef DDEBUG
	lprintf(lvl, "CoCoAdd2DAsync(dest = %p, ldest =%zu, src=%p, lsrc = %zu, rows = %zu, cols = %zu, loc = %d)\n",
		dest, ldest, src, lsrc, rows, cols, loc);
#endif
#ifdef TEST
	double cpu_timer = csecond();
#endif

  // Stable implementation with sync
  if(loc == -1){
    axpy_backend_in<double>* backend_axpy_wrapper = (axpy_backend_in<double>*) malloc(sizeof(struct axpy_backend_in));
    backend_axpy_wrapper->N = rows;
    backend_axpy_wrapper->incx = backend_axpy_wrapper->incy = 1;
    backend_axpy_wrapper->alpha = 1.0;
    backend_axpy_wrapper->dev_id = loc;
  	for(int colidx = 0; colidx < cols; colidx++){
  		VALUETYPE* x = &src[colidx*lsrc];
  		VALUETYPE* y = &dest[colidx*ldest];
  		backend_axpy_wrapper->x = (void**) &x;
  		backend_axpy_wrapper->y = (void**) &y;
  		backend_run_operation(backend_axpy_wrapper, "axpy", add_queue);
      add_queue->sync_barrier();
    }
    free(backend_axpy_wrapper);
  }
  // Better implementation without sync - not working in CPU reduction?
  else{
    for(int colidx = 0; colidx < cols; colidx++){
      axpy_backend_in<double>* backend_axpy_wrapper = (axpy_backend_in<double>*) malloc(sizeof(struct axpy_backend_in));
      backend_axpy_wrapper->N = rows;
      backend_axpy_wrapper->incx = backend_axpy_wrapper->incy = 1;
      backend_axpy_wrapper->alpha = 1.0;
      backend_axpy_wrapper->dev_id = loc;
      VALUETYPE* x = &src[colidx*lsrc];
      VALUETYPE* y = &dest[colidx*ldest];
      backend_axpy_wrapper->x = (void**) &x;
      backend_axpy_wrapper->y = (void**) &y;
      backend_run_operation(backend_axpy_wrapper, "axpy", add_queue);
      add_queue->add_host_func((void*)&CoCoFreeAllocAsync, (void*)backend_axpy_wrapper);
    }
    add_queue->sync_barrier();
  }

#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
  lprintf(lvl, "CoCoAdd2D(loc=%d, cols=%d, rows=%d): t_reduce_add = %lf ms\n", loc, cols, rows, cpu_timer*1000);
#endif
}

template void CoCoAdd2D<double>(double* dest, long int ldest, double* src, long int lsrc,
	long int rows, long int cols, short loc, CQueue_p add_queue);

typedef struct CoCoMemcpyReduce2D_pthread_wrap{
  void* reduce_buffer;
  void* dest;
  long int ldest;
  void* src;
  long int lsrc, rows, cols;
  short elemSize, loc_dest, loc_src, reduce_buf_it;
  void* Tile_lock;
  CQueue_p src_reduce_queue, dest_reduce_queue;
}* CoCoMemcpyReduce2D_pthread_wrap_p;

void* CoCoMemcpyReduce2DWrapped(void* wrapped_data){
  CoCoMemcpyReduce2D_pthread_wrap_p unwrap = (CoCoMemcpyReduce2D_pthread_wrap_p) wrapped_data;
  void* reduce_buffer = unwrap->reduce_buffer;
  short reduce_buf_it = unwrap->reduce_buf_it;
  void* dest = unwrap->dest;
  long int ldest = unwrap->ldest;
  void* src = unwrap->src;
  long int lsrc = unwrap->lsrc, rows = unwrap->rows, cols = unwrap->cols;
  short elemSize = unwrap->elemSize, loc_dest = unwrap->loc_dest, loc_src = unwrap->loc_src;
  void* Tile_lock = unwrap->Tile_lock;
  CQueue_p src_reduce_queue = unwrap->src_reduce_queue,
    dest_reduce_queue = unwrap->dest_reduce_queue;
  short lvl = 5;
  short loc_dest_idx = (loc_dest == -1)? LOC_NUM - 1: loc_dest;
#ifdef DEBUG
  lprintf(lvl, "CoCoMemcpyReduce2DAsync(buf = %p, dest=%p, ldest =%zu, src=%p, lsrc = %zu, rows = %zu, cols = %zu, elemsize = %d, loc_dest = %d, loc_src = %d)\n",
    reduce_buffer, dest, ldest, src, lsrc, rows, cols, elemSize, loc_dest, loc_src);
  lprintf(lvl, "Blocking until reduce_block_lock[%d]\n", reduce_buf_it*128 + loc_dest_idx);
#endif
#ifdef TEST
  double cpu_timer = csecond();
#endif

  CoCoQueueLock((void*)&reduce_block_lock[reduce_buf_it*128 + loc_dest_idx]);

#ifdef TEST
  cpu_timer = csecond() - cpu_timer;
  lprintf(lvl, "CoCoMemcpyReduce2DAsync(buf = %p, loc_dest = %d, loc_src = %d): Blocked waiting for reduce_block_lock[%d] lock: %lf ms\n",
    reduce_buffer, loc_dest, loc_src, reduce_buf_it*128 + loc_dest_idx, cpu_timer*1000);
  cpu_timer = csecond();
#endif

  CoCoMemcpy2DAsync(reduce_buffer, lsrc, src, lsrc, rows, cols, elemSize, loc_dest, loc_src, src_reduce_queue);
  src_reduce_queue->sync_barrier();

#ifdef DEBUG
  lprintf(lvl, "CoCoMemcpyReduce2DAsync(dest=%d, src=%d): Blocking until Tile_lock(%p)\n", loc_dest, loc_src, Tile_lock);
#endif

  CoCoQueueLock((void*)Tile_lock);

#ifdef TEST
  cpu_timer = csecond() - cpu_timer;
  lprintf(lvl, "CoCoMemcpyReduce2DAsync(buf = %p, loc_dest = %d, loc_src = %d): Blocked waiting for Tile lock: %lf ms\n",
    reduce_buffer, loc_dest, loc_src, cpu_timer*1000);
  cpu_timer = csecond();
#endif

  CoCoAdd2D<VALUE_TYPE>( (VALUE_TYPE*) dest, ldest, (VALUE_TYPE*) reduce_buffer, lsrc, rows, cols, loc_dest, dest_reduce_queue);

  CoCoQueueUnlock((void*)Tile_lock);
  CoCoQueueUnlock((void*)&reduce_block_lock[reduce_buf_it*128 + loc_dest_idx]);

#ifdef TEST
  cpu_timer = csecond() - cpu_timer;
  lprintf(lvl, "CoCoMemcpyReduce2D(loc_src=%d, loc_dest=%d, cols=%d, rows=%d): t_reduce_total = %lf ms\n", loc_src, loc_dest, cols, rows, cpu_timer*1000);
#endif

  return wrapped_data;
}

// Asunchronous Memcpy in internal buffer AND reduce to dest between two locations WITHOUT synchronous errorchecking. Use with caution.
void CoCoMemcpyReduce2DAsync(void* reduce_buffer, short reduce_buf_it, void* dest, long int ldest, void* src, long int lsrc,
	long int rows, long int cols, short elemSize, short loc_dest, short loc_src, void* Tile_lock, CQueue_p src_reduce_queue, CQueue_p dest_reduce_queue){
    int s;
    void* res;
    while(__sync_lock_test_and_set (&thread_lock, 1));
    if (thread_ctr == THREAD_POOL_SZ){
        for (int idx = 0; idx < THREAD_POOL_SZ; idx++){
          s = pthread_join(thread_pool[idx], &res);
          free(res);
          if (s != 0) error("CoCoMemcpyReduce2D: pthread_join(thread_pool[%d]) failed with exit value %d", idx, s);
        }
        thread_ctr = 0;
    }
    else if (thread_ctr > THREAD_POOL_SZ)
      error("CoCoMemcpyReduce2D: thread_ctr exeeded pool size somehow\n");
    int local_ctr = thread_ctr++;
    __sync_lock_release(&thread_lock);
    s = pthread_attr_init(&thread_pool_attr[local_ctr]);
  	if (s != 0) error("CoCoMemcpyReduce2D: pthread_attr_init failed s=%d\n", s);
  	CoCoMemcpyReduce2D_pthread_wrap_p thread_dev_data = (CoCoMemcpyReduce2D_pthread_wrap_p)
      malloc(sizeof(struct CoCoMemcpyReduce2D_pthread_wrap));
    thread_dev_data->reduce_buffer = reduce_buffer;
    thread_dev_data->reduce_buf_it = reduce_buf_it;
    thread_dev_data->dest = dest;
    thread_dev_data->ldest = ldest;
    thread_dev_data->src = src;
    thread_dev_data->lsrc = lsrc;
    thread_dev_data->rows = rows;
    thread_dev_data->cols = cols;
    thread_dev_data->elemSize = elemSize;
    thread_dev_data->loc_dest = loc_dest;
    thread_dev_data->loc_src = loc_src;
    thread_dev_data->Tile_lock = Tile_lock;
    thread_dev_data->src_reduce_queue = src_reduce_queue;
    thread_dev_data->dest_reduce_queue = dest_reduce_queue;
  	s = pthread_create(&thread_pool[local_ctr], &thread_pool_attr[local_ctr], &CoCoMemcpyReduce2DWrapped, thread_dev_data);
}

void CoCoMemcpyReduceAsync(void* reduce_buffer, short reduce_buf_it, void* dest, void* src, long long bytes,
	short loc_dest, short loc_src, void* Tile_lock, CQueue_p src_reduce_queue, CQueue_p dest_reduce_queue){
   error("CoCoMemcpyReduceAsync: Not implemented (should it?)\n");
 }

void CoCoReduceSyncThreads(){
  int s;
  void* res;
  for (int idx = 0; idx < thread_ctr; idx++){
    s = pthread_join(thread_pool[idx], &res);
    free(res);
    if (s != 0) error("CoCoReduceSyncThreads: pthread_join(thread_pool[%d]) failed with exit value %d", idx, s);
  }
  thread_ctr = 0;
}

#endif
