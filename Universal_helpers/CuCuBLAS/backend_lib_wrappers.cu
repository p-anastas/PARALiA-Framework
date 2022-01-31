///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The DGEMM CoCopeLia implementation.
///

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <pthread.h>
#include <cblas.h>

#include "backend_wrappers.hpp"

cublasHandle_t handle[128] = {NULL};
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

#ifdef MULTIDEVICE_REDUCTION_ENABLE

void CoCoQueueUnlock(void* wrapped_lock){
#ifdef ENABLE_MUTEX_LOCKING
	((std::mutex*)wrapped_lock)->unlock();
#else
  int* intptr = (int*) wrapped_lock;
  *intptr = 0;
#endif

#ifdef DEBUG
  lprintf(6, "CoCoSetFlag(%p) ran succesfully.\n", wrapped_lock);
#endif
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
  //add_queue->sync_barrier();
  free(backend_axpy_wrapper);
#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
  lprintf(lvl, "CoCoAdd2D(loc=%d, cols=%d, rows=%d): t_reduce_add = %lf ms\n", loc, cols, rows, cpu_timer*1000);
#endif
}

template void CoCoAdd2D<double>(double* dest, size_t ldest, double* src, size_t lsrc,
	size_t rows, size_t cols, short loc, CQueue_p add_queue);

typedef struct CoCoMemcpyReduce2D_pthread_wrap{
  void* reduce_buffer;
  void* dest;
  size_t ldest;
  void* src;
  size_t lsrc, rows, cols;
  short elemSize, loc_dest, loc_src, reduce_buf_it;
  void* Tile_lock;
  CQueue_p reduce_queue;
}* CoCoMemcpyReduce2D_pthread_wrap_p;

void* CoCoMemcpyReduce2DWrapped(void* wrapped_data){
  CoCoMemcpyReduce2D_pthread_wrap_p unwrap = (CoCoMemcpyReduce2D_pthread_wrap_p) wrapped_data;
  void* reduce_buffer = unwrap->reduce_buffer;
  short reduce_buf_it = unwrap->reduce_buf_it;
  void* dest = unwrap->dest;
  size_t ldest = unwrap->ldest;
  void* src = unwrap->src;
  size_t lsrc = unwrap->lsrc, rows = unwrap->rows, cols = unwrap->cols;
  short elemSize = unwrap->elemSize, loc_dest = unwrap->loc_dest, loc_src = unwrap->loc_src;
  void* Tile_lock = unwrap->Tile_lock;
  CQueue_p reduce_queue = unwrap->reduce_queue;
  short lvl = 5;
#ifdef DEBUG
  lprintf(lvl, "CoCoMemcpyReduce2DAsync(buf = %p, dest=%p, ldest =%zu, src=%p, lsrc = %zu, rows = %zu, cols = %zu, elemsize = %d, loc_dest = %d, loc_src = %d)\n",
    reduce_buffer, dest, ldest, src, lsrc, rows, cols, elemSize, loc_dest, loc_src);
  lprintf(lvl, "Blocking until CoCoSetFlag(%p)\n", Tile_lock);
#endif
#ifdef TEST
  double cpu_timer = csecond();
#endif

#ifdef ENABLE_MUTEX_LOCKING
  reduce_block_mutex[reduce_buf_it*128 + loc_src].lock();
#else
  while(__sync_lock_test_and_set (&reduce_block_lock[reduce_buf_it*128 +loc_src], 1));
#endif

#ifdef TEST
  cpu_timer = csecond() - cpu_timer;
  lprintf(lvl, "CoCoMemcpyReduce2DAsync(buf = %p, loc_dest = %d, loc_src = %d): Blocked waiting for reduce_block_lock[%d] lock: %lf ms\n",
    reduce_buffer, loc_dest, loc_src, loc_src, cpu_timer*1000);
  cpu_timer = csecond();
#endif

  CoCoMemcpy2DAsync(reduce_buffer, lsrc, src, lsrc, rows, cols, elemSize, loc_dest, loc_src, reduce_queue);
  reduce_queue->sync_barrier();

#ifdef ENABLE_MUTEX_LOCKING
  ((std::mutex*)Tile_lock)->lock();
#else
  while(__sync_lock_test_and_set ((int*)Tile_lock, 1));
#endif
#ifdef TEST
  cpu_timer = csecond() - cpu_timer;
  lprintf(lvl, "CoCoMemcpyReduce2DAsync(buf = %p, loc_dest = %d, loc_src = %d): Blocked waiting for Tile lock: %lf ms\n",
    reduce_buffer, loc_dest, loc_src, cpu_timer*1000);
  cpu_timer = csecond();
#endif
  CoCoAdd2D<VALUE_TYPE>( (VALUE_TYPE*) dest, ldest, (VALUE_TYPE*) reduce_buffer, lsrc, rows, cols, loc_dest, reduce_queue);

  #ifdef ENABLE_MUTEX_LOCKING
    ((std::mutex*)Tile_lock)->unlock();
  #else
    __sync_lock_release((int*)Tile_lock);
  #endif

  #ifdef ENABLE_MUTEX_LOCKING
    reduce_block_mutex[reduce_buf_it*128 + loc_src].unlock();
  #else
    __sync_lock_release(&reduce_block_lock[reduce_buf_it*128 + loc_src]);
  #endif

#ifdef TEST
  cpu_timer = csecond() - cpu_timer;
  lprintf(lvl, "CoCoMemcpyReduce2D(loc_src=%d, loc_dest=%d, cols=%d, rows=%d): t_reduce_total = %lf ms\n", loc_src, loc_dest, cols, rows, cpu_timer*1000);
#endif
  return wrapped_data;
}

// Asunchronous Memcpy in internal buffer AND reduce to dest between two locations WITHOUT synchronous errorchecking. Use with caution.
void CoCoMemcpyReduce2DAsync(void* reduce_buffer, short reduce_buf_it, void* dest, size_t ldest, void* src, size_t lsrc,
	size_t rows, size_t cols, short elemSize, short loc_dest, short loc_src, void* Tile_lock, CQueue_p reduce_queue){
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
    thread_dev_data->reduce_queue = reduce_queue;
  	s = pthread_create(&thread_pool[local_ctr], &thread_pool_attr[local_ctr], &CoCoMemcpyReduce2DWrapped, thread_dev_data);
}

void CoCoMemcpyReduceAsync(void* reduce_buffer, short reduce_buf_it, void* dest, void* src, long long bytes,
	short loc_dest, short loc_src, void* Tile_lock, CQueue_p reduce_queue){
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
