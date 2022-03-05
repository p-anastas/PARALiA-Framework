///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The headers for functions for general use throught CoCoPeLia
///

#ifndef UNIHELPERS_BACK_H
#define UNIHELPERS_BACK_H

#include <cuda.h>
#include "cublas_v2.h"
#include <cblas.h>

#include <unihelpers.hpp>

typedef struct gemm_backend_in{
	char TransA,  TransB;
	int M, N, K, ldA, ldB, ldC;
	VALUE_TYPE alpha,beta;
	void **A, **B, **C;
	short dev_id;
}* gemm_backend_in_p;

typedef struct axpy_backend_in{
	int N, incx, incy;
	VALUE_TYPE alpha;
	void **x, **y;
	short dev_id;
}* axpy_backend_in_p;

/// Initalize backend (handle[s]) in dev_id
void backend_init(short dev_id, CQueue_p h2d_q, CQueue_p d2h_q, CQueue_p exec_q);

/// Free backend (handle[s]) in dev_id
void backend_free(short dev_id);

/// Select and run a wrapped operation (e.g. gemm, axpy) depending on opname
void backend_run_operation(void* backend_data, const char* opname, CQueue_p run_queue);

#ifdef MULTIDEVICE_REDUCTION_ENABLE

// Asunchronous 2D Memcpy in internal buffer AND reduce to dest between two locations WITHOUT synchronous errorchecking. Use with caution.
void CoCoMemcpyReduce2DAsync(void* reduce_buffer, short reduce_buf_it, void* dest, size_t ldest, void* src, size_t lsrc, size_t rows, size_t cols,
	short elemSize, short loc_dest, short loc_src, void* Tile_lock, CQueue_p src_reduce_queue, CQueue_p dest_reduce_queue);

// Asunchronous Memcpy in internal buffer AND reduce to dest between two locations WITHOUT synchronous errorchecking. Use with caution.
void CoCoMemcpyReduceAsync(void* reduce_buffer, short reduce_buf_it, void* dest, void* src, long long bytes,
	short loc_dest, short loc_src, void* Tile_lock, CQueue_p src_reduce_queue, CQueue_p dest_reduce_queue);

// Blocks until all reduce threads are complete and free temp structs.
void CoCoReduceSyncThreads();

/// Asunchronous add 2D (for block reduce)
template<typename VALUETYPE>
void CoCoAdd2Dc(VALUETYPE* dest, size_t ldest, VALUETYPE* src, size_t lsrc,
	size_t rows, size_t cols, short loc, CQueue_p add_queue);

#endif

void TransposeTranslate(char TransChar, CBLAS_TRANSPOSE* cblasFlag, cublasOperation_t* cuBLASFlag, size_t* ldim, size_t dim1, size_t dim2);

cublasOperation_t OpCblasToCublas(CBLAS_TRANSPOSE src);
CBLAS_TRANSPOSE OpCublasToCblas(cublasOperation_t src);
cublasOperation_t OpCharToCublas(char src);
CBLAS_TRANSPOSE OpCharToCblas(char src);
char PrintCublasOp(cublasOperation_t src);

/// Internally used utils TODO: Is this the correct way softeng wise?
void cudaCheckErrors();

// Lock wrapped_lock. This functions is fired in a queue to lock when it reaches that point.
void CoCoQueueLock(void* wrapped_lock);
// Unlock wrapped_lock. This functions is fired in a queue to unlock when it reaches that point.
void CoCoQueueUnlock(void* wrapped_lock);

// Struct containing an int pointer and an int for Asynchronous set
typedef struct Ptr_and_int{
	int* int_ptr;
	int val;
}* Ptr_and_int_p;
void CoCoSetInt(void* wrapped_ptr_and_val);

void CoCoFreeAllocAsync(void* backend_data);

void cublas_wrap_daxpy(void* backend_data, void* queue_wrap_p);
void cublas_wrap_saxpy(void* backend_data, void* queue_wrap_p);
void cublas_wrap_dgemm(void* backend_data, void* queue_wrap_p);
void cublas_wrap_sgemm(void* backend_data, void* queue_wrap_p);

void cblas_wrap_daxpy(void* backend_data);
void cblas_wrap_saxpy(void* backend_data);
void cblas_wrap_dgemm(void* backend_data);
void cblas_wrap_sgemm(void* backend_data);

#endif
