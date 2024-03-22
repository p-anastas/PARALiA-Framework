///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The headers for functions for general use throught CoCoPeLia
///

#ifndef UNIHELPERS_BACK_H
#define UNIHELPERS_BACK_H

#include <cuda.h>
#include "cublas_v2.h"

#include <unihelpers.hpp>

template<typename VALUETYPE> class gemm_backend_in{
public:
	char TransA,  TransB;
	int M, N, K, ldA, ldB, ldC;
	VALUETYPE alpha,beta;
	void **A, **B, **C;
	short dev_id;
};

template<typename VALUETYPE> class gemv_backend_in{
public:
	char TransA,  incx, incy;
	int M, N, ldA;
	VALUETYPE alpha,beta;
	void **A, **x, **y;
	short dev_id;
};

template<typename VALUETYPE> class axpy_backend_in{
public:
		int N, incx, incy;
	VALUETYPE alpha;
	void **x, **y;
	short dev_id;
};

template<typename VALUETYPE> class dot_backend_in{
public:
	int N, incx, incy;
	void **x, **y;
	VALUETYPE* result;
	short dev_id;
};


/// Initalize backend (handle[s]) in dev_id
void backend_init(short dev_id, CQueue_p h2d_q, CQueue_p d2h_q, CQueue_p exec_q);

/// Free backend (handle[s]) in dev_id
void backend_free(short dev_id);

/// Select and run a wrapped operation (e.g. gemm, axpy) depending on opname
void backend_run_operation(void* backend_data, const char* opname, CQueue_p run_queue);

#ifdef MULTIDEVICE_REDUCTION_ENABLE

// Asunchronous 2D Memcpy in internal buffer AND reduce to dest between two locations WITHOUT synchronous errorchecking. Use with caution.
void CoCoMemcpyReduce2DAsync(void* reduce_buffer, short reduce_buf_it, void* dest, long int ldest, void* src, long int lsrc, long int rows, long int cols,
	short elemSize, short loc_dest, short loc_src, void* Tile_lock, CQueue_p src_reduce_queue, CQueue_p dest_reduce_queue);

// Asunchronous Memcpy in internal buffer AND reduce to dest between two locations WITHOUT synchronous errorchecking. Use with caution.
void CoCoMemcpyReduceAsync(void* reduce_buffer, short reduce_buf_it, void* dest, void* src, long long bytes,
	short loc_dest, short loc_src, void* Tile_lock, CQueue_p src_reduce_queue, CQueue_p dest_reduce_queue);

// Blocks until all reduce threads are complete and free temp structs.
void CoCoReduceSyncThreads();

/// Asunchronous add 2D (for block reduce)
template<typename VALUETYPE>
void CoCoAdd2Dc(VALUETYPE* dest, long int ldest, VALUETYPE* src, long int lsrc,
	long int rows, long int cols, short loc, CQueue_p add_queue);

#endif

#ifdef BUILD_BLAS_WRAPPERS_FLAG
#include <cblas.h>

void TransposeTranslate(char TransChar, CBLAS_TRANSPOSE* cblasFlag, cublasOperation_t* cuBLASFlag, long int* ldim, long int dim1, long int dim2);

cublasOperation_t OpCblasToCublas(CBLAS_TRANSPOSE src);
CBLAS_TRANSPOSE OpCublasToCblas(cublasOperation_t src);
cublasOperation_t OpCharToCublas(char src);
CBLAS_TRANSPOSE OpCharToCblas(char src);
char PrintCublasOp(cublasOperation_t src);

#endif

/// Internally used utils TODO: Is this the correct way softeng wise?
void cudaCheckErrors();

// Lock wrapped_lock. This functions is fired in a queue to lock when it reaches that point.
void CoCoQueueLock(void* wrapped_lock);
// Unlock wrapped_lock. This functions is fired in a queue to unlock when it reaches that point.
void CoCoQueueUnlock(void* wrapped_lock);

// Struct containing an int pointer
typedef struct Ptr_atomic_int{
	std::atomic<int>* ato_int_ptr;
}* Ptr_atomic_int_p;
void CoCoIncAsync(void* wrapped_ptr_int);
void CoCoDecAsync(void* wrapped_ptr_int);

// Struct containing an int pointer and an int for Asynchronous set
typedef struct Ptr_and_int{
	int* int_ptr;
	int val;
}* Ptr_and_int_p;
void CoCoSetInt(void* wrapped_ptr_and_val);

// Struct containing a void pointer and a void for Asynchronous set
typedef struct Ptr_and_parent{
	void** ptr_parent;
	void* ptr_val;
}* Ptr_and_parent_p;
void CoCoSetPtr(void* wrapped_ptr_and_parent);

void CoCoSetTimerAsync(void* wrapped_timer_Ptr);

void CoCoFreeAllocAsync(void* backend_data);

void cublas_wrap_ddot(void* backend_data, void* queue_wrap_p);
void cublas_wrap_daxpy(void* backend_data, void* queue_wrap_p);
void cublas_wrap_saxpy(void* backend_data, void* queue_wrap_p);

void cublas_wrap_dgemv(void* backend_data, void* queue_wrap_p);

void cublas_wrap_dgemm(void* backend_data, void* queue_wrap_p);
void cublas_wrap_sgemm(void* backend_data, void* queue_wrap_p);

void cblas_wrap_ddot(void* backend_data);
void cblas_wrap_daxpy(void* backend_data);
void cblas_wrap_saxpy(void* backend_data);

void cblas_wrap_dgemv(void* backend_data);

void cblas_wrap_dgemm(void* backend_data);
void cblas_wrap_sgemm(void* backend_data);

#endif
