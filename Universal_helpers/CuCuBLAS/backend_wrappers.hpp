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
void backend_run_operation(void* backend_data, const char* opname);

// Asunchronous Memcpy in internal buffer AND reduce to dest between two locations WITHOUT synchronous errorchecking. Use with caution.
void CoCoMemcpyReduce2D(void* reduce_buffer, void* dest, size_t ldest, void* src, size_t lsrc, size_t rows, size_t cols,
	short elemSize, short loc_dest, short loc_src, CQueue_p reduce_queue);

/// Asunchronous add 2D (for block reduce)
template<typename VALUETYPE>
void CoCoAdd2Dc(VALUETYPE* dest, size_t ldest, VALUETYPE* src, size_t lsrc,
	size_t rows, size_t cols, short loc, CQueue_p add_queue);
  
void TransposeTranslate(char TransChar, CBLAS_TRANSPOSE* cblasFlag, cublasOperation_t* cuBLASFlag, size_t* ldim, size_t dim1, size_t dim2);

cublasOperation_t OpCblasToCublas(CBLAS_TRANSPOSE src);
CBLAS_TRANSPOSE OpCublasToCblas(cublasOperation_t src);
cublasOperation_t OpCharToCublas(char src);
CBLAS_TRANSPOSE OpCharToCblas(char src);
char PrintCublasOp(cublasOperation_t src);

/// Internally used utils TODO: Is this the correct way softeng wise?
void cudaCheckErrors();

#endif
