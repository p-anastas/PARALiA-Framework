///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The external wrapper for CoCoPelia + wrapped cuBLASXt
///
#ifndef COCOPELIA_H
#define COCOPELIA_H

#include "Autotuning_runtime.hpp"

/// The CoCopeLia Dgemm implementation. A prediction model is used to select a tile from the micro-benchmarked tile candidates with CoCopeLia_optimize3.
ATC_p CoCopeLiaDgemm(char TransA,  char TransB, size_t M, size_t N, size_t K,
	double alpha, double* A, size_t ldA, double* B, size_t ldB, double beta, double* C, size_t ldC);

/// A modification of CoCopeLiaDgemm but with a given T (mainly for performance/debug purposes)
ATC_p CoCopeLiaDgemmControled(char TransA,  char TransB, size_t M, size_t N, size_t K,
	double alpha, double* A, size_t ldA, double* B, size_t ldB, double beta, double* C, size_t ldC, ATC_p predef_control_values);

/// The CoCopeLia Dgemm implementation. A prediction model is used to select a tile from the micro-benchmarked tile candidates with CoCopeLia_optimize3.
ATC_p CoCopeLiaDaxpy(size_t N, VALUE_TYPE alpha,
	VALUE_TYPE* x, size_t incx, VALUE_TYPE* y, size_t incy);

/// A modification of CoCopeLiaDgemm but with a given T (mainly for performance/debug purposes)
ATC_p CoCopeLiaDaxpyControled(size_t N, VALUE_TYPE alpha,
	VALUE_TYPE* x, size_t incx, VALUE_TYPE* y, size_t incy, ATC_p predef_control_values);

///Deallocates the GPU-allocated cache buffer at target device
void CoCopeLiaDevCacheFree(short dev_id);

#endif
