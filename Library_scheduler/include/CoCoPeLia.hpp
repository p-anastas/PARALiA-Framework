///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The external wrapper for CoCoPelia + wrapped cuBLASXt
///
#ifndef COCOPELIA_H
#define COCOPELIA_H

#include "Autotuning_runtime.hpp"

/// The CoCopeLia Dgemm implementation. A prediction model is used to select a tile from the micro-benchmarked tile candidates with CoCopeLia_optimize3.
ATC_p CoCopeLiaDgemm(char TransA,  char TransB, long int M, long int N, long int K,
	double alpha, double* A, long int ldA, double* B, long int ldB, double beta, double* C, long int ldC);

/// A modification of CoCopeLiaDgemm but with a given T (mainly for performance/debug purposes)
ATC_p CoCopeLiaDgemmControled(char TransA,  char TransB, long int M, long int N, long int K,
	double alpha, double* A, long int ldA, double* B, long int ldB, double beta, double* C, long int ldC, ATC_p predef_control_values);

/// The CoCopeLia Dgemm implementation. A prediction model is used to select a tile from the micro-benchmarked tile candidates with CoCopeLia_optimize3.
ATC_p CoCopeLiaDaxpy(long int N, VALUE_TYPE alpha,
	VALUE_TYPE* x, long int incx, VALUE_TYPE* y, long int incy);

/// A modification of CoCopeLiaDgemm but with a given T (mainly for performance/debug purposes)
ATC_p CoCopeLiaDaxpyControled(long int N, VALUE_TYPE alpha,
	VALUE_TYPE* x, long int incx, VALUE_TYPE* y, long int incy, ATC_p predef_control_values);

///Deallocates the GPU-allocated cache buffer at target device
void CoCopeLiaDevCacheFree(short dev_id);

#endif
