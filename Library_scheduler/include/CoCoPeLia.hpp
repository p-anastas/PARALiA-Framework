///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The external wrapper for CoCoPelia + wrapped cuBLASXt
///
#ifndef COCOPELIA_H
#define COCOPELIA_H

#include "Autotuner.hpp"

/// The PARALia Dgemm implementation.
ATC_p PARALiaDgemm(char TransA,  char TransB, long int M, long int N, long int K,
	double alpha, double* A, long int ldA, double* B, long int ldB, double beta, double* C, long int ldC);

/// A modification of PARALiaDgemm but with a given T (mainly for performance/debug purposes)
ATC_p PARALiaDgemmControled(char TransA,  char TransB, long int M, long int N, long int K,
	double alpha, double* A, long int ldA, double* B, long int ldB, double beta, double* C, long int ldC, ATC_p predef_control_values);

/// The PARALia Dgemm implementation.
ATC_p PARALiaSgemm(char TransA,  char TransB, long int M, long int N, long int K,
	float alpha, float* A, long int ldA, float* B, long int ldB, float beta, float* C, long int ldC);

/// A modification of PARALiaDgemm but with a given T (mainly for performance/debug purposes)
ATC_p PARALiaSgemmControled(char TransA,  char TransB, long int M, long int N, long int K,
	float alpha, float* A, long int ldA, float* B, long int ldB, float beta, float* C, long int ldC, ATC_p predef_control_values);

/// The PARALia Daxpy implementation.
ATC_p PARALiaDaxpy(long int N, VALUE_TYPE alpha,
	VALUE_TYPE* x, long int incx, VALUE_TYPE* y, long int incy);

/// A modification of PARALiaDaxpy but with a given T (mainly for performance/debug purposes)
ATC_p PARALiaDaxpyControled(long int N, VALUE_TYPE alpha,
	VALUE_TYPE* x, long int incx, VALUE_TYPE* y, long int incy, ATC_p predef_control_values);

///Deallocates the GPU-allocated cache buffer at target device
void CoCopeLiaDevCacheFree(short dev_id);

#endif
