///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The external wrapper for PARALiA + wrapped cuBLASXt
///
#ifndef PARALiA_H
#define PARALiA_H

#include "Autotuner.hpp"

typedef class Buffer* Buffer_p;
typedef class Decomposer* Decomposer_p; 
class Subkernel;

typedef class ProblemMetadata{
public:	
	ATC_p autotuner;
	const char* problem_name;
	void* problem_wrap; 
	int decom_num;
	Decomposer_p decom[10];
	int sk_num; 
	Subkernel** subkernel_list; 
	int sk_dev_num[LOC_NUM]; 
	Subkernel** subkernel_dev_list[LOC_NUM]; 
	Buffer_p SAB[LOC_NUM]; 
}* PMD_p; 

extern PMD_p PMD_cache[PROBLEM_MD_CACHE]; 
extern int PMD_cache_entries; 

/// The PARALiA Dgemm implementation.
ATC_p PARALiADgemm(char TransA,  char TransB, long int M, long int N, long int K,
	double alpha, double* A, long int ldA, double* B, long int ldB, double beta, double* C, long int ldC);

/// A modification of PARALiADgemm but with a given T (mainly for performance/debug purposes)
ATC_p PARALiADgemmControled(char TransA,  char TransB, long int M, long int N, long int K,
	double alpha, double* A, long int ldA, double* B, long int ldB, double beta, double* C, long int ldC, ATC_p predef_control_values);

/// The PARALiA Dgemm implementation.
ATC_p PARALiASgemm(char TransA,  char TransB, long int M, long int N, long int K,
	float alpha, float* A, long int ldA, float* B, long int ldB, float beta, float* C, long int ldC);

/// A modification of PARALiADgemm but with a given T (mainly for performance/debug purposes)
ATC_p PARALiASgemmControled(char TransA,  char TransB, long int M, long int N, long int K,
	float alpha, float* A, long int ldA, float* B, long int ldB, float beta, float* C, long int ldC, ATC_p predef_control_values);

ATC_p PARALiADgemv(char TransA, long int M, long int N,	double alpha, double* A, long int ldA,
	double* x, long int incx, double beta, double* y, long int incy);

/// A modification of PARALiADgemv but with a given T (mainly for performance/debug purposes)
ATC_p PARALiADgemvControled(char TransA, long int M, long int N, double alpha, double* A, long int ldA,
	double* x, long int incx, double beta, double* y, long int incy, ATC_p predef_control_values);

/// The PARALiA Daxpy implementation.
ATC_p PARALiADaxpy(long int N, double alpha,
	double* x, long int incx, double* y, long int incy);

/// A modification of PARALiADaxpy but with a given T (mainly for performance/debug purposes)
ATC_p PARALiADaxpyControled(long int N, double alpha,
	double* x, long int incx, double* y, long int incy, ATC_p predef_control_values);

/// The PARALiA Ddot implementation.
ATC_p PARALiADdot(long int N, double* x, long int incx,
	double* y, long int incy, double* result);

/// A modification of PARALiADdot but with a given T (mainly for performance/debug purposes)
ATC_p PARALiADdotControled(long int N, double* x, long int incx,
	double* y, long int incy, double* result, ATC_p predef_control_values);

///Deallocates the GPU-allocated cache buffer at target device
void PARALiADevCacheFree(short dev_id);

#endif
