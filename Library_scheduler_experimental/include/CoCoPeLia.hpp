///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The external wrapper for CoCoPelia + wrapped cuBLASXt
///
#ifndef COCOPELIA_H
#define COCOPELIA_H

typedef struct CoControl{
	int T = 0;
	int* dev_ids = NULL;
	int dev_num = -1;
	long long cache_limit = 0;
}* CoControl_p;

/// Return a string represenation of the given CoControl
char* CoControlPrint(CoControl_p input);

/// Return a string with the active Cmake implemetation flag used
char* CoCoImplementationPrint();

/// Return a string with the active Cmake Subkernel Distribution flag
char* CoCoDistributionPrint();

/// The CoCopeLia Dgemm implementation. A prediction model is used to select a tile from the micro-benchmarked tile candidates with CoCopeLia_optimize3.
CoControl_p CoCopeLiaDgemm(char TransA,  char TransB, size_t M, size_t N, size_t K,
	double alpha, double* A, size_t ldA, double* B, size_t ldB, double beta, double* C, size_t ldC);

/// A modification of CoCopeLiaDgemm but with a given T (mainly for performance/debug purposes)
CoControl_p CoCopeLiaDgemmControled(char TransA,  char TransB, size_t M, size_t N, size_t K,
	double alpha, double* A, size_t ldA, double* B, size_t ldB, double beta, double* C, size_t ldC, CoControl_p predef_control_values);

/// The CoCopeLia Dgemm implementation. A prediction model is used to select a tile from the micro-benchmarked tile candidates with CoCopeLia_optimize3.
CoControl_p CoCopeLiaDaxpy(size_t N, VALUE_TYPE alpha,
	VALUE_TYPE* x, size_t incx, VALUE_TYPE* y, size_t incy);

/// A modification of CoCopeLiaDgemm but with a given T (mainly for performance/debug purposes)
CoControl_p CoCopeLiaDaxpyControled(size_t N, VALUE_TYPE alpha,
	VALUE_TYPE* x, size_t incx, VALUE_TYPE* y, size_t incy, CoControl_p predef_control_values);

///Deallocates the GPU-allocated cache buffer at target device
void CoCopeLiaDevCacheFree(short dev_id);

#endif
