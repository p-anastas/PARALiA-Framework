///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The external wrapper for CoCoPelia + wrapped cuBLASXt
///
#ifndef BACKENDLIBSWRAPPED_H
#define BACKENDLIBSWRAPPED_H

/// cuBLASXt wrappers of Dgemm
double cuBLASXtDgemmWrap(char TransA,  char TransB, long int M, long int N, long int K,
  double alpha, double* A, long int ldA, double* B, long int ldB, double beta, double* C,
  long int ldC, long int T, double cpu_ratio, short dev_num, int dev_ids[]);

/// cuBLAS wrappers of Daxpy
double cuBLASDaxpyWrap(long int N, double alpha, double* x, long int incx,
  double* y, long int incy, double cpu_ratio, short dev_num, int dev_ids[]);

#endif
