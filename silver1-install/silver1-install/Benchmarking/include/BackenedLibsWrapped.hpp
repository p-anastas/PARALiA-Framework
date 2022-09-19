///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The external wrapper for CoCoPelia + wrapped cuBLASXt
///
#ifndef BACKENDLIBSWRAPPED_H
#define BACKENDLIBSWRAPPED_H

/// cuBLASXt wrappers of Dgemm
double cuBLASXtDgemmWrap(char TransA,  char TransB, size_t M, size_t N, size_t K,
  double alpha, double* A, size_t ldA, double* B, size_t ldB, double beta, double* C,
  size_t ldC, size_t T, double cpu_ratio, short dev_num, int dev_ids[]);

/// cuBLAS wrappers of Daxpy
double cuBLASDaxpyWrap(size_t N, double alpha, double* x, size_t incx,
  double* y, size_t incy, double cpu_ratio, short dev_num, int dev_ids[]);

#endif
