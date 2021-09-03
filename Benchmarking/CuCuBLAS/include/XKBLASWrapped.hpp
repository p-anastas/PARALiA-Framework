///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The external wrapper for CoCoPelia + wrapped cuBLASXt
///
#ifndef XKBLASWRAPPED_H
#define XKBLASWRAPPED_H

void XKBLASFlushGPUBuf(short dev_num, int dev_ids[] );
/// XKBLAS wrappers for performance evaluation
double XKBLASDgemmWrap(char TransA,  char TransB, size_t M, size_t N, size_t K, double alpha, double* A, size_t ldA, double* B, size_t ldB, double beta, double* C, size_t ldC, size_t T, double cpu_ratio, short dev_num, int dev_ids[]);

#endif
