///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The external wrapper for CoCoPelia + wrapped cuBLASXt
///
#ifndef BLASXWRAPPED_H
#define BLASXWRAPPED_H

#include <cblas.h>

void BLASxFlushGPUBuf(short dev_num, int dev_ids[] );
/// BLASx wrappers for performance evaluation
double BLASxDgemmWrap(char TransA,  char TransB, size_t M, size_t N, size_t K, double alpha, double* A, size_t ldA, double* B, size_t ldB, double beta, double* C, size_t ldC, size_t T, double cpu_ratio, short dev_num, int dev_ids[]);
double BLASxExDgemmWrap(char TransA,  char TransB, size_t M, size_t N, size_t K, double alpha, double* A, size_t ldA, double* B, size_t ldB, double beta, double* C, size_t ldC, size_t T, double cpu_ratio, short dev_num, int dev_ids[]);

#endif
