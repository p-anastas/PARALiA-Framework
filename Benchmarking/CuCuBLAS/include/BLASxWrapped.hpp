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
double BLASxDgemmWrap(char TransA,  char TransB, long int M, long int N, long int K, double alpha, double* A, long int ldA, double* B, long int ldB, double beta, double* C, long int ldC, long int T, double cpu_ratio, short dev_num, int dev_ids[]);
double BLASxExDgemmWrap(char TransA,  char TransB, long int M, long int N, long int K, double alpha, double* A, long int ldA, double* B, long int ldB, double beta, double* C, long int ldC, long int T, double cpu_ratio, short dev_num, int dev_ids[]);

#endif
