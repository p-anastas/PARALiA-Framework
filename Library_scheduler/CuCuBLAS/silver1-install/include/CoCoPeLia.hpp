///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The external wrapper for CoCoPelia + wrapped cuBLASXt
///
#ifndef COCOPELIA_H
#define COCOPELIA_H

/// The CoCopeLia Dgemm implementation. A prediction model is used to select a tile from the micro-benchmarked tile candidates with CoCopeLia_optimize3.
void CoCopeLiaDgemm(char TransA,  char TransB, size_t M, size_t N, size_t K, double alpha, double* A, size_t ldA, double* B, size_t ldB, double beta, double* C, size_t ldC);

/// A modification of CoCopeLiaDgemm but with a given T (mainly for performance/debug perpuses)
void CoCopeLiaDgemmTin(char TransA,  char TransB, size_t M, size_t N, size_t K, double alpha, double* A, size_t ldA, double* B, size_t ldB, double beta, double* C, size_t ldC, size_t Tin);

///Clears all GPU mem allocation done by CoCopeLiaDgemm at target device
void CoCopeLiaDgemm_flush_gpu_mem_buf(short dev_id);
#endif
