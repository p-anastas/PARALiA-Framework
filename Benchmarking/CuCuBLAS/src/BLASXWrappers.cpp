///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief BLAS lvl 3 wrappers for benchmarks.
///
#include <cassert>
#include <cublasXt.h>
#include <cblas.h>

#include "backend_wrappers.hpp"
#include <BLASxModifiedcblas.h>
#include "CoCoPeLia.hpp"
#include "unihelpers.hpp"

void BLASxFlushGPUBuf(short dev_num, int dev_ids[] ){
	BLASx_LRU_free(dev_num, dev_ids);
}

double BLASxDgemmWrap(char TransA, char TransB, long int M, long int N, long int K, double alpha, double* A, long int ldA, double* B, long int ldB, double beta, double* C, long int ldC, long int T, double cpu_ratio, short dev_num, int dev_ids[] ){
	short lvl = 1;
	double total_t = csecond();
#ifdef DEBUG
	lprintf(lvl-1, "|-----> BLASxDgemmWrap(%c,%c,%zu,%zu,%zu,%lf,A(%d),%zu,B(%d),%zu,%lf,C(%d),%zu)\n",
		TransA, TransB, M, N, K, alpha, CoCoGetPtrLoc(A), ldA,
		CoCoGetPtrLoc(B), ldB, beta, CoCoGetPtrLoc(C), ldC);
#endif

#ifdef TEST
	lprintf(lvl-1, "|-----> BLASxDgemmWrap\n");
	double cpu_timer = csecond();
#endif
	CBLAS_TRANSPOSE cpu_op_A = OpCharToCblas(TransA), cpu_op_B = OpCharToCblas(TransB);
	BLASx_dgemm(CblasColMajor,cpu_op_A,cpu_op_B,M,N,K,alpha,A,ldA, B,ldB, beta,C, ldC, T, dev_num, dev_ids);
	CoCoSyncCheckErr();
#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "BLASx execution time -> t_kernel = %lf ms\n", cpu_timer*1000);
#endif

	CoCoSyncCheckErr();
	total_t = csecond() - total_t;
	return total_t;

}

double BLASxExDgemmWrap(char TransA, char TransB, long int M, long int N, long int K, double alpha, double* A, long int ldA, double* B, long int ldB, double beta, double* C, long int ldC, long int T, double cpu_ratio, short dev_num, int dev_ids[] ){
	short lvl = 1;
	double total_t = csecond();
#ifdef DEBUG
	lprintf(lvl-1, "|-----> BLASxExDgemmWrap(%c,%c,%zu,%zu,%zu,%lf,A(%d),%zu,B(%d),%zu,%lf,C(%d),%zu)\n",
		TransA, TransB, M, N, K, alpha, CoCoGetPtrLoc(A), ldA,
		CoCoGetPtrLoc(B), ldB, beta, CoCoGetPtrLoc(C), ldC);
#endif

#ifdef TEST
	lprintf(lvl-1, "|-----> BLASxExDgemmWrap\n");
	double cpu_timer = csecond();
#endif
	CBLAS_TRANSPOSE cpu_op_A = OpCharToCblas(TransA), cpu_op_B = OpCharToCblas(TransB);
	BLASx_gpubuf_dgemm(CblasColMajor,cpu_op_A,cpu_op_B,M,N,K,alpha,A,ldA, B,ldB, beta,C, ldC, T, dev_num, dev_ids);
	CoCoSyncCheckErr();
#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "BLASx execution time -> t_kernel = %lf ms\n", cpu_timer*1000);
#endif

	CoCoSyncCheckErr();
	total_t = csecond() - total_t;
	return total_t;

}
