///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief BLAS lvl 3 wrappers for benchmarks.
///
#include <cassert>
#include <cublasXt.h>

#include "CoCoPeLiaLibBackened.hpp"
#include "unihelpers.hpp"
#include "xkblas.h" // Must be put after unihelpers cause of cblas def clashes...

void xkHopMemcpyPrint(); 

double XKBLASDgemmWrap(char TransA,  char TransB, long int M, long int N, long int K, double alpha, double* A, long int ldA, double* B, long int ldB, double beta, double* C, long int ldC, long int T, double cpu_ratio, short dev_num, int dev_ids[]){
	short lvl = 1;
	double total_t = csecond();
#ifdef DEBUG
	lprintf(lvl-1, "|-----> XKBLASDgemmWrap(%c,%c,%zu,%zu,%zu,%lf,A(%d),%zu,B(%d),%zu,%lf,C(%d),%zu)\n",
		TransA, TransB, M, N, K, alpha, CoCopeLia_ptr_check_cuda_9_2(A), ldA,
		CoCopeLia_ptr_check_cuda_9_2(B), ldB, beta, CoCopeLia_ptr_check_cuda_9_2(C), ldC);
#endif

#ifdef TEST
	lprintf(lvl-1, "|-----> XKBLASDgemmWrap\n");
	double cpu_timer = csecond();
#endif
	CBLAS_TRANSPOSE cpu_op_A = OpCharToCblas(TransA), cpu_op_B = OpCharToCblas(TransB);
	cblas_dgemm(CblasColMajor, cpu_op_A,cpu_op_B,M,N,K,alpha,A,ldA, B,ldB, beta,C, ldC); //, T, dev_num, dev_ids);
	cudaCheckErrors();
#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "XKBLAS execution time -> t_kernel = %lf ms\n", cpu_timer*1000);
#endif
	cudaCheckErrors();
	total_t = csecond() - total_t;
#ifdef TTEST
	xkHopMemcpyPrint();
#endif

	return total_t;

}

void XKBLASFlushGPUBuf(short dev_num, int dev_ids[] ){
	;// TODO: Is there anything to flush actually?
}
