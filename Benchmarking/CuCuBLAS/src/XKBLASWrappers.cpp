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


double XKBLASDgemmWrap(char TransA,  char TransB, size_t M, size_t N, size_t K, double alpha, double* A, size_t ldA, double* B, size_t ldB, double beta, double* C, size_t ldC, size_t T, double cpu_ratio, short dev_num, int dev_ids[]){
	short lvl = 1; 
	double total_t = csecond();
#ifdef DEBUG
	lprintf(lvl-1, "|-----> BLASxDgemmWrap(%c,%c,%zu,%zu,%zu,%lf,A(%d),%zu,B(%d),%zu,%lf,C(%d),%zu)\n", 
		TransA, TransB, M, N, K, alpha, CoCopeLia_ptr_check_cuda_9_2(A), ldA,
		CoCopeLia_ptr_check_cuda_9_2(B), ldB, beta, CoCopeLia_ptr_check_cuda_9_2(C), ldC);
#endif

#ifdef TEST
	lprintf(lvl-1, "|-----> BLASxDgemmWrap\n");
	double cpu_timer = csecond();
#endif
	CBLAS_TRANSPOSE cpu_op_A = OpCharToCblas(TransA), cpu_op_B = OpCharToCblas(TransB);
	xkblas_dgemm_async(cpu_op_A,cpu_op_B,M,N,K,&alpha,A,ldA, B,ldB, &beta,C, ldC); //, T, dev_num, dev_ids);
	cudaCheckErrors();
#ifdef TEST
	cpu_timer = csecond() - cpu_timer; 
	lprintf(lvl, "BLASx execution time -> t_kernel = %lf ms\n", cpu_timer*1000);
#endif

	cudaCheckErrors();
	total_t = csecond() - total_t;
	return total_t;

}

void XKBLASFlushGPUBuf(short dev_num, int dev_ids[] ){
	;// TODO: Is there anything to flush actually?
}


