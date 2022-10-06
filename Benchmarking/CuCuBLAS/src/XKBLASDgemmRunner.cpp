///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The start of Zawarudo
///

#include "backend_wrappers.hpp"
#include "unihelpers.hpp"
#include "CoCoPeLia.hpp"
//#include "XKBLASWrapped.hpp"
#include "BackenedLibsWrapped.hpp"
#include "Testing.hpp"

#define KAAPI_NO_DEFAULT_BLAS_ENUM
#ifdef __cplusplus
extern "C"{
#endif
#include "xkblas.h" // Must be put after unihelpers cause of cblas def clashes...
#ifdef __cplusplus
}
#endif

#define CBLASXT_MAX_SAFE_TILE 10000

double XKBLASDgemmWrap(char TransA,  char TransB, long int M, long int N, long int K, double alpha, double* A, long int ldA, double* B, long int ldB, double beta, double* C, long int ldC, long int T, double cache_limit, short dev_num, int dev_ids[]){
	short lvl = 1;
	double total_t = csecond();
#ifdef DEBUG
	lprintf(lvl-1, "|-----> XKBLASDgemmWrap(%c,%c,%zu,%zu,%zu,%lf,A(%d),%zu,B(%d),%zu,%lf,C(%d),%zu)\n",
		TransA, TransB, M, N, K, alpha, CoCoGetPtrLoc(A), ldA,
		CoCoGetPtrLoc(B), ldB, beta, CoCoGetPtrLoc(C), ldC);
#endif

#ifdef TEST
	lprintf(lvl-1, "|-----> XKBLASDgemmWrap\n");
	double cpu_timer = csecond();
#endif
	CBLAS_TRANSPOSE cpu_op_A = OpCharToCblas(TransA), cpu_op_B = OpCharToCblas(TransB);
	//cblas_dgemm(CblasColMajor, cpu_op_A,cpu_op_B,M,N,K,alpha,A,ldA, B,ldB, beta,C, ldC); //, T, dev_num, dev_ids);
	xkblas_dgemm_async(cpu_op_A,cpu_op_B,M,N,K,&alpha,A,ldA, B,ldB, &beta,C, ldC);
	xkblas_memory_coherent_async(0, 0, M, N, C, ldC, sizeof(double));
	xkblas_sync();
	CoCoSyncCheckErr();
#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "XKBLAS execution time -> t_kernel = %lf ms\n", cpu_timer*1000);
#endif

	CoCoSyncCheckErr();
	total_t = csecond() - total_t;
	return total_t;

}

void XKBLASFlushGPUBuf(){
	xkblas_memory_free();
}

int main(const int argc, const char *argv[]) {


	char TransA, TransB;
  	double alpha, beta;
	long int M, N, K;
	short A_loc, B_loc, C_loc, C_out_loc;

	ATC_p predef_control_values = NULL;
	ParseInputLvl3(argc, argv, &predef_control_values, &TransA, &TransB, &alpha, &beta, &M, &N, &K, &A_loc, &B_loc, &C_loc, &C_out_loc);

	char *filename = (char *) malloc(2048* sizeof(char));
	if (predef_control_values!= NULL){
		if(predef_control_values->T > 0) {
			if (predef_control_values->T > M || predef_control_values->T > N || predef_control_values->T > K)
				error("Given Tin=%ld bigger than problem dim\n", predef_control_values->T);
			else if (predef_control_values->T > M/1.5 && predef_control_values->T > N/1.5 && predef_control_values->T > K/1.5)
				error("Given Tin=%ld bigger than all problem dims/1.5\n", predef_control_values->T);
		}
			sprintf(filename, "%s/XKBLASDgemmRunner_predefined_vals_%s_%s_%s.log",
				TESTLIBDIR, CoCoDistributionPrint(), CoCoImplementationPrint(), VERSION);
	}
	else sprintf(filename, "%s/XKBLASDgemmRunner_%s_%s_%s.log",
		TESTLIBDIR, CoCoDistributionPrint(), CoCoImplementationPrint(), VERSION);

	long int XKBLAS_tile;
	if (predef_control_values!= NULL && predef_control_values->T > 0)
		error("XKBLASDgemmRunner: XKBLAS not modified to accept T\n");
	else XKBLAS_tile = 1600; // The best performing static tile for our machine
	double cache_limit;
	if (predef_control_values!= NULL && predef_control_values->cache_limit > 0)
		error("XKBLASDgemmRunner: XKBLAS not modified to accept cache_limit\n");
	else cache_limit = -1;
	int dev_num, *dev_ids;
	if (predef_control_values!= NULL && predef_control_values->active_unit_num > 0)
		error("XKBLASDgemmRunner: XKBLAS not modified to accept devices from within script\n");
	else{
#ifdef ENABLE_CPU_WORKLOAD
		dev_num = DEV_NUM-1; /// Don't use CPU.
#else
		dev_num = DEV_NUM;
#endif
		dev_ids = (int*) malloc(dev_num*sizeof(int));
		for (int i = 0; i < dev_num; i++) dev_ids[i] = deidxize(i);
	}
	if(!predef_control_values) predef_control_values = new ATC();
#ifdef CHECKLOG
	CheckLogLvl3(filename, predef_control_values, TransA, TransB, alpha, beta, M, N, K, A_loc, B_loc, C_loc, C_out_loc);
#endif
	/// Matrix Layouts for CPU GEMM
	CBLAS_TRANSPOSE cpu_op_A, cpu_op_B;    // CblasNoTrans, CblasTrans
	cublasOperation_t gpu_op_A, gpu_op_B; // CUBLAS_OP_N, CUBLAS_OP_T

	long int ldA, ldB, ldC = M;
	TransposeTranslate(TransA, &cpu_op_A, &gpu_op_A, &ldA, M, K);
	TransposeTranslate(TransB, &cpu_op_B, &gpu_op_B, &ldB, K, N);

	/// Local Timers
	double cpu_timer = csecond();

	fprintf(stderr, "\nAllocating memory...");

	double *A, *B, *C;
	// allocate in device if loc = 0, otherwise allocate in pinned memory for benchmarks
	A = (double*) CoCoMalloc(M * K*sizeof(double), A_loc);
	B = (double*) CoCoMalloc(N * K*sizeof(double), B_loc);
	C = (double*) CoCoMalloc(M * N*sizeof(double), C_loc);

	CoCoSyncCheckErr();
	cpu_timer  = csecond() - cpu_timer;
	fprintf(stderr, "done.\nAlloc time:\t%lf ms\n\n",  cpu_timer  * 1000);
	cpu_timer = csecond();
	fprintf(stderr, "Initializing to random values (VALIDATE)...");
	CoCoVecInit(A, K * M, 42, A_loc);
	CoCoVecInit(B, K * N, 43, B_loc);
	CoCoVecInit(C, M * N, 44, C_loc);
	CoCoSyncCheckErr();
	cpu_timer  = csecond() - cpu_timer ;
	fprintf(stderr, "done.\nInit time:\t%lf ms\n\n",  cpu_timer  * 1000);

#ifdef RUNVALIDATION
	double *C_out, *C_out1, *C_buf;
	C_out  = (double*) malloc(M * N*sizeof(double));
	C_out1  = (double*) malloc(M * N*sizeof(double));
	C_buf  = (double*) malloc(M * N*sizeof(double));

	CoCoMemcpy(C_buf, C,  M * N *sizeof(double), -2, C_loc);

	// Validate with cuBLASXt (questionable but CPU validation can be slower by at least a factor)
	{

	cuBLASXtDgemmWrap(TransA,  TransB, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC,  (long int) fmin(fmin(fmin(M,N),K)/2,CBLASXT_MAX_SAFE_TILE), 0, dev_num, dev_ids);
	}
	CoCoMemcpy(C_out1, C,  M * N *sizeof(double), -2, C_loc);
 	CoCoMemcpy(C, C_buf,  M * N *sizeof(double), C_loc, -2);

	// Call for Validate
	cpu_timer = csecond();
	XKBLASDgemmWrap(TransA,  TransB, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC,  XKBLAS_tile, cache_limit, dev_num, dev_ids);
	CoCoSyncCheckErr();
	cpu_timer  = csecond() - cpu_timer;
	XKBLASFlushGPUBuf();

	CoCoMemcpy(C_out, C,  M * N *sizeof(double), -2, C_loc);

 	if(Dtest_equality(C_out1, C_out, M * N) < 9) error("Insufficient accuracy for benchmarks\n");

 	CoCoMemcpy(C, C_buf,  M * N *sizeof(double), C_loc, -2);
	free(C_out);
	free(C_out1);
	//free(C_buf);
#endif


	// First call for additional overhead counting
	cpu_timer = csecond();
	XKBLASDgemmWrap(TransA,  TransB, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC,  XKBLAS_tile, cache_limit, dev_num, dev_ids);
	CoCoSyncCheckErr();
	cpu_timer  = csecond() - cpu_timer;
	StoreLogLvl3(filename, predef_control_values, TransA, TransB, alpha, beta, M, N, K, A_loc, B_loc, C_loc, C_out_loc, cpu_timer);
	xkblas_memory_invalidate_caches();
	double first_over_t = cpu_timer;

	double min_t = first_over_t, max_t = 0, avg_t = 0;
	cpu_timer = csecond();
	short bench_it = 100;
	if ( M >= 8192 || N >= 8192 || K >= 8192) bench_it = 10;
	for(int it = 0; it < bench_it; it++){
		cpu_timer = csecond();
		XKBLASDgemmWrap(TransA,  TransB, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC,  XKBLAS_tile, cache_limit, dev_num, dev_ids);
		CoCoSyncCheckErr();
		cpu_timer = csecond() - cpu_timer;
		xkblas_memory_invalidate_caches();
		CoCoSyncCheckErr();
		StoreLogLvl3(filename, predef_control_values, TransA, TransB, alpha, beta, M, N, K, A_loc, B_loc, C_loc, C_out_loc, cpu_timer);
		if ( cpu_timer < min_t ) min_t = cpu_timer;
		if ( cpu_timer > max_t ) max_t = cpu_timer;
		avg_t += cpu_timer;
	}
	avg_t/=bench_it;
	fprintf(stderr, "XKBLAS (%s):\n\tfirst_it_t = %lf ms ( %lf Gflops/s )\n\tavg_t = %lf ms ( %lf Gflops/s )\n\tmin_t = %lf ms ( %lf Gflops/s )\n\tmax_t = %lf ms ( %lf Gflops/s )\n",
	predef_control_values->print_csv(),
	first_over_t  * 1000, Gval_per_s(gemm_flops(M,N,K),first_over_t),
	avg_t  * 1000, Gval_per_s(gemm_flops(M,N,K),avg_t),
	min_t  * 1000, Gval_per_s(gemm_flops(M,N,K),min_t),
	max_t  * 1000, Gval_per_s(gemm_flops(M,N,K),max_t));

	XKBLASFlushGPUBuf();
	CoCoSyncCheckErr();
	CoCoFree(A, A_loc);
	CoCoFree(B, B_loc);
	CoCoFree(C, C_loc);
	return 0;
}
