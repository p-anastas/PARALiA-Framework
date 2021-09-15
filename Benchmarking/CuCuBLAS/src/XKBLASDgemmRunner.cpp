///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The start of Zawarudo
///

#include "unihelpers.hpp"
#include "CoCoPeLia.hpp"
//#include "XKBLASWrapped.hpp"
#include "cuBLASXtWrapped.hpp"

#define KAAPI_NO_DEFAULT_BLAS_ENUM
#ifdef __cplusplus
extern "C"{
#endif 
#include "xkblas.h" // Must be put after unihelpers cause of cblas def clashes...
#ifdef __cplusplus
}
#endif

#define CBLASXT_MAX_SAFE_TILE 10000

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
	//cblas_dgemm(CblasColMajor, cpu_op_A,cpu_op_B,M,N,K,alpha,A,ldA, B,ldB, beta,C, ldC); //, T, dev_num, dev_ids);
	xkblas_dgemm_async(cpu_op_A,cpu_op_B,M,N,K,&alpha,A,ldA, B,ldB, &beta,C, ldC);
	xkblas_memory_coherent_async(0, 0, M, N, C, ldC, sizeof(double));
	xkblas_sync();
	cudaCheckErrors();
#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "BLASx execution time -> t_kernel = %lf ms\n", cpu_timer*1000);
#endif

	cudaCheckErrors();
	total_t = csecond() - total_t;
	return total_t;

}

void XKBLASFlushGPUBuf(){
	xkblas_memory_free();
}

int main(const int argc, const char *argv[]) {


	char TransA, TransB; 
  	double alpha, beta;
	size_t M, N, K;
	short A_loc, B_loc, C_loc, C_out_loc;

	CoControl_p predef_control_values = NULL, return_values = (CoControl_p) malloc(sizeof(struct CoControl)) ;
	ParseInputLvl3(argc, argv, &predef_control_values, &TransA, &TransB, &alpha, &beta, &M, &N, &K, &A_loc, &B_loc, &C_loc, &C_out_loc);

	char *filename = (char *) malloc(256* sizeof(char));
	if (predef_control_values!= NULL){ 
		if(predef_control_values->T > 0) {
			if (predef_control_values->T > M || predef_control_values->T > N || predef_control_values->T > K) error("Given Tin=%d bigger than problem dim\n", predef_control_values->T); 
			else if (predef_control_values->T > M/1.5 && predef_control_values->T > N/1.5 && predef_control_values->T > K/1.5) error("Given Tin=%d bigger than all problem dims/1.5\n", predef_control_values->T);
			sprintf(filename, "%s/XKBLASDgemmRunner_predefined_vals_%s.log", TESTLIBDIR, VERSION);	
		}
	}
	else sprintf(filename, "%s/XKBLASDgemmRunner_%s.log", TESTLIBDIR, VERSION);

	size_t XKBLAS_tile;
	if (predef_control_values!= NULL && predef_control_values->T > 0) error("XKBLASDgemmRunner: XKBLAS not modified to accept T\n");
	else return_values->T = XKBLAS_tile = -1; // The best performing static tile for our machine
	double cpu_ratio;
	if (predef_control_values!= NULL && predef_control_values->cpu_ratio > 0) error("XKBLASDgemmRunner: XKBLAS not modified to accept cpu_ratio\n");
	else return_values->cpu_ratio = cpu_ratio = -1; 
	int dev_num, *dev_ids;
	if (predef_control_values!= NULL && predef_control_values->dev_num > 0) error("XKBLASDgemmRunner: XKBLAS not modified to accept devices from within script\n");
	else{
		return_values->dev_num = -1;
		dev_ids = (int*) malloc(dev_num*sizeof(int));
		for (int i = 0; i < dev_num; i++) dev_ids[i] = -1;
		return_values->dev_ids = dev_ids;
	}
#ifdef CHECKLOG
	CheckLogLvl3(filename, return_values, TransA, TransB, alpha, beta, M, N, K, A_loc, B_loc, C_loc, C_out_loc);
#endif
	/// Matrix Layouts for CPU GEMM
	CBLAS_TRANSPOSE cpu_op_A, cpu_op_B;    // CblasNoTrans, CblasTrans
	cublasOperation_t gpu_op_A, gpu_op_B; // CUBLAS_OP_N, CUBLAS_OP_T
	
	size_t ldA, ldB, ldC = M;
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

	cudaCheckErrors();
	cpu_timer  = csecond() - cpu_timer;
	fprintf(stderr, "done.\nAlloc time:\t%lf ms\n\n",  cpu_timer  * 1000);
	cpu_timer = csecond();
	fprintf(stderr, "Initializing to random values (VALIDATE)..."); 
	CoCoVecInit(A, K * M, 42, A_loc);
	CoCoVecInit(B, K * N, 43, B_loc);
	CoCoVecInit(C, M * N, 44, C_loc);
	cudaCheckErrors();
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
	int dev_ids[DEV_NUM];
	for (int i = 0; i < DEV_NUM; i++) dev_ids[i] = i; 
	cuBLASXtDgemmWrap(TransA,  TransB, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC,  (size_t) fmin(fmin(fmin(M,N),K)/2,CBLASXT_MAX_SAFE_TILE), 0, DEV_NUM, dev_ids);
	}
	CoCoMemcpy(C_out1, C,  M * N *sizeof(double), -2, C_loc);
 	CoCoMemcpy(C, C_buf,  M * N *sizeof(double), C_loc, -2);

	// Call for Validate
	cpu_timer = csecond();
	XKBLASDgemmWrap(TransA,  TransB, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC,  XKBLAS_tile, cpu_ratio, dev_num, dev_ids);
	cudaCheckErrors();
	cpu_timer  = csecond() - cpu_timer;
	XKBLASFlushGPUBuf();

	CoCoMemcpy(C_out, C,  M * N *sizeof(double), -2, C_loc);

 	if(Dtest_equality(C_out1, C_out, M * N) < 9) error("Insufficient accuracy for benchmarks\n");

 	CoCoMemcpy(C, C_buf,  M * N *sizeof(double), C_loc, -2);
	free(C_out);
	free(C_out1);
	free(C_buf);
#endif


	// First call for additional overhead counting
	cpu_timer = csecond();
	XKBLASDgemmWrap(TransA,  TransB, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC,  XKBLAS_tile, cpu_ratio, dev_num, dev_ids);
	cudaCheckErrors();
	cpu_timer  = csecond() - cpu_timer;
	StoreLogLvl3(filename, return_values, TransA, TransB, alpha, beta, M, N, K, A_loc, B_loc, C_loc, C_out_loc, cpu_timer); 
	xkblas_memory_invalidate_caches();
	double first_over_t = cpu_timer; 

	double min_t = first_over_t, max_t = 0, avg_t = 0;
	cpu_timer = csecond();
	short bench_it = 100;
	if ( M >= 8192 || N >= 8192 || K >= 8192) bench_it = 10; 
	for(int it = 0; it < bench_it; it++){
		cpu_timer = csecond();
		XKBLASDgemmWrap(TransA,  TransB, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC,  XKBLAS_tile, cpu_ratio, dev_num, dev_ids);
		cudaCheckErrors();
		cpu_timer = csecond() - cpu_timer;
		xkblas_memory_invalidate_caches();
		cudaCheckErrors();
		StoreLogLvl3(filename, return_values, TransA, TransB, alpha, beta, M, N, K, A_loc, B_loc, C_loc, C_out_loc, cpu_timer); 
		if ( cpu_timer < min_t ) min_t = cpu_timer;
		if ( cpu_timer > max_t ) max_t = cpu_timer;
		avg_t += cpu_timer;
	}
	avg_t/=bench_it;
	fprintf(stderr, "XKBLAS (%s):\n\tfirst_it_t = %lf ms ( %lf Gflops/s )\n\tavg_t = %lf ms ( %lf Gflops/s )\n\tmin_t = %lf ms ( %lf Gflops/s )\n\tmax_t = %lf ms ( %lf Gflops/s )\n", 
	CoControlPrint(return_values),
	first_over_t  * 1000, Gval_per_s(dgemm_flops(M,N,K),first_over_t),
	avg_t  * 1000, Gval_per_s(dgemm_flops(M,N,K),avg_t),
	min_t  * 1000, Gval_per_s(dgemm_flops(M,N,K),min_t),
	max_t  * 1000, Gval_per_s(dgemm_flops(M,N,K),max_t));
		
	XKBLASFlushGPUBuf();
	cudaCheckErrors();
	CoCoFree(A, A_loc);
	CoCoFree(B, B_loc);
	CoCoFree(C, C_loc); 
	return 0;
}
