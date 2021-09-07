///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The start of Zawarudo
///

#include "unihelpers.hpp"
#include "CoCoPeLia.hpp"
#include "cuBLASXtWrapped.hpp"
//#include "testing.hpp"

#define CBLASXT_MAX_SAFE_TILE 10000

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
			else if (predef_control_values->T > CBLASXT_MAX_SAFE_TILE) error("Given Tin=%d bigger than CBLASXT_MAX_SAFE_TILE\n", predef_control_values->T);
		}
		sprintf(filename, "%s/cuBLASXtDgemmRunner_predefined_vals_%s.log", TESTLIBDIR, VERSION);	
	}
	else sprintf(filename, "%s/cuBLASXtDgemmRunner_%s.log", TESTLIBDIR, VERSION);

	size_t cublasXt_tile;
	if (predef_control_values!= NULL && predef_control_values->T > 0) return_values->T = cublasXt_tile = predef_control_values->T;
	else return_values->T = cublasXt_tile = (size_t) fmin(fmin(fmin(M,N),K)/2,CBLASXT_MAX_SAFE_TILE); 
	double cpu_ratio;
	if (predef_control_values!= NULL && predef_control_values->cpu_ratio > 0) return_values->cpu_ratio = cpu_ratio = predef_control_values->cpu_ratio;
	else return_values->cpu_ratio = cpu_ratio = 0; 
	int dev_num, *dev_ids;
	if (predef_control_values!= NULL && predef_control_values->dev_num > 0){
		return_values->dev_num = dev_num = predef_control_values->dev_num;
		return_values->dev_ids = dev_ids = predef_control_values->dev_ids;
	}
	else{
		return_values->dev_num = dev_num = DEV_NUM;
		dev_ids = (int*) malloc(dev_num*sizeof(int));
		for (int i = 0; i < dev_num; i++) dev_ids[i] = i;
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

	// First call with T set to half the smaller problem dim (unless predefined or larger than CBLASXT_MAX_SAFE_TILE) 
	cpu_timer = csecond();
	cuBLASXtDgemmWrap(TransA,  TransB, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC,  cublasXt_tile, cpu_ratio, dev_num, dev_ids);
	cudaCheckErrors();
	cpu_timer  = csecond() - cpu_timer;
	double best_standard_tile_t = cpu_timer; 

	if (!(predef_control_values!= NULL && predef_control_values->T > 0)){
		// Second call with T set to smaller problem dim ( can be better for small problems with fat/thin matrices)
		size_t cublasXt_min_dim = (size_t) fmin(fmin(fmin(M,N),K),CBLASXT_MAX_SAFE_TILE); 
		cpu_timer = csecond();
		cuBLASXtDgemmWrap(TransA,  TransB, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC,  cublasXt_min_dim, cpu_ratio, dev_num, dev_ids);
		cudaCheckErrors();
		cpu_timer  = csecond() - cpu_timer;
		if ( cpu_timer < best_standard_tile_t) {
			cublasXt_tile = cublasXt_min_dim;
			best_standard_tile_t = cpu_timer;
		}
			
	}
#ifdef CHECKLOG
	CheckLogLvl3(filename, return_values, TransA, TransB, alpha, beta, M, N, K, A_loc, B_loc, C_loc, C_out_loc);
#endif
	double cublasXt_t = best_standard_tile_t; 
	if (predef_control_values!= NULL && predef_control_values->T > 0){
		fprintf(stderr,"Running CUBLASXT DGEMM-> M = %zu, N = %zu, K = %zu, T = %zu\n", M, N, K, cublasXt_tile);
		cpu_timer  = csecond();
		cuBLASXtDgemmWrap(TransA, TransB, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC,  cublasXt_tile, cpu_ratio, dev_num, dev_ids);
		cudaCheckErrors();
		cpu_timer  = csecond() - cpu_timer;
		fprintf(stderr, "Total time:\t%lf ms\n", cpu_timer  * 1000);
		if (cublasXt_t > cpu_timer) cublasXt_t = cpu_timer; 
	}
	else {
		for (size_t T_trial = (((size_t)fmax(fmin(fmin(M/8,N/8),K/8),1024))/1024)*1024; T_trial <= fmin(fmin(fmin(M,N),K),CBLASXT_MAX_SAFE_TILE); T_trial+=1024) if (M >= T_trial*1.5 || N >= T_trial*1.5 || K >= T_trial*1.5){
			fprintf(stderr,"Running CUBLASXT DGEMM-> M = %zu, N = %zu, K = %zu, T = %zu\n", M, N, K, T_trial);
			cpu_timer  = csecond();
			cuBLASXtDgemmWrap(TransA,  TransB, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC,  T_trial, cpu_ratio, dev_num, dev_ids);
			cudaCheckErrors();
			cpu_timer  = csecond() - cpu_timer;
			fprintf(stderr, "Total time:\t%lf ms\n", cpu_timer  * 1000);
			if (cpu_timer < cublasXt_t){
				cublasXt_t = cpu_timer;
				cublasXt_tile = T_trial;
			}
		}
		fprintf(stderr, "\nCUBLASXT DGEMM T_best = %zu : t = %lf ms ( %lf Gflops/s )\n\n", cublasXt_tile, cublasXt_t  * 1000, Gval_per_s(dgemm_flops(M,N,K),cublasXt_t));
	}
	return_values->T = cublasXt_tile;
	fprintf(stderr,"Running CUBLASXT DGEMM-> M = %zu, N = %zu, K = %zu T_best = %zu\n", M, N, K, cublasXt_tile);
#ifdef CHECKLOG
	CheckLogLvl3(filename, return_values, TransA, TransB, alpha, beta, M, N, K, A_loc, B_loc, C_loc, C_out_loc);
#endif
	double min_t = cublasXt_t, max_t = 0, avg_t = 0;
	cpu_timer = csecond();
	short bench_it = 100;
	if ( M >= 8192 || N >= 8192 || K >= 8192) bench_it = 10; 
	for(int it = 0; it < bench_it; it++){
		cpu_timer = csecond();
		cuBLASXtDgemmWrap(TransA,  TransB, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC,  cublasXt_tile, cpu_ratio, dev_num, dev_ids);
		cudaCheckErrors();
		cpu_timer = csecond() - cpu_timer;
		StoreLogLvl3(filename, return_values, TransA, TransB, alpha, beta, M, N, K, A_loc, B_loc, C_loc, C_out_loc, cpu_timer); 
		if ( cpu_timer < min_t ) min_t = cpu_timer;
		if ( cpu_timer > max_t ) max_t = cpu_timer;
		avg_t += cpu_timer;
	}
	avg_t/=bench_it;
	fprintf(stderr, "cuBLASXt (%s):\n\tavg_t = %lf ms ( %lf Gflops/s )\n\tmin_t = %lf ms ( %lf Gflops/s )\n\tmax_t = %lf ms ( %lf Gflops/s )\n", 
	CoControlPrint(return_values),
	avg_t  * 1000, Gval_per_s(dgemm_flops(M,N,K),avg_t),
	min_t  * 1000, Gval_per_s(dgemm_flops(M,N,K),min_t),
	max_t  * 1000, Gval_per_s(dgemm_flops(M,N,K),max_t));

	cudaCheckErrors();
	CoCoFree(A, A_loc);
	CoCoFree(B, B_loc);
	CoCoFree(C, C_loc); 
	return 0;
}
