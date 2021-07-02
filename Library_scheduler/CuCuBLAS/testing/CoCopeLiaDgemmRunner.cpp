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

	short dev_id;

	char TransA, TransB; 
  	double alpha, beta;
	size_t M, N, K;
	short A_loc, B_loc, C_loc, C_out_loc;
	int Tin; 
	double cpu_ratio; 

	ParseInputLvl3(argc, argv, &dev_id, &TransA, &TransB, &alpha, &beta, &M, &N, &K, &A_loc, &B_loc, &C_loc, &C_out_loc, &Tin, &cpu_ratio);

	char *filename = (char *) malloc(256* sizeof(char));
	if (Tin > 0) {
		if (Tin > M || Tin > N || Tin > K) error("Given Tin=%d bigger than problem dim\n", Tin); 
		else if (Tin > M/1.5 && Tin > N/1.5 && Tin > K/1.5) error("Given Tin=%d bigger than all problem dims/1.5\n", Tin);
		sprintf(filename, "%s/CoCopeLiaDgemmRunner.log", TESTLIBDIR);	
	}
	else sprintf(filename, "%s/CoCopeLiaDgemmRunner.log", TESTLIBDIR);	

	//CheckLogLvl3(filename, dev_id, TransA, TransB, alpha, beta, M, N, K, A_loc, B_loc, C_loc, C_out_loc, -1, cpu_ratio);
	
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

	double *C_out, *C_out1;
	C_out  = (double*) malloc(M * N*sizeof(double));
	C_out1  = (double*) malloc(M * N*sizeof(double));

	CoCoMemcpy(C_out1, C,  M * N *sizeof(double), -2, C_loc);

	// First call for Validate and/or additional overhead counting
	cpu_timer = csecond();
	if (Tin > 0) CoCopeLiaDgemmTin(TransA, TransB, M, N, K, alpha, A, ldA, B, ldB, beta, C , ldC, Tin);
	else CoCopeLiaDgemm(TransA, TransB, M, N, K, alpha, A, ldA, B, ldB, beta, C , ldC);
	cudaCheckErrors();
	cpu_timer  = csecond() - cpu_timer;
	fprintf(stderr, "First call time:\t%lf ms\n\n", cpu_timer  * 1000);
	double first_over_t = cpu_timer; 

 	CoCoMemcpy(C_out, C,  M * N *sizeof(double), -2, C_loc);

	double min_t = first_over_t, max_t = 0, avg_t = 0;
	cpu_timer = csecond();
	short bench_it = 100;
	if ( M >= 8192 || N >= 8192 || K >= 8192) bench_it = 10; 
	for(int it = 0; it < bench_it; it++){
		cpu_timer = csecond();
		if (Tin > 0) CoCopeLiaDgemmTin(TransA, TransB, M, N, K, alpha, A, ldA, B, ldB, beta, C , ldC, Tin);
		else CoCopeLiaDgemm(TransA, TransB, M, N, K, alpha, A, ldA, B, ldB, beta, C , ldC);
		cudaCheckErrors();
		cpu_timer = csecond() - cpu_timer;
		if ( cpu_timer < min_t ) min_t = cpu_timer;
		if ( cpu_timer > max_t ) max_t = cpu_timer;
		avg_t += cpu_timer;
	}
	avg_t/=bench_it;
	fprintf(stderr, "CoCopeLia :\n\tavg_t = %lf ms ( %lf Gflops/s )\n\tmin_t = %lf ms ( %lf Gflops/s )\n\tmax_t = %lf ms ( %lf Gflops/s )\n", 
	avg_t  * 1000, Gval_per_s(dgemm_flops(M,N,K),avg_t),
	min_t  * 1000, Gval_per_s(dgemm_flops(M,N,K),min_t),
	max_t  * 1000, Gval_per_s(dgemm_flops(M,N,K),max_t));


	CoCopeLiaDgemm_flush_gpu_mem_buf(0);
	
 	CoCoMemcpy(C, C_out1,  M * N *sizeof(double), C_loc, -2);
	short dev_num = 2; 
	int dev_ids[dev_num] = {0,1};
	cuBLASXtDgemmWrap(TransA,  TransB, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC,  (size_t) fmin(fmin(fmin(M,N),K)/2,CBLASXT_MAX_SAFE_TILE), 0, dev_num, dev_ids);
	CoCoMemcpy(C_out1, C,  M * N *sizeof(double), -2, C_loc);
 	if(Dtest_equality(C_out1, C_out, M * N) < 9) error("Insufficient accuracy for benchmarks\n");

	StoreLogLvl3(filename, dev_id, TransA, TransB, alpha, beta, M, N, K, A_loc, B_loc, C_loc, C_out_loc, -1, cpu_ratio, avg_t, min_t, max_t); 
	cudaCheckErrors();
	CoCoFree(A, A_loc);
	CoCoFree(B, B_loc);
	CoCoFree(C, C_loc); 
	return 0;
}
