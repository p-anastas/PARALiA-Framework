///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The start of Zawarudo
///

#include "unihelpers.hpp"
#include "CoCoPeLia.hpp"
#include "BackenedLibsWrapped.hpp"
#include "Testing.hpp"

#include "backend_wrappers.hpp"

#define CBLASXT_MAX_SAFE_TILE 10000

int main(const int argc, const char *argv[]) {
	char TransA, TransB;
  	double alpha, beta;
	long int M, N, K;
	short A_loc, B_loc, C_loc, C_out_loc;
	ATC_p predef_control_values = NULL, return_values = NULL;
	ParseInputLvl3(argc, argv, &predef_control_values, &TransA, &TransB, &alpha, &beta, &M, &N, &K, &A_loc, &B_loc, &C_loc, &C_out_loc);

	char *filename = (char *) malloc(1024* sizeof(char));
	if (predef_control_values!= NULL){
		error("CoCoPeLiaDgemmRunnerBest: I am not supposed to be used with specific inputs. You probably need CoCoPeLiaDgemmRunner\n");
	}
	else sprintf(filename, "%s/CoCoPeLiaDgemmRunnerBest_%s_%s_%s.log",
		TESTLIBDIR, CoCoDistributionPrint(), CoCoImplementationPrint(), VERSION);
#ifdef CHECKLOG
	CheckLogLvl3(filename, predef_control_values, TransA, TransB, alpha, beta, M, N, K, A_loc, B_loc, C_loc, C_out_loc);
#endif
	predef_control_values = new ATC();
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
	double *C_buf;
	C_buf  = (double*) malloc(M * N*sizeof(double));
	CoCoMemcpy(C_buf, C,  M * N *sizeof(double), -2, C_loc);
#endif

	long int best_T = (long int) fmin(fmin(fmin(M/LOC_NUM,N/LOC_NUM),K),CBLASXT_MAX_SAFE_TILE);
	predef_control_values-> cache_limit = 0;
	predef_control_values->active_unit_num = -1;
	predef_control_values-> T = best_T;
	// Warmup
	for(int it = 0; it < 10; it++){
		return_values = CoCopeLiaDgemmControled(TransA, TransB, M, N, K, alpha, A, ldA, B, ldB, beta, C , ldC, predef_control_values);
		CoCoSyncCheckErr();
	}

	cpu_timer  = csecond();
	return_values = CoCopeLiaDgemmControled(TransA, TransB, M, N, K, alpha, A, ldA, B, ldB, beta, C , ldC, predef_control_values);
	CoCoSyncCheckErr();
	cpu_timer  = csecond() - cpu_timer;

	short bench_it = 10;
	double best_t = cpu_timer;
	for (long int T_trial = (((long int)fmax(fmin(fmin(M/32,N/32),K/32),512))/512)*512; T_trial <= (long int) fmin(fmin(fmin(M/sqrt(DEV_NUM),N/sqrt(DEV_NUM)),K),CBLASXT_MAX_SAFE_TILE); T_trial+=512){
			fprintf(stderr,"Running CoCopeLia DGEMM-> M = %zu, N = %zu, K = %zu, T = %zu\n", M, N, K, T_trial);
			predef_control_values-> T = T_trial;
			cpu_timer  = csecond();
			for(int it = 0; it < bench_it; it++){
				return_values = CoCopeLiaDgemmControled(TransA, TransB, M, N, K, alpha, A, ldA, B, ldB, beta, C , ldC, predef_control_values);
				CoCoSyncCheckErr();
			}
			cpu_timer  = (csecond() - cpu_timer)/bench_it;
			fprintf(stderr, "Total time:\t%lf ms\n", cpu_timer  * 1000);
			if (cpu_timer < best_t){
				best_t = cpu_timer;
				best_T = T_trial;
			}
	}
	fprintf(stderr, "\nCoCopeLia DGEMM T_best = %zu : t = %lf ms ( %lf Gflops/s )\n\n", best_T, best_t  * 1000, Gval_per_s(gemm_flops(M,N,K),best_t));
	predef_control_values-> T = best_T;
	for (int i = 0; i< LOC_NUM; i++) CoCopeLiaDevCacheFree(deidxize(i));

#ifdef RUNVALIDATION
	double *C_out, *C_out1;
	C_out  = (double*) malloc(M * N*sizeof(double));
	C_out1  = (double*) malloc(M * N*sizeof(double));

 	CoCoMemcpy(C, C_buf,  M * N *sizeof(double), C_loc, -2);

	// Call for Validate
	if (predef_control_values!= NULL) return_values = CoCopeLiaDgemmControled(TransA, TransB, M, N, K, alpha, A, ldA, B, ldB, beta, C , ldC, predef_control_values);
	else return_values = CoCopeLiaDgemm(TransA, TransB, M, N, K, alpha, A, ldA, B, ldB, beta, C , ldC);
	CoCoSyncCheckErr();
	for (int i = 0; i< LOC_NUM; i++) CoCopeLiaDevCacheFree(deidxize(i));

 	CoCoMemcpy(C_out, C,  M * N *sizeof(double), -2, C_loc);
 	CoCoMemcpy(C, C_buf,  M * N *sizeof(double), C_loc, -2);

	// Validate with cuBLASXt (questionable but CPU validation can be slower by at least a factor)
	int dev_ids[DEV_NUM];
	for (int i = 0; i < DEV_NUM; i++) dev_ids[i] = i;
	cuBLASXtDgemmWrap(TransA,  TransB, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC,  (long int) fmin(fmin(fmin(M,N),K)/2,CBLASXT_MAX_SAFE_TILE), 0, DEV_NUM, dev_ids);
	CoCoMemcpy(C_out1, C,  M * N *sizeof(double), -2, C_loc);
 	if(Dtest_equality(C_out1, C_out, M * N) < 9) error("Insufficient accuracy for benchmarks\n");

 	CoCoMemcpy(C, C_buf,  M * N *sizeof(double), C_loc, -2);
	free(C_out);
	free(C_out1);
	free(C_buf);
#endif

	cpu_timer = csecond();
	// First call for Validate
	if (predef_control_values!= NULL) return_values = CoCopeLiaDgemmControled(TransA, TransB, M, N, K, alpha, A, ldA, B, ldB, beta, C , ldC, predef_control_values);
	else return_values = CoCopeLiaDgemm(TransA, TransB, M, N, K, alpha, A, ldA, B, ldB, beta, C , ldC);
	CoCoSyncCheckErr();
	cpu_timer  = csecond() - cpu_timer;

#ifdef CHECKLOG
	CheckLogLvl3(filename, return_values, TransA, TransB, alpha, beta, M, N, K, A_loc, B_loc, C_loc, C_out_loc);
#endif
	// Store the time required for the first call (+ 1-time overheads etc)
	//StoreLogLvl3(filename, return_values, TransA, TransB, alpha, beta, M, N, K, A_loc, B_loc, C_loc, C_out_loc, cpu_timer);

	double first_over_t = cpu_timer;

	double min_t = first_over_t, max_t = 0, avg_t = 0;
	cpu_timer = csecond();
  bench_it = 100;
	if ( M >= 8192 || N >= 8192 || K >= 8192) bench_it = 10;
	for(int it = 0; it < bench_it; it++){
		cpu_timer = csecond();
		if (predef_control_values!= NULL) return_values = CoCopeLiaDgemmControled(TransA, TransB, M, N, K, alpha, A, ldA, B, ldB, beta, C , ldC, predef_control_values);
		else return_values = CoCopeLiaDgemm(TransA, TransB, M, N, K, alpha, A, ldA, B, ldB, beta, C , ldC);
		CoCoSyncCheckErr();
		cpu_timer = csecond() - cpu_timer;
		StoreLogLvl3(filename, return_values, TransA, TransB, alpha, beta, M, N, K, A_loc, B_loc, C_loc, C_out_loc, cpu_timer);
		if ( cpu_timer < min_t ) min_t = cpu_timer;
		if ( cpu_timer > max_t ) max_t = cpu_timer;
		avg_t += cpu_timer;
	}
	avg_t/=bench_it;
	fprintf(stderr, "CoCopeLia (%s):\n\tfirst_it_t = %lf ms ( %lf Gflops/s )\n\tavg_t = %lf ms ( %lf Gflops/s )\n\tmin_t = %lf ms ( %lf Gflops/s )\n\tmax_t = %lf ms ( %lf Gflops/s )\n",
	return_values->print_csv(),
	first_over_t  * 1000, Gval_per_s(gemm_flops(M,N,K),first_over_t),
	avg_t  * 1000, Gval_per_s(gemm_flops(M,N,K),avg_t),
	min_t  * 1000, Gval_per_s(gemm_flops(M,N,K),min_t),
	max_t  * 1000, Gval_per_s(gemm_flops(M,N,K),max_t));

	for (int i = 0; i< LOC_NUM; i++) CoCopeLiaDevCacheFree(deidxize(i));

	CoCoSyncCheckErr();
	CoCoFree(A, A_loc);
	CoCoFree(B, B_loc);
	CoCoFree(C, C_loc);
	return 0;
}
