///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief A cblasDgemm micro-benchmark
///

#include <cassert>
#include "microbenchmarks.hpp"
#include <cblas.h>
//TODO: This should at some point be removed (some fuctions require wrapping)
#include "backend_wrappers.hpp"

void report_run(char* filename, size_t M, size_t N, size_t K, double mean_t, double margin_err, size_t sample_sz, double bench_t){

	FILE* fp = fopen(filename,"a");
	if (!fp) error("report_run: LogFile failed to open");
   	fprintf(fp,"%zu,%zu,%zu, %e,%e,%zu,%e\n", M, N, K, mean_t, margin_err, sample_sz, bench_t);
        fclose(fp); 
}

int main(const int argc, const char *argv[]) {

  	double alpha, beta;
  	alpha = 1.1234, beta = 1.2345;

	char TransA = 'X', TransB = 'X'; 
  	int ctr = 1;

	switch (argc) {
	case (3):
		TransA = argv[ctr++][0];
		TransB = argv[ctr++][0]; 
		break;
	default:
		error("Incorrect input arguments. Usage: ./correct_run TransA TransB\n");
  	}

	// Define the max size of a benchmark kernel to run on this machine. FIXME: currently defined by GPU device 0, totaly inefficient
	size_t maxDim = CoCoGetMaxDimSqAsset2D(3, sizeof(double), STEP_TRANS, 0);

	char *filename = (char *) malloc(256* sizeof(char));
	sprintf(filename, "%s/Benchmark-Results/cblasDgemm_TransA-%c_TransB-%c_%s.log", DEPLOYDB, TransA, TransB, VERSION);
	check_benchmark(filename);

	size_t ldA = maxDim, ldB = maxDim, ldC = maxDim;

	/// Matrix Layouts for CPU GEMM
	CBLAS_TRANSPOSE cpu_op_A, cpu_op_B;    // CblasNoTrans, CblasTrans
	cublasOperation_t dummy1; // CUBLAS_OP_N, CUBLAS_OP_T
	
	TransposeTranslate(TransA, &cpu_op_A, &dummy1, &ldA, maxDim, maxDim);
	TransposeTranslate(TransB, &cpu_op_B, &dummy1, &ldB, maxDim, maxDim);

	// TODO: Benchmarks are conducted on pinned host memory, since this must be the case for CPU-GPU-trans async execution
	fprintf(stderr, "\nAllocating pinned host memory...");
	double cpu_timer = csecond();

	double *A_p, *B_p, *C_p;
  	A_p = (double*) CoCoMalloc(maxDim * maxDim * sizeof(double), -1);
  	B_p = (double*) CoCoMalloc(maxDim * maxDim * sizeof(double), -1);
  	C_p = (double*) CoCoMalloc(maxDim * maxDim * sizeof(double), -1);
	CoCoSyncCheckErr();

	cpu_timer  = csecond() - cpu_timer ;
	fprintf(stderr, "done.\nAlloc time:\t%lf ms\n\n",  cpu_timer  * 1000);

	fprintf(stderr, "Initializing to random values..."); 
	cpu_timer = csecond();

	CoCoVecInit(A_p, maxDim * maxDim, 42, -1);
	CoCoVecInit(B_p, maxDim * maxDim, 42, -1);
	CoCoVecInit(C_p, maxDim * maxDim, 42, -1);


	CoCoSyncCheckErr();
	cpu_timer  = csecond() - cpu_timer ;	
	fprintf(stderr, "done.\nInit time:\t%lf ms\n\n",  cpu_timer  * 1000);

	fprintf(stderr, "\nMatrix details: A(%s) B(%s) C(%s) -> Mmax = %zu, Nmax = %zu, Kmax = %zu\n",
            print_mem(COL_MAJOR), print_mem(COL_MAJOR), print_mem(COL_MAJOR), maxDim, maxDim, maxDim);
	fprintf(stderr, "Constants: alpha = %lf, beta = %lf\n", alpha, beta);

	// Warmup 
	for ( int itt = 0; itt <1; itt++) cblas_dgemm(CblasColMajor, cpu_op_A, cpu_op_B, maxDim, maxDim, maxDim, alpha, A_p, ldA, B_p, ldB, beta, C_p, ldC);

	CoCoSyncCheckErr();
#ifdef AUTO_BENCH_USE_BOOST
	double cblas_t_vals[MICRO_MAX_ITER], cblas_t_sum, cblas_t_mean, bench_t, error_margin; 
	size_t bench_ctr = 0, sample_sz, step = STEP_BLAS3;
	for (size_t T = MIN_DIM_BLAS3; T < maxDim + 1; T+=step){
		if (T >= step * 16) step *=2; 
		fprintf(stderr,"Running cblasDgemm-> square T = %zu:\n", T);
		cblas_t_mean = cblas_t_sum = error_margin = 0;
		sample_sz = 0; 
		bench_t = csecond();
		double std_dev = 0; 
		for (sample_sz = 1; sample_sz < MICRO_MAX_ITER + 1; sample_sz++) {	
			cpu_timer = csecond();
			cblas_dgemm(CblasColMajor, cpu_op_A, cpu_op_B, T, T, T, alpha, A_p, ldA, B_p, ldB, beta, C_p, ldC);
			cpu_timer  = csecond() - cpu_timer ;
			cblas_t_vals[sample_sz-1] = cpu_timer;
			cblas_t_sum += cblas_t_vals[sample_sz-1];
			cblas_t_mean = cblas_t_sum/sample_sz; 
			if (sample_sz < 2) continue;
			for (int i = 0; i < sample_sz; i++) std_dev += pow(cblas_t_vals[i] - cblas_t_mean, 2);
			std_dev /= sample_sz;
    			std_dev = sqrt(std_dev);
			boost::math::students_t dist(sample_sz - 1);
			double Td = boost::math::quantile(boost::math::complement(dist, alphaCI / 2)); //T
			error_margin = Td*std_dev/sqrt(sample_sz); //T
			//fprintf(stderr, "\tItter %zu:\t mean=%lf, std_dev = %lf, Error margin =%lf\n", sample_sz, cblas_t_mean , std_dev, error_margin);
			if (sample_sz > MICRO_MIN_ITER && error_margin/cblas_t_mean  * 100 <= 5) break; 
		}
		bench_t = csecond() - bench_t;
		fprintf(stderr, "Microbenchmark (M = N = K = %zu) complete:\t mean_exec_t=%lf ms ( %.1lf Gflops/s ), Error Margin (percentage of mean) = %lf %%, Itter = %zu, Microbench_t = %lf\n\n", T, cblas_t_mean  * 1000, Gval_per_s(dgemm_flops(T,T,T), cblas_t_mean), error_margin/cblas_t_mean  * 100, sample_sz, bench_t);
		CoCoSyncCheckErr();

		report_run(filename, T, T, T, cblas_t_mean, error_margin, sample_sz, bench_t); 
		bench_ctr++;
	}
#else
	double  bench_t, cblas_t_av, cblas_t_min , cblas_t_max; 
	size_t bench_ctr = 0, step = STEP_BLAS3;
	for (size_t T = MIN_DIM_BLAS3; T < maxDim + 1; T+=step){
		if (T >= step * 16) step *=2; 
		fprintf(stderr,"Running cblasDgemm-> square T = %zu:\n", T);
		cblas_t_av = cblas_t_max = 0;
		cblas_t_min = 1e9;
		bench_t = csecond();
		for (int itt = 0; itt < ITER; itt ++) {
			cpu_timer = csecond();
			cblas_dgemm(CblasColMajor, cpu_op_A, cpu_op_B, T, T, T, alpha, A_p, ldA, B_p, ldB, beta, C_p, ldC);
			cpu_timer  = csecond() - cpu_timer ;
			cblas_t_av += cpu_timer;
			if (cpu_timer > cblas_t_max) cblas_t_max = cpu_timer; 
			if (cpu_timer < cblas_t_min) cblas_t_min = cpu_timer; 
		}
		bench_t = csecond() - bench_t;
		cblas_t_av /= ITER;
		fprintf(stderr, "GPU exec time:\t Average=%lf ms, Min = %lf ms, Max = %lf ms\n", cblas_t_av  * 1000, cblas_t_min  * 1000, cblas_t_max  * 1000);
		CoCoSyncCheckErr();

		report_run(filename, T, T, T, cblas_t_av, fmax(cblas_t_max - cblas_t_av, cblas_t_av - cblas_t_min), ITER, bench_t); 
		bench_ctr++;
	}
#endif
	fprintf(stderr, "Ran %zu Benchmarks.Finallizing...\n", bench_ctr);
	return 0;
}
