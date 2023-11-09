///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief A cublasDgemm micro-benchmark
///

#include <cassert>
#include "microbenchmarks.hpp"

//TODO: This should at some point be removed (some fuctions require wrapping)
#include "backend_wrappers.hpp"

#ifdef ENABLE_POWA
/// Energy measuring
#include "nvem.hpp"
#endif

void report_run(char* filename, long int M, long int N, long int K, double mean_t, double W_avg, double Joules, double margin_err, long int sample_sz, double bench_t){

	FILE* fp = fopen(filename,"a");
	if (!fp) error("report_run: LogFile failed to open");
   	fprintf(fp,"%d,%d,%d, %e,%e,%e,%e,%zu,%e\n", M, N, K, mean_t, W_avg, Joules, margin_err, sample_sz, bench_t);
        fclose(fp);
}

int main(const int argc, const char *argv[]) {

  	double alpha, beta;
  	alpha = 1.1234, beta = 1.2345;

	char TransA, TransB;
  	int ctr = 1, dev_id;

	switch (argc) {
	case (4):
		dev_id = atoi(argv[ctr++]);
		TransA = argv[ctr++][0];
		TransB = argv[ctr++][0];
		break;
	default:
		error("Incorrect input arguments. Usage: ./correct_run dev_id TransA TransB\n");
  	}

	/// Set device
	CoCoPeLiaSelectDevice(dev_id);
	if (dev_id != -1){
		size_t free_cuda_mem, max_cuda_mem;
		massert(cudaSuccess == cudaMemGetInfo(&free_cuda_mem, &max_cuda_mem),
			"dgemm_microbench_gpu: cudaMemGetInfo failed");
		if (free_cuda_mem < 1.0*PROBLEM_GPU_PERCENTAGE/100*max_cuda_mem)
			error("dgemm_microbench_gpu: Free memory is much less than max ( Free: %zu, Max: %zu ),\
			device under utilization\n", free_cuda_mem, max_cuda_mem);
	}

	char *filename = (char *) malloc(1024* sizeof(char));
	sprintf(filename, "%s/Benchmark-Results/cublasDgemm_dev-%d_TransA-%c_TransB-%c_%s.log",
		DEPLOYDB, dev_id, TransA, TransB, VERSION);
	check_benchmark(filename);

	// Define the max size of a benchmark kernel to run on this machine.
	long int maxDim = CoCoGetMaxDimSqAsset2D(3, sizeof(double), STEP_TRANS, dev_id);

	long int ldA = maxDim, ldB = maxDim, ldC = maxDim;

	/// Matrix Layouts for CPU GEMM
	CBLAS_TRANSPOSE cpu_op_A, cpu_op_B;    // CblasNoTrans, CblasTrans
	cublasOperation_t gpu_op_A, gpu_op_B; // CUBLAS_OP_N, CUBLAS_OP_T

	TransposeTranslate(TransA, &cpu_op_A, &gpu_op_A, &ldA, maxDim, maxDim);
	TransposeTranslate(TransB, &cpu_op_B, &gpu_op_B, &ldB, maxDim, maxDim);

	cublasHandle_t handle0;
 	cudaStream_t host_stream;

	if(dev_id != -1){
  	cudaStreamCreate(&host_stream);
		assert(CUBLAS_STATUS_SUCCESS == cublasCreate(&handle0));
		assert(CUBLAS_STATUS_SUCCESS == cublasSetStream(handle0, host_stream));
	    //void* WS = CoCoMalloc(4*1024*1024*sizeof(double), dev_id);
        //assert(CUBLAS_STATUS_SUCCESS == cublasSetWorkspace(handle0, WS, 4*1024*1024*sizeof(double)));
    }

	fprintf(stderr, "\nAllocating memory...");
	double cpu_timer = csecond();

	double *A_buf, *B_buf, *C_buf;
  	A_buf = (double*) CoCoMalloc(maxDim * maxDim * sizeof(double), dev_id, 0);
  	B_buf = (double*) CoCoMalloc(maxDim * maxDim * sizeof(double), dev_id, 0);
  	C_buf = (double*) CoCoMalloc(maxDim * maxDim * sizeof(double), dev_id, 1);
	CoCoSyncCheckErr();

	cpu_timer  = csecond() - cpu_timer ;
	fprintf(stderr, "done.\nAlloc time (locs = [%d, %d, %d]):\t%lf ms\n\n",  CoCoGetPtrLoc(A_buf) ,  CoCoGetPtrLoc(B_buf) ,CoCoGetPtrLoc(C_buf) , cpu_timer  * 1000);

	fprintf(stderr, "Initializing to random values...");
	cpu_timer = csecond();

	CoCoVecInit(A_buf, maxDim * maxDim, 42, dev_id);
	CoCoVecInit(B_buf, maxDim * maxDim, 42, dev_id);
	CoCoVecInit(C_buf, maxDim * maxDim, 42, dev_id);


	CoCoSyncCheckErr();
	cpu_timer  = csecond() - cpu_timer ;
	fprintf(stderr, "done.\nInit time:\t%lf ms\n\n",  cpu_timer  * 1000);

	fprintf(stderr, "\nMatrix details: A(%s) B(%s) C(%s) -> Mmax = %d, Nmax = %d, Kmax = %d\n",
            print_mem(COL_MAJOR), print_mem(COL_MAJOR), print_mem(COL_MAJOR), maxDim, maxDim, maxDim);
	fprintf(stderr, "Constants: alpha = %lf, beta = %lf\n", alpha, beta);

	// Warmup
	for ( int itt = 0; itt <1; itt++){
		if(dev_id != -1){
			assert(CUBLAS_STATUS_SUCCESS == cublasDgemm(handle0, gpu_op_A, gpu_op_B,
				maxDim, maxDim, maxDim, &alpha, A_buf, ldA, B_buf, ldB, &beta, C_buf, ldC));
			cudaStreamSynchronize(host_stream);
		}
		else cblas_dgemm(CblasColMajor, cpu_op_A, cpu_op_B,
			maxDim, maxDim, maxDim, alpha, A_buf, ldA, B_buf, ldB, beta, C_buf, ldC);
	}
	CoCoSyncCheckErr();
#ifdef AUTO_BENCH_USE_BOOST
	double cublas_t_vals[MICRO_MAX_ITER], cublas_t_sum, cublas_t_mean, bench_t, error_margin;
	long int bench_ctr = 0, sample_sz, step = STEP_BLAS3;
	for (long int T = MIN_DIM_BLAS3; T < MAX_DIM_BLAS3; T+=step){
		if (T >= step * 16) step *=2;
		fprintf(stderr,"Running cublasDgemm-> square T = %d:\n", T);
		cublas_t_mean = cublas_t_sum = error_margin = 0;
		sample_sz = 0;
#ifdef ENABLE_POWA
		char powa_filename[256];
		sprintf(powa_filename, "dgemm_microbench_powa.log");
		NvemStats_p nvem_data;
		if(dev_id != -1) NvemStartMeasure(dev_id, powa_filename, 0);
#endif
		bench_t = csecond();
		double std_dev = 0;
		for (sample_sz = 1; sample_sz < MICRO_MAX_ITER + 1; sample_sz++) {
			cpu_timer = csecond();
			if(dev_id != -1){
				assert(CUBLAS_STATUS_SUCCESS == cublasDgemm(handle0, gpu_op_A, gpu_op_B,
					T, T, T, &alpha, A_buf, ldA, B_buf, ldB, &beta, C_buf, ldC));
				cudaStreamSynchronize(host_stream);
			}
			else cblas_dgemm(CblasColMajor, cpu_op_A, cpu_op_B,
				T, T, T, alpha, A_buf, ldA, B_buf, ldB, beta, C_buf, ldC);
			cpu_timer  = csecond() - cpu_timer;
			cublas_t_vals[sample_sz-1] = cpu_timer;
			cublas_t_sum += cublas_t_vals[sample_sz-1];
			cublas_t_mean = cublas_t_sum/sample_sz;
			if (sample_sz < 2) continue;
			for (int i = 0; i < sample_sz; i++) std_dev += pow(cublas_t_vals[i] - cublas_t_mean, 2);
			std_dev /= sample_sz;
    			std_dev = sqrt(std_dev);
			boost::math::students_t dist(sample_sz - 1);
			double Td = boost::math::quantile(boost::math::complement(dist, alphaCI / 2)); //T
			error_margin = Td*std_dev/sqrt(sample_sz); //T
			//fprintf(stderr, "\tItter %d:\t mean=%lf, std_dev = %lf, Error margin =%lf\n", sample_sz, cublas_t_mean , std_dev, error_margin);
			if (sample_sz > MICRO_MIN_ITER && error_margin/cublas_t_mean  * 100 <= 5) break;
		}
		bench_t = csecond() - bench_t;
		CoCoSyncCheckErr();
#ifdef ENABLE_POWA
		if(dev_id != -1) nvem_data = NvemStopMeasure(dev_id, "Energy measure dgemm_microbench");
		else{
			nvem_data = (NvemStats_p) malloc(sizeof(struct nvem_results));
			strcpy(nvem_data->name, "Energy measure dgemm_microbench");
			nvem_data->sensor_ticks = -1;
			nvem_data->total_bench_t = bench_t;
			nvem_data->W_avg = CPU_W_PREDEF;
			nvem_data->J_estimated = nvem_data->W_avg*nvem_data->total_bench_t;
		}
		double W_avg = nvem_data->W_avg, J_estimated = nvem_data->J_estimated/sample_sz;
		fprintf(stderr, "Microbenchmark (M = N = K = %zu) complete:\t mean_exec_t=%lf ms ( %.1lf Gflops/s )\
			, Energy: ( %lf Watt -> %lf J), Error Margin (percentage of mean) = %lf %, Itter = %d, Microbench_t = %lf\n\n",
			T, cublas_t_mean  * 1000, Gval_per_s(gemm_flops(T,T,T), cublas_t_mean),
			W_avg, J_estimated, error_margin/cublas_t_mean  * 100, sample_sz, bench_t);
		report_run(filename, T, T, T, cublas_t_mean, W_avg, J_estimated, error_margin, sample_sz, bench_t);
#else
		fprintf(stderr, "Microbenchmark (M = N = K = %zu) complete:\t mean_exec_t=%lf ms ( %.1lf Gflops/s )\
			, Error Margin (percentage of mean) = %lf %, Itter = %d, Microbench_t = %lf\n\n",
			T, cublas_t_mean  * 1000, Gval_per_s(gemm_flops(T,T,T), cublas_t_mean),
			error_margin/cublas_t_mean  * 100, sample_sz, bench_t);
		report_run(filename, T, T, T, cublas_t_mean, -1, -1, error_margin, sample_sz, bench_t);
#endif
		bench_ctr++;
	}
#else
	double  bench_t, cublas_t_av, cublas_t_min , cublas_t_max;
	long int bench_ctr = 0, step = STEP_BLAS3;
	for (long int T = MIN_DIM_BLAS3; T < MAX_DIM_BLAS3; T+=step){
		if (T >= step * 16) step *=2;
		fprintf(stderr,"Running cublasDgemm-> square T = %d:\n", T);
		cublas_t_av = cublas_t_max = 0;
		cublas_t_min = 1e9;
#ifdef ENABLE_POWA
		char powa_filename[256];
		sprintf(powa_filename, "dgemm_microbench_powa.log");
		NvemStats_p nvem_data;
		if(dev_id != -1) NvemStartMeasure(dev_id, powa_filename, 0);
#endif
		bench_t = csecond();
		for (int itt = 0; itt < ITER; itt ++) {
			cpu_timer = csecond();
			if(dev_id != -1){
				assert(CUBLAS_STATUS_SUCCESS == cublasDgemm(handle0, gpu_op_A, gpu_op_B,
					T, T, T, &alpha, A_buf, ldA, B_buf, ldB, &beta, C_buf, ldC));
				cudaStreamSynchronize(host_stream);
			}
			else cblas_dgemm(CblasColMajor, cpu_op_A, cpu_op_B,
				T, T, T, alpha, A_buf, ldA, B_buf, ldB, beta, C_buf, ldC);
			cpu_timer  = csecond() - cpu_timer ;
			cublas_t_av += cpu_timer;
			if (cpu_timer > cublas_t_max) cublas_t_max = cpu_timer;
			if (cpu_timer < cublas_t_min) cublas_t_min = cpu_timer;
		}
		bench_t = csecond() - bench_t;
		cublas_t_av /= ITER;
		CoCoSyncCheckErr();
#ifdef ENABLE_POWA
		if(dev_id != -1) nvem_data = NvemStopMeasure(dev_id, "Energy measure dgemm_microbench");
		else{
			nvem_data = (NvemStats_p) malloc(sizeof(struct nvem_results));
			strcpy(nvem_data->name, "Energy measure dgemm_microbench");
			nvem_data->sensor_ticks = -1;
			nvem_data->total_bench_t = bench_t;
			nvem_data->W_avg = CPU_W_PREDEF;
			nvem_data->J_estimated = nvem_data->W_avg*nvem_data->total_bench_t;
		}
		double W_avg = nvem_data->W_avg, J_estimated = nvem_data->J_estimated/ITER;
		fprintf(stderr, "GPU exec time:\t Average=%lf ms, Min = %lf ms, Max = %lf ms, Energy: ( %lf Watt -> %lf J)\n",
			cublas_t_av  * 1000, cublas_t_min  * 1000, cublas_t_max  * 1000, W_avg, J_estimated);

		report_run(filename, T, T, T, cublas_t_av, W_avg, J_estimated, fmax(cublas_t_max - cublas_t_av,
				cublas_t_av - cublas_t_min), ITER, bench_t);
#else
		fprintf(stderr, "GPU exec time:\t Average=%lf ms, Min = %lf ms, Max = %lf ms\n",
			cublas_t_av  * 1000, cublas_t_min  * 1000, cublas_t_max  * 1000);
		report_run(filename, T, T, T, cublas_t_av, -1, -1, fmax(cublas_t_max - cublas_t_av,
				cublas_t_av - cublas_t_min), ITER, bench_t);
#endif

		bench_ctr++;
	}
#endif
	fprintf(stderr, "Ran %d Benchmarks.Finallizing...\n", bench_ctr);
	return 0;
}
