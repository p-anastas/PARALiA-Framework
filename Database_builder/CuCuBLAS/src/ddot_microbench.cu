///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
/// \author THeodoridis Aristomenis (atheodor@cslab.ece.ntua.gr)
///
/// \brief A cublasDdot micro-benchmark
///

#include <cassert>
#include "microbenchmarks.hpp"

//TODO: This should at some point be removed (some fuctions require wrapping)
#include "backend_wrappers.hpp"

#ifdef ENABLE_POWA
/// Energy measuring
#include "nvem.hpp"
#endif

void report_run(char* filename, long int N, double mean_t, double W_avg, double Joules, double margin_err, long int sample_sz, double bench_t){

	FILE* fp = fopen(filename,"a");
	if (!fp) error("report_run: LogFile failed to open");
   	fprintf(fp,"%d, %e,%e,%e,%e,%zu,%e\n", N, mean_t, W_avg, Joules, margin_err, sample_sz, bench_t);
        fclose(fp);
}

int main(const int argc, const char *argv[]) {

  	double alpha;
  	alpha = 1.1234;

  	int ctr = 1, dev_id;

	long int incx = 1, incy = 1;

	switch (argc) {
	case (2):
		dev_id = atoi(argv[ctr++]);
		break;
	default:
		error("Incorrect input arguments. Usage: ./correct_run dev_id\n");
  	}

	/// Set device
	CoCoPeLiaSelectDevice(dev_id);
	if (dev_id != -1){
		size_t free_cuda_mem, max_cuda_mem;
		massert(cudaSuccess == cudaMemGetInfo(&free_cuda_mem, &max_cuda_mem),
			"ddot_microbench_gpu: cudaMemGetInfo failed");
		if (free_cuda_mem < 1.0*PROBLEM_GPU_PERCENTAGE/100*max_cuda_mem)
			error("ddot_microbench_gpu: Free memory is much less than max ( Free: %zu, Max: %zu ),\
			device under utilization\n", free_cuda_mem, max_cuda_mem);
	}

	char *filename = (char *) malloc(1024* sizeof(char));
	sprintf(filename, "%s/Benchmark-Results/cublasDdot_dev-%d_%s.log", DEPLOYDB, dev_id, VERSION);


	check_benchmark(filename);


	// Define the max size of a benchmark kernel to run on this machine.
	long int maxDim = CoCoGetMaxDimAsset1D(2, sizeof(double), STEP_BLAS1, dev_id);

	cublasHandle_t handle0;
 	cudaStream_t host_stream;

  	cudaStreamCreate(&host_stream);
	assert(CUBLAS_STATUS_SUCCESS == cublasCreate(&handle0));
	assert(CUBLAS_STATUS_SUCCESS == cublasSetStream(handle0, host_stream));

	fprintf(stderr, "\nAllocating device memory...");
	double cpu_timer = csecond();

    double result;
	double *x_dev, *y_dev;
  	x_dev = (double*) CoCoMalloc(maxDim * sizeof(double), dev_id);
  	y_dev = (double*) CoCoMalloc(maxDim * sizeof(double), dev_id);

	CoCoSyncCheckErr();
	cpu_timer  = csecond() - cpu_timer ;
	fprintf(stderr, "done.\nAlloc time:\t%lf ms\n\n",  cpu_timer  * 1000);

	fprintf(stderr, "Initializing to random values...");
	cpu_timer = csecond();

	CoCoVecInit(x_dev, maxDim, 42, dev_id);
	CoCoVecInit(y_dev, maxDim, 43, dev_id);

	CoCoSyncCheckErr();
	cpu_timer  = csecond() - cpu_timer ;
	fprintf(stderr, "done.\nInit time:\t%lf ms\n\n",  cpu_timer  * 1000);

	fprintf(stderr, "\nTile details: x(inc=%d) y(inc=%d) -> maxDim = %d\n", 1, 1, maxDim);

	fprintf(stderr, "Constants: alpha = %lf\n", alpha);

	// Warmup
	for ( int itt = 0; itt <1; itt++){
		if(dev_id != -1){
			assert(CUBLAS_STATUS_SUCCESS == cublasDdot(handle0, maxDim, x_dev, incx, y_dev, incy, &result));
			cudaStreamSynchronize(host_stream);
		}
		else cblas_ddot(maxDim, x_dev, incx, y_dev, incy);


	}
	CoCoSyncCheckErr();
#ifdef AUTO_BENCH_USE_BOOST
	double cublas_t_vals[MICRO_MAX_ITER], cublas_t_sum, cublas_t_mean, bench_t, error_margin;
	long int bench_ctr = 0, sample_sz, step = STEP_BLAS1;
	for (long int T = MIN_DIM_BLAS1; T < maxDim + 1; T+=step){
		if (T >= step * 16) step *=2;
		fprintf(stderr,"Running cublasDdot-> T = %d:\n", T);
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
				assert(CUBLAS_STATUS_SUCCESS == cublasDdot(handle0, T, x_dev, incx, y_dev, incy, &result));
				cudaStreamSynchronize(host_stream);
			}
			else cblas_ddot(T, x_dev, incx, y_dev, incy);
			cpu_timer  = csecond() - cpu_timer ;
			cublas_t_vals[sample_sz-1] = cpu_timer;
			cublas_t_sum += cublas_t_vals[sample_sz-1];
			cublas_t_mean = cublas_t_sum/sample_sz;
			if (sample_sz < 2) continue;
			for (int i = 0; i < sample_sz; i++) std_dev += pow(cublas_t_vals[i] - cublas_t_mean, 2);
			std_dev /= sample_sz;
    			std_dev = sqrt(std_dev);
			boost::math::students_t dist(sample_sz - 1);
			double T = boost::math::quantile(boost::math::complement(dist, alphaCI / 2));
			error_margin = T*std_dev/sqrt(sample_sz);
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
		fprintf(stderr, "Microbenchmark (N = %zu) complete:\t mean_exec_t=%lf ms ( %.1lf Gflops/s )\
			, Energy: ( %lf Watt -> %lf J), Error Margin (percentage of mean) = %lf %, Itter = %d, Microbench_t = %lf\n\n",
			T, cublas_t_mean  * 1000, Gval_per_s(dot_flops(T), cublas_t_mean),
			W_avg, J_estimated, error_margin/cublas_t_mean  * 100, sample_sz, bench_t);
		report_run(filename, T, cublas_t_mean, W_avg, J_estimated, error_margin, sample_sz, bench_t);
#else
		fprintf(stderr, "Microbenchmark (N = %zu) complete:\t mean_exec_t=%lf ms ( %.1lf Gflops/s )\
			, Error Margin (percentage of mean) = %lf %, Itter = %d, Microbench_t = %lf\n\n",
			T, cublas_t_mean  * 1000, Gval_per_s(dot_flops(T), cublas_t_mean),
			error_margin/cublas_t_mean  * 100, sample_sz, bench_t);
		report_run(filename, T, cublas_t_mean, -1, -1, error_margin, sample_sz, bench_t);
#endif
		bench_ctr++;
	}
#else
	double  bench_t, cublas_t_av, cublas_t_min , cublas_t_max;
	long int bench_ctr = 0, step = STEP_BLAS1;
	for (long int T = MIN_DIM_BLAS1; T < maxDim + 1; T+=step){
		if (T >= step * 16) step *=2;
		fprintf(stderr,"Running cublasDdot-> T = %d:\n", T);
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
				assert(CUBLAS_STATUS_SUCCESS == cublasDdot(handle0, T, x_dev, incx, y_dev, incy, &result));
				cudaStreamSynchronize(host_stream);
			}
			else cblas_ddot(T, x_dev, incx, y_dev, incy);
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
		fprintf(stderr, "Microbenchmark (N = %zu) complete:\t mean_exec_t=%lf ms ( %.1lf Gflops/s )\
			, Energy: ( %lf Watt -> %lf J), Error Margin (percentage of mean) = %lf %, Itter = %d, Microbench_t = %lf\n\n",
			T, cublas_t_mean  * 1000, Gval_per_s(dot_flops(T), cublas_t_mean),
			W_avg, J_estimated, error_margin/cublas_t_mean  * 100, ITER, bench_t);
		report_run(filename, T, cublas_t_av, W_avg, J_estimated,
			fmax(cublas_t_max - cublas_t_av, cublas_t_av - cublas_t_min), ITER, cublas_t_max);
#else
		fprintf(stderr, "Microbenchmark (N = %zu) complete:\t mean_exec_t=%lf ms ( %.1lf Gflops/s )\
			, Error Margin (percentage of mean) = %lf %, Itter = %d, Microbench_t = %lf\n\n",
			T, cublas_t_mean  * 1000, Gval_per_s(dot_flops(T), cublas_t_mean),
			error_margin/cublas_t_mean  * 100, ITER, bench_t);
		report_run(filename, T, cublas_t_av,
			fmax(cublas_t_max - cublas_t_av, cublas_t_av - cublas_t_min), ITER, cublas_t_max);
#endif
		bench_ctr++;
	}
#endif
	fprintf(stderr, "Ran %d Benchmarks.Finallizing...\n", bench_ctr);
	return 0;
}
