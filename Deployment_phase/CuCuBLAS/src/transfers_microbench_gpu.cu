///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief A transfer micro-benchmark from->to for a) contiguous transfers, b) non-cont square transfers, c) full bidirectional overlapped transfers 
///

#include <unistd.h>
#include <cassert>
#include "microbenchmarks.hpp"

void report_run(char* filename, size_t dim_1, size_t dim_2, double mean_t, double margin_err, size_t sample_sz, double mean_t_bid, double margin_err_bid, size_t sample_sz_bid, double bench_t){

	FILE* fp = fopen(filename,"a");
	if (!fp) error("report_run: LogFile failed to open");
   	fprintf(fp,"%d,%d, %e,%e,%zu, %e,%e,%zu, %e\n", dim_1, dim_2, mean_t, margin_err, sample_sz, mean_t_bid, margin_err_bid, sample_sz_bid, bench_t);
        fclose(fp); 
}

int main(const int argc, const char *argv[]) {

  	int ctr = 1, samples, dev_id, dev_count;

	short from, to; 
	size_t minDim = MIN_DIM_TRANS, maxDim = 0, step = STEP_TRANS; 

	switch (argc) {
	case (3):
		to = atoi(argv[ctr++]);
		from = atoi(argv[ctr++]);
		break;
	default:
		error("Incorrect input arguments. Usage: ./correct_run to from\n");
  	}

	char *filename = (char *) malloc(256* sizeof(char));
	sprintf(filename, "%s/Benchmark-Results/CoCoMemcpy2DAsync_to-%d_from-%d.log", DEPLOYDB, to, from);
	check_benchmark(filename);

	// Define the max size of a benchmark kernel to run on this machine. 
	maxDim = CoCopeLiaGetMaxSqdimLvl3(3, sizeof(double), STEP_TRANS); 

	fprintf(stderr,"\nTransfer benchmark@%s %s->%s : (%d,%d) with step %d\n", TESTBED, print_loc(from), print_loc(to), minDim, maxDim, step);

	cudaGetDeviceCount(&dev_count);

	if (minDim < 1) error("Transfer Microbench: Bytes must be > 0"); 
	else if ( dev_count < from + 1) error("Transfer Microbench: Src device does not exist"); 
	else if ( dev_count < to + 1) error("Transfer Microbench: Dest device does not exist"); 

	void* src, *dest, *rev_src, *rev_dest; 

	//Only model pinned memory transfers from host to dev and visa versa
  	if (from < 0 && to < 0) error("Transfer Microbench: Both locations are in host");
	else if (from == -2 || to == -2) error("Transfer Microbench: Not pinned memory (synchronous)");
  	else if ( from >= 0 && to >= 0){
		short dev_id[2], num_devices = 2;
		dev_id[0] = from;
		dev_id[1] = to;
		// Check/Enable peer access between participating GPUs
		CoCoPeLiaEnableGPUPeer(0, dev_id, num_devices); 
		// Check/Enable peer access between participating GPUs
		CoCoPeLiaEnableGPUPeer(1, dev_id, num_devices); 
	}
	else if(from >= 0) cudaSetDevice(from);
	else if(to >= 0) cudaSetDevice(to);
	
	size_t ldsrc, ldest = ldsrc = maxDim + 1; 

	src = CoCoMalloc(maxDim*(maxDim+1)*sizeof(double), from);
	dest =  CoCoMalloc(maxDim*(maxDim+1)*sizeof(double), to);
	rev_src = CoCoMalloc(maxDim*(maxDim+1)*sizeof(double), to);
	rev_dest = CoCoMalloc(maxDim*(maxDim+1)*sizeof(double), from);

	CoCoVecInit((double*)src, maxDim*(maxDim+1), 42, from);
	CoCoVecInit((double*)rev_src, maxDim*(maxDim+1), 43, to);

	cudaStream_t stream, reverse_stream;
	cudaStreamCreate(&stream);
	cudaStreamCreate(&reverse_stream);
	cudaCheckErrors();
	fprintf(stderr, "Warming up...\n");
	/// Warmup.
	for (int it = 0; it < 10; it++) CoCoMemcpy2DAsync(dest, ldest, src, ldsrc, maxDim, maxDim, sizeof(double), to, from, stream);
	cudaCheckErrors();
#ifdef AUTO_BENCH_USE_BOOST
	double cpu_timer, transfer_t_vals[MICRO_MAX_ITER], transfer_t_sum, transfer_t_mean, bench_t, error_margin; 
	double transfer_t_bid_sum, transfer_t_bid_mean, error_margin_bid; 
	size_t sample_sz, sample_sz_bid;
	gpu_timer_p cuda_timer = gpu_timer_init();
	for (size_t dim = minDim; dim < maxDim+1; dim+=step){
		if (dim >= step * 16) step*=2;
		transfer_t_sum = transfer_t_mean = bench_t = error_margin = 0;
		fprintf(stderr, "Cublas-chunk Link %s->%s (Chunk %dx%d):\n", print_loc(from), print_loc(to), dim, dim);
		sample_sz = 0; 
		bench_t = csecond();
		double std_dev = 0; 
		for (sample_sz = 1; sample_sz < MICRO_MAX_ITER + 1; sample_sz++) {	
			cpu_timer = csecond();
			CoCoMemcpy2DAsync(dest, ldest, src, ldsrc, dim, dim, sizeof(double), to, from, stream);
			cudaStreamSynchronize(stream);
			cpu_timer  = csecond() - cpu_timer ;
			transfer_t_vals[sample_sz-1] = cpu_timer;
			transfer_t_sum += transfer_t_vals[sample_sz-1];
			transfer_t_mean = transfer_t_sum/sample_sz; 
			if (sample_sz < 2) continue;
			for (int i = 0; i < sample_sz; i++) std_dev += pow(transfer_t_vals[i] - transfer_t_mean, 2);
			std_dev /= sample_sz;
    			std_dev = sqrt(std_dev);
			boost::math::students_t dist(sample_sz - 1);
			double Td = boost::math::quantile(boost::math::complement(dist, alphaCI / 2));
			error_margin = Td*std_dev/sqrt(sample_sz);
			//fprintf(stderr, "\tItter %d:\t mean=%lf, std_dev = %lf, Error margin =%lf\n", sample_sz, cublas_t_mean , std_dev, error_margin);
			if (sample_sz > MICRO_MIN_ITER && error_margin/transfer_t_mean  * 100 <= 5) break; 
		}
		bench_t = csecond() - bench_t;
		fprintf(stderr, "Microbenchmark (dim1 = dim2 = %zu) complete:\t mean_exec_t=%lf ms  ( %lf Gb/s), Error Margin (percentage of mean) = %lf %, Itter = %d, Microbench_t = %lf\n\n", dim, transfer_t_mean  * 1000, Gval_per_s(dim*dim*8, transfer_t_mean), error_margin/transfer_t_mean  * 100, sample_sz, bench_t);
		cudaCheckErrors();

		transfer_t_bid_sum = transfer_t_bid_mean = error_margin_bid = 0;
		fprintf(stderr, "Reverse overlapped Link %s->%s (Chunk %dx%d):\n", print_loc(from), print_loc(to), dim, dim);
		sample_sz_bid = 0; 
		bench_t = csecond() - bench_t;
		std_dev = 0; 
		for (sample_sz_bid = 1; sample_sz_bid < MICRO_MAX_ITER + 1; sample_sz_bid++) {	
			for (int rep = 0; rep < 10 ; rep++) CoCoMemcpy2DAsync(rev_dest, ldest, rev_src, ldsrc, dim, dim, sizeof(double), from, to, reverse_stream);
			gpu_timer_start(cuda_timer, stream);
			CoCoMemcpy2DAsync(dest, ldest, src, ldsrc, dim, dim, sizeof(double), to, from, stream);
			gpu_timer_stop(cuda_timer, stream);
			cudaCheckErrors();
			transfer_t_vals[sample_sz_bid-1] = gpu_timer_get(cuda_timer)/1000;
			transfer_t_bid_sum += transfer_t_vals[sample_sz_bid-1];
			transfer_t_bid_mean = transfer_t_bid_sum/sample_sz_bid; 
			if (sample_sz_bid < 2) continue;
			for (int i = 0; i < sample_sz_bid; i++) std_dev += pow(transfer_t_vals[i] - transfer_t_bid_mean, 2);
			std_dev /= sample_sz_bid;
    			std_dev = sqrt(std_dev);
			boost::math::students_t dist(sample_sz_bid - 1);
			double Td = boost::math::quantile(boost::math::complement(dist, alphaCI / 2));
			error_margin_bid = Td*std_dev/sqrt(sample_sz_bid);
			//fprintf(stderr, "\tItter %d:\t mean=%lf, std_dev = %lf, Error margin =%lf\n", sample_sz_bid, cublas_t_mean , std_dev, error_margin_bid);
			if (sample_sz_bid > MICRO_MIN_ITER && error_margin_bid/transfer_t_bid_mean  * 100 <= 5) break; 
		}
		bench_t = csecond() - bench_t;
		fprintf(stderr, "Microbenchmark (dim1 = dim2 = %zu) complete:\t mean_exec_t=%lf ms  ( %lf Gb/s), Error Margin (percentage of mean) = %lf %, Itter = %d, Microbench_t = %lf\n\n", dim, transfer_t_bid_mean  * 1000, Gval_per_s(dim*dim*8, transfer_t_bid_mean), error_margin_bid/transfer_t_bid_mean  * 100, sample_sz_bid, bench_t);
		cudaCheckErrors();

		report_run(filename, dim, dim , transfer_t_mean, error_margin, sample_sz, transfer_t_bid_mean, error_margin_bid, sample_sz_bid, bench_t); 
	}
#else
	/// Local Timers 
	double cpu_timer, t_sq_av, t_sq_min, t_sq_max, t_sq_bid_av, t_sq_bid_min, t_sq_bid_max, bench_t;
	gpu_timer_p cuda_timer = gpu_timer_init();
	for (size_t dim = minDim; dim < maxDim+1; dim+=step){
		if (dim >= step * 16) step*=2;
		t_sq_av = t_sq_max = t_sq_bid_av = t_sq_bid_max = bench_t= 0;
		t_sq_min = t_sq_bid_min = 1e9; 
		fprintf(stderr, "Cublas-chunk Link %s->%s (Chunk %dx%d):\n", print_loc(from), print_loc(to), dim, dim);
		bench_t = csecond();
		for (int it = 0; it < ITER ; it++) {
			cpu_timer = - csecond();
			CoCoMemcpy2DAsync(dest, ldest, src, ldsrc, dim, dim, sizeof(double), to, from, stream);
			cudaStreamSynchronize(stream);
			cpu_timer = csecond() + cpu_timer;
			t_sq_av += cpu_timer;
			if (cpu_timer > t_sq_max) t_sq_max = cpu_timer; 
			if (cpu_timer < t_sq_min) t_sq_min = cpu_timer; 
		}
		cudaCheckErrors();
		t_sq_av = t_sq_av/ITER;
		fprintf(stderr, "Transfer time:\t Average=%lf ms ( %lf Gb/s), Min = %lf ms, Max = %lf ms\n", t_sq_av  * 1000, Gval_per_s(dim*dim*8, t_sq_av), t_sq_min  * 1000, t_sq_max  * 1000);

		fprintf(stderr, "Reverse overlapped Link %s->%s (Chunk %dx%d):\n", print_loc(from), print_loc(to), dim, dim);
		for (int it = 0; it < ITER ; it++) {
			for (int rep = 0; rep < 10 ; rep++) CoCoMemcpy2DAsync(rev_dest, ldest, rev_src, ldsrc, dim, dim, sizeof(double), from, to, reverse_stream);
			gpu_timer_start(cuda_timer, stream);
			CoCoMemcpy2DAsync(dest, ldest, src, ldsrc, dim, dim, sizeof(double), to, from, stream);
			gpu_timer_stop(cuda_timer, stream);
			cudaCheckErrors();
			t_sq_bid_av += gpu_timer_get(cuda_timer);
			if (gpu_timer_get(cuda_timer) > t_sq_bid_max) t_sq_bid_max = gpu_timer_get(cuda_timer); 
			if (gpu_timer_get(cuda_timer) < t_sq_bid_min) t_sq_bid_min = gpu_timer_get(cuda_timer); 
		}
		cudaCheckErrors();
		t_sq_bid_av = t_sq_bid_av/ITER/1000;
		t_sq_bid_min/= 1000;
		t_sq_bid_max/= 1000;
		bench_t = csecond() - bench_t;
		fprintf(stderr, "Transfer time:\t Average=%lf ms ( %lf Gb/s), Min = %lf ms, Max = %lf ms\n", t_sq_bid_av  * 1000, Gval_per_s(dim*dim*8, t_sq_bid_av), t_sq_bid_min  * 1000, t_sq_bid_max  * 1000);
		report_run(filename, dim, dim, t_sq_av, fmax(t_sq_max - t_sq_av, t_sq_av - t_sq_min), ITER, t_sq_bid_av, fmax(t_sq_bid_max - t_sq_bid_av, t_sq_bid_av - t_sq_bid_min), ITER, bench_t);

	}
#endif
	CoCoFree(&src, from);
	CoCoFree(&dest, to); 
	CoCoFree(&rev_src, to);
	CoCoFree(&rev_dest, from); 
	return 0;
}
