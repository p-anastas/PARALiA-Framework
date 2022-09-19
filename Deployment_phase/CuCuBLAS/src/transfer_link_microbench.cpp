///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief A transfer microbenchmark aiming to capture how link overlap effects their bandwidth
///

#include <unistd.h>
#include <cassert>
#include "microbenchmarks.hpp"

/// TODO: This is the number of other-link transfers used for overlaping.
/// The benchmark results are correct ONLY if Link_t > MAX_ASSUMED_OTHER_LINK_TIMES_FASTER * Other_link_t.
/// For current systems 10 is sufficient - larger multipliers increase total benchmark time.
#define MAX_ASSUMED_OTHER_LINK_TIMES_FASTER 10

void report_run(char* filename, size_t dim_1, size_t dim_2, double mean_t, double margin_err, size_t sample_sz, double bench_t){

	FILE* fp = fopen(filename,"a");
	if (!fp) error("report_run: LogFile failed to open");
   	fprintf(fp,"%d,%d, %e,%e,%zu, %e\n", dim_1, dim_2, mean_t, margin_err, sample_sz, bench_t);
        fclose(fp);
}

int main(const int argc, const char *argv[]) {

  int ctr = 1, samples, dev_id, dev_count;

	short loc_src = -2, loc_dest = -2;
	size_t minDim = MIN_DIM_TRANS, maxDim = 0, step = STEP_TRANS;

	switch (argc) {
	case (3):
		loc_dest = atoi(argv[ctr++]);
		loc_src = atoi(argv[ctr++]);
		break;
	default:
		error("Incorrect input arguments. Usage: ./correct_run to loc_src\n");
  	}

	if (loc_src == loc_dest) error("Transfer benchmark@%s %d->%d: Same device\n",TESTBED, loc_src, loc_dest);

	char *filename = (char *) malloc(1024 * sizeof(char));
	sprintf(filename, "%s/Benchmark-Results/CuCuBLAS_link_benchmark_loc_dest-%d_loc_src-%d_%s.log", DEPLOYDB, loc_dest, loc_src, VERSION);
	char *filename_over = (char *) malloc(1024 * sizeof(char));
	sprintf(filename_over, "%s/Benchmark-Results/CuCuBLAS_link_overlap_benchmark_loc_dest-%d_loc_src-%d_%s.log", DEPLOYDB, loc_dest, loc_src, VERSION);
	check_benchmark(filename);
	check_benchmark(filename_over);

	// Define the max size of a benchmark kernel to run on this machine.
	maxDim = std::min(CoCoGetMaxDimSqAsset2D(3, sizeof(double), STEP_TRANS, loc_dest),CoCoGetMaxDimSqAsset2D(3, sizeof(double), STEP_TRANS, loc_src))/2 ;

	fprintf(stderr,"\nTransfer benchmark@%s %d->%d : (%d,%d) with step %d\n", TESTBED, loc_src, loc_dest, minDim, maxDim, step);


	if (minDim < 1) error("Transfer Microbench: Bytes must be > 0");
  void* src, *dest, *rev_src, *rev_dest;

  //Only model pinned memory transfers loc_src host loc_dest dev and visa versa
 	if (loc_src < 0 && loc_dest < 0) error("Transfer Microbench: Both locations are in host");
  else if (loc_src == -2 || loc_dest == -2) error("Transfer Microbench: Not pinned memory (synchronous)");
	short num_devices = LOC_NUM;
	for(int d=0; d < LOC_NUM; d++){
		// Check/Enable peer access between participating GPUs
		CoCoEnableLinks(d, num_devices);
	}

	void* unit_buffs[2*LOC_NUM];
  size_t ldsrc, ldest = ldsrc = maxDim + 1;
	short elemSize = sizeof(double);

	for(short dev_id_idx = 0 ; dev_id_idx < LOC_NUM; dev_id_idx++){
		double timer = csecond();
  	unit_buffs[2*dev_id_idx] = CoCoMalloc(maxDim*(maxDim+1)*elemSize, deidxize(dev_id_idx));
		unit_buffs[2*dev_id_idx+1] = CoCoMalloc(maxDim*(maxDim+1)*elemSize, deidxize(dev_id_idx));
		timer = csecond() - timer;
		//fprintf(stderr, "Allocation of 2 Tiles (dim1 = dim2 = %zu) in unit_id = %2d complete:\t alloc_timer=%lf ms\n",
		//maxDim, deidxize(dev_id_idx), timer  * 1000);

	}

	for(short dev_id_idx = 0 ; dev_id_idx < LOC_NUM; dev_id_idx++){
		double timer = csecond();
	  CoCoVecInit((double*)unit_buffs[2*dev_id_idx], maxDim*(maxDim+1), 42, deidxize(dev_id_idx));
	  CoCoVecInit((double*)unit_buffs[2*dev_id_idx+1], maxDim*(maxDim+1), 43, deidxize(dev_id_idx));
		timer = csecond() - timer;
		//fprintf(stderr, "Initialization of 2 Tiles in unit_id = %2d complete:\t init_timer=%lf ms\n",
		//maxDim, deidxize(dev_id_idx), timer  * 1000);
	}

  CQueue_p transfer_queue_list[LOC_NUM][LOC_NUM] = {{NULL}};
	for(short dev_id_idx = 0 ; dev_id_idx < LOC_NUM; dev_id_idx++){
		for(short dev_id_idy = 0 ; dev_id_idy < LOC_NUM; dev_id_idy++)
			if(dev_id_idy!=dev_id_idx){
					if (!transfer_queue_list[dev_id_idx][dev_id_idy]){
						//printf("dev_id = %d, dev_id_idx = %d, dev_id_idy = %d, LOC_NUM = %d\n", dev_id, dev_id_idx, dev_id_idy, LOC_NUM);
						short queue_id = (dev_id_idy == LOC_NUM - 1) ? deidxize(dev_id_idx): deidxize(dev_id_idy);
						transfer_queue_list[dev_id_idx][dev_id_idy] = new CommandQueue(queue_id);
				}
			}
			else transfer_queue_list[dev_id_idx][dev_id_idy] = NULL;
	}

	fprintf(stderr, "Warming up...\n");
	/// Warmup.
	for (int it = 0; it < 10; it++) CoCoMemcpy2DAsync(unit_buffs[2*idxize(loc_dest)], ldest,
									unit_buffs[2*idxize(loc_src)], ldsrc,
									maxDim, maxDim, elemSize,
									loc_dest, loc_src, transfer_queue_list[idxize(loc_dest)][idxize(loc_src)]);
	transfer_queue_list[idxize(loc_dest)][idxize(loc_src)]->sync_barrier();
	CoCoSyncCheckErr();
	size_t dim;
#ifdef AUTO_BENCH_USE_BOOST
	double cpu_timer, transfer_t_vals[MICRO_MAX_ITER], transfer_t_sum, transfer_t_mean = 0, bench_t, error_margin;
	double transfer_t_bid_sum, transfer_t_bid_mean, error_margin_bid;
	size_t sample_sz, sample_sz_bid;
	CoCoPeLiaSelectDevice(loc_dest);
	Event_timer_p device_timer = new Event_timer(loc_dest);
	for (dim = minDim; dim < MAX_DIM_BLAS3; dim+=step){ // maxDim+1
		if (dim >= step * 8) step*=2;
		if (dim > maxDim) break;
		transfer_t_sum = transfer_t_mean = bench_t = error_margin = 0;
		fprintf(stderr, "Cublas-chunk Link %d->%d (Chunk %dx%d):\n", loc_src, loc_dest, dim, dim);
		sample_sz = 0;
		bench_t = csecond();
		double std_dev = 0;
		for (sample_sz = 1; sample_sz < MICRO_MAX_ITER + 1; sample_sz++) {
			cpu_timer = csecond();
			CoCoMemcpy2DAsync(unit_buffs[2*idxize(loc_dest)], ldest,
											unit_buffs[2*idxize(loc_src)], ldsrc,
											dim, dim, elemSize,
											loc_dest, loc_src, transfer_queue_list[idxize(loc_dest)][idxize(loc_src)]);
			transfer_queue_list[idxize(loc_dest)][idxize(loc_src)]->sync_barrier();
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
		CoCoSyncCheckErr();

		report_run(filename, dim, dim , transfer_t_mean, error_margin, sample_sz, bench_t);
	}
#else
	/// Local Timers
	double cpu_timer, transfer_t_mean = 0, t_sq_min, t_sq_max, bench_t;
	CoCoPeLiaSelectDevice(loc_dest);
	Event_timer_p device_timer = new Event_timer(loc_dest);
	for (dim = minDim; dim < maxDim+1; dim+=step){
		if (dim >= step * 8) step*=2;
		if (dim > maxDim) break;
		transfer_t_mean = t_sq_max = t_sq_bid_av = t_sq_bid_max = bench_t= 0;
		t_sq_min = t_sq_bid_min = 1e9;
		fprintf(stderr, "Cublas-chunk Link %d->%d (Chunk %dx%d):\n", loc_src, loc_dest, dim, dim);
		bench_t = csecond();
		for (int it = 0; it < ITER ; it++) {
			cpu_timer = - csecond();
			CoCoMemcpy2DAsync(unit_buffs[2*idxize(loc_dest)], ldest,
											unit_buffs[2*idxize(loc_src)], ldsrc,
											dim, dim, elemSize,
											loc_dest, loc_src, transfer_queue_list[idxize(loc_dest)][idxize(loc_src)]);
			transfer_queue_list[idxize(loc_dest)][idxize(loc_src)]->sync_barrier();
			cpu_timer = csecond() + cpu_timer;
			transfer_t_mean += cpu_timer;
			if (cpu_timer > t_sq_max) t_sq_max = cpu_timer;
			if (cpu_timer < t_sq_min) t_sq_min = cpu_timer;
		}
		CoCoSyncCheckErr();
		bench_t = csecond() - bench_t;
		transfer_t_mean = transfer_t_mean/ITER;
		fprintf(stderr, "Transfer time:\t Average=%lf ms ( %lf Gb/s), Min = %lf ms, Max = %lf ms\n", transfer_t_mean  * 1000, Gval_per_s(dim*dim*8, transfer_t_mean), t_sq_min  * 1000, t_sq_max  * 1000);
		report_run(filename, dim, dim, transfer_t_mean, fmax(t_sq_max - transfer_t_mean, transfer_t_mean - t_sq_min), ITER, bench_t);
	}
#endif

	double shared_timer;
	for(short dev_id_idx = 0 ; dev_id_idx < LOC_NUM; dev_id_idx++){
		for(short dev_id_idy = 0 ; dev_id_idy < LOC_NUM; dev_id_idy++){
			if (dev_id_idx == dev_id_idy ||
				(dev_id_idx == idxize(loc_dest) && dev_id_idy == idxize(loc_src))) continue;
			bench_t = csecond();
			for(int itt = 0 ; itt < MAX_ASSUMED_OTHER_LINK_TIMES_FASTER; itt++) CoCoMemcpy2DAsync(unit_buffs[2*dev_id_idx+1], ldest,
										unit_buffs[2*dev_id_idy+1], ldsrc,
										dim, dim, elemSize,
										deidxize(dev_id_idx), deidxize(dev_id_idy), transfer_queue_list[dev_id_idx][dev_id_idy]);
			device_timer->start_point(transfer_queue_list[idxize(loc_dest)][idxize(loc_src)]);

			CoCoMemcpy2DAsync(unit_buffs[2*idxize(loc_dest)], ldest,
										unit_buffs[2*idxize(loc_src)], ldsrc,
										dim, dim, elemSize,
										loc_dest, loc_src, transfer_queue_list[idxize(loc_dest)][idxize(loc_src)]);
			device_timer->stop_point(transfer_queue_list[idxize(loc_dest)][idxize(loc_src)]);

			shared_timer = device_timer->sync_get_time()/1000;
			transfer_queue_list[idxize(loc_dest)][idxize(loc_src)]->sync_barrier();
			transfer_queue_list[dev_id_idx][dev_id_idy]->sync_barrier();
			CoCoSyncCheckErr();
			bench_t = csecond() - bench_t;
			//fprintf(stderr, "Shared Link (%d->%d) transfer complete:\t shared_timer=%lf ms  ( %lf Gb/s)\n\n",
			//	deidxize(dev_id_idy), deidxize(dev_id_idx), shared_timer  * 1000, Gval_per_s(maxDim*maxDim*elemSize, shared_timer));

			if (transfer_t_mean < shared_timer*(1-NORMALIZE_NEAR_SPLIT_LIMIT)) fprintf(stderr, "Link(%2d->%2d) & Link(%2d->%2d) partially shared: Shared_BW: %1.2lf %\n\n",
				loc_src, loc_dest, deidxize(dev_id_idy), deidxize(dev_id_idx), 100*transfer_t_mean/shared_timer);

			report_run(filename_over, deidxize(dev_id_idx), deidxize(dev_id_idy), transfer_t_mean, shared_timer, MAX_ASSUMED_OTHER_LINK_TIMES_FASTER, bench_t);

		}
	}

	CoCoSyncCheckErr();
	for(short dev_id_idx = 0 ; dev_id_idx < LOC_NUM; dev_id_idx++){
		double timer = csecond();
	  CoCoFree(unit_buffs[2*dev_id_idx], deidxize(dev_id_idx));
	  CoCoFree(unit_buffs[2*dev_id_idx+1], deidxize(dev_id_idx));
		timer = csecond() - timer;
		//fprintf(stderr, "De-allocation of 2 Tiles in unit_id = %2d complete:\t init_timer=%lf ms\n",
		//maxDim, deidxize(dev_id_idx), timer  * 1000);
	}
  return 0;
}
