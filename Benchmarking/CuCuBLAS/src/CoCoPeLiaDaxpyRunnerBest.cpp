///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The start of Zawarudo
///

#include "backend_wrappers.hpp"

#include "unihelpers.hpp"
#include "CoCoPeLia.hpp"
#include "Testing.hpp"
#include "BackenedLibsWrapped.hpp"

#define DAXPY_MAX_SAFE_TILE 100000000

int main(const int argc, const char *argv[]) {
  double alpha;
	size_t N;
	short x_loc, y_loc, x_out_loc, y_out_loc;
	size_t incx, incy;

	CoControl_p predef_control_values = NULL, return_values = NULL;
	ParseInputLvl1(argc, argv, &predef_control_values, &alpha, &N, &incx, &incy, &x_loc, &y_loc, &x_out_loc, &y_out_loc);

	char *filename = (char *) malloc(256* sizeof(char));
	if (predef_control_values!= NULL){
		error("CoCoPeLiaDaxpyRunnerBest: I am not supposed to be used with specific inputs. You probably need CoCoPeLiaDaxpyRunner\n");
	}
	else sprintf(filename, "%s/CoCoPeLiaDaxpyRunnerBest_%s_%s_%s.log",
		TESTLIBDIR, CoCoDistributionPrint(), CoCoImplementationPrint(), VERSION);
#ifdef CHECKLOG
	CheckLogLvl1(filename, predef_control_values, alpha, N, incx, incy, x_loc, y_loc, x_out_loc, y_out_loc);
#endif
	predef_control_values = (CoControl_p) malloc(sizeof(CoControl_p*));

	/// Local Timers
	double cpu_timer = csecond();

	fprintf(stderr, "\nAllocating memory...");

	double *x, *y;
	// allocate in device if loc = 0, otherwise allocate in pinned memory for benchmarks
	x = (double*) CoCoMalloc(N * incx *sizeof(double), x_loc);
	y = (double*) CoCoMalloc(N * incy *sizeof(double), y_loc);

	CoCoSyncCheckErr();
	cpu_timer  = csecond() - cpu_timer;
	fprintf(stderr, "done.\nAlloc time:\t%lf ms\n\n",  cpu_timer  * 1000);
	cpu_timer = csecond();
	fprintf(stderr, "Initializing to random values (VALIDATE)...");
	CoCoVecInit(x, N * incx, 42, x_loc);
	CoCoVecInit(y, N * incy, 43, y_loc);
	CoCoSyncCheckErr();
	cpu_timer  = csecond() - cpu_timer ;
	fprintf(stderr, "done.\nInit time:\t%lf ms\n\n",  cpu_timer  * 1000);

#ifdef RUNVALIDATION
	double *y_buf;
	y_buf  = (double*) malloc(N * incy *sizeof(double));
	CoCoMemcpy(y_buf, y,  N * incy *sizeof(double), -2, y_loc);
#endif

	size_t best_T = (size_t) fmin(N/DEV_NUM,DAXPY_MAX_SAFE_TILE);
	predef_control_values-> cache_limit = 0;
	predef_control_values-> dev_num = -1;
	predef_control_values-> T = best_T;
	// Warmup
	for(int it = 0; it < 10; it++){
		return_values = CoCopeLiaDaxpyControled(N, alpha, x, incx, y, incy, predef_control_values);
		CoCoSyncCheckErr();
	}

	cpu_timer  = csecond();
	return_values = CoCopeLiaDaxpyControled(N, alpha, x, incx, y, incy, predef_control_values);
	CoCoSyncCheckErr();
	cpu_timer  = csecond() - cpu_timer;

	short bench_it = 10;
	double best_t = 10e9;//cpu_timer;
	for (size_t T_trial = (((size_t)fmax(N/(32*32),512*512))/(512*512))*(512*512);
		T_trial <= (size_t) fmin(N/DEV_NUM,DAXPY_MAX_SAFE_TILE); T_trial+=(512*512)){
			fprintf(stderr,"Running CoCopeLia DAXPY-> N = %zu, T = %zu\n", N, T_trial);
			predef_control_values->T = T_trial;
			cpu_timer  = csecond();
			for(int it = 0; it < bench_it; it++){
				return_values = CoCopeLiaDaxpyControled(N, alpha, x, incx, y, incy, predef_control_values);
				CoCoSyncCheckErr();
			}
			cpu_timer  = (csecond() - cpu_timer)/bench_it;
			fprintf(stderr, "Total time:\t%lf ms\n", cpu_timer  * 1000);
			if (cpu_timer < best_t){
				best_t = cpu_timer;
				best_T = T_trial;
			}
	}
	fprintf(stderr, "\nCoCopeLia DAXPY T_best = %zu : t = %lf ms ( %lf Gflops/s )\n\n", best_T, best_t  * 1000, Gval_per_s(daxpy_flops(N),best_t));
	predef_control_values-> T = best_T;
	for (int i = 0; i< LOC_NUM; i++) CoCopeLiaDevCacheFree(deidxize(i));

#ifdef RUNVALIDATION
	double *y_out, *y_out1;
	y_out  = (double*) malloc(N * incy *sizeof(double));
	y_out1  = (double*) malloc(N * incy *sizeof(double));

	CoCoMemcpy(y, y_buf, N * incy *sizeof(double), y_loc, -2);

	// Call for Validate
	if (predef_control_values!= NULL) return_values = CoCopeLiaDaxpyControled(N, alpha, x, incx, y, incy, predef_control_values);
	else return_values = CoCopeLiaDaxpy(N, alpha, x, incx, y, incy);
	CoCoSyncCheckErr();
	for (int i = 0; i< LOC_NUM; i++) CoCopeLiaDevCacheFree(deidxize(i));

	CoCoMemcpy(y_out, y, N * incy *sizeof(double), -2, y_loc);
	CoCoMemcpy(y, y_buf, N * incy *sizeof(double), y_loc, -2);

	// Validate with cuBLAS in device 0 (questionable but CPU validation can be slower)
	int dev_ids[1] = {0};

	double *x_dev, *y_dev;
	if (x_loc == 0) x_dev = x;
	else {
    x_dev = (double*) CoCoMalloc(N * incx *sizeof(double), 0);
    CoCoMemcpy(x_dev, x,  N * incx *sizeof(double), 0, x_loc);
  }
	if (y_loc == 0) y_dev = y;
	else{
    y_dev = (double*) CoCoMalloc(N * incy *sizeof(double), 0);
    CoCoMemcpy(y_dev, y,  N * incy *sizeof(double), 0, y_loc);
  }

	cuBLASDaxpyWrap(N, alpha, x_dev, incx, y_dev, incy, 0 , 1, dev_ids);
	CoCoSyncCheckErr();
	CoCoMemcpy(y_out1, y_dev,  N * incy *sizeof(double), -2, 0);
 	if(Dtest_equality(y_out1, y_out,  N * incy) < 9) error("Insufficient accuracy for benchmarks\n");

 	CoCoMemcpy(y, y_buf,  N * incy *sizeof(double), y_loc, -2);
	free(y_out);
	free(y_out1);
	free(y_buf);
	if (x_loc) CoCoFree(x_dev, 0);
	if (y_loc) CoCoFree(y_dev, 0);

#endif

	cpu_timer = csecond();
	// First call for Validate
	if (predef_control_values!= NULL) return_values = CoCopeLiaDaxpyControled(N, alpha, x, incx, y, incy, predef_control_values);
	else return_values = CoCopeLiaDaxpy(N, alpha, x, incx, y, incy);
	CoCoSyncCheckErr();
	cpu_timer  = csecond() - cpu_timer;

#ifdef CHECKLOG
	CheckLogLvl1(filename, predef_control_values, alpha, N, incx, incy, x_loc, y_loc, x_out_loc, y_out_loc);
#endif
	// Store the time required for the first call (+ 1-time overheads etc)
	StoreLogLvl1(filename, predef_control_values, alpha, N, incx, incy, x_loc, y_loc, x_out_loc, y_out_loc, cpu_timer);

	double first_over_t = cpu_timer;

	double min_t = first_over_t, max_t = 0, avg_t = 0;
	cpu_timer = csecond();
  bench_it = 100;
	if ( N >= 8192*8192 ) bench_it = 10;
	for(int it = 0; it < bench_it; it++){
		cpu_timer = csecond();
		if (predef_control_values!= NULL) return_values = CoCopeLiaDaxpyControled(N, alpha, x, incx, y, incy, predef_control_values);
		else return_values = CoCopeLiaDaxpy(N, alpha, x, incx, y, incy);
		CoCoSyncCheckErr();
		cpu_timer = csecond() - cpu_timer;
		StoreLogLvl1(filename, predef_control_values, alpha, N, incx, incy, x_loc, y_loc, x_out_loc, y_out_loc, cpu_timer);
		if ( cpu_timer < min_t ) min_t = cpu_timer;
		if ( cpu_timer > max_t ) max_t = cpu_timer;
		avg_t += cpu_timer;
	}
	avg_t/=bench_it;
	fprintf(stderr, "CoCopeLia Daxpy(%s):\n\tfirst_it_t = %lf ms ( %lf Gflops/s )\n\tavg_t = %lf ms ( %lf Gflops/s )\n\tmin_t = %lf ms ( %lf Gflops/s )\n\tmax_t = %lf ms ( %lf Gflops/s )\n",
	CoControlPrint(return_values),
	first_over_t  * 1000, Gval_per_s(daxpy_flops(N),first_over_t),
	avg_t  * 1000, Gval_per_s(daxpy_flops(N),avg_t),
	min_t  * 1000, Gval_per_s(daxpy_flops(N),min_t),
	max_t  * 1000, Gval_per_s(daxpy_flops(N),max_t));

	for (int i = 0; i< LOC_NUM; i++) CoCopeLiaDevCacheFree(deidxize(i));

	CoCoSyncCheckErr();
	CoCoFree(x, x_loc);
	CoCoFree(y, y_loc);
	return 0;
}
