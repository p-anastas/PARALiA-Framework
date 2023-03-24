///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
/// \author Theodoridis Aristomenis (atheodor@cslab.ece.ntua.gr)
///
/// \brief The start of Zawarudo
///

#include "unihelpers.hpp"
#include "PARALiA.hpp"
#include "BackenedLibsWrapped.hpp"
#include "Testing.hpp"

#include "backend_wrappers.hpp"

#define CBLASXT_MAX_SAFE_TILE 10000

int main(const int argc, const char *argv[]) {
	char TransA;
  	double alpha, beta;
	long int M_in = -2, N_in = -1, x_dim = -1, y_dim = -1;
	short A_loc, x_loc, y_loc, y_out_loc;
    long int incx, incy;
	ATC_p predef_control_values = NULL, return_values = NULL;
	ParseInputLvl2(argc, argv, &predef_control_values, &TransA, &alpha, &beta, &M_in, &N_in, &incx, &incy, &A_loc, &x_loc, &y_loc, &y_out_loc);

	char *filename = (char *) malloc(1024* sizeof(char));
	if (predef_control_values!= NULL){
		if(predef_control_values->T > 0) {
			if (predef_control_values->T > M_in || predef_control_values->T > N_in )
				error("Given Tin=%ld bigger than problem dim\n", predef_control_values->T);
			else if (predef_control_values->T > M_in/1.5 && predef_control_values->T > N_in/1.5)
				warning("Given Tin=%ld bigger than all problem dims/1.5\n", predef_control_values->T);
		}
		sprintf(filename, "%s/CoCoPeLiaDgemvRunner_predefined_vals_%s_%s_%s.log",
			TESTLIBDIR, CoCoDistributionPrint(), CoCoImplementationPrint(), VERSION);
	}
	else sprintf(filename, "%s/CoCoPeLiaDgemvRunner_%s_%s_%s.log",
		TESTLIBDIR, CoCoDistributionPrint(), CoCoImplementationPrint(), VERSION);
#ifdef CHECKLOG
	CheckLogLvl2(filename, predef_control_values, TransA, alpha, beta, M, N, incx, incy, A_loc, B_loc, C_loc, C_out_loc);
#endif
	/// Matrix Layouts for CPU GEMV
	CBLAS_TRANSPOSE cpu_op_A;    // CblasNoTrans, CblasTrans
	cublasOperation_t gpu_op_A; // CUBLAS_OP_N, CUBLAS_OP_T

	long int ldA = M_in;
	TransposeTranslate(TransA, &cpu_op_A, &gpu_op_A, &ldA, M_in, N_in);
	// Transpose translate changes ldA based on TransA, which is not intended in gemv. 
	ldA = M_in; 
	if (TransA == 'N'){
		x_dim = N_in;
		y_dim = M_in;
	}
	else{
		x_dim = M_in;
		y_dim = N_in;
	}

	/// Local Timers
	double cpu_timer = csecond();

	fprintf(stderr, "\nAllocating memory...");

	double *A, *x, *y;
	// allocate in device if loc = 0, otherwise allocate in pinned memory for benchmarks
	A = (double*) CoCoMalloc(M_in * N_in *sizeof(double), A_loc);
    x = (double*) CoCoMalloc(x_dim * incx *sizeof(double), x_loc);
	y = (double*) CoCoMalloc(y_dim * incy *sizeof(double), y_loc);

	CoCoSyncCheckErr();
	cpu_timer  = csecond() - cpu_timer;
	fprintf(stderr, "done.\nAlloc time:\t%lf ms\n\n",  cpu_timer  * 1000);
	cpu_timer = csecond();
	fprintf(stderr, "Initializing to random values (VALIDATE)...");
	CoCoVecInit(A, N_in * M_in, 42, A_loc);
    CoCoVecInit(x, x_dim * incx, 43, x_loc);
	CoCoVecInit(y, y_dim * incy, 44, y_loc);
	CoCoSyncCheckErr();
	cpu_timer  = csecond() - cpu_timer ;
	fprintf(stderr, "done.\nInit time:\t%lf ms\n\n",  cpu_timer  * 1000);

#ifdef RUNVALIDATION
    double *y_buf;
    y_buf  = (double*) malloc(y_dim * incy *sizeof(double));
    CoCoMemcpy(y_buf, y,  y_dim * incy *sizeof(double), -2, y_loc);

    double *y_out, *y_out1;
    y_out  = (double*) malloc(y_dim * incy *sizeof(double));
    y_out1  = (double*) malloc(y_dim * incy *sizeof(double));

    CoCoMemcpy(y, y_buf, y_dim * incy *sizeof(double), y_loc, -2);

	// Call for Validate
	if (predef_control_values!= NULL) return_values = PARALiADgemvControled(TransA, M_in, N_in, alpha, A, ldA, x, incx, beta, y, incy, predef_control_values);
	else error("Autotuning parameters for dgemv not implemented.\n");//return_values = PARALiADgemv(TransA, M, N, alpha, A, ldA, x, incx, beta, y, incy);
	CoCoSyncCheckErr();
	for (int i = 0; i< LOC_NUM; i++) PARALiADevCacheFree(deidxize(i));

    CoCoMemcpy(y_out, y, y_dim * incy *sizeof(double), -2, y_loc);
	CoCoMemcpy(y, y_buf, y_dim * incy *sizeof(double), y_loc, -2);

    // Validate with cuBLAS in device 0 (questionable but CPU validation can be slower)
	int dev_ids[1] = {0};

	double *A_dev, *x_dev, *y_dev;
    if (A_loc == 0) A_dev = A;
	else {
		A_dev = (double*) CoCoMalloc(M_in * N_in *sizeof(double), 0);
		CoCoMemcpy(A_dev, A,  M_in * N_in *sizeof(double), 0, A_loc);
	}
	if (x_loc == 0) x_dev = x;
	else {
		x_dev = (double*) CoCoMalloc(x_dim * incx *sizeof(double), 0);
		CoCoMemcpy(x_dev, x,  x_dim * incx *sizeof(double), 0, x_loc);
	}
	if (y_loc == 0) y_dev = y;
	else{
		y_dev = (double*) CoCoMalloc(y_dim * incy *sizeof(double), 0);
		CoCoMemcpy(y_dev, y,  y_dim * incy *sizeof(double), 0, y_loc);
	}

	cuBLASDgemvWrap(TransA, M_in, N_in, alpha, A_dev, ldA, x_dev, incx, beta, y_dev, incy, 0 , 1, dev_ids);
	CoCoSyncCheckErr();
	CoCoMemcpy(y_out1, y_dev,  y_dim * incy *sizeof(double), -2, 0);
	if(Dtest_equality(y_out1, y_out,  y_dim * incy) < 9) error("Insufficient accuracy for benchmarks\n");

    CoCoMemcpy(y, y_buf,  y_dim * incy *sizeof(double), y_loc, -2);
    free(y_out);
	free(y_out1);
	free(y_buf);
	if (x_loc) CoCoFree(x_dev, 0);
	if (y_loc) CoCoFree(y_dev, 0);
#endif

	cpu_timer = csecond();
	if (predef_control_values!= NULL) return_values = PARALiADgemvControled(TransA, M_in, N_in, alpha, A, ldA, x, incx, beta, y, incy, predef_control_values);
	else error("Autotuning parameters for dgemv not implemented.\n");//return_values = PARALiADgemv(TransA, TransB, M, N, K, alpha, A, ldA, B, ldB, beta, C , ldC);
	CoCoSyncCheckErr();
	cpu_timer  = csecond() - cpu_timer;

#ifdef CHECKLOG
	CheckLogLvl3(filename, return_values, TransA, alpha, beta, M, N, A_loc, x_loc, y_loc, y_out_loc);
#endif
	// Store the time required for the first call (+ 1-time overheads etc)
	//StoreLogLvl3(filename, return_values, TransA, TransB, alpha, beta, M, N, K, A_loc, B_loc, C_loc, C_out_loc, cpu_timer);

	double first_over_t = cpu_timer;

	short warmup_bench_it = 100;
	if ( M_in >= 20000 && N_in >= 20000 ) warmup_bench_it = 20;
	for(int it = 0; it < warmup_bench_it; it++){
		if (predef_control_values!= NULL) return_values = PARALiADgemvControled(TransA, M_in, N_in, alpha, A, ldA, x, incx, beta, y, incy, predef_control_values);
		else error("Autotuning parameters for dgemv not implemented.\n");//return_values = PARALiADgemv(TransA, TransB, M, N, K, alpha, A, ldA, B, ldB, beta, C , ldC);
	}
	CoCoSyncCheckErr();

	double min_t = first_over_t, max_t = 0, avg_t = 0;
	cpu_timer = csecond();
	short bench_it = 100;
	if ( M_in >= 20000 && N_in >= 20000) bench_it = 20;
	for(int it = 0; it < bench_it; it++){
		cpu_timer = csecond();
		if (predef_control_values!= NULL) return_values = PARALiADgemvControled(TransA, M_in, N_in, alpha, A, ldA, x, incx, beta, y, incy, predef_control_values);
		else error("Autotuning parameters for dgemv not implemented.\n");//return_values = PARALiADgemv(TransA, TransB, M, N, K, alpha, A, ldA, B, ldB, beta, C , ldC);
		CoCoSyncCheckErr();
		cpu_timer = csecond() - cpu_timer;
		StoreLogLvl2(filename, return_values, TransA, alpha, beta, M_in, N_in, incx, incy, A_loc, x_loc, y_loc, y_out_loc, cpu_timer, return_values->pred_t, return_values->pred_J);
		if ( cpu_timer < min_t ) min_t = cpu_timer;
		if ( cpu_timer > max_t ) max_t = cpu_timer;
		avg_t += cpu_timer;
	}
	avg_t/=bench_it;
	fprintf(stderr, "CoCopeLia (%s):\n\tfirst_it_t = %lf ms ( %lf Gflops/s )\n\tavg_t = %lf ms ( %lf Gflops/s )\n\tmin_t = %lf ms ( %lf Gflops/s )\n\tmax_t = %lf ms ( %lf Gflops/s )\n",
	return_values->print_csv(),
	first_over_t  * 1000, Gval_per_s(gemv_flops(M_in,N_in),first_over_t),
	avg_t  * 1000, Gval_per_s(gemv_flops(M_in,N_in),avg_t),
	min_t  * 1000, Gval_per_s(gemv_flops(M_in,N_in),min_t),
	max_t  * 1000, Gval_per_s(gemv_flops(M_in,N_in),max_t));

	for (int i = 0; i< LOC_NUM; i++) PARALiADevCacheFree(deidxize(i));

	CoCoSyncCheckErr();
	CoCoFree(A, A_loc);
    CoCoFree(x, x_loc);
	CoCoFree(y, y_loc);
	return 0;
}
