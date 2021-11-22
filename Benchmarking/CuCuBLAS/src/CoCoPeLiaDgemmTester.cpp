///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The start of Zawarudo
///

#include "backend_wrappers.hpp"
#include "unihelpers.hpp"
#include "CoCoPeLia.hpp"
#include "cuBLASXtWrapped.hpp"

#define CBLASXT_MAX_SAFE_TILE 10000

int main(const int argc, const char *argv[]) {

	short run_large_flag, mode;

	char TransA, TransB;
  	double alpha, beta;
	size_t M, N, K, T;
	short A_loc, B_loc, C_loc, C_out_loc;
	double cpu_ratio = 0;
	size_t ldA, ldB, ldC;

	if(argc != 2) error("Incorrect input arguments. Usage: ./correct_run run_large_flag\n");
	// Control Parameters
	run_large_flag = atoi(argv[1]);

	/// Local Timers
	double cpu_timer = csecond();
	fprintf(stderr, "CoCoPeLiaDgemmTester: Initallizing tests for CoCopeLiaDgemmTile with run_large_flag = %d\n", run_large_flag);

#ifndef DEV_NUM
#define DEV_NUM 1
#endif
	int dev_ids[DEV_NUM];
	for(int i = 0; i< DEV_NUM; i++) dev_ids[i] = i;
	fprintf(stderr, "\n==============================================================================================================================\n");
	fprintf(stderr, "CoCoPeLiaDgemmTester: Allocating CPU buffers...->100 MB...");
	ldA = ldB = ldC = M = N = K = 8192;

	A_loc = B_loc = C_loc = -1;

	double *A, *B, *C;
	A = (double*) CoCoMalloc(M * K*sizeof(double), A_loc);
	B = (double*) CoCoMalloc(N * K*sizeof(double), B_loc);
	C = (double*) CoCoMalloc(M * N*sizeof(double), C_loc);

	CoCoSyncCheckErr();
	cpu_timer  = csecond() - cpu_timer;
	fprintf(stderr, "done.\nAlloc time:\t%lf ms\n\n",  cpu_timer  * 1000);
	cpu_timer = csecond();
	fprintf(stderr, "Initializing to random values...");
	CoCoVecInit(A, K * M, 42, A_loc);
	CoCoVecInit(B, K * N, 43, B_loc);
	CoCoVecInit(C, M * N, 44, C_loc);
	CoCoSyncCheckErr();
	cpu_timer  = csecond() - cpu_timer ;
	fprintf(stderr, "done.\nInit time:\t%lf ms\n\n",  cpu_timer  * 1000);

	double *C_comp = (double*) CoCoMalloc(M * N*sizeof(double), C_loc);
	CoCoMemcpy(C_comp, C,  M * N *sizeof(double), -1, -1);

	fprintf(stderr, "\n==============================================================================================================================\n");
	fprintf(stderr, "CoCoPeLiaDgemmTester: Testing Square Problems < 100 MB:\n\n");
	TransA = TransB = 'N';
	alpha = 1.23;
	beta = 0.9876;
	for (int dim = 256; dim <= M; dim*=2){
		fprintf(stderr, "M=N=K=%d: Gflops/s -> ", dim);
		cpu_timer = csecond();
		CoCopeLiaDgemm(TransA, TransB, dim, dim, dim, alpha, A, ldA, B, ldB, beta, C , ldC);
		CoCoSyncCheckErr();
		cpu_timer  = csecond() - cpu_timer;
		double comp_flops = Gval_per_s(dgemm_flops(dim,dim,dim),cpu_timer);
		fprintf(stderr, "CoCopeLia: %.1lf, ", comp_flops);
		cpu_timer = csecond();
		T = fmin(dim,fmin(dim,dim))/2;

		cuBLASXtDgemmWrap(TransA, TransB, dim, dim, dim, alpha, A, ldA, B, ldB, beta, C_comp, ldC,  T, cpu_ratio, DEV_NUM, dev_ids);
		CoCoSyncCheckErr();
		cpu_timer  = csecond() - cpu_timer;
		fprintf(stderr, "cuBLASXT: %.1lf\n", Gval_per_s(dgemm_flops(dim,dim,dim),cpu_timer));
		if (comp_flops < Gval_per_s(dgemm_flops(dim,dim,dim),cpu_timer)) warning("Inferior Perf to cublasXt\n");
		if(Dtest_equality(C, C_comp, dim * dim) < 5) error("Insufficient accuracy for benchmarks\n");

		CoCoSyncCheckErr();
	}

	CoCoVecInit(C, M * N, 44, C_loc);
	CoCoMemcpy(C_comp, C,  M * N *sizeof(double), -1, -1);

	fprintf(stderr, "\n==============================================================================================================================\n");
	fprintf(stderr, "CoCoPeLiaDgemmTester: Testing Non-Square Problems < 100 MB:\n\n");
	alpha = 1.23;
	beta = 0.9876;
	for (int dim1 = 256; dim1 <= M; dim1*=4) for (int dim2 = 256; dim2 <= N; dim2*=4) for (int dim3 = 256; dim3 <= K; dim3*=4) if ( dim1 != dim2 || dim2 != dim3 || dim1!= dim3){
		fprintf(stderr, "M=%d,N=%d,K=%d: Gflops/s -> ", dim1, dim2, dim3);
		cpu_timer = csecond();
		CoCopeLiaDgemm(TransA, TransB, dim1, dim2, dim3, alpha, A, ldA, B, ldB, beta, C , ldC);
		CoCoSyncCheckErr();
		cpu_timer  = csecond() - cpu_timer;
		double comp_flops =  Gval_per_s(dgemm_flops(dim1,dim2,dim3),cpu_timer);
		fprintf(stderr, "CoCopeLia: %.1lf, ", comp_flops);
		cpu_timer = csecond();
		T = fmin(dim1,fmin(dim2,dim3))/2;
		cuBLASXtDgemmWrap(TransA, TransB, dim1, dim2, dim3, alpha, A, ldA, B, ldB, beta, C_comp, ldC,  T, cpu_ratio, DEV_NUM, dev_ids);
		CoCoSyncCheckErr();
		cpu_timer  = csecond() - cpu_timer;
		fprintf(stderr, "cuBLASXT: %.1lf\n", Gval_per_s(dgemm_flops(dim1,dim2,dim3),cpu_timer));
		if (comp_flops < Gval_per_s(dgemm_flops(dim1,dim2,dim3),cpu_timer)) warning("Inferior Perf to cublasXt\n");
		if(Dtest_equality(C, C_comp, dim1 * dim2) < 5) error("Insufficient accuracy for benchmarks\n");
		CoCoSyncCheckErr();
	}

	CoCoVecInit(C, M * N, 44, C_loc);
	CoCoMemcpy(C_comp, C,  M * N *sizeof(double), -1, -1);

	fprintf(stderr, "\n==============================================================================================================================\n");
	fprintf(stderr, "CoCoPeLiaDgemmTester: Testing Weird Dimension Problems < 100 MB:\n\n");
	alpha = 1.23;
	beta = 0.9876;
	for (int dim1 = 289; dim1 <= M; dim1*=4) for (int dim2 = 353; dim2 <= N; dim2*=4) for (int dim3 = 307; dim3 <= K; dim3*=4) if ( dim1 != dim2 || dim2 != dim3 || dim1!= dim3){
		fprintf(stderr, "M=%d,N=%d,K=%d: Gflops/s -> ", dim1, dim2, dim3);
		cpu_timer = csecond();
		CoCopeLiaDgemm(TransA, TransB, dim1, dim2, dim3, alpha, A, ldA, B, ldB, beta, C , ldC);
		CoCoSyncCheckErr();
		cpu_timer  = csecond() - cpu_timer;
		double comp_flops =  Gval_per_s(dgemm_flops(dim1,dim2,dim3),cpu_timer);
		fprintf(stderr, "CoCopeLia: %.1lf, ", comp_flops);
		cpu_timer = csecond();
		T = fmin(dim1,fmin(dim2,dim3))/2;
		cuBLASXtDgemmWrap(TransA, TransB, dim1, dim2, dim3, alpha, A, ldA, B, ldB, beta, C_comp, ldC,  T, cpu_ratio, DEV_NUM, dev_ids);
		CoCoSyncCheckErr();
		cpu_timer  = csecond() - cpu_timer;
		fprintf(stderr, "cuBLASXT: %.1lf\n", Gval_per_s(dgemm_flops(dim1,dim2,dim3),cpu_timer));
		if (comp_flops < Gval_per_s(dgemm_flops(dim1,dim2,dim3),cpu_timer)) warning("Inferior Perf to cublasXt\n");
		if(Dtest_equality(C, C_comp, dim1 * dim2) < 5) error("Insufficient accuracy for benchmarks\n");
		CoCoSyncCheckErr();
	}

	CoCoVecInit(C, M * N, 44, C_loc);
	CoCoMemcpy(C_comp, C,  M * N *sizeof(double), -1, -1);

	fprintf(stderr, "\n==============================================================================================================================\n");
	fprintf(stderr, "CoCoPeLiaDgemmTester: Testing Transpose < 100 MB:\n\n");
	TransA = TransB = 'T';
	alpha = 1.23;
	beta = 0.9876;
	for  (int dim1 = 289; dim1 <= M; dim1*=4) for (int dim2 = 353; dim2 <= N; dim2*=4) for (int dim3 = 307; dim3 <= K; dim3*=4){
		fprintf(stderr, "M=%d,N=%d,K=%d: Gflops/s -> ", dim1, dim2, dim3);
		cpu_timer = csecond();
		CoCopeLiaDgemm(TransA, TransB, dim1, dim2, dim3, alpha, A, ldA, B, ldB, beta, C , ldC);
		CoCoSyncCheckErr();
		cpu_timer  = csecond() - cpu_timer;
		double comp_flops =  Gval_per_s(dgemm_flops(dim1,dim2,dim3),cpu_timer);
		fprintf(stderr, "CoCopeLia: %.1lf, ", comp_flops);
		cpu_timer = csecond();
		T = fmin(dim1,fmin(dim2,dim3))/2;
		cuBLASXtDgemmWrap(TransA, TransB, dim1, dim2, dim3, alpha, A, ldA, B, ldB, beta, C_comp, ldC,  T, cpu_ratio, DEV_NUM, dev_ids);
		CoCoSyncCheckErr();
		cpu_timer  = csecond() - cpu_timer;
		fprintf(stderr, "cuBLASXT: %.1lf\n", Gval_per_s(dgemm_flops(dim1,dim2,dim3),cpu_timer));
		if (comp_flops < Gval_per_s(dgemm_flops(dim1,dim2,dim3),cpu_timer)) warning("Inferior Perf to cublasXt\n");
		if(Dtest_equality(C, C_comp, dim1 * dim2) < 5) error("Insufficient accuracy for benchmarks\n");
		CoCoSyncCheckErr();
	}
	CoCoFree(A, A_loc);
	CoCoFree(B, B_loc);
	CoCoFree(C, C_loc);
	CoCoFree(C_comp, C_loc);
	CoCoSyncCheckErr();

	cpu_timer = csecond();
	// allocate in device GPU memory for benchmarks
	for (int i = 0; i< DEV_NUM; i++){
		short dev_id = dev_ids[i];
		fprintf(stderr, "\n==============================================================================================================================\n");
		fprintf(stderr, "CoCoPeLiaDgemmTester: Allocating GPU buffers...->100 MB...");
		cpu_timer = csecond();
		A_loc = B_loc = C_loc = dev_id;
		A = (double*) CoCoMalloc(M * K*sizeof(double), A_loc);
		B = (double*) CoCoMalloc(N * K*sizeof(double), B_loc);
		C = (double*) CoCoMalloc(M * N*sizeof(double), C_loc);
		C_comp = (double*) CoCoMalloc(M * N*sizeof(double), C_loc);

		double* C_host_buf, * C_host_comp_buf;
		C_host_buf =  (double*) CoCoMalloc(M * N*sizeof(double), -2);
		C_host_comp_buf =  (double*) CoCoMalloc(M * N*sizeof(double), -2);
		CoCoSyncCheckErr();
		cpu_timer  = csecond() - cpu_timer;
		fprintf(stderr, "done.\nAlloc time:\t%lf ms\n\n",  cpu_timer  * 1000);

		cpu_timer = csecond();
		fprintf(stderr, "Initializing to random values...");
		CoCoVecInit(A, K * M, 42, A_loc);
		CoCoVecInit(B, K * N, 43, B_loc);
		CoCoVecInit(C, M * N, 44, C_loc);
		CoCoMemcpy(C_host_comp_buf, C,  M * N *sizeof(double), -2, C_loc);
		CoCoMemcpy(C_comp, C_host_comp_buf,  M * N *sizeof(double), C_loc, -2);
		CoCoSyncCheckErr();
		cpu_timer  = csecond() - cpu_timer ;
		fprintf(stderr, "done.\nInit time:\t%lf ms\n\n",  cpu_timer  * 1000);

		fprintf(stderr, "\n==============================================================================================================================\n");
		fprintf(stderr, "CoCoPeLiaDgemmTester: Testing Matrices In GPU(%d) mem < 100 MB:\n\n", dev_id);
		TransA = TransB = 'N';
		alpha = 1.23;
		beta = 0.9876;
		for (int dim1 = 289; dim1 <= M; dim1*=4) for (int dim2 = 353; dim2 <= N; dim2*=4) for (int dim3 = 307; dim3 <= K; dim3*=4){
			fprintf(stderr, "M=%d,N=%d,K=%d: Gflops/s -> ", dim1, dim2, dim3);
			cpu_timer = csecond();
			CoCopeLiaDgemm(TransA, TransB, dim1, dim2, dim3, alpha, A, ldA, B, ldB, beta, C , ldC);
			CoCoSyncCheckErr();
			cpu_timer  = csecond() - cpu_timer;
			double comp_flops =  Gval_per_s(dgemm_flops(dim1,dim2,dim3),cpu_timer);
			fprintf(stderr, "CoCopeLia: %.1lf, ", comp_flops);
			cpu_timer = csecond();
			T = fmin(dim1,fmin(dim2,dim3))/2;
			cuBLASXtDgemmWrap(TransA, TransB, dim1, dim2, dim3, alpha, A, ldA, B, ldB, beta, C_comp, ldC,  T, cpu_ratio, DEV_NUM, dev_ids);
			CoCoSyncCheckErr();
			cpu_timer  = csecond() - cpu_timer;
			fprintf(stderr, "cuBLASXT: %.1lf\n", Gval_per_s(dgemm_flops(dim1,dim2,dim3),cpu_timer));
			if (comp_flops < Gval_per_s(dgemm_flops(dim1,dim2,dim3),cpu_timer)) warning("Inferior Perf to cublasXt\n");
			CoCoMemcpy(C_host_buf, C,  dim1 * dim2 *sizeof(double), -2, C_loc);
			CoCoMemcpy(C_host_comp_buf, C_comp,  dim1 * dim2 *sizeof(double), -2, C_loc);
			if(Dtest_equality(C_host_buf, C_host_comp_buf, dim1 * dim2) < 5) error("Insufficient accuracy for benchmarks\n");
			CoCoSyncCheckErr();
		}

		fprintf(stderr, "\n==============================================================================================================================\n");
		fprintf(stderr, "CoCoPeLiaDgemmTester: Testing Matrices In GPU(%d) mem + Transpose < 100 MB:\n\n", dev_id);
		TransA = TransB = 'T';
		alpha = 1.23;
		beta = 0.9876;
		for (int dim1 = 289; dim1 <= M; dim1*=4) for (int dim2 = 353; dim2 <= N; dim2*=4) for (int dim3 = 307; dim3 <= K; dim3*=4){
			fprintf(stderr, "M=%d,N=%d,K=%d: Gflops/s -> ", dim1, dim2, dim3);
			cpu_timer = csecond();
			CoCopeLiaDgemm(TransA, TransB, dim1, dim2, dim3, alpha, A, ldA, B, ldB, beta, C , ldC);
			CoCoSyncCheckErr();
			cpu_timer  = csecond() - cpu_timer;
			double comp_flops =  Gval_per_s(dgemm_flops(dim1,dim2,dim3),cpu_timer);
			fprintf(stderr, "CoCopeLia: %.1lf, ", comp_flops);
			cpu_timer = csecond();
			T = fmin(dim1,fmin(dim2,dim3))/2;
			cuBLASXtDgemmWrap(TransA, TransB, dim1, dim2, dim3, alpha, A, ldA, B, ldB, beta, C_comp, ldC,  T, cpu_ratio, DEV_NUM, dev_ids);
			CoCoSyncCheckErr();
			cpu_timer  = csecond() - cpu_timer;
			fprintf(stderr, "cuBLASXT: %.1lf\n", Gval_per_s(dgemm_flops(dim1,dim2,dim3),cpu_timer));
			if (comp_flops < Gval_per_s(dgemm_flops(dim1,dim2,dim3),cpu_timer)) warning("Inferior Perf to cublasXt\n");
			CoCoMemcpy(C_host_buf, C,  dim1 * dim2 *sizeof(double), -2, C_loc);
			CoCoMemcpy(C_host_comp_buf, C_comp,  dim1 * dim2 *sizeof(double), -2, C_loc);
			if(Dtest_equality(C_host_buf, C_host_comp_buf, dim1 * dim2) < 5) error("Insufficient accuracy for benchmarks\n");
			CoCoSyncCheckErr();
		}

		CoCoFree(A, A_loc);
		CoCoFree(B, B_loc);
		CoCoFree(C, C_loc);
		CoCoFree(C_comp, C_loc);
		CoCoFree(C_host_buf, -2);
		CoCoFree(C_host_comp_buf, -2);
	}
	if (run_large_flag){
		ldA = ldB = ldC = M = N = K = (size_t) 1.5*CoCoGetMaxDimSqAsset2D(3, sizeof(double), 256, 0);
		fprintf(stderr, "\n==============================================================================================================================\n");
		fprintf(stderr, "CoCoPeLiaDgemmTester: Allocating CPU buffers...-> %.3lf GB:", dgemm_memory(M,N,K,1,1,2)/1e9);
		cpu_timer = csecond();

		A_loc = B_loc = C_loc = -1;

		double *A, *B, *C;
		A = (double*) CoCoMalloc(M * K*sizeof(double), A_loc);
		B = (double*) CoCoMalloc(N * K*sizeof(double), B_loc);
		C = (double*) CoCoMalloc(M * N*sizeof(double), C_loc);

		CoCoSyncCheckErr();
		cpu_timer  = csecond() - cpu_timer;
		fprintf(stderr, "done.\nAlloc time:\t%lf ms\n\n",  cpu_timer  * 1000);
		cpu_timer = csecond();
		fprintf(stderr, "Initializing to random values...");
		CoCoVecInit(A, K * M, 42, A_loc);
		CoCoVecInit(B, K * N, 43, B_loc);
		CoCoVecInit(C, M * N, 44, C_loc);
		CoCoSyncCheckErr();
		cpu_timer  = csecond() - cpu_timer ;
		fprintf(stderr, "done.\nInit time:\t%lf ms\n\n",  cpu_timer  * 1000);

		double *C_comp = (double*) malloc(M * N*sizeof(double));
		CoCoMemcpy(C_comp, C,  M * N *sizeof(double), -2, -2);

		fprintf(stderr, "\n==============================================================================================================================\n");
		fprintf(stderr, "CoCoPeLiaDgemmTester: Testing Square Problem: %.3lf GB:\n\n", dgemm_memory(M,N,K,1,1,1)/1e9);
		TransA = TransB = 'N';
		alpha = 1.23;
		beta = 0.9876;
		fprintf(stderr, "M=%zu, N=%zu, K=%zu: Gflops/s -> ", M, N, K);
		cpu_timer = csecond();
		CoControl_p return_values = NULL;
		return_values = CoCopeLiaDgemm(TransA, TransB, M, N, K, alpha, A, ldA, B, ldB, beta, C , ldC);
		CoCoSyncCheckErr();
		cpu_timer  = csecond() - cpu_timer;
		for (int i = 0; i< return_values->dev_num; i++) CoCopeLiaDgemm_flush_gpu_mem_buf(return_values->dev_ids[i]);
		CoCoSyncCheckErr();
		double comp_flops = Gval_per_s(dgemm_flops(M,N,K),cpu_timer);
		fprintf(stderr, "CoCopeLia: %.1lf, ", comp_flops);
		cpu_timer = csecond();
		T = fmin(M,fmin(N,K))/4;
		cuBLASXtDgemmWrap(TransA, TransB, M, N, K, alpha, A, ldA, B, ldB, beta, C_comp, ldC,  T, cpu_ratio, DEV_NUM, dev_ids);
		CoCoSyncCheckErr();
		cpu_timer  = csecond() - cpu_timer;
		fprintf(stderr, "cuBLASXT: %.1lf\n", Gval_per_s(dgemm_flops(M,N,K),cpu_timer));
		if (comp_flops < Gval_per_s(dgemm_flops(M,N,K),cpu_timer)) warning("Inferior Perf to cublasXt\n");
		if(Dtest_equality(C, C_comp, M * N) < 5) error("Insufficient accuracy for benchmarks\n");
		CoCoSyncCheckErr();

		CoCoVecInit(C, M * N, 44, C_loc);
		CoCoMemcpy(C_comp, C,  M * N *sizeof(double), -2, -2);

		fprintf(stderr, "\n==============================================================================================================================\n");
		M = (size_t) M/1.24223;
		N = (size_t) N/1.34645;
		K = (size_t) K/2.18321;
		fprintf(stderr, "CoCoPeLiaDgemmTester: Testing Weird Non-Square Problem: %.3lf GB:\n\n", dgemm_memory(M,N,K,1,1,1)/1e9);
		alpha = 1.23;
		beta = 0.9876;
		fprintf(stderr, "M=%zu, N=%zu, K=%zu: Gflops/s -> ", M, N, K);
		cpu_timer = csecond();
		return_values = CoCopeLiaDgemm(TransA, TransB, M, N, K, alpha, A, ldA, B, ldB, beta, C , ldC);
		CoCoSyncCheckErr();
		cpu_timer  = csecond() - cpu_timer;
		for (int i = 0; i< return_values->dev_num; i++) CoCopeLiaDgemm_flush_gpu_mem_buf(return_values->dev_ids[i]);
		CoCoSyncCheckErr();
		comp_flops = Gval_per_s(dgemm_flops(M,N,K),cpu_timer);
		fprintf(stderr, "CoCopeLia: %.1lf, ", comp_flops);
		cpu_timer = csecond();
		T = fmin(M,fmin(N,K))/4;
		cuBLASXtDgemmWrap(TransA, TransB, M, N, K, alpha, A, ldA, B, ldB, beta, C_comp, ldC,  T, cpu_ratio, DEV_NUM, dev_ids);
		CoCoSyncCheckErr();
		cpu_timer  = csecond() - cpu_timer;
		fprintf(stderr, "cuBLASXT: %.1lf\n", Gval_per_s(dgemm_flops(M,N,K),cpu_timer));
		if (comp_flops < Gval_per_s(dgemm_flops(M,N,K),cpu_timer)) warning("Inferior Perf to cublasXt\n");
		if(Dtest_equality(C, C_comp, M * N) < 5) error("Insufficient accuracy for benchmarks\n");
		CoCoSyncCheckErr();

		CoCoVecInit(C, M * N, 44, C_loc);
		CoCoMemcpy(C_comp, C,  M * N *sizeof(double), -2, -2);

		fprintf(stderr, "\n==============================================================================================================================\n");
		fprintf(stderr, "CoCoPeLiaDgemmTester: Testing Large Transpose\n\n");
		TransA = TransB = 'T';
		alpha = 1.23;
		beta = 0.9876;
		fprintf(stderr, "M=%zu, N=%zu, K=%zu: Gflops/s -> ", M, N, K);
		cpu_timer = csecond();
		return_values = CoCopeLiaDgemm(TransA, TransB, M, N, K, alpha, A, ldA, B, ldB, beta, C , ldC);
		CoCoSyncCheckErr();
		cpu_timer  = csecond() - cpu_timer;
		for (int i = 0; i< return_values->dev_num; i++) CoCopeLiaDgemm_flush_gpu_mem_buf(return_values->dev_ids[i]);
		CoCoSyncCheckErr();
		comp_flops = Gval_per_s(dgemm_flops(M,N,K),cpu_timer);
		fprintf(stderr, "CoCopeLia: %.1lf, ", comp_flops);
		cpu_timer = csecond();
		T = fmin(M,fmin(N,K))/4;
		cuBLASXtDgemmWrap(TransA, TransB, M, N, K, alpha, A, ldA, B, ldB, beta, C_comp, ldC,  T, cpu_ratio, DEV_NUM, dev_ids);
		CoCoSyncCheckErr();
		cpu_timer  = csecond() - cpu_timer;
		fprintf(stderr, "cuBLASXT: %.1lf\n", Gval_per_s(dgemm_flops(M,N,K),cpu_timer));
		if (comp_flops < Gval_per_s(dgemm_flops(M,N,K),cpu_timer)) warning("Inferior Perf to cublasXt\n");
		if(Dtest_equality(C, C_comp, M * N) < 5) error("Insufficient accuracy for benchmarks\n");
		CoCoSyncCheckErr();

		CoCoFree(A, A_loc);
		CoCoFree(B, B_loc);
		CoCoFree(C, C_loc);
		CoCoFree(C_comp, -2);
		CoCoSyncCheckErr();

		A_loc = 0;
		if (DEV_NUM == 1){
			B_loc = C_loc = 0;
			ldA = ldB = ldC = M = N = K = (size_t) CoCoGetMaxDimSqAsset2D(4, sizeof(double), 256, 0);
		}
		else if (DEV_NUM == 2){
			B_loc = 0;
			C_loc = 1;
			ldA = ldB = ldC = M = N = K = (size_t) CoCoGetMaxDimSqAsset2D(2, sizeof(double), 256, 0);
		}
		else if (DEV_NUM > 2){
			B_loc = 1;
			C_loc = 2;
			ldA = ldB = ldC = M = N = K = (size_t) CoCoGetMaxDimSqAsset2D(2, sizeof(double), 256, 0);
		}

		fprintf(stderr, "\n==============================================================================================================================\n");
		fprintf(stderr, "CoCoPeLiaDgemmTester: Allocating Mixed GPU buffers...-> A(dev=%d) : %.3lf GB, B(dev=%d) : %.3lf GB, C(dev=%d) : %.3lf GB(x2 for check):", A_loc, M*K*sizeof(double)/1e9, B_loc, K*N*sizeof(double)/1e9, C_loc, M*N*sizeof(double)/1e9);
		cpu_timer = csecond();

		A = (double*) CoCoMalloc(M * K*sizeof(double), A_loc);
		B = (double*) CoCoMalloc(N * K*sizeof(double), B_loc);
		C = (double*) CoCoMalloc(M * N*sizeof(double), C_loc);
		C_comp = (double*) CoCoMalloc(M * N*sizeof(double), C_loc);

		double* C_host_buf, * C_host_comp_buf;
		C_host_buf =  (double*) CoCoMalloc(M * N*sizeof(double), -2);
		C_host_comp_buf =  (double*) CoCoMalloc(M * N*sizeof(double), -2);
		CoCoSyncCheckErr();
		cpu_timer  = csecond() - cpu_timer;
		fprintf(stderr, "done.\nAlloc time:\t%lf ms\n\n",  cpu_timer  * 1000);

		cpu_timer = csecond();
		fprintf(stderr, "Initializing to random values...");
		CoCoVecInit(A, K * M, 42, A_loc);
		CoCoVecInit(B, K * N, 43, B_loc);
		CoCoVecInit(C, M * N, 44, C_loc);
		CoCoMemcpy(C_host_comp_buf, C,  M * N *sizeof(double), -2, C_loc);
		CoCoMemcpy(C_comp, C_host_comp_buf,  M * N *sizeof(double), C_loc, -2);
		CoCoSyncCheckErr();
		cpu_timer  = csecond() - cpu_timer ;
		fprintf(stderr, "done.\nInit time:\t%lf ms\n\n",  cpu_timer  * 1000);

		fprintf(stderr, "\n==============================================================================================================================\n");
		fprintf(stderr, "CoCoPeLiaDgemmTester: Testing Large Matrices In GPU\n\n");
		TransA = TransB = 'N';
		alpha = 1.23;
		beta = 0.9876;
		fprintf(stderr, "M=%zu,N=%zu,K=%zu: Gflops/s -> ", M, N, K);
		cpu_timer = csecond();
		return_values = CoCopeLiaDgemm(TransA, TransB, M, N, K, alpha, A, ldA, B, ldB, beta, C , ldC);
		CoCoSyncCheckErr();
		cpu_timer  = csecond() - cpu_timer;
		for (int i = 0; i< return_values->dev_num; i++) CoCopeLiaDgemm_flush_gpu_mem_buf(return_values->dev_ids[i]);
		CoCoSyncCheckErr();
		comp_flops =  Gval_per_s(dgemm_flops(M,N,K),cpu_timer);
		fprintf(stderr, "CoCopeLia: %.1lf, ", comp_flops);
		cpu_timer = csecond();
		T = fmin(M,fmin(N,K))/4;
		cuBLASXtDgemmWrap(TransA, TransB, M, N, K, alpha, A, ldA, B, ldB, beta, C_comp, ldC,  T, cpu_ratio, DEV_NUM, dev_ids);
		CoCoSyncCheckErr();
		cpu_timer  = csecond() - cpu_timer;
		fprintf(stderr, "cuBLASXT: %.1lf\n", Gval_per_s(dgemm_flops(M, N, K),cpu_timer));
		if (comp_flops < Gval_per_s(dgemm_flops(M, N, K),cpu_timer)) warning("Inferior Perf to cublasXt\n");
		CoCoMemcpy(C_host_buf, C,  M * N *sizeof(double), -2, C_loc);
		CoCoMemcpy(C_host_comp_buf, C_comp,  M * N  *sizeof(double), -2, C_loc);
		if(Dtest_equality(C_host_buf, C_host_comp_buf, M * N ) < 5) error("Insufficient accuracy for benchmarks\n");
		CoCoSyncCheckErr();

		fprintf(stderr, "\n==============================================================================================================================\n");
		fprintf(stderr, "CoCoPeLiaDgemmTester: Testing Large Matrices In GPUmem + Transpose\n\n");
		TransA = TransB = 'T';
		alpha = 1.23;
		beta = 0.9876;
		fprintf(stderr, "M=%zu,N=%zu,K=%zu: Gflops/s -> ", M, N, K);
		cpu_timer = csecond();
		return_values = CoCopeLiaDgemm(TransA, TransB, M, N, K, alpha, A, ldA, B, ldB, beta, C , ldC);
		CoCoSyncCheckErr();
		cpu_timer  = csecond() - cpu_timer;
		for (int i = 0; i< return_values->dev_num; i++) CoCopeLiaDgemm_flush_gpu_mem_buf(return_values->dev_ids[i]);
		CoCoSyncCheckErr();
		comp_flops =  Gval_per_s(dgemm_flops(M,N,K),cpu_timer);
		fprintf(stderr, "CoCopeLia: %.1lf, ", comp_flops);
		cpu_timer = csecond();
		T = fmin(M,fmin(N,K))/4;
		cuBLASXtDgemmWrap(TransA, TransB, M, N, K, alpha, A, ldA, B, ldB, beta, C_comp, ldC,  T, cpu_ratio, DEV_NUM, dev_ids);
		CoCoSyncCheckErr();
		cpu_timer  = csecond() - cpu_timer;
		fprintf(stderr, "cuBLASXT: %.1lf\n", Gval_per_s(dgemm_flops(M, N, K),cpu_timer));
		if (comp_flops < Gval_per_s(dgemm_flops(M, N, K),cpu_timer)) warning("Inferior Perf to cublasXt\n");
		CoCoMemcpy(C_host_buf, C,  M * N *sizeof(double), -2, C_loc);
		CoCoMemcpy(C_host_comp_buf, C_comp,  M * N  *sizeof(double), -2, C_loc);
		if(Dtest_equality(C_host_buf, C_host_comp_buf, M * N ) < 5) error("Insufficient accuracy for benchmarks\n");
		CoCoSyncCheckErr();

		CoCoFree(A, A_loc);
		CoCoFree(B, B_loc);
		CoCoFree(C, C_loc);
		CoCoFree(C_comp, C_loc);
		CoCoFree(C_host_buf, -2);
		CoCoFree(C_host_comp_buf, -2);
	}
	return 0;
}
