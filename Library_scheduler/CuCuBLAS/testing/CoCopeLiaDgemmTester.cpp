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

	short dev_id, mode;

	char TransA, TransB; 
  	double alpha, beta;
	size_t M, N, K, T;
	short A_loc, B_loc, C_loc, C_out_loc;
	double cpu_ratio = 0; 
	size_t ldA, ldB, ldC;

	if(argc != 2) error("Incorrect input arguments. Usage: ./correct_run dev_id\n");
	// Control Parameters
	dev_id = atoi(argv[1]);

	/// Local Timers 
	double cpu_timer = csecond();
	fprintf(stderr, "CoCopeLiaDgemmTester: Initallizing tests for CoCopeLiaDgemmTile in device with id = %d\n", dev_id);
	fprintf(stderr, "CoCopeLiaDgemmTester: Allocating CPU buffers...->100 MB...");
	ldA = ldB = ldC = M = N = K = 8192; 

	A_loc = B_loc = C_loc = -1; 
	
	double *A, *B, *C;
	A = (double*) CoCoMalloc(M * K*sizeof(double), A_loc);
	B = (double*) CoCoMalloc(N * K*sizeof(double), B_loc);
	C = (double*) CoCoMalloc(M * N*sizeof(double), C_loc);

	cudaCheckErrors();
	cpu_timer  = csecond() - cpu_timer;
	fprintf(stderr, "done.\nAlloc time:\t%lf ms\n\n",  cpu_timer  * 1000);
	cpu_timer = csecond();
	fprintf(stderr, "Initializing to random values..."); 
	CoCoVecInit(A, K * M, 42, A_loc);
	CoCoVecInit(B, K * N, 43, B_loc);
	CoCoVecInit(C, M * N, 44, C_loc);
	cudaCheckErrors();
	cpu_timer  = csecond() - cpu_timer ;
	fprintf(stderr, "done.\nInit time:\t%lf ms\n\n",  cpu_timer  * 1000);
		
	double *C_comp = (double*) malloc(M * N*sizeof(double));
	CoCoMemcpy(C_comp, C,  M * N *sizeof(double), -2, -2);

	fprintf(stderr, "CoCopeLiaDgemmTester: Testing Square Problems < 100 MB:\n\n"); 
	TransA = TransB = 'N';
	alpha = 1.23;
	beta = 0.9876;
	for (int dim = 256; dim <= M; dim*=2){
		fprintf(stderr, "M=N=K=%d: Gflops/s -> ", dim); 
		cpu_timer = csecond();
		CoCopeLiaDgemm(TransA, TransB, dim, dim, dim, alpha, A, ldA, B, ldB, beta, C , ldC);
		cudaCheckErrors();
		cpu_timer  = csecond() - cpu_timer;
		double comp_flops = Gval_per_s(dgemm_flops(dim,dim,dim),cpu_timer);
		fprintf(stderr, "CoCopeLia: %.1lf, ", comp_flops);
		cpu_timer = csecond();
		T = fmin(dim,fmin(dim,dim))/2; 
		short dev_num = 2; 
		int dev_ids[dev_num] = {0,1};
		cuBLASXtDgemmWrap(TransA, TransB, dim, dim, dim, alpha, A, ldA, B, ldB, beta, C_comp, ldC,  T, cpu_ratio, dev_num, dev_ids);	
		cudaCheckErrors();
		cpu_timer  = csecond() - cpu_timer;
		fprintf(stderr, "cuBLASXT: %.1lf\n", Gval_per_s(dgemm_flops(dim,dim,dim),cpu_timer));
		if (comp_flops < Gval_per_s(dgemm_flops(dim,dim,dim),cpu_timer)) warning("Inferior Perf to cublasXt");
		if(Dtest_equality(C, C_comp, dim * dim) < 5) error("Insufficient accuracy for benchmarks\n");

		cudaCheckErrors();
	}

	CoCoVecInit(C, M * N, 44, C_loc);
	CoCoMemcpy(C_comp, C,  M * N *sizeof(double), -2, -2);

	fprintf(stderr, "CoCopeLiaDgemmTester: Testing Non-Square Problems < 100 MB:\n\n"); 
	alpha = 1.23;
	beta = 0.9876;
	for (int dim1 = 256; dim1 <= M; dim1*=4) for (int dim2 = 256; dim2 <= N; dim2*=4) for (int dim3 = 256; dim3 <= K; dim3*=4) if ( dim1 != dim2 || dim2 != dim3 || dim1!= dim3){
		fprintf(stderr, "M=%d,N=%d,K=%d: Gflops/s -> ", dim1, dim2, dim3); 
		cpu_timer = csecond();
		CoCopeLiaDgemm(TransA, TransB, dim1, dim2, dim3, alpha, A, ldA, B, ldB, beta, C , ldC);
		cudaCheckErrors();
		cpu_timer  = csecond() - cpu_timer;
		double comp_flops =  Gval_per_s(dgemm_flops(dim1,dim2,dim3),cpu_timer); 
		fprintf(stderr, "CoCopeLia: %.1lf, ", comp_flops);
		cpu_timer = csecond();
		T = fmin(dim1,fmin(dim2,dim3))/2; 
		short dev_num = 2; 
		int dev_ids[dev_num] = {0,1};
		cuBLASXtDgemmWrap(TransA, TransB, dim1, dim2, dim3, alpha, A, ldA, B, ldB, beta, C_comp, ldC,  T, cpu_ratio, dev_num, dev_ids);	
		cudaCheckErrors();
		cpu_timer  = csecond() - cpu_timer;
		fprintf(stderr, "cuBLASXT: %.1lf\n", Gval_per_s(dgemm_flops(dim1,dim2,dim3),cpu_timer));
		if (comp_flops < Gval_per_s(dgemm_flops(dim1,dim2,dim3),cpu_timer)) warning("Inferior Perf to cublasXt");
		if(Dtest_equality(C, C_comp, dim1 * dim2) < 5) error("Insufficient accuracy for benchmarks\n");
		cudaCheckErrors();
	}

	CoCoVecInit(C, M * N, 44, C_loc);
	CoCoMemcpy(C_comp, C,  M * N *sizeof(double), -2, -2);

	fprintf(stderr, "CoCopeLiaDgemmTester: Testing Weird Dimension Problems < 100 MB:\n\n"); 
	alpha = 1.23;
	beta = 0.9876;
	for (int dim1 = 289; dim1 <= M; dim1*=4) for (int dim2 = 353; dim2 <= N; dim2*=4) for (int dim3 = 307; dim3 <= K; dim3*=4) if ( dim1 != dim2 || dim2 != dim3 || dim1!= dim3){
		fprintf(stderr, "M=%d,N=%d,K=%d: Gflops/s -> ", dim1, dim2, dim3); 
		cpu_timer = csecond();
		CoCopeLiaDgemm(TransA, TransB, dim1, dim2, dim3, alpha, A, ldA, B, ldB, beta, C , ldC);
		cudaCheckErrors();
		cpu_timer  = csecond() - cpu_timer;
		double comp_flops =  Gval_per_s(dgemm_flops(dim1,dim2,dim3),cpu_timer); 
		fprintf(stderr, "CoCopeLia: %.1lf, ", comp_flops);
		cpu_timer = csecond();
		T = fmin(dim1,fmin(dim2,dim3))/2; 
		short dev_num = 2; 
		int dev_ids[dev_num] = {0,1};
		cuBLASXtDgemmWrap(TransA, TransB, dim1, dim2, dim3, alpha, A, ldA, B, ldB, beta, C_comp, ldC,  T, cpu_ratio, dev_num, dev_ids);	
		cudaCheckErrors();
		cpu_timer  = csecond() - cpu_timer;
		fprintf(stderr, "cuBLASXT: %.1lf\n", Gval_per_s(dgemm_flops(dim1,dim2,dim3),cpu_timer));
		if (comp_flops < Gval_per_s(dgemm_flops(dim1,dim2,dim3),cpu_timer)) warning("Inferior Perf to cublasXt");
		if(Dtest_equality(C, C_comp, dim1 * dim2) < 5) error("Insufficient accuracy for benchmarks\n");
		cudaCheckErrors();
	}

	CoCoVecInit(C, M * N, 44, C_loc);
	CoCoMemcpy(C_comp, C,  M * N *sizeof(double), -2, -2);

	fprintf(stderr, "CoCopeLiaDgemmTester: Testing Transpose < 100 MB:\n\n"); 
	TransA = TransB = 'T';
	alpha = 1.23;
	beta = 0.9876;
	for  (int dim1 = 289; dim1 <= M; dim1*=4) for (int dim2 = 353; dim2 <= N; dim2*=4) for (int dim3 = 307; dim3 <= K; dim3*=4){
		fprintf(stderr, "M=%d,N=%d,K=%d: Gflops/s -> ", dim1, dim2, dim3); 
		cpu_timer = csecond();
		CoCopeLiaDgemm(TransA, TransB, dim1, dim2, dim3, alpha, A, ldA, B, ldB, beta, C , ldC);
		cudaCheckErrors();
		cpu_timer  = csecond() - cpu_timer;
		double comp_flops =  Gval_per_s(dgemm_flops(dim1,dim2,dim3),cpu_timer); 
		fprintf(stderr, "CoCopeLia: %.1lf, ", comp_flops);
		cpu_timer = csecond();
		T = fmin(dim1,fmin(dim2,dim3))/2; 
		short dev_num = 2; 
		int dev_ids[dev_num] = {0,1};
		cuBLASXtDgemmWrap(TransA, TransB, dim1, dim2, dim3, alpha, A, ldA, B, ldB, beta, C_comp, ldC,  T, cpu_ratio, dev_num, dev_ids);	
		cudaCheckErrors();
		cpu_timer  = csecond() - cpu_timer;
		fprintf(stderr, "cuBLASXT: %.1lf\n", Gval_per_s(dgemm_flops(dim1,dim2,dim3),cpu_timer));
		if (comp_flops < Gval_per_s(dgemm_flops(dim1,dim2,dim3),cpu_timer)) warning("Inferior Perf to cublasXt");
		if(Dtest_equality(C, C_comp, dim1 * dim2) < 5) error("Insufficient accuracy for benchmarks\n");
		cudaCheckErrors();
	}
	CoCoFree(A, A_loc);
	CoCoFree(B, B_loc);
	CoCoFree(C, C_loc); 
	CoCoFree(C_comp, -2); 
	cudaCheckErrors();

	cpu_timer = csecond();
	// allocate in device GPU memory for benchmarks
	A_loc = B_loc = C_loc = dev_id;
	A = (double*) CoCoMalloc(M * K*sizeof(double), A_loc);
	B = (double*) CoCoMalloc(N * K*sizeof(double), B_loc);
	C = (double*) CoCoMalloc(M * N*sizeof(double), C_loc);
	C_comp = (double*) CoCoMalloc(M * N*sizeof(double), C_loc);

	double* C_host_buf, * C_host_comp_buf;
	C_host_buf =  (double*) CoCoMalloc(M * N*sizeof(double), -2);
	C_host_comp_buf =  (double*) CoCoMalloc(M * N*sizeof(double), -2);
	cudaCheckErrors();
	cpu_timer  = csecond() - cpu_timer;
	fprintf(stderr, "done.\nAlloc time:\t%lf ms\n\n",  cpu_timer  * 1000);

	cpu_timer = csecond();
	fprintf(stderr, "Initializing to random values..."); 
	CoCoVecInit(A, K * M, 42, A_loc);
	CoCoVecInit(B, K * N, 43, B_loc);
	CoCoVecInit(C, M * N, 44, C_loc);
	CoCoMemcpy(C_host_comp_buf, C,  M * N *sizeof(double), -2, C_loc);
	CoCoMemcpy(C_comp, C_host_comp_buf,  M * N *sizeof(double), C_loc, -2);
	cudaCheckErrors();
	cpu_timer  = csecond() - cpu_timer ;
	fprintf(stderr, "done.\nInit time:\t%lf ms\n\n",  cpu_timer  * 1000);

	fprintf(stderr, "CoCopeLiaDgemmTester: Testing Matrices In GPU mem < 100 MB:\n\n"); 
	TransA = TransB = 'N';
	alpha = 1.23;
	beta = 0.9876;
	for (int dim1 = 289; dim1 <= M; dim1*=4) for (int dim2 = 353; dim2 <= N; dim2*=4) for (int dim3 = 307; dim3 <= K; dim3*=4){
		fprintf(stderr, "M=%d,N=%d,K=%d: Gflops/s -> ", dim1, dim2, dim3); 
		cpu_timer = csecond();
		CoCopeLiaDgemm(TransA, TransB, dim1, dim2, dim3, alpha, A, ldA, B, ldB, beta, C , ldC);
		cudaCheckErrors();
		cpu_timer  = csecond() - cpu_timer;
		double comp_flops =  Gval_per_s(dgemm_flops(dim1,dim2,dim3),cpu_timer); 
		fprintf(stderr, "CoCopeLia: %.1lf, ", comp_flops);
		cpu_timer = csecond();
		T = fmin(dim1,fmin(dim2,dim3))/2; 
		short dev_num = 2; 
		int dev_ids[dev_num] = {0,1};
		cuBLASXtDgemmWrap(TransA, TransB, dim1, dim2, dim3, alpha, A, ldA, B, ldB, beta, C_comp, ldC,  T, cpu_ratio, dev_num, dev_ids);	
		cudaCheckErrors();
		cpu_timer  = csecond() - cpu_timer;
		fprintf(stderr, "cuBLASXT: %.1lf\n", Gval_per_s(dgemm_flops(dim1,dim2,dim3),cpu_timer));
		if (comp_flops < Gval_per_s(dgemm_flops(dim1,dim2,dim3),cpu_timer)) warning("Inferior Perf to cublasXt");
		CoCoMemcpy(C_host_buf, C,  dim1 * dim2 *sizeof(double), -2, C_loc);
		CoCoMemcpy(C_host_comp_buf, C_comp,  dim1 * dim2 *sizeof(double), -2, C_loc);
		if(Dtest_equality(C_host_buf, C_host_comp_buf, dim1 * dim2) < 5) error("Insufficient accuracy for benchmarks\n");
		cudaCheckErrors();
	}

	fprintf(stderr, "CoCopeLiaDgemmTester: Testing Matrices In GPU mem + Transpose < 100 MB:\n\n"); 
	TransA = TransB = 'T';
	alpha = 1.23;
	beta = 0.9876;
	for (int dim1 = 289; dim1 <= M; dim1*=4) for (int dim2 = 353; dim2 <= N; dim2*=4) for (int dim3 = 307; dim3 <= K; dim3*=4){
		fprintf(stderr, "M=%d,N=%d,K=%d: Gflops/s -> ", dim1, dim2, dim3); 
		cpu_timer = csecond();
		CoCopeLiaDgemm(TransA, TransB, dim1, dim2, dim3, alpha, A, ldA, B, ldB, beta, C , ldC);
		cudaCheckErrors();
		cpu_timer  = csecond() - cpu_timer;
		double comp_flops =  Gval_per_s(dgemm_flops(dim1,dim2,dim3),cpu_timer); 
		fprintf(stderr, "CoCopeLia: %.1lf, ", comp_flops);
		cpu_timer = csecond();
		T = fmin(dim1,fmin(dim2,dim3))/2; 
		short dev_num = 2; 
		int dev_ids[dev_num] = {0,1};
		cuBLASXtDgemmWrap(TransA, TransB, dim1, dim2, dim3, alpha, A, ldA, B, ldB, beta, C_comp, ldC,  T, cpu_ratio, dev_num, dev_ids);	
		cudaCheckErrors();
		cpu_timer  = csecond() - cpu_timer;
		fprintf(stderr, "cuBLASXT: %.1lf\n", Gval_per_s(dgemm_flops(dim1,dim2,dim3),cpu_timer));
		if (comp_flops < Gval_per_s(dgemm_flops(dim1,dim2,dim3),cpu_timer)) warning("Inferior Perf to cublasXt");
		CoCoMemcpy(C_host_buf, C,  dim1 * dim2 *sizeof(double), -2, C_loc);
		CoCoMemcpy(C_host_comp_buf, C_comp,  dim1 * dim2 *sizeof(double), -2, C_loc);
		if(Dtest_equality(C_host_buf, C_host_comp_buf, dim1 * dim2) < 5) error("Insufficient accuracy for benchmarks\n");
		cudaCheckErrors();
	}

	CoCoFree(A, A_loc);
	CoCoFree(B, B_loc);
	CoCoFree(C, C_loc);
	CoCoFree(C_comp, C_loc);
	CoCoFree(C_host_buf, -2);
	CoCoFree(C_host_comp_buf, -2);

	return 0;
}
