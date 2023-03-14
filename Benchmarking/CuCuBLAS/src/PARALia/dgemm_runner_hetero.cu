///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief This is a special runner used to emulate a GPU load. It loops forever running a specifid dgemm load, and should be used with all locs at the execution device enforced by dev_id_token
///

#include "unihelpers.hpp"
#include "PARALiA.hpp"
#include "BackenedLibsWrapped.hpp"
#include "Testing.hpp"

#include <cuda.h>
#include "cublas_v2.h"

#include "backend_wrappers.hpp"

#define CBLASXT_MAX_SAFE_TILE 10000

int main(const int argc, const char *argv[]) {
	char TransA, TransB;
  	double alpha, beta;
	long int M, N, K;
	short A_loc, B_loc, C_loc, C_out_loc;
	ATC_p predef_control_values = NULL, return_values = NULL;
	ParseInputLvl3(argc, argv, &predef_control_values, &TransA, &TransB, &alpha, &beta, &M, &N, &K, &A_loc, &B_loc, &C_loc, &C_out_loc);

	/// Matrix Layouts for CPU GEMM
	CBLAS_TRANSPOSE cpu_op_A, cpu_op_B;    // CblasNoTrans, CblasTrans
	cublasOperation_t gpu_op_A, gpu_op_B; // CUBLAS_OP_N, CUBLAS_OP_T

	long int ldA, ldB, ldC = M;
	TransposeTranslate(TransA, &cpu_op_A, &gpu_op_A, &ldA, M, K);
	TransposeTranslate(TransB, &cpu_op_B, &gpu_op_B, &ldB, K, N);

	double *A, *B, *C;
	// allocate in device if loc = 0, otherwise allocate in pinned memory for benchmarks
	A = (double*) CoCoMalloc(M * K*sizeof(double), A_loc);
	B = (double*) CoCoMalloc(N * K*sizeof(double), B_loc);
	C = (double*) CoCoMalloc(M * N*sizeof(double), C_loc);

	CoCoSyncCheckErr();
	CoCoVecInit(A, K * M, 42, A_loc);
	CoCoVecInit(B, K * N, 43, B_loc);
	CoCoVecInit(C, M * N, 44, C_loc);
	CoCoSyncCheckErr();
	if (predef_control_values!= NULL) return_values = PARALiADgemmControled(TransA, TransB, M, N, K, alpha, A, ldA, B, ldB, beta, C , ldC, predef_control_values);
	else return_values = PARALiADgemm(TransA, TransB, M, N, K, alpha, A, ldA, B, ldB, beta, C , ldC);
	CoCoSyncCheckErr();

	int bench_it = 1;
	for(int it = 0; it < bench_it; it=it){
		if (predef_control_values!= NULL) return_values = PARALiADgemmControled(TransA, TransB, M, N, K, alpha, A, ldA, B, ldB, beta, C , ldC, predef_control_values);
		else return_values = PARALiADgemm(TransA, TransB, M, N, K, alpha, A, ldA, B, ldB, beta, C , ldC);
		CoCoSyncCheckErr();
	}


	for (int i = 0; i< LOC_NUM; i++) PARALiADevCacheFree(deidxize(i));

	CoCoSyncCheckErr();
	CoCoFree(A, A_loc);
	CoCoFree(B, B_loc);
	CoCoFree(C, C_loc);
	return 0;
}
