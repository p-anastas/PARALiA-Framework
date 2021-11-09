///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The DGEMM CoCopeLia implementation using the new mission-agent-asset C++ classes.
///

#include <cblas.h>

// FIXME: Must remove any calls from this since its the backend of utils (should be wrapped).
//#include "backend_wrappers.hpp"
// FIXME: This header should be "backend_wrappers.hpp" but there is a clash with temp wrappers for Deployment. Must fix when they are removed. 
#include "backend_lib_wrappers.hpp"
#include "CoCoPeLiaModel.hpp"
#include "CoCoPeLia.hpp"
#include "unihelpers.hpp"
#include "Asset.hpp"

/// TODO: Works for systems with up to 128 devices, not 'completely' future-proof
BLAS3GPUBufPtr GloBuf[128];
CoCoModel_p glob_model;
struct CoControl predef_vals;
CoControl_p used_vals = NULL;

void CoCopeLiaDgemmAgent(char TransA, char TransB, size_t M, size_t N, size_t K, double alpha, double* A, size_t ldA, double* B, size_t ldB, double beta, double* C, size_t ldC, short dev_id)
{}

typedef struct gemm_in{
	char TransA,  TransB;
	size_t M, N, K, ldA, ldB, ldC;
	double alpha, *A, *B, beta, *C;
	short dev_id;
}* pthread_gemm_data_p;

void* CoCopeLiaDgemmAgentVoid(void* compressed_gemm_data){
	pthread_gemm_data_p gemm_data = (pthread_gemm_data_p)compressed_gemm_data;
	CoCopeLiaDgemmAgent(gemm_data->TransA,  gemm_data->TransB, gemm_data->M, gemm_data->N, gemm_data->K, gemm_data->alpha, gemm_data->A, gemm_data->ldA, 
	gemm_data->B, gemm_data->ldB, gemm_data->beta, gemm_data->C, gemm_data->ldC, gemm_data->dev_id);
}


/// A dgemm wrapper including auto-tuning of T and cpu_ratio, as well as device management
CoControl_p CoCopeLiaDgemm(char TransA,  char TransB, size_t M, size_t N, size_t K, double alpha, double* A, size_t ldA, double* B, size_t ldB, double beta, double* C, size_t ldC)
{
	short lvl = 1; 
#ifdef DEBUG
	lprintf(lvl-1, "|-----> CoCopeLiaDgemm(%c,%c,%zu,%zu,%zu,%lf,A(%d),%zu,B(%d),%zu,%lf,C(%d),%zu)\n", 
		TransA, TransB, M, N, K, alpha, CoCoGetPtrLoc(A), ldA,
		CoCoGetPtrLoc(B), ldB, beta, CoCoGetPtrLoc(C), ldC);
#endif

#ifdef TEST
	lprintf(lvl-1, "|-----> CoCopeLiaDgemm\n");
	double cpu_timer = csecond();
#endif

    Asset2D<double> A_asset = Asset2D<double>( A, M, K, ldA);
    Asset2D<double> B_asset = Asset2D<double>( B, K, N, ldB);
    Asset2D<double> C_asset = Asset2D<double>( B, M, N, ldC);

	pthread_attr_t attr;
	int s = pthread_attr_init(&attr);
	if (s != 0) error("CoCopeLiaDgemm: pthread_attr_init failed s=%d\n", s);
	
	//if (stack_size > 0) { 
	//s = pthread_attr_setstacksize(&attr, stack_size);
	//       if (s != 0)handle_error_en(s, "pthread_attr_setstacksize");
	//}
        
    pthread_t asset_thread_id[3];

	A_asset.prepareAsync(&asset_thread_id[0], attr);
	B_asset.prepareAsync(&asset_thread_id[1], attr);
	C_asset.prepareAsync(&asset_thread_id[2], attr);

	void* res;
	for(int i=0; i<3;i++){
		s = pthread_join(asset_thread_id[i], &res);
		if (s != 0) error("CoCopeLiaDgemm: pthread_join failed with exit value %d", s);
		free(res);      /* Free memory allocated by thread */
	}

#ifdef TEST
	cpu_timer = csecond() - cpu_timer; 
	lprintf(lvl, "Preparing assets (parallel with pthreads) -> t_prep = %lf ms\n", cpu_timer*1000);
	cpu_timer = csecond();
#endif

	int prev_devID; 
	cudaGetDevice(&prev_devID); 

	short num_devices, *dev_id = NULL;
	if (predef_vals.dev_num > 0){
		num_devices = predef_vals.dev_num;
		dev_id = (short*) malloc (num_devices*sizeof(short));
		for (int i =0; i < num_devices; i++) dev_id[i] = predef_vals.dev_ids[i];
#ifdef TEST
		lprintf(lvl, "Running on %d devices with dev_ids=[ ", num_devices);
		for (int i =0; i < num_devices; i++) fprintf(stderr, "%d ", predef_vals.dev_ids[i]);
		fprintf(stderr, "]\n");
#endif
	}
	else if (predef_vals.dev_num == 0) error("CoCopeLiaDgemm: CPU-only version not implemented (why should it?)\n");
	else{
		num_devices = DEV_NUM;
		dev_id = (short*) malloc (num_devices*sizeof(short));
		for (int i =0; i < num_devices; i++) dev_id[i] = i;
	}

	if(used_vals == NULL) {
		used_vals = (CoControl_p) malloc(sizeof(struct CoControl));
		used_vals->dev_ids = NULL;
	}
	used_vals->dev_num = num_devices;
	if(used_vals->dev_ids != NULL)  free(used_vals->dev_ids);
	used_vals->dev_ids = (int*) malloc(num_devices*sizeof(int));
	for (int d = 0; d< num_devices; d++) used_vals->dev_ids[d] = dev_id[d];
	
	pthread_t thread_id[num_devices];
	pthread_gemm_data_p gemm_data_tmp[num_devices];

	// Here would be the place for Agent distribution to devices. 
	// Working implementation similar to old : Each agent works on part of the problem, asigns to opperatives subproblems
	// TODO: Idea 1 - Simulate the full execution step to step using thew expected times (instead of runtime scheduler), and asign operatives TO agents respectively to minimize communication. 
	
	double t_pred[num_devices], t_total = 0;
 	t_pred[0] = 0.5;
	t_pred[1] = 0.5;
	t_total = 1;


	size_t temp_M = M, M_sum = 0, temp_N = N, N_sum = 0; 
	long long A_ptr_offset_dev = 0, B_ptr_offset_dev = 0, C_ptr_offset_dev = 0;
	for(int i=0; i<num_devices;i++){

		// Check/Enable peer access between participating devices
		CoCoEnableLinks(i, dev_id, num_devices); 

		/// Split M dim.
		temp_M = (size_t) M*t_pred[i]/t_total; 
		if ( i == num_devices - 1) temp_M = M - M_sum; 
        	if (TransA == 'N') A_ptr_offset_dev = M_sum;
		else A_ptr_offset_dev = M_sum*ldA;
        	C_ptr_offset_dev = M_sum;
		M_sum += temp_M;		
/*
		/// Split N dim.
		temp_N = (size_t) N*t_pred[i]/t_total; 
		if ( i == num_devices - 1) temp_N = N - N_sum; 
        	if (gpu_op_B == CUBLAS_OP_N) B_ptr_offset_dev = N_sum*ldB;
		else B_ptr_offset_dev = N_sum;
		C_ptr_offset_dev = N_sum*ldC;
		N_sum += temp_N;
*/
		gemm_data_tmp[i] = (pthread_gemm_data_p) malloc(sizeof(struct gemm_in));
		gemm_data_tmp[i]->TransA = TransA;
		gemm_data_tmp[i]->TransB = TransB;
		gemm_data_tmp[i]->M = temp_M;
		gemm_data_tmp[i]->N = temp_N;
		gemm_data_tmp[i]->K = K;
		gemm_data_tmp[i]->A = A + A_ptr_offset_dev;
		gemm_data_tmp[i]->B = B + B_ptr_offset_dev;
		gemm_data_tmp[i]->C = C + C_ptr_offset_dev;
		gemm_data_tmp[i]->alpha = alpha;
		gemm_data_tmp[i]->beta = beta;
		gemm_data_tmp[i]->ldA = ldA;
		gemm_data_tmp[i]->ldB = ldB;
		gemm_data_tmp[i]->ldC = ldC;
		gemm_data_tmp[i]->dev_id = dev_id[i];

		s = pthread_create(&thread_id[i], &attr,
                                  &CoCopeLiaDgemmAgentVoid, gemm_data_tmp[i]);
	}
	for(int i=0; i<num_devices;i++){
		s = pthread_join(thread_id[i], &res);
		if (s != 0) error("CoCopeLiaDgemm: pthread_join failed with exit value %d", s);
		free(res);      /* Free memory allocated by thread */
	}
	cudaSetDevice(prev_devID);

#ifdef TEST
	cpu_timer = csecond();
#endif
        //if(pinA) cudaHostUnregister(A);
        //if(pinB) cudaHostUnregister(B);
        //if(pinC) cudaHostUnregister(C);
#ifdef TEST
	cpu_timer = csecond() - cpu_timer; 
	lprintf(lvl, "Unregistering matrices -> t_unpin = %lf ms\n", cpu_timer*1000);
#endif

#ifdef DEBUG
	lprintf(lvl-1, "<-----|\n"); 
#endif
#ifdef TEST
	lprintf(lvl-1, "<-----|\n"); 
#endif
	return used_vals;
}

/// A modification of CoCopeLiaDgemm but with given parameters (mainly for performance/debug purposes)
CoControl_p CoCopeLiaDgemmControled(char TransA,  char TransB, size_t M, size_t N, size_t K, double alpha, double* A, size_t ldA, double* B, size_t ldB, double beta, double* C, size_t ldC, CoControl_p predef_control_values){
	if (predef_control_values == NULL) return CoCopeLiaDgemm(TransA, TransB,  M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);
	predef_vals.T = predef_control_values->T; 
	predef_vals.dev_ids = predef_control_values->dev_ids; 
	predef_vals.dev_num = predef_control_values->dev_num; 
	predef_vals.cpu_ratio = predef_control_values->cpu_ratio; 
	return CoCopeLiaDgemm(TransA, TransB,  M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);
}

