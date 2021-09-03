///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The DGEMM CoCopeLia implementation.
///

#include <cblas.h>

#include "CoCoPeLiaLibBackened.hpp"
#include "CoCoPeLiaModel.hpp"
#include "CoCoPeLia.hpp"
#include "unihelpers.hpp"

/// TODO: Works for systems with up to 128 devices, not 'completely' future-proof
BLAS3GPUBufPtr GloBuf[128];
CoCoModel_p glob_model;
struct CoControl predef_vals;
CoControl_p used_vals = NULL;

void CoCopeLiaDgemm_flush_gpu_mem_buf(short dev_id){
	short lvl = 3; 
#ifdef DEBUG
	lprintf(lvl-1, "|-----> CoCopeLiaDgemm_flush_gpu_mem_buf(dev_id=%d)\n", dev_id);
	lprintf(lvl, "Clearing (presumably) %zu MB\n\n", (size_t) GloBuf[dev_id]->gpu_mem_buf_sz/1024/1024);
#endif
	cudaFree(GloBuf[dev_id]->gpu_mem_buf);
	free(GloBuf[dev_id]);
	GloBuf[dev_id] = NULL;
	cudaCheckErrors();
}

void CoCopelia_split_dims(size_t* M_memparts, size_t* N_memparts, size_t* K_memparts, size_t M, size_t N, size_t K, long long avail_bytes, short A_loc, short B_loc, short C_loc)
{
	short lvl = 3; 
#ifdef DEBUG
	lprintf(lvl-1, "|-----> CoCopelia_split_dims(&M_memparts,&N_memparts,&K_memparts,%zu,%zu,%zu,%zu MB,%d,%d,%d)\n", M, N, K, avail_bytes/1024/1024, A_loc, B_loc, C_loc);
#endif
#ifdef TEST
	lprintf(lvl-1, "|-----> CoCopelia_split_dims\n");
#endif
	if (!(A_loc || B_loc || C_loc)) return; 
	size_t candM = M/ *M_memparts, candN = N/ *N_memparts, candK = K/ *K_memparts; 
	while (avail_bytes < dgemm_memory(candM, candN, candK,A_loc,B_loc,C_loc)){
		if(A_loc || C_loc) candM = M/ *M_memparts;
		else candM = 0; 
		if(B_loc || C_loc) candN = N/ *N_memparts;
		else candN = 0; 
		if(A_loc || B_loc) candK = K/ *K_memparts;
		else candK = 0; 

		if (candM >= (size_t) fmax(candN, candK)) *M_memparts+=1;
		else if (candN >= (size_t) fmax(candM, candK)) *N_memparts+=1;
		else if (candK >= (size_t) fmax(candM, candN)) *K_memparts+=1;
	}
#ifdef DEBUG
	lprintf(lvl-1, "<-----|\n"); 
#endif
	return;
}

void CoCopeLiaDgemmTile(kernel3_p in_kernel, int T){
	short lvl = 3; 
	int curId; cudaGetDevice(&curId);
	massert(T > 0, "CoCopeLiaDgemmTile: invalid T=%d\n", T);
#ifdef DEBUG
	lprintf(lvl-1, "|-----> CoCopeLiaDgemmTile(in_kernel,%zu)\n", T);
	if ( curId != in_kernel->devId) lprintf(lvl, "Device change initiated(%d->%d)\n",curId, in_kernel->devId);
#endif
#ifdef TEST
	lprintf(lvl-1, "|-----> CoCopeLiaDgemmTile\n");
	double timer = csecond(); 
#endif
	//cudaSetDevice(in_kernel->devId);

	if (T > in_kernel->Ms || T > in_kernel->Ns || T > in_kernel->Ks) error("CoCopeLiaDgemmTile: T greater than dim"); 

	size_t kernel_num; 

	kernel3_p* kernels = CoCopeLiaDgemmSubkernelCreateGrid(in_kernel, T, T, T, &kernel_num);
	cudaCheckErrors();
#ifdef TEST
	timer = csecond() - timer;
	lprintf(lvl, "Subkernel Grid created: t_grid = %lf ms\n" , timer*1000);
	timer = csecond(); 
	cudaEvent_t start_firing; 
	cudaEventCreateWithFlags(&start_firing, cudaEventDefault);
	cudaEventRecord(start_firing);
#endif
	size_t ker, Mp = in_kernel->MgridSz, Np = in_kernel->NgridSz, Kp = in_kernel->KgridSz;
	for (int mi = 0; mi < Mp; mi++){
		for (int ni = 0; ni< Np; ni++){ 
			for (int ki = 0; ki < Kp; ki++){
				ker = mi*Np*Kp + ni*Kp + ki;  
				CoCopeLia_Dgemm_subkernel_async(kernels[ker]);
			}
		}
	}
	cudaCheckErrors();
#ifdef TEST
	timer = csecond() - timer;
	lprintf(lvl, "Subkernels complete: t_comp = %lf ms\n" , timer*1000);
	cudaEvent_t prev_data_sent = start_firing, prev_exec =  kernels[0]->data_avail; 
	for (int ker = 0; ker < kernel_num; ker++){
		float t_send, t_exec, temp, t_gpu_idle = 0;
		cudaEventElapsedTime(&t_send, prev_data_sent, kernels[ker]->data_avail);
		cudaEventElapsedTime(&temp, prev_exec, kernels[ker]->gemm_complete);
		cudaEventElapsedTime(&t_exec, kernels[ker]->data_avail, kernels[ker]->gemm_complete);
		if (!ker) t_gpu_idle = t_send; 
		else if (t_exec <= temp) t_gpu_idle = max(0.0, temp - t_exec); 
		t_exec = min(t_exec, temp); 
		lprintf(lvl, "Subkernel(%d): t_h2d = %f ms, t_exec = %f ms, t_gpu_idle = %f ms\n" , ker, t_send, t_exec, t_gpu_idle);
		//cudaCheckErrors();
		prev_data_sent = kernels[ker]->data_avail; 
		prev_exec = kernels[ker]->gemm_complete;
	}
	timer = csecond(); 
#endif
	for (int ker = 0; ker < kernel_num; ker++)CoCopeLia_Dgemm_subkernel_destroy(kernels[ker]);
	cudaCheckErrors();
#ifdef TEST
	timer = csecond() - timer;
	lprintf(lvl, "Subkernels destroyed: t_dest = %lf ms\n" , timer*1000);
	lprintf(lvl-1, "<-----|\n"); 
#endif
#ifdef DEBUG
	lprintf(lvl-1, "<-----|\n");
#endif
	return;
}

void CoCopeLiaDgemmDevice(cublasOperation_t gpu_op_A,  cublasOperation_t gpu_op_B, size_t M, size_t N, size_t K, double alpha, double* A, size_t ldA, double* B, size_t ldB, double beta, double* C, size_t ldC, short dev_id){ 

	cudaSetDevice(dev_id);
	short lvl = 2; 

	short A_loc, B_loc, C_loc; 
	A_loc = CoCopeLia_ptr_check_cuda_9_2(A);
	B_loc = CoCopeLia_ptr_check_cuda_9_2(B);
	C_loc = CoCopeLia_ptr_check_cuda_9_2(C);	
	cudaGetLastError();
	short A_remote_f = CoCoPeLiaGetRemoteFlag(A_loc, dev_id), 
		B_remote_f = CoCoPeLiaGetRemoteFlag(B_loc, dev_id), 
		C_remote_f = CoCoPeLiaGetRemoteFlag(C_loc, dev_id), 
		Cout_remote_f = CoCoPeLiaGetRemoteFlag(C_loc, dev_id);
#ifdef DEBUG
	lprintf(lvl-1, "|-----> CoCopeLiaDgemmDevice(%c,%c,%zu,%zu,%zu,%lf,A(%d),%zu,B(%d),%zu,%lf,C(%d),%zu,Globuf[%d]=%s,%d)\n", 
		PrintCublasOp(gpu_op_A), PrintCublasOp(gpu_op_B), M, N, K, alpha, CoCopeLia_ptr_check_cuda_9_2(A), ldA,
		CoCopeLia_ptr_check_cuda_9_2(B), ldB, beta, CoCopeLia_ptr_check_cuda_9_2(C), ldC, 
		dev_id, NULL == GloBuf[dev_id] ? "uninitialized" : "initialized", dev_id);
#endif

#ifdef TEST
	lprintf(lvl-1, "|-----> CoCopeLiaDgemmDevice(dev_id=%d)\n", dev_id);
	double cpu_timer = csecond();
#endif
	if (NULL == GloBuf[dev_id]){
		GloBuf[dev_id] = (BLAS3GPUBufPtr) malloc(sizeof(struct globuf));
		GloBuf[dev_id]->gpu_mem_buf_sz = 0; 
	}

	long long mem_foot = dgemm_memory(M,N,K,A_remote_f,B_remote_f,C_remote_f);
#ifdef DEBUG
	lprintf(lvl, "====================================\n");
	lprintf(lvl, "GPU mem management:\n", (size_t) mem_foot/1024/1024);
	lprintf(lvl, " -Mem required for matrices: %zu MB\n", (size_t) mem_foot/1024/1024);
#endif
	size_t free_cuda_mem, max_cuda_mem; 
	massert(cudaSuccess == cudaMemGetInfo(&free_cuda_mem, &max_cuda_mem), "CoCopeLiaDgemm: cudaMemGetInfo failed"); 
	size_t problem_avail_mem = free_cuda_mem + GloBuf[dev_id]->gpu_mem_buf_sz;
	// Problem Fits-in-GPU case
	if (mem_foot < problem_avail_mem){
#ifdef DEBUG
		lprintf(lvl, " -Problem fits in GPU\n");
		if (GloBuf[dev_id]->gpu_mem_buf_sz >= mem_foot) lprintf(lvl, "-GPU buf available: %zu MB\n", (size_t) mem_foot/1024/1024);
		else if (GloBuf[dev_id]->gpu_mem_buf_sz > 0) lprintf(lvl, "-Smaller GPU buf available -> resizing : %zu -> %zu MB\n", (size_t) GloBuf[dev_id]->gpu_mem_buf_sz/1024/1024, (size_t) mem_foot/1024/1024);
		else if (GloBuf[dev_id]->gpu_mem_buf_sz == 0) lprintf(lvl, "-Initializing new buffer: %zu MB\n",(size_t) mem_foot/1024/1024);
#endif
		if (GloBuf[dev_id]->gpu_mem_buf_sz >= mem_foot);
		else if (GloBuf[dev_id]->gpu_mem_buf_sz > 0){
			CoCoFree(GloBuf[dev_id]->gpu_mem_buf, dev_id);
			GloBuf[dev_id]->gpu_mem_buf = CoCoMalloc(mem_foot, dev_id);
			GloBuf[dev_id]->gpu_mem_buf_sz = mem_foot; 
		}
		else if (GloBuf[dev_id]->gpu_mem_buf_sz == 0){
			GloBuf[dev_id]->gpu_mem_buf = CoCoMalloc(mem_foot, dev_id);
			GloBuf[dev_id]->gpu_mem_buf_sz = mem_foot; 
		}
		else error("Unknown memory case");
		if (A_remote_f) GloBuf[dev_id]->dgemm_A_offset = 0;
		if (B_remote_f) GloBuf[dev_id]->dgemm_B_offset = (A_remote_f)*M*K*sizeof(double);  
		if (C_remote_f) GloBuf[dev_id]->dgemm_C_offset = (A_remote_f)*M*K*sizeof(double)+ (B_remote_f)*N*K*sizeof(double); 
		cudaCheckErrors();
#ifdef TEST
		cpu_timer = csecond() - cpu_timer; 
		lprintf(lvl, "Memory management: t_mem = %lf ms\n", dev_id, cpu_timer*1000);
		cpu_timer = csecond(); 
#endif

		CoCoModel_p model = NULL;

		size_t T = 0;
		if(predef_vals.T <= 0){ 
			model = CoCoPeLiaModelInit(dev_id, "Dgemm", 'X', PrintCublasOp(gpu_op_A), PrintCublasOp(gpu_op_B), M, N, K, A_remote_f, B_remote_f, C_remote_f, A_remote_f, B_remote_f, C_remote_f, ldA, ldB, ldC);
#ifdef TEST
			cpu_timer = csecond() - cpu_timer; 
			lprintf(lvl, "Model Initialization: t_mod_init = %lf ms\n", dev_id, cpu_timer*1000);
			cpu_timer = csecond(); 
#endif
			tunableParams_p pred_p = CoCoPeLiaModelOptimizeTile(model, COCOPELIA_REUSE);
			T = pred_p->T;

#ifdef TEST

			cpu_timer = csecond() - cpu_timer; 
			lprintf(lvl, "Model Selected T=%zu with t_predicted = %lf ms : t_mod_opt = %lf ms\n", T, pred_p->pred_t*1000, cpu_timer*1000);
			cpu_timer = csecond(); 
#endif

#ifdef DEBUG
			lprintf(lvl, "Model Selected T=%zu : t_predicted = %lf ms\n", T, pred_p->pred_t*1000);
			lprintf(lvl, "====================================\n");
#endif
/* TODO: Pipeline emulation takes time currently, must optimize or stick to prediction model as initially (D) intended
			pred_p = CoCoPeLiaModelOptimizeTile(model, COCOPELIA_PIPELINE_EMULATE);
			T = pred_p->T;
#ifdef TEST
			cpu_timer = csecond() - cpu_timer; 
			lprintf(lvl, "Pipeline Sim. selected T=%zu : t_predicted = %lf ms\n", T, pred_p->pred_t*1000);
			lprintf(lvl, "Pipeline Optimize: t_pipe_opt = %lf ms\n", dev_id, cpu_timer*1000);
			cpu_timer = csecond(); 
#endif
#ifdef DEBUG
			lprintf(lvl, "Pipeline Sim. selected T=%zu : t_predicted = %lf ms\n", T, pred_p->pred_t*1000);
			lprintf(lvl, "====================================\n");
#endif
*/
		}
		else{
			T = predef_vals.T; 
#ifdef DEBUG
			lprintf(lvl, "====================================\n");
			lprintf(lvl, "Using predefined T=%zu\n", T);
			lprintf(lvl, "====================================\n");
#endif
		}

		if(used_vals == NULL) {
			used_vals = (CoControl_p) malloc(sizeof(struct CoControl));
			used_vals->dev_ids = NULL;
		}
		used_vals->T = T;

		kernel3_p kernel = CoCopeLiaDgemmSubkernelInit(gpu_op_A, gpu_op_B, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC, C, ldC, GloBuf[dev_id], dev_id);
		cudaCheckErrors();
#ifdef TEST
		cpu_timer = csecond() - cpu_timer; 
		lprintf(lvl, "Subkernel Initialization: t_subker_init = %lf ms\n", dev_id, cpu_timer*1000);
		cpu_timer = csecond(); 
#endif
		CoCopeLiaDgemmTile(kernel, T);
		cudaCheckErrors();
#ifdef TEST
		cpu_timer = csecond() - cpu_timer; 
		lprintf(lvl, "Subkernel offload: t_offload = %lf ms\n", dev_id, cpu_timer*1000);
#endif
	}
	else{
		size_t Mp, Np, Kp = Mp = Np = 1; 
		CoCopelia_split_dims(&Mp, &Np, &Kp, M, N, K, problem_avail_mem,A_remote_f,B_remote_f,C_remote_f);
		long long adjusted_mem_foot = dgemm_memory(M/Mp + M%Mp,N/Np + N%Np,K/Kp + K%Kp,A_remote_f,B_remote_f,C_remote_f);
#ifdef DEBUG
		lprintf(lvl, " -Problem does not fit in GPU: %d MB required vs %d MB available\n", (size_t) mem_foot/1024/1024, (size_t) problem_avail_mem/1024/1024);
		if (GloBuf[dev_id]->gpu_mem_buf_sz >= adjusted_mem_foot)lprintf(lvl, " -GPU buf available: %zu MB\n", (size_t) adjusted_mem_foot/1024/1024);
		else if (GloBuf[dev_id]->gpu_mem_buf_sz == 0) lprintf(lvl, " -Initalizing partial buffer : %zu MB\n", (size_t) adjusted_mem_foot/1024/1024);
		else if (GloBuf[dev_id]->gpu_mem_buf_sz > 0) lprintf(lvl, " -Smaller GPU buf available -> resizing : %zu -> %zu MB\n", 
								(size_t) GloBuf[dev_id]->gpu_mem_buf_sz/1024/1024, (size_t) adjusted_mem_foot/1024/1024);
#endif
		if (GloBuf[dev_id]->gpu_mem_buf_sz >= adjusted_mem_foot);
		else if (GloBuf[dev_id]->gpu_mem_buf_sz == 0){	 
			GloBuf[dev_id]->gpu_mem_buf = CoCoMalloc(adjusted_mem_foot, dev_id);
			GloBuf[dev_id]->gpu_mem_buf_sz = adjusted_mem_foot; 
		}
		else if (GloBuf[dev_id]->gpu_mem_buf_sz > 0){
			CoCoFree(GloBuf[dev_id]->gpu_mem_buf, dev_id);
			GloBuf[dev_id]->gpu_mem_buf = CoCoMalloc(adjusted_mem_foot, dev_id);
			GloBuf[dev_id]->gpu_mem_buf_sz = adjusted_mem_foot; 
		}
		else error("Unknown memory case");
		if (A_remote_f) GloBuf[dev_id]->dgemm_A_offset = 0;
		if (B_remote_f) GloBuf[dev_id]->dgemm_B_offset = (A_remote_f)*(M/Mp + M%Mp)*(K/Kp + K%Kp)*sizeof(double);  
		if (C_remote_f) GloBuf[dev_id]->dgemm_C_offset = (A_remote_f)*(M/Mp + M%Mp)*(K/Kp + K%Kp)*sizeof(double)+ (B_remote_f)*(N/Np + N%Np)*(K/Kp + K%Kp)*sizeof(double); 
		cudaCheckErrors();
#ifdef TEST

		cpu_timer = csecond() - cpu_timer; 
		lprintf(lvl, "Memory management for dev_id = %d : t_mem = %lf ms\n", dev_id, cpu_timer*1000);
		cpu_timer = csecond(); 
#endif

#ifdef DEBUG
		lprintf(lvl, "====================================\n");
#endif

		CoCoModel_p model = NULL;

		long long A_ptr_offset, B_ptr_offset, C_ptr_offset; 	
		long long prev_A_ptr_offset = -1 , prev_B_ptr_offset = -1, prev_C_ptr_offset = -1;
		double t_pred_total_model = 0, t_pred_total_pipe = 0;
		double t_total = csecond();


		size_t T;
		if(predef_vals.T <= 0){
			model = CoCoPeLiaModelInit(dev_id, "Dgemm", 'X', PrintCublasOp(gpu_op_A), PrintCublasOp(gpu_op_B), M/Mp, N/Np, K/Kp, A_remote_f, B_remote_f, C_remote_f, A_remote_f, B_remote_f, C_remote_f, ldA, ldB, ldC);
#ifdef TEST
			cpu_timer = csecond() - cpu_timer; 
			lprintf(lvl, "Model Initialization: t_mod_init = %lf ms\n", dev_id, cpu_timer*1000);
			cpu_timer = csecond(); 
#endif
			size_t T_model, T_pipeline;
			tunableParams_p pred_p = CoCoPeLiaModelOptimizeTile(model, COCOPELIA_REUSE);
			T_model = pred_p->T;
			
#ifdef TEST
			cpu_timer = csecond() - cpu_timer; 
			lprintf(lvl, "Model Selected T=%zu with t_predicted = %lf ms : t_mod_opt = %lf ms\n", T_model, pred_p->pred_t*1000, cpu_timer*1000);
			cpu_timer = csecond(); 
#endif
#ifdef DEBUG
			lprintf(lvl, "Model Selected T=%zu : t_predicted = %lf ms\n", T_model, pred_p->pred_t*1000);

#endif
			t_pred_total_model = pred_p->pred_t*(Mp*Np*Kp);
/* TODO: Pipeline emulation takes time currently, must optimize or stick to prediction model as initially (D) intended
			pred_p = CoCoPeLiaModelOptimizeTile(model, COCOPELIA_REUSE);
			T_pipeline = pred_p->T;
#ifdef TEST
			cpu_timer = csecond() - cpu_timer; 
			lprintf(lvl, "Pipeline Sim. selected T=%zu : t_predicted = %lf ms\n", T_pipeline, pred_p->pred_t*1000);
			lprintf(lvl, "Pipeline Optimize: t_pipe_opt = %lf ms\n", dev_id, cpu_timer*1000);
#endif
#ifdef DEBUG
			lprintf(lvl, "Pipeline Sim. selected T=%zu : t_predicted = %lf ms\n", T_pipeline, pred_p->pred_t*1000);
#endif
			t_pred_total_pipe= pred_p->pred_t*(Mp*Np*Kp);
			T = T_pipeline; 
*/
			T = T_model;
		}
		else{
			T = predef_vals.T; 
#ifdef DEBUG
			lprintf(lvl, "Using predefined T=%zu\n", T);
#endif
		}

		if(used_vals == NULL) {
			used_vals = (CoControl_p) malloc(sizeof(struct CoControl));
			used_vals->dev_ids = NULL;
		}
		used_vals->T = T;

		/// Flag which is used to ALWAYS include reuse
		short reverseK = 0, reverseN = 0; 
		for (int mi = 0; mi < Mp; mi++){
		for (int ni_lie = 0; ni_lie < Np; ni_lie++){ 
			int ni; 
			if(reverseN) ni = Np-1 -ni_lie; 
			else ni = ni_lie;
		for (int ki_lie = 0; ki_lie < Kp; ki_lie++){
			int ki; 
			if(reverseK) ki = Kp-1 -ki_lie; 
			else ki = ki_lie; 
#ifdef TEST
			cpu_timer = csecond(); 
#endif
			size_t tempM = M/Mp, tempN = N/Np, tempK = K/Kp, tempLdA = ldA, tempLdB = ldB, tempLdC = ldC, tempOutLdC = ldC; 

        		if (gpu_op_A == CUBLAS_OP_N) A_ptr_offset = mi*tempM + ki*tempK*ldA;
			else A_ptr_offset = mi*tempM*ldA + ki*tempK;
        		
			if (gpu_op_B == CUBLAS_OP_N) B_ptr_offset = ni*tempN*ldB + tempK*ki;
			else B_ptr_offset = ni*tempN + tempK*ki*ldB;

        		C_ptr_offset = ni*tempN*ldC + mi*tempM;
	
			double use_beta = beta, *A_in, *B_in, *C_in, *C_out; 
			if (mi == Mp - 1) tempM = M/Mp+M%Mp; 
			if (ni == Np - 1) tempN = N/Np+N%Np; 
			if (ki == Kp - 1) tempK = K/Kp+K%Kp;

			if (A_remote_f && prev_A_ptr_offset == A_ptr_offset){
				A_in = (double*) (GloBuf[dev_id]->gpu_mem_buf + GloBuf[dev_id]->dgemm_A_offset);
				if (gpu_op_A == CUBLAS_OP_N) tempLdA = tempM; 
				else tempLdA = tempK; 
			}
			else A_in = A+A_ptr_offset; 

			if (B_remote_f && prev_B_ptr_offset == B_ptr_offset){
				B_in = (double*) (GloBuf[dev_id]->gpu_mem_buf + GloBuf[dev_id]->dgemm_B_offset);
				if (gpu_op_B == CUBLAS_OP_N) tempLdB = tempK; 
				else tempLdB = tempN; 
			}
			else B_in = B+B_ptr_offset; 

			if ( C_remote_f && prev_C_ptr_offset == C_ptr_offset){
				C_in = (double*) (GloBuf[dev_id]->gpu_mem_buf + GloBuf[dev_id]->dgemm_C_offset);
				tempLdC = tempM; 		
			}
			else C_in = C+C_ptr_offset; 

			if (prev_C_ptr_offset == C_ptr_offset) use_beta = 1; 
			if (ki_lie == Kp - 1 || !C_remote_f) C_out = C+C_ptr_offset; 
			else{ 
				C_out = (double*) (GloBuf[dev_id]->gpu_mem_buf + GloBuf[dev_id]->dgemm_C_offset);
				tempOutLdC = tempM; 

			}

			kernel3_p kernel = CoCopeLiaDgemmSubkernelInit(gpu_op_A, gpu_op_B, tempM, tempN, tempK, alpha, A_in, tempLdA, B_in, tempLdB, use_beta, C_in, tempLdC, C_out, tempOutLdC, GloBuf[dev_id], dev_id);
			cudaCheckErrors();
#ifdef TEST
			cpu_timer = csecond() - cpu_timer; 
			lprintf(lvl, "Subkernel Initialization: t_init = %lf ms\n", cpu_timer*1000);
			cpu_timer = csecond(); 
#endif
			CoCopeLiaDgemmTile(kernel, T);
			cudaCheckErrors();
#ifdef TEST
			cpu_timer = csecond() - cpu_timer; 
			lprintf(lvl, "Subkernel Offload: t_offload = %lf ms\n", cpu_timer*1000);
#endif
#ifdef DEBUG
			lprintf(lvl, "====================================\n");
#endif
			prev_A_ptr_offset = A_ptr_offset;
			prev_B_ptr_offset = B_ptr_offset;
			prev_C_ptr_offset = C_ptr_offset;
		}
		if(reverseK) reverseK = 0; 
		else reverseK = 1; 
		}
		if(reverseN) reverseN = 0; 
		else reverseN = 1; 
		}

		t_total = csecond() - t_total; 
#ifdef TEST
		lprintf(lvl, "Total time: t_total = %lf ms\n", t_total*1000); 
#endif	
#ifdef DEBUG
		lprintf(lvl, "====================================\n");
		//if(!predef_vals.T) lprintf(lvl, "Model vs Pipeline: Error %lf vs %lf\n", (t_pred_total_model - t_total)/t_total, (t_pred_total_pipe - t_total)/t_total);
		if(predef_vals.T <= 0) lprintf(lvl, "Model Error %lf\n", (t_pred_total_model - t_total)/t_total);
#endif
	}
#ifdef DEBUG
	lprintf(lvl-1, "<-----|\n");
#endif 
#ifdef TEST
	lprintf(lvl-1, "<-----|\n"); 
#endif
	return;
}

typedef struct gemm_in{
	cublasOperation_t gpu_op_A,  gpu_op_B;
	size_t M, N, K, ldA, ldB, ldC;
	double alpha, *A, *B, beta, *C;
	short dev_id;
}* pthread_gemm_data_p;

void* CoCopeLiaDgemmDeviceVoid(void* compressed_gemm_data){
	pthread_gemm_data_p gemm_data = (pthread_gemm_data_p)compressed_gemm_data;
	CoCopeLiaDgemmDevice(gemm_data->gpu_op_A,  gemm_data->gpu_op_B, gemm_data->M, gemm_data->N, gemm_data->K, gemm_data->alpha, gemm_data->A, gemm_data->ldA, 
	gemm_data->B, gemm_data->ldB, gemm_data->beta, gemm_data->C, gemm_data->ldC, gemm_data->dev_id);
}

/// A dgemm wrapper including auto-tuning of T and cpu_ratio, as well as device management
CoControl_p CoCopeLiaDgemm(char TransA,  char TransB, size_t M, size_t N, size_t K, double alpha, double* A, size_t ldA, double* B, size_t ldB, double beta, double* C, size_t ldC)
{
	short lvl = 1; 
#ifdef DEBUG
	lprintf(lvl-1, "|-----> CoCopeLiaDgemm(%c,%c,%zu,%zu,%zu,%lf,A(%d),%zu,B(%d),%zu,%lf,C(%d),%zu)\n", 
		TransA, TransB, M, N, K, alpha, CoCopeLia_ptr_check_cuda_9_2(A), ldA,
		CoCopeLia_ptr_check_cuda_9_2(B), ldB, beta, CoCopeLia_ptr_check_cuda_9_2(C), ldC);
#endif

#ifdef TEST
	lprintf(lvl-1, "|-----> CoCopeLiaDgemm\n");
	double cpu_timer = csecond();
#endif
	CoControl_p used_params;

	cudaPointerAttributes attributes;
	short pinA, pinB, pinC = pinB = pinA = 0;
	/// If CUDA does not recognise ptr, its not pinned (neither in GPU)
	if (cudaSuccess!=cudaPointerGetAttributes(&attributes, A)) pinA = 1; 
	if (cudaSuccess!=cudaPointerGetAttributes(&attributes, B)) pinB = 1;
	if (cudaSuccess!=cudaPointerGetAttributes(&attributes, C)) pinC = 1;
	cudaGetLastError();

	/// Pin any un-pinned host pointer
        if(pinA) cudaHostRegister(A,sizeof(double)*M*K,cudaHostRegisterPortable);
        if(pinB) cudaHostRegister(B,sizeof(double)*K*N,cudaHostRegisterPortable);
        if(pinC) cudaHostRegister(C,sizeof(double)*M*N,cudaHostRegisterPortable);

#ifdef DEBUG
	lprintf(lvl, "pinA=%d, pinB=%d, pinC=%d\n", pinA, pinB, pinC);
#endif

#ifdef TEST
	cpu_timer = csecond() - cpu_timer; 
	lprintf(lvl, "Pinning host matrices -> t_pin = %lf ms\n", cpu_timer*1000);
	cpu_timer = csecond();
#endif

	int prev_devID; 
	cudaGetDevice(&prev_devID); 
	cublasOperation_t gpu_op_A  = OpCharToCublas(TransA),  gpu_op_B = OpCharToCublas(TransB);

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

	double t_pred[num_devices], t_total = 0;
 	//t_pred[0] = 0.5;
	//t_pred[1] = 0.5;
	//t_total = 1;

	//. TODO: Naive split way which assumes cutting will not effect performance differently for devices. Naive but sufficient.
	for(int i=0; i<num_devices;i++){
		short A_loc, B_loc, C_loc; 
		A_loc = CoCopeLia_ptr_check_cuda_9_2(A);
		B_loc = CoCopeLia_ptr_check_cuda_9_2(B);
		C_loc = CoCopeLia_ptr_check_cuda_9_2(C);	
		cudaGetLastError();
		short A_remote_f = CoCoPeLiaGetRemoteFlag(A_loc, dev_id[i]), 
			B_remote_f = CoCoPeLiaGetRemoteFlag(B_loc, dev_id[i]), 
			C_remote_f = CoCoPeLiaGetRemoteFlag(C_loc, dev_id[i]), 
			Cout_remote_f = CoCoPeLiaGetRemoteFlag(C_loc, dev_id[i]);
		CoCoModel_p model = CoCoPeLiaModelInit(dev_id[i], "Dgemm", 'X', TransA, TransB, M, N, K, A_remote_f, B_remote_f, C_remote_f, A_remote_f, B_remote_f, C_remote_f, ldA, ldB, ldC);
		tunableParams_p pred_p = CoCoPeLiaModelOptimizeTile(model, COCOPELIA_REUSE);
		t_total += t_pred[i] = pred_p->pred_t;
	}

    	// Initialize thread creation attributes.
	pthread_attr_t attr;
	int s = pthread_attr_init(&attr);
	if (s != 0) error("CoCopeLiaDgemm: pthread_attr_init failed s=%d\n", s);
	//if (stack_size > 0) { 
	//s = pthread_attr_setstacksize(&attr, stack_size);
        //       if (s != 0)handle_error_en(s, "pthread_attr_setstacksize");
        //}

	size_t temp_M = M, M_sum = 0, temp_N = N, N_sum = 0; 
	long long A_ptr_offset_dev = 0, B_ptr_offset_dev = 0, C_ptr_offset_dev = 0;
	for(int i=0; i<num_devices;i++){

		// Check/Enable peer access between participating GPUs
		CoCoPeLiaEnableGPUPeer(i, dev_id, num_devices); 

		/// Split M dim.
		temp_M = (size_t) M*t_pred[i]/t_total; 
		if ( i == num_devices - 1) temp_M = M - M_sum; 
        	if (gpu_op_A == CUBLAS_OP_N) A_ptr_offset_dev = M_sum;
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
		gemm_data_tmp[i]->gpu_op_A = gpu_op_A;
		gemm_data_tmp[i]->gpu_op_B = gpu_op_B;
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
                                  &CoCopeLiaDgemmDeviceVoid, gemm_data_tmp[i]);
		
	}
	void* res;
	for(int i=0; i<num_devices;i++){
		s = pthread_join(thread_id[i], &res);
		if (s != 0) error("CoCopeLiaDgemm: pthread_join failed with exit value %d", s);
		free(res);      /* Free memory allocated by thread */
	}
	cudaSetDevice(prev_devID);


#ifdef TEST
	cpu_timer = csecond();
#endif
        if(pinA) cudaHostUnregister(A);
        if(pinB) cudaHostUnregister(B);
        if(pinC) cudaHostUnregister(C);
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

