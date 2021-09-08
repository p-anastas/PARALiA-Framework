///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The CoCopeLia sub-kernel Tile scheduler functions. 
///

#include <cassert>

#include "CoCoPeLiaLibBackened.hpp"
#include "unihelpers.hpp"

cudaStream_t h2d_stream[128] = {NULL}, d2h_stream[128] = {NULL}, exec_stream[128] = {NULL};
cublasHandle_t handle[128] = {NULL};

kernel3_p CoCopeLiaDgemmSubkernelInit(cublasOperation_t gpu_op_A, cublasOperation_t gpu_op_B, size_t Ms, size_t Ns, size_t Ks, double alpha, double* A, size_t ldA, double* B, size_t ldB, double beta, double* Cin, size_t ldCin, double* Cout, size_t ldCout, BLAS3GPUBufPtr GloBuf, short devId){
  	
  	
	short lvl = 3; 

	kernel3_p kernel = (kernel3_p)malloc(sizeof(struct subkernel3_gen));

	kernel->devId = devId;

#ifdef DEBUG
	lprintf(lvl-1, "|-----> CoCopeLiaDgemmSubkernelInit(%c,%c,%zu,%zu,%zu,%lf,A(%d),%zu,B(%d),%zu,%lf,Cin(%d),%zu,Cout(%d),%zu,Globuf(sz=%zuMB),%d)\n", 
		PrintCublasOp(gpu_op_A), PrintCublasOp(gpu_op_B), Ms, Ns, Ks, alpha, CoCopeLia_ptr_check_cuda_9_2(A), ldA,
		CoCopeLia_ptr_check_cuda_9_2(B), ldB, beta, CoCopeLia_ptr_check_cuda_9_2(Cin), ldCin, 
		CoCopeLia_ptr_check_cuda_9_2(Cout), ldCout, NULL == GloBuf ? 0 : (size_t)GloBuf->gpu_mem_buf_sz/1024/1024, kernel->devId);
#endif

	kernel->gpu_op_A = gpu_op_A;
	kernel->gpu_op_B = gpu_op_B;

  	kernel->Ms = Ms;
  	kernel->Ns = Ns;
  	kernel->Ks = Ks;

	kernel->alpha = alpha; 
	kernel->beta = beta; 

	kernel->As = A; 
	kernel->Bs = B; 
	kernel->CsIn = Cin; 
	kernel->CsOut = Cout; 
	kernel->ldAs = ldA;
	kernel->ldBs = ldB;
	kernel->ldCsIn = ldCin;
	kernel->ldCsOut = ldCout;

	kernel->Asloc = CoCopeLia_ptr_check_cuda_9_2(A);
	kernel->Bsloc = CoCopeLia_ptr_check_cuda_9_2(B);
	kernel->Csloc = CoCopeLia_ptr_check_cuda_9_2(Cin);
	kernel->CsOutloc = CoCopeLia_ptr_check_cuda_9_2(Cout);

	if(kernel->Asloc != kernel->devId && GloBuf) kernel->AdevBuf = (double*) (GloBuf->gpu_mem_buf + GloBuf->dgemm_A_offset);
	else kernel->AdevBuf = NULL; 
	if(kernel->Bsloc != kernel->devId && GloBuf) kernel->BdevBuf = (double*) (GloBuf->gpu_mem_buf + GloBuf->dgemm_B_offset);
	else kernel->BdevBuf = NULL; 
	//FIXME: why the || kernel->CsOutloc? Seems wrong
	if((kernel->Csloc != kernel->devId || kernel->CsOutloc) && GloBuf) kernel->CdevBuf = (double*) (GloBuf->gpu_mem_buf + GloBuf->dgemm_C_offset);
	else kernel->CdevBuf = NULL; 

	if (devId == kernel->Asloc){
		kernel->Aker = kernel->As;
		kernel->ldAker = kernel->ldAs;
	}
	else{
		kernel->Aker = kernel->AdevBuf;
		if (gpu_op_A == CUBLAS_OP_N) kernel->ldAker = Ms;
		else kernel->ldAker = Ks;
	}

	if (devId == kernel->Bsloc){
		kernel->Bker = kernel->Bs;
		kernel->ldBker = kernel->ldBs;
	}
	else{
		kernel->Bker = kernel->BdevBuf;
		if (gpu_op_B == CUBLAS_OP_N) kernel->ldBker = Ks;
		else kernel->ldBker = Ns;
	}

	if (devId == kernel->Csloc){
		kernel->Cker = kernel->CsIn;
		kernel->ldCker = kernel->ldCsIn;
	}
	else{
		kernel->Cker = kernel->CdevBuf;
		kernel->ldCker = Ms;

	}

  	if (!h2d_stream[devId]) cudaStreamCreate(&h2d_stream[devId]);
  	if (!d2h_stream[devId]) cudaStreamCreate(&d2h_stream[devId]);
  	if (!exec_stream[devId]) cudaStreamCreate(&exec_stream[devId]);
	if (!handle[devId]){
		assert(CUBLAS_STATUS_SUCCESS == cublasCreate(&(handle[devId])));
		assert(CUBLAS_STATUS_SUCCESS == cublasSetStream(handle[devId], exec_stream[devId]));
	}

	cudaEventCreateWithFlags(&kernel->data_avail, cudaEventDefault);
	cudaEventCreateWithFlags(&kernel->gemm_complete, cudaEventDefault);

#ifdef DEBUG
	lprintf(lvl-1, "<-----|\n"); 
#endif
	return kernel;
}

kernel3_p* CoCopeLiaDgemmSubkernelCreateGrid(kernel3_p kernel, size_t MblockSz, size_t NblockSz, size_t KblockSz, size_t* kernelNum){

	short lvl = 4; 
	/// Generalize for not exact tiles
	size_t Mlast = kernel->Ms%MblockSz, Nlast = kernel->Ns%NblockSz, Klast= kernel->Ks%KblockSz; 
	kernel->MblockSz = MblockSz;
	kernel->NblockSz = NblockSz;
	kernel->KblockSz = KblockSz;
	kernel->MgridSz = kernel->Ms/MblockSz;
	kernel->NgridSz = kernel->Ns/NblockSz;
	kernel->KgridSz = kernel->Ks/KblockSz;
	if (Mlast > MblockSz/4) kernel->MgridSz++;
	else Mlast+=MblockSz;
	if (Nlast > NblockSz/4) kernel->NgridSz++;
	else Nlast+=NblockSz;
	if (Klast > KblockSz/4) kernel->KgridSz++;
	else Klast+=KblockSz;
#ifdef DEBUG
	lprintf(lvl-1, "|-----> CoCopeLiaDgemmSubkernelCreateGrid(kernel,%zu,%zu,%zu)\n", MblockSz, NblockSz, KblockSz);
	lprintf(lvl,"MgridSz = %zu, NgridSz = %zu, KgridSz = %zu\n", kernel->MgridSz, kernel->NgridSz, kernel->KgridSz);
	lprintf(lvl,"Mlast = %zu, Nlast = %zu, Klast = %zu\n", Mlast, Nlast, Klast);
#endif

	size_t current_ctr, ptr_offset, gpu_ptr_offset, out_ptr_offset;

	*kernelNum = kernel->MgridSz*kernel->NgridSz*kernel->KgridSz;

	kernel3_p* kernels = (kernel3_p*) malloc(*kernelNum*sizeof(kernel3_p)); 

	size_t MtempSz = kernel->MblockSz, NtempSz = kernel->NblockSz, KtempSz = kernel->KblockSz;

	for (int mi = 0; mi < kernel->MgridSz; mi++)
	{
		if ( mi == kernel->MgridSz - 1) MtempSz = Mlast;
		else MtempSz = kernel->MblockSz; 
		for (int ni = 0 ; ni < kernel->NgridSz; ni++){
			if ( ni == kernel->NgridSz - 1) NtempSz = Nlast;
			else NtempSz = kernel->NblockSz; 
			for (int ki = 0; ki < kernel->KgridSz; ki++){
        			if ( ki == kernel->KgridSz - 1) KtempSz = Klast;
				else KtempSz = kernel->KblockSz; 
        			current_ctr = mi*kernel->NgridSz*kernel->KgridSz + ni*kernel->KgridSz + ki; 
				kernels[current_ctr] = CoCopeLiaDgemmSubkernelClone(kernel, mi, ni, ki, MtempSz, NtempSz, KtempSz);
			}
		}
	}
#ifdef DEBUG
	lprintf(lvl-1, "<-----|\n");
#endif
	return kernels; 
}

kernel3_p CoCopeLiaDgemmSubkernelClone(const kernel3_p in_kernel, size_t MgridIdx, size_t NgridIdx, size_t KgridIdx, size_t Ms, size_t Ns, size_t Ks){
	kernel3_p out_kernel = (kernel3_p)malloc(sizeof(struct subkernel3_gen));

	short lvl = 5; 

#ifdef DEBUG
	lprintf(lvl-1, "|-----> CoCopeLiaDgemmSubkernelClone(in_kernel, %zu,%zu,%zu,%zu,%zu,%zu)\n", MgridIdx, NgridIdx, KgridIdx, Ms, Ns, Ks);
#endif

	out_kernel->gpu_op_A = in_kernel->gpu_op_A;
	out_kernel->gpu_op_B = in_kernel->gpu_op_B;

	out_kernel->Ms = Ms;
	out_kernel->Ns = Ns;
	out_kernel->Ks = Ks;

	out_kernel->alpha = in_kernel->alpha;
	if (0 == KgridIdx) out_kernel->beta = in_kernel->beta;
	else out_kernel->beta = 1.0;

	out_kernel->devId = in_kernel->devId;

	out_kernel->Asloc = in_kernel->Asloc;
	out_kernel->Bsloc = in_kernel->Bsloc;
	out_kernel->Csloc = in_kernel->Csloc;
	out_kernel->CsOutloc = in_kernel->CsOutloc;

	out_kernel->MgridIdx = MgridIdx;
	out_kernel->NgridIdx = NgridIdx;
	out_kernel->KgridIdx = KgridIdx;

	size_t ptr_offset, gpu_ptr_offset, out_ptr_offset;

	out_kernel->ldAs = in_kernel->ldAs;
	out_kernel->ldBs = in_kernel->ldBs;
	out_kernel->ldCsIn = in_kernel->ldCsIn;
	out_kernel->ldCsOut = in_kernel->ldCsOut;

	out_kernel->ldAker = in_kernel->ldAker;
	out_kernel->ldBker = in_kernel->ldBker;
	out_kernel->ldCker = in_kernel->ldCker;

	if (out_kernel->gpu_op_A == CUBLAS_OP_N) {
		ptr_offset = MgridIdx*in_kernel->MblockSz + KgridIdx*in_kernel->KblockSz*out_kernel->ldAs;
		gpu_ptr_offset = MgridIdx*in_kernel->MblockSz + KgridIdx*in_kernel->KblockSz*out_kernel->ldAker;
	}
	else{
		ptr_offset = MgridIdx*in_kernel->MblockSz*out_kernel->ldAs + KgridIdx*in_kernel->KblockSz;
		gpu_ptr_offset = MgridIdx*in_kernel->MblockSz*out_kernel->ldAker + KgridIdx*in_kernel->KblockSz;
	}
	out_kernel->Aker = in_kernel->Aker + gpu_ptr_offset;
	out_kernel->As = in_kernel->As + ptr_offset;

	if (out_kernel->gpu_op_B == CUBLAS_OP_N){
		ptr_offset = NgridIdx*in_kernel->NblockSz*out_kernel->ldBs + KgridIdx*in_kernel->KblockSz;
		gpu_ptr_offset = NgridIdx*in_kernel->NblockSz*out_kernel->ldBker + in_kernel->KblockSz*KgridIdx;
	}
	else{
		ptr_offset = NgridIdx*in_kernel->NblockSz + KgridIdx*in_kernel->KblockSz*out_kernel->ldBs;
		gpu_ptr_offset = NgridIdx*in_kernel->NblockSz+ in_kernel->KblockSz*KgridIdx*out_kernel->ldBker ;
	}
	out_kernel->Bker = in_kernel->Bker + gpu_ptr_offset;
	out_kernel->Bs = in_kernel->Bs + ptr_offset;

	ptr_offset = NgridIdx*in_kernel->NblockSz*out_kernel->ldCsIn + MgridIdx*in_kernel->MblockSz;
	out_ptr_offset = NgridIdx*in_kernel->NblockSz*out_kernel->ldCsOut + MgridIdx*in_kernel->MblockSz;
	gpu_ptr_offset = NgridIdx*in_kernel->NblockSz*out_kernel->ldCker + MgridIdx*in_kernel->MblockSz;

	out_kernel->Cker = in_kernel->Cker + gpu_ptr_offset;
	out_kernel->CsIn = in_kernel->CsIn + ptr_offset;
	out_kernel->CsOut = in_kernel->CsOut + out_ptr_offset;

	if (KgridIdx == in_kernel->KgridSz-1) out_kernel->CsOutMaster = 1;
	else out_kernel->CsOutMaster = 0;

#ifdef DEBUG
	lprintf(lvl, "NgridIdx=%d, MgridIdx=%d, KgridIdx=%d, CsOutMaster=%d\n", out_kernel->NgridIdx, out_kernel->MgridIdx, out_kernel->KgridIdx, out_kernel->CsOutMaster);
#endif

	cudaEventCreateWithFlags(&out_kernel->data_avail, cudaEventDefault);
	cudaEventCreateWithFlags(&out_kernel->gemm_complete, cudaEventDefault);


#ifdef DEBUG
	lprintf(lvl-1, "<-----|\n");
#endif
	return out_kernel;
}

void CoCopeLia_Dgemm_subkernel_async(kernel3_p kernel){

	short lvl = 4; 
#ifdef DEBUG
	lprintf(lvl-1, "|-----> CoCopeLia_Dgemm_subkernel_async(kernel)\n");
	lprintf(lvl, "NgridIdx=%d, MgridIdx=%d, KgridIdx=%d, CsOutMaster=%d\n", kernel->NgridIdx, kernel->MgridIdx, kernel->KgridIdx, kernel->CsOutMaster);
#endif
	if (!kernel->NgridIdx && kernel->Asloc != kernel->devId && kernel->alpha){ 
		if (kernel->gpu_op_A == CUBLAS_OP_N) CoCoMemcpy2DAsync(kernel->Aker, kernel->ldAker, kernel->As, kernel->ldAs, kernel->Ms, kernel->Ks, sizeof(double), kernel->devId, kernel->Asloc, h2d_stream[kernel->devId]);	
		else CoCoMemcpy2DAsync(kernel->Aker, kernel->ldAker, kernel->As, kernel->ldAs, kernel->Ks, kernel->Ms, sizeof(double), kernel->devId, kernel->Asloc, h2d_stream[kernel->devId]);	
	}
	//cudaCheckErrors();

	if (!kernel->MgridIdx && kernel->Bsloc != kernel->devId && kernel->alpha){
		if (kernel->gpu_op_B == CUBLAS_OP_N) CoCoMemcpy2DAsync(kernel->Bker, kernel->ldBker, kernel->Bs, kernel->ldBs, kernel->Ks, kernel->Ns, sizeof(double), kernel->devId, kernel->Bsloc, h2d_stream[kernel->devId]);
		else CoCoMemcpy2DAsync(kernel->Bker, kernel->ldBker, kernel->Bs, kernel->ldBs, kernel->Ns, kernel->Ks, sizeof(double), kernel->devId, kernel->Bsloc, h2d_stream[kernel->devId]);
	}
	//cudaCheckErrors();

	if (!kernel->KgridIdx && kernel->Csloc != kernel->devId && kernel->beta){
		CoCoMemcpy2DAsync(kernel->Cker, kernel->ldCker, kernel->CsIn, kernel->ldCsIn, kernel->Ms, kernel->Ns, sizeof(double), kernel->devId, kernel->Csloc, h2d_stream[kernel->devId]);
	}
	cudaEventRecord(kernel->data_avail, h2d_stream[kernel->devId]);
	cudaStreamWaitEvent(exec_stream[kernel->devId], kernel->data_avail,0);
	

	assert(CUBLAS_STATUS_SUCCESS == cublasDgemm(handle[kernel->devId], kernel->gpu_op_A, kernel->gpu_op_B, kernel->Ms, kernel->Ns, kernel->Ks, &kernel->alpha, kernel->Aker, kernel->ldAker, kernel->Bker, kernel->ldBker, &kernel->beta, kernel->Cker, kernel->ldCker));
	cudaEventRecord(kernel->gemm_complete, exec_stream[kernel->devId]);
	if (kernel->CsOutMaster && kernel->CsOutloc != kernel->devId) {
		cudaStreamWaitEvent(d2h_stream[kernel->devId], kernel->gemm_complete,0);
		CoCoMemcpy2DAsync(kernel->CsOut, kernel->ldCsOut, kernel->Cker, kernel->ldCker, kernel->Ms, kernel->Ns, sizeof(double), kernel->CsOutloc, kernel->devId, d2h_stream[kernel->devId]);
	
	}
	//cudaCheckErrors();
#ifdef DEBUG
	lprintf(lvl-1, "<-----|\n");
#endif
	return ;
}

/// Destroys given Dgemm subkernel.
void CoCopeLia_Dgemm_subkernel_destroy(kernel3_p kernel){
	assert(CUBLAS_STATUS_SUCCESS == cudaEventDestroy(kernel->data_avail));
	assert(CUBLAS_STATUS_SUCCESS == cudaEventDestroy(kernel->gemm_complete));

	free(kernel);
	
}

/*
/// Initializes a Dgemm subkernel with given dimensions, and creates the 3 overlap CUDA streams if needed. 
kernel3_p CoCopeLia_Dgemm_subkernel_init(kernel3_p parent, size_t Ms, size_t Ns, size_t Ks) {
  	
	kernel3_p kernel = (kernel3_p)malloc(sizeof(struct subkernel3_gen));

	kernel->device = device;
	kernel->A_loc = A_loc;
	kernel->B_loc = B_loc;
	kernel->C_loc = C_loc;
	kernel->C_out_loc = C_out_loc;

  	kernel->Ms = Ms;
  	kernel->Ns = Ns;
  	kernel->Ks = Ks;

	kernel->gpu_op_A = gpu_op_A;
	kernel->ldA = ldA;
	if (!A_loc) kernel->ldAdevBuf = ldA;
	else if (gpu_op_A == CUBLAS_OP_N) kernel->ldAdevBuf = M;
	else kernel->ldAdevBuf = K;

	kernel->gpu_op_B = gpu_op_B;
	kernel->ldB = ldB;
	if (!B_loc) kernel->ldBdevBuf = ldB;
	else  if (gpu_op_B == CUBLAS_OP_N) kernel->ldBdevBuf = K;
	else kernel->ldBdevBuf = N;


	kernel->ldC = ldC;
	kernel->ldC_out = ldC_out;
	if (!C_loc) kernel->ldCdevBuf = ldC;
	else kernel->ldCdevBuf = M;


  	if (!h2d_stream) cudaStreamCreate(&h2d_stream);
  	if (!d2h_stream) cudaStreamCreate(&d2h_stream);
  	if (!exec_stream) cudaStreamCreate(&exec_stream);
	if (!handle){
		assert(CUBLAS_STATUS_SUCCESS == cublasCreate(&handle));
		assert(CUBLAS_STATUS_SUCCESS == cublasSetStream(handle, exec_stream));
	}

	if (!h2d_complete) cudaEventCreateWithFlags(&h2d_complete, cudaEventDisableTiming);
	cudaEventCreateWithFlags(&kernel->data_avail, cudaEventDisableTiming);
	cudaEventCreateWithFlags(&kernel->gemm_complete, cudaEventDisableTiming);

return kernel;
}

/// Puts a Dgemm sub-kernel's input and execution on the corresponding stream pipelines.
void CoCopeLia_Dgemm_subkernel_async(double alpha, double beta, kernel3_p kernel, short d2hWaitForH2d){
  	if (!h2d_stream) cudaStreamCreate(&h2d_stream);
  	if (!d2h_stream) cudaStreamCreate(&d2h_stream);
  	if (!exec_stream) cudaStreamCreate(&exec_stream);
	if (!handle){
		assert(CUBLAS_STATUS_SUCCESS == cublasCreate(&handle));
		assert(CUBLAS_STATUS_SUCCESS == cublasSetStream(handle, exec_stream));
	}

	if (!h2d_complete) cudaEventCreateWithFlags(&h2d_complete, cudaEventDisableTiming);

				if (!ni) kernel->AT_master = 1;
				else kernel->AT_master = 0;
				if (!mi) kernel->BT_master = 1;
				else kernel->BT_master = 0;
				if (!ki) kernel->CT_master = 1;
				else kernel->CT_master = 0;
				if (ki == K_parts-1) kernel->CT_out_master = 1;
				else kernel->CT_out_master = 0;

	if(!kernel->A_dev && alpha)error("CoCoBLAS_Dgemm_subkernel_async: A_dev buffer unallocated");
	else if(!kernel->B_dev && alpha )error("CoCoBLAS_Dgemm_subkernel_async: B_dev buffer unallocated");
	else if(!kernel->C_dev)error("CoCoBLAS_Dgemm_subkernel_async: C_dev buffer unallocated");

	if (kernel->AT_master && kernel->A_loc && alpha){ 
		if (kernel->gpu_op_A == CUBLAS_OP_N) assert(CUBLAS_STATUS_SUCCESS == cublasSetMatrixAsync(kernel->Ms, kernel->Ks, sizeof(double), kernel->As, kernel->ldA, kernel->A_dev, kernel->ldAdevBuf, h2d_stream));	
		else assert(CUBLAS_STATUS_SUCCESS == cublasSetMatrixAsync(kernel->Ks, kernel->Ms, sizeof(double), kernel->As, kernel->ldA, kernel->A_dev, kernel->ldAdevBuf, h2d_stream));
	}

	if (kernel->BT_master && kernel->B_loc && alpha){
		if (kernel->gpu_op_B == CUBLAS_OP_N) assert(CUBLAS_STATUS_SUCCESS == cublasSetMatrixAsync(kernel->Ks, kernel->Ns, sizeof(double), kernel->Bs, kernel->ldB, kernel->B_dev, kernel->ldBdevBuf, h2d_stream));
		else assert(CUBLAS_STATUS_SUCCESS == cublasSetMatrixAsync(kernel->Ns, kernel->Ks, sizeof(double), kernel->Bs, kernel->ldB, kernel->B_dev, kernel->ldBdevBuf, h2d_stream));
	}

	if (kernel->CT_master && kernel->C_loc && beta) assert(CUBLAS_STATUS_SUCCESS == cublasSetMatrixAsync(kernel->Ms, kernel->Ns, sizeof(double), kernel->Cs, kernel->ldC, kernel->C_dev, kernel->ldCdevBuf, h2d_stream));
	cudaEventRecord(kernel->data_avail, h2d_stream);
	//cudaCheckErrors();
	
	if (d2hWaitForH2d) cudaEventRecord(h2d_complete, h2d_stream);

	cudaStreamWaitEvent(exec_stream, kernel->data_avail,0);
	assert(CUBLAS_STATUS_SUCCESS == cublasDgemm(handle, kernel->gpu_op_A, kernel->gpu_op_B, kernel->Ms, kernel->Ns, kernel->Ks, &alpha, kernel->A_dev, kernel->ldAdevBuf, kernel->B_dev, kernel->ldBdevBuf, &beta, kernel->C_dev, kernel->ldCdevBuf));
	if (kernel->CT_out_master) cudaEventRecord(kernel->gemm_complete, exec_stream);
	//cudaCheckErrors();

	return ;
}

/// Puts a Dgemm sub-kernel's output on the corresponding stream pipeline.
void CoCopeLia_Dgemm_subkernel_out(kernel3_p kernel)
{
	cudaStreamWaitEvent(d2h_stream, kernel->gemm_complete,0);
	cudaStreamWaitEvent(d2h_stream, h2d_complete,0);
	assert(CUBLAS_STATUS_SUCCESS == cublasGetMatrixAsync(kernel->Ms, kernel->Ns, sizeof(double), kernel->C_dev, kernel->ldCdevBuf, kernel->C_out, kernel->ldC_out, d2h_stream));
	//cudaCheckErrors();
}

/// Initializes an Sgemm subkernel with given dimensions, and creates the 3 overlap CUDA streams if needed. 
kernel3f_p CoCopeLia_Sgemm_subkernel_init(size_t M, size_t N, size_t K, size_t Ms, size_t Ns, size_t Ks, short A_loc, short B_loc, short C_loc, short device) {
  	
	kernel3f_p kernel = (kernel3f_p)malloc(sizeof(struct subkernel3f_gen));

	kernel->device = device;
	kernel->A_loc = A_loc;
	kernel->B_loc = B_loc;
	kernel->C_loc = C_loc;

  	kernel->Ms = Ms;
  	kernel->Ns = Ns;
  	kernel->Ks = Ks;

	kernel->ldA = M;
	kernel->ldAdevBuf = M;
	kernel->gpu_op_A = CUBLAS_OP_N;

	kernel->ldB = K;
	kernel->ldBdevBuf = K;
	kernel->gpu_op_B = CUBLAS_OP_N;

	kernel->ldC = M;
	kernel->ldCdevBuf = M;


  	if (!h2d_stream) cudaStreamCreate(&h2d_stream);
  	if (!d2h_stream) cudaStreamCreate(&d2h_stream);
  	if (!exec_stream) cudaStreamCreate(&exec_stream);
	if (!handle){
		assert(CUBLAS_STATUS_SUCCESS == cublasCreate(&handle));
		assert(CUBLAS_STATUS_SUCCESS == cublasSetStream(handle, exec_stream));
	}

	if (!h2d_complete) cudaEventCreateWithFlags(&h2d_complete, cudaEventDisableTiming);
	cudaEventCreateWithFlags(&kernel->data_avail, cudaEventDisableTiming);
	cudaEventCreateWithFlags(&kernel->gemm_complete, cudaEventDisableTiming);

return kernel;
}

/// Destroys given Sgemm subkernel.
void CoCopeLia_Sgemm_subkernel_destroy(kernel3f_p kernel){
	// TODO: For now use only one device;
	int dev_id; cudaGetDevice(&dev_id);

	assert(CUBLAS_STATUS_SUCCESS == cudaEventDestroy(kernel->data_avail));
	assert(CUBLAS_STATUS_SUCCESS == cudaEventDestroy(kernel->gemm_complete));
	
}

/// Puts an Sgemm sub-kernel's input and execution on the corresponding stream pipelines.
void CoCopeLia_Sgemm_subkernel_async(float alpha, float beta, kernel3f_p kernel, short d2hWaitForH2d){
	if(!kernel->A_dev && alpha)error("CoCoBLAS_Dgemm_subkernel_async: A_dev buffer unallocated");
	else if(!kernel->B_dev && alpha )error("CoCoBLAS_Dgemm_subkernel_async: B_dev buffer unallocated");
	else if(!kernel->C_dev)error("CoCoBLAS_Dgemm_subkernel_async: C_dev buffer unallocated");

	if (kernel->AT_master && kernel->A_loc && alpha) assert(CUBLAS_STATUS_SUCCESS == cublasSetMatrixAsync(kernel->Ms, kernel->Ks, sizeof(float), kernel->As, kernel->ldA, kernel->A_dev, kernel->ldAdevBuf, h2d_stream));
	if (kernel->BT_master && kernel->B_loc && alpha) assert(CUBLAS_STATUS_SUCCESS == cublasSetMatrixAsync(kernel->Ks, kernel->Ns, sizeof(float), kernel->Bs, kernel->ldB, kernel->B_dev, kernel->ldBdevBuf, h2d_stream));
	if (kernel->CT_master && kernel->C_loc && beta) assert(CUBLAS_STATUS_SUCCESS == cublasSetMatrixAsync(kernel->Ms, kernel->Ns, sizeof(float), kernel->Cs, kernel->ldC, kernel->C_dev, kernel->ldCdevBuf, h2d_stream));
	cudaEventRecord(kernel->data_avail, h2d_stream);
	//cudaCheckErrors();
	
	if (d2hWaitForH2d) cudaEventRecord(h2d_complete, h2d_stream);

	cudaStreamWaitEvent(exec_stream, kernel->data_avail,0);
	assert(CUBLAS_STATUS_SUCCESS == cublasSgemm(handle, kernel->gpu_op_A, kernel->gpu_op_B, kernel->Ms, kernel->Ns, kernel->Ks, &alpha, kernel->A_dev, kernel->ldAdevBuf, kernel->B_dev, kernel->ldBdevBuf, &beta, kernel->C_dev, kernel->ldCdevBuf));
	if (kernel->CT_out_master) cudaEventRecord(kernel->gemm_complete, exec_stream);
	//cudaCheckErrors();

	return ;
}

/// Puts an Sgemm sub-kernel's output on the corresponding stream pipeline.
void CoCopeLia_Sgemm_subkernel_out(kernel3f_p kernel)
{
	cudaStreamWaitEvent(d2h_stream, kernel->gemm_complete,0);
	cudaStreamWaitEvent(d2h_stream, h2d_complete,0);
	assert(CUBLAS_STATUS_SUCCESS == cublasGetMatrixAsync(kernel->Ms, kernel->Ns, sizeof(float), kernel->C_dev, kernel->ldCdevBuf, kernel->Cs, kernel->ldC, d2h_stream));
	//cudaCheckErrors();
}
*/

