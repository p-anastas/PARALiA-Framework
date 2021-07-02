///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief Some CUDA function calls with added error-checking
///
#ifndef COCOBLAS_SUBKERNELS_H
#define COCOBLAS_SUBKERNELS_H

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <pthread.h>

#define NUM_DEVICES 1

short CoCoPeLiaGetRemoteFlag(short data_loc, short dev_id);

short CoCopeLia_ptr_check_cuda_9_2(const void * in_ptr);

short CoCopeLia_ptr_check_cuda_11(const void * in_ptr);

/* global variable declaration */
typedef struct globuf{
	short dev_id; 
	void * gpu_mem_buf = NULL;
	long long gpu_mem_buf_sz = 0; 
	long long dgemm_A_offset = -1, dgemm_B_offset = -1, dgemm_C_offset = -1;
}* BLAS3GPUBufPtr;

typedef struct subkernel3_gen {
	cublasOperation_t gpu_op_A, gpu_op_B; 
	size_t Ms, Ns, Ks;
	double alpha, beta; 
	/// The initial (->start) Subkernel Matrices. 
	double* As, *Bs, *CsIn, *CsOut;
	size_t ldAs, ldBs, ldCsIn, ldCsOut; 

	/// The GPU buffers for each matrix, if required. 
	double * AdevBuf, * BdevBuf, *CdevBuf;

	/// The GPU pointers used in the actual kernel calls (Xker infered from Xs, XdevBuf). 
	double * Aker, * Bker, *Cker;
	size_t ldAker, ldBker, ldCker; 

	short Asloc, Bsloc, Csloc, CsOutloc;

	size_t MgridSz, NgridSz, KgridSz;
	size_t MblockSz, NblockSz, KblockSz;

	short MgridIdx, NgridIdx, KgridIdx;
	short AsMaster, BsMaster, CsMaster, CsOutMaster;

	short devId; 
	cudaEvent_t data_avail, gemm_complete;
 
	
	
} * kernel3_p;

void CoCopeLiaDgemmTile(kernel3_p in_kernel, int T); 

kernel3_p CoCopeLiaDgemmSubkernelInit(cublasOperation_t gpu_op_A, cublasOperation_t gpu_op_B, size_t M, size_t N, size_t K, double alpha, double* A, size_t ldA, double* B, size_t ldB, double beta, double* Cin, size_t ldCin, double* Cout, size_t ldCout, BLAS3GPUBufPtr GloBuf, short dev_id);

kernel3_p* CoCopeLiaDgemmSubkernelCreateGrid(kernel3_p kernel, size_t MblockSz, size_t NblockSz, size_t KblockSz, size_t* kernelNum);

kernel3_p CoCopeLiaDgemmSubkernelClone(const kernel3_p parent, size_t MgridIdx, size_t NgridIdx, size_t KgridIdx, size_t Ms, size_t Ns, size_t Ks);

//double* CoCopeLiaDgemmSubkernelGetPtr(kernel3_p kernel, char name);

//void CoCopeLiaDgemmSubkernelGetSz(kernel3_p kernel, size_t* Ms, size_t* Ns, size_t* Ks);
//void CoCopeLiaDgemmSubkernelSetSz(kernel3_p kernel, size_t Ms, size_t Ns, size_t Ks);

void CoCopeLia_Dgemm_subkernel_async(kernel3_p kernel);
//void CoCopeLia_Dgemm_subkernel_out(kernel3_p kernel);

void CoCopeLia_Dgemm_subkernel_destroy(kernel3_p kernel);


/// The CoCopeLia Dgemm tiled implementation.
void CoCopeLiaDgemmTile(cublasOperation_t gpu_op_A,  cublasOperation_t gpu_op_B, size_t M, size_t N, size_t K, double alpha, double* A, size_t ldA, double* B, size_t ldB, double beta, double* C_in, size_t ldC, double* C_out, int Tin, short dev_id);

#endif
