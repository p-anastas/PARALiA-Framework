///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief Some CUDA function calls with added error-checking
///

#include <cstdio>
#include <typeinfo>
#include <float.h>
#include <curand.h>

#include "unihelpers.hpp"

const char *print_mem(mem_layout mem) {
  if (mem == ROW_MAJOR)
    return "Row major";
  else if (mem == COL_MAJOR)
    return "Col major";
  else
    return "ERROR";
}

const char *print_loc(short loc) {

  int dev_count;
  massert(CUBLAS_STATUS_SUCCESS == cudaGetDeviceCount(&dev_count), "print_loc: cudaGetDeviceCount failed");

  if (loc == -2)  return "Host"; 
  else if (loc == -1 || loc == -3)  return "Pinned Host";
  else if (loc < dev_count) return "Device";
  else return "ERROR";
}

void print_devices() {
  cudaDeviceProp properties;
  int nDevices = 0;
  massert(CUBLAS_STATUS_SUCCESS == cudaGetDeviceCount(&nDevices), "print_devices: cudaGetDeviceCount failed");
  printf("Found %d Devices: \n\n", nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaGetDeviceProperties(&properties, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", properties.name);
    printf("  Memory Clock Rate (MHz): %d\n",
           properties.memoryClockRate / 1024);
    printf("  Memory Bus Width (bits): %d\n", properties.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n",
           2.0 * properties.memoryClockRate * (properties.memoryBusWidth / 8) /
               1.0e6);
    if (properties.major >= 3)
      printf("  Unified Memory support: YES\n\n");
    else
      printf("  Unified Memory support: NO\n\n");
  }
}

void cudaCheckErrors() {
  cudaError_t errSync = cudaGetLastError();
  cudaError_t errAsync = cudaDeviceSynchronize();
  if (errSync != cudaSuccess)
    printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
  if (errAsync != cudaSuccess)
    printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
}

gpu_timer_p gpu_timer_init() {
  gpu_timer_p timer = (gpu_timer_p)malloc(sizeof(struct gpu_timer));
  cudaEventCreate(&timer->start);
  cudaEventCreate(&timer->stop);
  return timer;
}

void gpu_timer_start(gpu_timer_p timer, cudaStream_t stream) { cudaEventRecord(timer->start, stream); }

void gpu_timer_stop(gpu_timer_p timer, cudaStream_t stream) { cudaEventRecord(timer->stop, stream); }

float gpu_timer_get(gpu_timer_p timer) {
  cudaEventSynchronize(timer->stop);
  cudaEventElapsedTime(&timer->ms, timer->start, timer->stop);
  return timer->ms;
}

size_t CoCopeLiaGetMaxSqdimLvl3(short matrixNum, short dsize, size_t step){
	size_t free_cuda_mem, max_cuda_mem; 
	massert(cudaSuccess == cudaMemGetInfo(&free_cuda_mem, &max_cuda_mem), "CoCopeLiaGetMaxSqdimLvl3: cudaMemGetInfo failed"); 

	// Define the max size of a benchmark kernel to run on this machine. 
	size_t maxDim = (( (size_t) sqrt(max_cuda_mem/(3*dsize))/1.5 ) / step) * step;
	return maxDim;
}

size_t CoCopeLiaGetMaxVecdimLvl1(short vecNum, short dsize, size_t step){
	size_t free_cuda_mem, max_cuda_mem; 
	massert(cudaSuccess == cudaMemGetInfo(&free_cuda_mem, &max_cuda_mem), "CoCopeLiaGetMaxSqdimLvl3: cudaMemGetInfo failed"); 

	size_t maxDim = (( (size_t) max_cuda_mem/(3*dsize)/2 ) / step) * step;
	return maxDim;
}

void TransposeTranslate(char TransChar, CBLAS_TRANSPOSE* cblasFlag, cublasOperation_t* cuBLASFlag, size_t* ldim, size_t dim1, size_t dim2){
	if (TransChar == 'N'){
 		*cblasFlag = CblasNoTrans;
 		*cuBLASFlag = CUBLAS_OP_N;
		*ldim = dim1; 
	}
	else if (TransChar == 'T'){
 		*cblasFlag = CblasTrans;
 		*cuBLASFlag = CUBLAS_OP_T;
		*ldim = dim2; 
	}
	else if (TransChar == 'C'){
 		*cblasFlag = CblasConjTrans;
 		*cuBLASFlag = CUBLAS_OP_C;
		*ldim = dim2; 
	}
	else error("TransposeTranslate: %c is an invalid Trans flag", TransChar);
}


cublasOperation_t OpCblasToCublas(CBLAS_TRANSPOSE src)
{ 
	if(src == CblasNoTrans) return CUBLAS_OP_N;
	else if(src == CblasTrans) return CUBLAS_OP_T;
	else if(src == CblasConjTrans) return CUBLAS_OP_C;
	else error("OpCblasToCublas: Invalid Op\n"); 
}

cublasOperation_t OpCharToCublas(char src)
{
	if(src == 'N') return CUBLAS_OP_N;
	else if(src == 'T') return CUBLAS_OP_T;
	else if(src == 'C') return CUBLAS_OP_C;
	else error("OpCharToCublas: Invalid Op: %c\n", src); 
}

CBLAS_TRANSPOSE OpCharToCblas(char src)
{
	if(src == 'N') return CblasNoTrans;
	else if(src == 'T') return CblasTrans;
	else if(src == 'C') return CblasConjTrans;
	else error("OpCharToCblas: Invalid Op: %c\n", src); 
}

CBLAS_TRANSPOSE OpCublasToCblas(cublasOperation_t src)
{
	if(src == CUBLAS_OP_N) return CblasNoTrans;
	else if(src == CUBLAS_OP_T) return CblasTrans;
	else if(src == CUBLAS_OP_C) return CblasConjTrans;
	else error("OpCublasToCblas: Invalid Op\n"); 
}

char PrintCublasOp(cublasOperation_t src)
{
	
	if(src == CUBLAS_OP_N) return 'N';
	else if(src == CUBLAS_OP_T) return 'T';
	else if(src == CUBLAS_OP_C) return 'C';
	else error("PrintCublasOp: Invalid Op\n"); 
}

