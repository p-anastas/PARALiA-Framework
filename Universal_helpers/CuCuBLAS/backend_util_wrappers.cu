///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief Some CUDA function calls with added error-checking
///

#include <cstdio>
#include <typeinfo>
#include <float.h>
#include <curand.h>

#include "backend_wrappers.hpp"

/*void print_devices() {
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
*/

void CoCoSyncCheckErr(){
  cudaError_t errSync = cudaDeviceSynchronize();
  if (errSync != cudaSuccess)
    printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
}

void CoCoASyncCheckErr(){
  cudaError_t errAsync = cudaGetLastError();
  if (errAsync != cudaSuccess)
    printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
}

void cudaCheckErrors(){
	//CoCoASyncCheckErr();
	CoCoSyncCheckErr();
}

int CoCoPeLiaGetDevice(){
  int dev_id = -1;
  cudaError_t err = cudaGetDevice(&dev_id);
  massert(cudaSuccess == err,
    "CoCoPeLiaGetDevice: cudaGetDevice failed - %s\n", cudaGetErrorString(err));
  return dev_id;
}

void CoCoPeLiaSelectDevice(short dev_id){
  int dev_count;
  cudaError_t err = cudaGetDeviceCount(&dev_count);
  if(dev_id >= 0 && dev_id < dev_count){
  cudaError_t err = cudaSetDevice(dev_id);
  massert(cudaSuccess == err,
    "CoCoPeLiaSelectDevice(%d): cudaSetDevice(%d) failed - %s\n", dev_id, dev_id, cudaGetErrorString(err));
  }
  else if(dev_id == -1){  /// "Host" device loc id used by CoCoPeLia
    cudaSetDevice(0);
  }
  else error("CoCoPeLiaSelectDevice(%d): invalid dev_id\n", dev_id);
}

void CoCoPeLiaDevGetMemInfo(long long* free_dev_mem, long long* max_dev_mem){
  size_t free_dev_mem_tmp, max_dev_mem_tmp;
    int tmp_dev_id;
    cudaError_t err = cudaGetDevice(&tmp_dev_id);
    // TODO: For the CPU this function returns device 0 memory availability. Its a feature not a bug.
    massert(cudaSuccess == err,
      "CoCoPeLiaDevGetMemInfo: cudaGetDevice failed - %s\n", cudaGetErrorString(err));
    err = cudaMemGetInfo(&free_dev_mem_tmp, &max_dev_mem_tmp);
  	massert(cudaSuccess == err,
      "CoCoPeLiaDevGetMemInfo: cudaMemGetInfo failed - %s\n", cudaGetErrorString(err));
    *free_dev_mem = (long long) free_dev_mem_tmp;
    *max_dev_mem = (long long) max_dev_mem_tmp;
}

void TransposeTranslate(char TransChar, CBLAS_TRANSPOSE* cblasFlag, cublasOperation_t* cuBLASFlag, long int* ldim, long int dim1, long int dim2){
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
