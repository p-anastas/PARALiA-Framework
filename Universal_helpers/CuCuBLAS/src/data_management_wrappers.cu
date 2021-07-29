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

void *gpu_malloc(long long count) {
  void *ret;
  massert(cudaMalloc(&ret, count) == cudaSuccess,
          cudaGetErrorString(cudaGetLastError()));
  return ret;
}

void *pin_malloc(long long count) {
  void *ret;
  massert(cudaMallocHost(&ret, count) == cudaSuccess,
          cudaGetErrorString(cudaGetLastError()));
  return ret;
}

void* CoCoMalloc(long long bytes, short loc){
  int count = 42;
  massert(CUBLAS_STATUS_SUCCESS == cudaGetDeviceCount(&count), "CoCoMalloc: cudaGetDeviceCount failed");
  void *ptr = NULL; 

  if (-2 == loc) {
    //fprintf(stderr, "Allocating %lld bytes to host...\n", bytes);
	ptr = (void*) malloc(bytes);
  }
  else if (-1 == loc) {
    //fprintf(stderr, "Allocating %lld bytes to pinned host...\n", bytes);
	ptr = pin_malloc(bytes);

  } else if (loc >= count || loc < 0)
    error("CoCoMalloc: Invalid device id/location\n");
  else {
    //fprintf(stderr, "Allocating %lld bytes to device(%d)...\n", bytes, loc);
    int prev_loc; cudaGetDevice(&prev_loc);	
    /// TODO: Can this ever happen in a healthy scenario?
    //if (prev_loc != loc) warning("CoCoMalloc: Malloc'ed memory in other device (Previous device: %d, Malloc in: %d)\n", prev_loc, loc);
    cudaSetDevice(loc);
    ptr = gpu_malloc(bytes);
  
	cudaCheckErrors();
    	if (prev_loc != loc){ 
		//warning("CoCoMalloc: Reseting device to previous: %d\n", prev_loc);
		cudaSetDevice(prev_loc);
	}
  }
  cudaCheckErrors();
  return ptr;
}

void gpu_free(void *gpuptr) {
  massert(cudaFree(gpuptr) == cudaSuccess,
          cudaGetErrorString(cudaGetLastError()));
}

void pin_free(void *gpuptr) {
  massert(cudaFreeHost(gpuptr) == cudaSuccess,
          cudaGetErrorString(cudaGetLastError()));
}

void CoCoFree(void * ptr, short loc){
  int count = 42;
  massert(CUBLAS_STATUS_SUCCESS == cudaGetDeviceCount(&count), "CoCoFree: cudaGetDeviceCount failed");

  if (-2 == loc) free(ptr);
  else if (-1 == loc) pin_free(ptr);
  else if (loc >= count || loc < 0) error("CoCoFree: Invalid device id/location\n");
  else {
	int prev_loc; cudaGetDevice(&prev_loc);	
	//if (prev_loc != loc) warning("CoCoFree: Freed memory in other device (Previous device: %d, Free in: %d)\n", prev_loc, loc);
    	cudaSetDevice(loc);
	gpu_free(ptr);
	cudaCheckErrors();
    	if (prev_loc != loc){ 
		//warning("CoCoFree: Reseting device to previous: %d\n", prev_loc);
		cudaSetDevice(prev_loc);
	}
  }
  cudaCheckErrors();
}

void CoCoMemcpy(void* dest, void* src, long long bytes, short loc_dest, short loc_src)
{
	int count = 42;
	massert(CUBLAS_STATUS_SUCCESS == cudaGetDeviceCount(&count), "CoCoMemcpy: cudaGetDeviceCount failed");
	massert(-3 < loc_dest && loc_dest < count, "CoCoMemcpy: Invalid destination device: %d/n", loc_dest);
	massert(-3 < loc_src && loc_src < count, "CoCoMemcpy: Invalid source device: %d/n", loc_src);
	
	enum cudaMemcpyKind kind = cudaMemcpyHostToHost;
	if (loc_src < 0 && loc_dest < 0) memcpy(dest, src, bytes);
	else if (loc_dest < 0) kind = cudaMemcpyDeviceToHost;
	else if (loc_src < 0) kind = cudaMemcpyHostToDevice;
	else kind = cudaMemcpyDeviceToDevice;

	massert(CUBLAS_STATUS_SUCCESS == cudaMemcpy(dest, src, bytes, kind), "CoCoMemcpy: cudaMemcpy from device src=%d to dest=%d failed\n", loc_src, loc_dest);
	cudaCheckErrors();
}

void CoCoMemcpyAsync(void* dest, void* src, long long bytes, short loc_dest, short loc_src, cudaStream_t stream)
{
	int count = 42;
	massert(CUBLAS_STATUS_SUCCESS == cudaGetDeviceCount(&count), "CoCoMemcpyAsync: cudaGetDeviceCount failed\n");
	massert(-2 < loc_dest && loc_dest < count, "CoCoMemcpyAsync: Invalid destination device: %d\n", loc_dest);
	massert(-2 < loc_src && loc_src < count, "CoCoMemcpyAsync: Invalid source device: %d\n", loc_src);
	
	enum cudaMemcpyKind kind;
	if (loc_src < 0 && loc_dest < 0) kind = cudaMemcpyHostToHost;
	else if (loc_dest < 0) kind = cudaMemcpyDeviceToHost;
	else if (loc_src < 0) kind = cudaMemcpyHostToDevice;
	else kind = cudaMemcpyDeviceToDevice;

	cudaMemcpyAsync(dest, src, bytes, kind, stream);
	//cudaCheckErrors();
}

template<typename VALUETYPE>
void CoCoVecInit(VALUETYPE *vec, long long length, int seed, short loc)
{
  int count = 42;
  cudaGetDeviceCount(&count);
  if (!vec) error("CoCoVecInit: vec is not allocated (correctly)\n");
  if (-2 == loc || -1 == loc) CoCoParallelVecInitHost(vec, length, seed);
  else if (loc >= count || loc < 0) error("CoCoVecInit: Invalid device id/location\n");
  else {
	int prev_loc; cudaGetDevice(&prev_loc);	

	//if (prev_loc != loc) warning("CoCoVecInit: Initialized vector in other device (Previous device: %d, init in: %d)\n", prev_loc, loc);
    	cudaSetDevice(loc);
	curandGenerator_t gen;
	/* Create pseudo-random number generator */
	massert(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) == cudaSuccess,
          cudaGetErrorString(cudaGetLastError()));
	/* Set seed */
	massert(curandSetPseudoRandomGeneratorSeed(gen, seed) == cudaSuccess,
          cudaGetErrorString(cudaGetLastError()));
	if (typeid(VALUETYPE) == typeid(float)) 
	  massert(curandGenerateUniform(gen, (float*) vec, length) == cudaSuccess,
            cudaGetErrorString(cudaGetLastError()));
	else if (typeid(VALUETYPE) == typeid(double)) 
	  massert(curandGenerateUniformDouble(gen, (double*) vec, length) == cudaSuccess,
            cudaGetErrorString(cudaGetLastError()));
	cudaCheckErrors();
    	if (prev_loc != loc){ 
		//warning("CoCoVecInit: Reseting device to previous: %d\n", prev_loc);
		cudaSetDevice(prev_loc);
	}
  }
  cudaCheckErrors();
}

template void CoCoVecInit<double>(double *vec, long long length, int seed, short loc);
template void CoCoVecInit<float>(float *vec, long long length, int seed, short loc);

void CoCoMemcpy2D(void* dest, size_t ldest, void* src, size_t lsrc, size_t rows, size_t cols, short elemSize, short loc_dest, short loc_src){
	int count = 42;
	massert(CUBLAS_STATUS_SUCCESS == cudaGetDeviceCount(&count), "CoCoMemcpy2D: cudaGetDeviceCount failed\n");
	massert(-2 < loc_dest && loc_dest < count, "CoCoMemcpy2D: Invalid destination device: %d\n", loc_dest);
	massert(-2 < loc_src && loc_src < count, "CoCoMemcpy2D: Invalid source device: %d\n", loc_src);

	enum cudaMemcpyKind kind;
	if (loc_src < 0 && loc_dest < 0) kind = cudaMemcpyHostToHost;
	else if (loc_dest < 0) kind = cudaMemcpyDeviceToHost;
	else if (loc_src < 0) kind = cudaMemcpyHostToDevice;
	else kind = cudaMemcpyDeviceToDevice;

	massert(cudaSuccess == cudaMemcpy2D(dest, ldest*elemSize, src, lsrc*elemSize, rows*elemSize, cols, kind),  "CoCoMemcpy2D: cudaMemcpy2D failed\n");
	//if (loc_src == -1 && loc_dest >=0) massert(CUBLAS_STATUS_SUCCESS == cublasSetMatrix(rows, cols, elemSize, src, lsrc, dest, ldest), "CoCoMemcpy2DAsync: cublasSetMatrix failed\n");
	//else if (loc_src >=0 && loc_dest == -1) massert(CUBLAS_STATUS_SUCCESS == cublasGetMatrix(rows, cols, elemSize, src, lsrc, dest, ldest),  "CoCoMemcpy2DAsync: cublasGetMatrix failed");

}

void CoCoMemcpy2DAsync(void* dest, size_t ldest, void* src, size_t lsrc, size_t rows, size_t cols, short elemSize, short loc_dest, short loc_src, cudaStream_t stream){
	int count = 42;
	massert(CUBLAS_STATUS_SUCCESS == cudaGetDeviceCount(&count), "CoCoMemcpy2DAsync: cudaGetDeviceCount failed\n");
	massert(-2 < loc_dest && loc_dest < count, "CoCoMemcpyAsync2D: Invalid destination device: %d\n", loc_dest);
	massert(-2 < loc_src && loc_src < count, "CoCoMemcpyAsync2D: Invalid source device: %d\n", loc_src);

	enum cudaMemcpyKind kind;
	if (loc_src < 0 && loc_dest < 0) kind = cudaMemcpyHostToHost;
	else if (loc_dest < 0) kind = cudaMemcpyDeviceToHost;
	else if (loc_src < 0) kind = cudaMemcpyHostToDevice;
	else kind = cudaMemcpyDeviceToDevice;

	cudaMemcpy2DAsync(dest, ldest*elemSize, src, lsrc*elemSize, rows*elemSize, cols, kind, stream);
	//if (loc_src == -1 && loc_dest >=0) massert(CUBLAS_STATUS_SUCCESS == cublasSetMatrixAsync(rows, cols, elemSize, src, lsrc, dest, ldest, stream), "CoCoMemcpy2DAsync: cublasSetMatrixAsync failed\n");
	//else if (loc_src >=0 && loc_dest == -1) massert(CUBLAS_STATUS_SUCCESS == cublasGetMatrixAsync(rows, cols, elemSize, src, lsrc, dest, ldest, stream),  "CoCoMemcpy2DAsync: cublasGetMatrixAsync failed");
}

void CoCoPeLiaEnableGPUPeer(short target_dev_i, short dev_ids[], short num_devices){
	short lvl = 2;
#ifdef DEBUG
	lprintf(lvl-1, "|-----> CoCoPeLiaEnableGPUPeer(%d,dev_ids,%d)\n", target_dev_i, num_devices);
#endif

#ifdef TEST
	lprintf(lvl-1, "|-----> CoCopeLiaDgemm\n");
	double cpu_timer = csecond();
#endif
	cudaSetDevice(dev_ids[target_dev_i]);
	for(int j=0; j<num_devices;j++){
		if (dev_ids[target_dev_i] == dev_ids[j]) continue;
		int can_access_peer; 
		massert(cudaSuccess == cudaDeviceCanAccessPeer(&can_access_peer, dev_ids[target_dev_i], dev_ids[j]), "CoCopeLiaDgemm: cudaDeviceCanAccessPeer failed\n");
		if(can_access_peer){ 
			cudaError_t check_peer = cudaDeviceEnablePeerAccess(dev_ids[j], 0);
			if(check_peer == cudaSuccess){ ;
#ifdef DEBUG
				lprintf(lvl, "Enabled Peer access for dev %d to dev %d\n", dev_ids[target_dev_i], dev_ids[j]);
#endif
			}
			else if (check_peer == cudaErrorPeerAccessAlreadyEnabled){
				cudaGetLastError();
#ifdef DEBUG
				lprintf(lvl, "Peer access already enabled for dev %d to dev %d\n", dev_ids[target_dev_i], dev_ids[j]);
#endif
			}
			else error("Enabling Peer access failed for %d to dev %d\n", dev_ids[target_dev_i], dev_ids[j]);
		}
	}
#ifdef TEST
	cpu_timer = csecond() - cpu_timer; 
	lprintf(lvl, "Utiilizing Peer access for dev %d -> t_enable =%lf ms\n", dev_ids[target_dev_i], 1000*cpu_timer);
	cpu_timer = csecond();
	lprintf(lvl-1, "<-----|\n"); 
#endif
#ifdef DEBUG
	lprintf(lvl-1, "<-----|\n"); 
#endif
}


