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

size_t CoCoGetMaxDimSqAsset2D(short Asset2DNum, short dsize, size_t step, short loc){
	size_t free_cuda_mem, max_cuda_mem;
	int prev_loc; cudaGetDevice(&prev_loc);
    /// TODO: Can this ever happen in a healthy scenario?
    //if (prev_loc != loc) warning("CoCoMalloc: Malloc'ed memory in other device (Previous device: %d, Malloc in: %d)\n", prev_loc, loc);
    cudaSetDevice(loc);
	massert(cudaSuccess == cudaMemGetInfo(&free_cuda_mem, &max_cuda_mem), "backend_get_max_dim_sq_Asset2D: cudaMemGetInfo failed");

	// Define the max size of a benchmark kernel to run on this machine.
	size_t maxDim = (( (size_t) sqrt((free_cuda_mem*PROBLEM_GPU_PERCENTAGE/100.0)/(Asset2DNum*dsize))) / step) * step;
	cudaSetDevice(prev_loc);
	return maxDim;
}

size_t CoCoGetMaxDimAsset1D(short Asset1DNum, short dsize, size_t step, short loc){
	size_t free_cuda_mem, max_cuda_mem;
	int prev_loc; cudaGetDevice(&prev_loc);
    /// TODO: Can this ever happen in a healthy scenario?
    //if (prev_loc != loc) warning("CoCoMalloc: Malloc'ed memory in other device (Previous device: %d, Malloc in: %d)\n", prev_loc, loc);
    cudaSetDevice(loc);
	massert(cudaSuccess == cudaMemGetInfo(&free_cuda_mem, &max_cuda_mem), "backend_get_max_dim_Asset1D: cudaMemGetInfo failed");

	size_t maxDim = (( (size_t) (free_cuda_mem*PROBLEM_GPU_PERCENTAGE/100.0)/(Asset1DNum*dsize)) / step) * step;
	cudaSetDevice(prev_loc);
	return maxDim;
}

short CoCoGetPtrLoc(const void * in_ptr)
{
#ifndef CUDA_VER
#error CUDA_VER Undefined!
#elif (CUDA_VER == 920)
	short loc = -2;
	cudaPointerAttributes ptr_att;
	if (cudaSuccess != cudaPointerGetAttributes(&ptr_att, in_ptr)) warning("CoCoBLAS_ptr_check_cuda_9_2: Pointer not visible to CUDA, host alloc or error");
	if (ptr_att.memoryType == cudaMemoryTypeHost) loc = -1;
	else if (ptr_att.memoryType == cudaMemoryTypeDevice) loc = ptr_att.device;
	else if (ptr_att.isManaged) loc = ptr_att.device;
	else error("CoCoBLAS_ptr_check_cuda_9_2: Invalid memory type");
	return loc;
#elif (CUDA_VER == 1100)
	short loc = -2;
	cudaPointerAttributes ptr_att;
	if (cudaSuccess != cudaPointerGetAttributes(&ptr_att, in_ptr)) warning("CoCopeLia_ptr_check_cuda_11: Pointer not visible to CUDA, host alloc or error");
	if (ptr_att.type == cudaMemoryTypeHost) loc = -1;
	else if (ptr_att.type == cudaMemoryTypeDevice) loc = ptr_att.device;
	// TODO: Unified memory is considered available in the GPU as cuBLASXt ( not bad, not great)
	else if (ptr_att.type == cudaMemoryTypeManaged) loc = ptr_att.device;
	else error("CoCoBLAS_ptr_check_cuda_11: Invalid memory type");
	return loc;
#else
#error Unknown CUDA_VER!
#endif
}

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

void CoCoMemcpyAsync(void* dest, void* src, long long bytes, short loc_dest, short loc_src, CQueue_p transfer_queue)
{
	cudaStream_t stream = *((cudaStream_t*)transfer_queue->cqueue_backend_ptr);
	int count = 42;
	massert(CUBLAS_STATUS_SUCCESS == cudaGetDeviceCount(&count), "CoCoMemcpyAsync: cudaGetDeviceCount failed\n");
	massert(-2 < loc_dest && loc_dest < count, "CoCoMemcpyAsync: Invalid destination device: %d\n", loc_dest);
	massert(-2 < loc_src && loc_src < count, "CoCoMemcpyAsync: Invalid source device: %d\n", loc_src);

	enum cudaMemcpyKind kind;
	if (loc_src < 0 && loc_dest < 0) kind = cudaMemcpyHostToHost;
	else if (loc_dest < 0) kind = cudaMemcpyDeviceToHost;
	else if (loc_src < 0) kind = cudaMemcpyHostToDevice;
	else kind = cudaMemcpyDeviceToDevice;

	massert(cudaSuccess == cudaMemcpyAsync(dest, src, bytes, kind, stream),
	"CoCoMemcpy2D: cudaMemcpyAsync failed\n");
	//cudaCheckErrors();
}

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

void CoCoMemcpy2DAsync(void* dest, size_t ldest, void* src, size_t lsrc, size_t rows, size_t cols, short elemSize, short loc_dest, short loc_src, CQueue_p transfer_queue){
	short lvl = 6;
#ifdef DDEBUG
	lprintf(lvl, "CoCoMemcpy2DAsync(dest=%p, ldest =%zu, src=%p, lsrc = %zu, rows = %zu, cols = %zu, elemsize = %d, loc_dest = %d, loc_src = %d)\n",
		dest, ldest, src, lsrc, rows, cols, elemSize, loc_dest, loc_src);
#endif
	cudaStream_t stream = *((cudaStream_t*)transfer_queue->cqueue_backend_ptr);
	int count = 42;
	massert(CUBLAS_STATUS_SUCCESS == cudaGetDeviceCount(&count), "CoCoMemcpy2DAsync: cudaGetDeviceCount failed\n");
	massert(-2 < loc_dest && loc_dest < count, "CoCoMemcpyAsync2D: Invalid destination device: %d\n", loc_dest);
	massert(-2 < loc_src && loc_src < count, "CoCoMemcpyAsync2D: Invalid source device: %d\n", loc_src);

	enum cudaMemcpyKind kind;
	if (loc_src < 0 && loc_dest < 0) kind = cudaMemcpyHostToHost;
	else if (loc_dest < 0) kind = cudaMemcpyDeviceToHost;
	else if (loc_src < 0) kind = cudaMemcpyHostToDevice;
	else kind = cudaMemcpyDeviceToDevice;

	massert(cudaSuccess == cudaMemcpy2DAsync(dest, ldest*elemSize, src, lsrc*elemSize,
		rows*elemSize, cols, kind, stream),  "CoCoMemcpy2D: cudaMemcpy2DAsync failed\n");
	//if (loc_src == -1 && loc_dest >=0) massert(CUBLAS_STATUS_SUCCESS == cublasSetMatrixAsync(rows, cols, elemSize, src, lsrc, dest, ldest, stream), "CoCoMemcpy2DAsync: cublasSetMatrixAsync failed\n");
	//else if (loc_src >=0 && loc_dest == -1) massert(CUBLAS_STATUS_SUCCESS == cublasGetMatrixAsync(rows, cols, elemSize, src, lsrc, dest, ldest, stream),  "CoCoMemcpy2DAsync: cublasGetMatrixAsync failed");
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

template<typename VALUETYPE>
void CoCoParallelVecInitHost(VALUETYPE *vec, long long length, int seed)
{
	srand(seed);
	//#pragma omp parallel for
	for (long long i = 0; i < length; i++) vec[i] = (VALUETYPE) Drandom();
}

template void CoCoParallelVecInitHost<double>(double *vec, long long length, int seed);
template void CoCoParallelVecInitHost<float>(float *vec, long long length, int seed);

void backend_enableGPUPeer(short target_dev_i, short dev_ids[], short num_devices){
	short lvl = 2;
#ifdef DEBUG
	lprintf(lvl-1, "|-----> CoCoPeLiaEnableGPUPeer(%d,dev_ids,%d)\n", target_dev_i, num_devices);
#endif

#ifdef TEST
	lprintf(lvl-1, "|-----> CoCoPeLiaEnableGPUPeer\n");
	double cpu_timer = csecond();
#endif
	cudaSetDevice(dev_ids[target_dev_i]);
	for(int j=0; j<num_devices;j++){
		if (dev_ids[target_dev_i] == dev_ids[j] || dev_ids[target_dev_i] == -1 || dev_ids[j] == -1) continue;
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

void CoCoEnableLinks(short target_dev_i, short dev_ids[], short num_devices){
	backend_enableGPUPeer(target_dev_i, dev_ids, num_devices);
}
