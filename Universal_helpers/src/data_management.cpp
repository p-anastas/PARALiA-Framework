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
#include "backend_wrappers.hpp"

void CoCoSyncCheckErr(){ backend_check_errors_Sync(); }

void CoCoASyncCheckErr(){ backend_check_errors_Async(); }

void CoCoEnableLinks(short target_dev_i, short dev_ids[], short num_devices) { backend_enable_links(target_dev_i, dev_ids, num_devices); }

void* CoCoMalloc(long long N_bytes, short loc){ backend_malloc ( N_bytes, loc); }

void CoCoFree(void * ptr, short loc){ backend_free(ptr, loc); }

void CoCoMemcpy(void* dest, void* src, long long N_bytes, short loc_dest, short loc_src) { backend_memcpy(dest, src, N_bytes, loc_dest, loc_src); }

void CoCoMemcpy2D(void* dest, size_t ldest, void* src, size_t lsrc, size_t rows, size_t cols, short elemSize, short loc_dest, short loc_src)
{ backend_memcpy2D(dest, ldest, src, lsrc, rows, cols, elemSize, loc_dest, loc_src); }

void CoCoMemcpyAsync(void* dest, void* src, long long N_bytes, short loc_dest, short loc_src, CQueue_p transfer_queue)
{ backend_memcpy_async(dest, src, N_bytes, loc_dest, loc_src, transfer_queue); } 

void CoCoMemcpy2DAsync(void* dest, size_t ldest, void* src, size_t lsrc, size_t rows, size_t cols, short elemSize, short loc_dest, short loc_src, CQueue_p transfer_queue)
{ backend_memcpy2D_async(dest, ldest, src, lsrc, rows, cols, elemSize, loc_dest, loc_src, transfer_queue); }

template<typename VALUETYPE>
extern void CoCoVecInit(VALUETYPE *vec, long long length, int seed, short loc){ backend_vec_init<VALUETYPE>(vec, length, seed, loc); }

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

size_t CoCoGetMaxDimSqAsset2D(short Asset2DNum, short dsize, size_t step, short loc){ backend_get_max_dim_sq_Asset2D(Asset2DNum, dsize, step, loc); }

size_t CoCoGetMaxDimAsset1D(short Asset1DNum, short dsize, size_t step, short loc){ backend_get_max_dim_Asset1D(Asset1DNum, dsize, step, loc); }

