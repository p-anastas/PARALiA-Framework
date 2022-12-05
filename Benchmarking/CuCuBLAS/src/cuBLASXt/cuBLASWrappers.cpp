///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief BLAS lvl 3 wrappers for benchmarks.
///
#include <cassert>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "CoCoPeLia.hpp"
#include "unihelpers.hpp"
#include "backend_wrappers.hpp"

double cuBLASDaxpyWrap(long int N, double alpha, double* x, long int incx, double* y, long int incy, double cpu_ratio, short dev_num, int dev_ids[] ){
	short lvl = 1;
	double total_t = csecond();
#ifdef DEBUG
	lprintf(lvl-1, "|-----> cuBLASDaxpyWrap(%zu,%lf,x(%d),%zu,y(%d),%zu)\n",
		N, alpha, CoCoGetPtrLoc(x), incx, CoCoGetPtrLoc(y), incy);
#endif

#ifdef TEST
	lprintf(lvl-1, "|-----> cuBLASDaxpyWrap\n");
	double cpu_timer = csecond();
#endif

  if (dev_num > 1) error("cuBLASDaxpyWrap: Not implemented for multiple devices\n");
  else if (cpu_ratio > 0) error("cuBLASDaxpyWrap: Not implemented for cpu-assisted execution\n");
  cudaSetDevice(dev_ids[0]);

	cublasStatus_t stat;
	cublasHandle_t handle0;

	/// Required allocations for device
	assert(CUBLAS_STATUS_SUCCESS == cublasCreate(&handle0));

#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "cuBLAS initialization -> t_init = %lf ms\n", cpu_timer*1000);
    	cpu_timer = csecond();
#endif
	assert(CUBLAS_STATUS_SUCCESS == cublasDaxpy(handle0, N, &alpha, x, incx, y, incy));
	CoCoSyncCheckErr();
#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "cuBLASDaxpyWrap execution time -> t_kernel = %lf ms\n", cpu_timer*1000);
#endif

	/// Free local buffers
	assert(CUBLAS_STATUS_SUCCESS == cublasDestroy(handle0));
	CoCoSyncCheckErr();
	total_t = csecond() - total_t;
	return total_t;
}
