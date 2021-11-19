///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The DGEMM CoCopeLia implementation.
///

#include <cblas.h>

#include "backend_lib_wrappers.hpp"
#include "unihelpers.hpp"

void CoCoPeLiaSelectDevice(short dev_id){
  cudaSetDevice(dev_id);
}
void CoCoPeLiaDevGetMemInfo(long long* free_dev_mem, long long* max_dev_mem){
  size_t free_dev_mem_tmp, max_dev_mem_tmp;
  	massert(cudaSuccess == cudaMemGetInfo(&free_dev_mem_tmp, &max_dev_mem_tmp), "CoCoPeLiaDevGetMemInfo: cudaMemGetInfo failed");
    *free_dev_mem = (long long) free_dev_mem_tmp;
    *max_dev_mem = (long long) max_dev_mem_tmp;
}
