///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The DGEMM CoCopeLia implementation.
///

#include <cblas.h>

#include "backend_lib_wrappers.hpp"
#include "unihelpers.hpp"

int CoCoPeLiaGetDevice(){
  int dev_id = -1;
  cudaError_t err = cudaGetDevice(&dev_id);
  massert(cudaSuccess == err,
    "CoCoPeLiaGetDevice: cudaGetDevice failed - %s\n", cudaGetErrorString(err));
  return dev_id;
}

void CoCoPeLiaSelectDevice(short dev_id){
  cudaSetDevice(dev_id);
  cudaError_t err = cudaSetDevice(dev_id);
  massert(cudaSuccess == err,
    "CoCoPeLiaSelectDevice: cudaSetDevice failed - %s\n", cudaGetErrorString(err));
}
void CoCoPeLiaDevGetMemInfo(long long* free_dev_mem, long long* max_dev_mem){
  size_t free_dev_mem_tmp, max_dev_mem_tmp;
    cudaError_t err = cudaMemGetInfo(&free_dev_mem_tmp, &max_dev_mem_tmp);
  	massert(cudaSuccess == err,
      "CoCoPeLiaDevGetMemInfo: cudaMemGetInfo failed - %s\n", cudaGetErrorString(err));
    *free_dev_mem = (long long) free_dev_mem_tmp;
    *max_dev_mem = (long long) max_dev_mem_tmp;
}
