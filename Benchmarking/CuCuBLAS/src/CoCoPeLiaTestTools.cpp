///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The start of Zawarudo
///

#include "unihelpers.hpp"
#include "CoCoPeLia.hpp"
#include "BackenedLibsWrapped.hpp"
#include "Testing.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <cstring>

#include "backend_wrappers.hpp"

#define CBLASXT_MAX_SAFE_TILE 10000

int main(const int argc, const char *argv[]) {

	int ctr = 1, samples, dev_id, dev_count;

  short loc_src = -2, loc_dest = -2;
  size_t TileDim = 2048;

  switch (argc) {
  case (3):
 	 loc_dest = atoi(argv[ctr++]);
 	 loc_src = atoi(argv[ctr++]);
 	 break;
  default:
 	 error("Incorrect input arguments. Usage: ./correct_run loc_dest loc_src\n");
 	 }

  if (loc_src == loc_dest) error("Transfer benchmark@%s %d->%d: Same device\n",TESTBED, loc_src, loc_dest);

  fprintf(stderr,"\nTransfer benchmark@%s %d->%d : (%d,%d)\n", TESTBED, loc_src, loc_dest, TileDim, TileDim);

  cudaGetDeviceCount(&dev_count);

  if (TileDim < 1) error("Transfer Microbench: Bytes must be > 0");
  else if ( dev_count < loc_src + 1) error("Transfer Microbench: Src device does not exist");
  else if ( dev_count < loc_dest + 1) error("Transfer Microbench: Dest device does not exist");

  void* src, *dest, *rev_src, *rev_dest;

  //Only model pinned memory transfers loc_src host loc_dest dev and visa versa
 	 if (loc_src < 0 && loc_dest < 0) error("Transfer Microbench: Both locations are in host");
  else if (loc_src == -2 || loc_dest == -2) error("Transfer Microbench: Not pinned memory (synchronous)");
 	 else if ( loc_src >= 0 && loc_dest >= 0){
 	 short dev_id[2], num_devices = 2;
 	 dev_id[0] = loc_src;
 	 dev_id[1] = loc_dest;
 	 // Check/Enable peer access between participating GPUs
 	 CoCoEnableLinks(0, dev_id, num_devices);
 	 // Check/Enable peer access between participating GPUs
 	 CoCoEnableLinks(1, dev_id, num_devices);
  }
  else if(loc_src >= 0) cudaSetDevice(loc_src);
  else if(loc_dest >= 0) cudaSetDevice(loc_dest);

  size_t ldsrc, ldest = ldsrc = TileDim + 1;
	short elemSize = sizeof(double);

  src = CoCoMalloc(TileDim*(TileDim+1)*elemSize, loc_src);
  dest =  CoCoMalloc(TileDim*(TileDim+1)*elemSize, loc_dest);
  rev_src = CoCoMalloc(TileDim*(TileDim+1)*elemSize, loc_dest);
  rev_dest = CoCoMalloc(TileDim*(TileDim+1)*elemSize, loc_src);

	void* host_buf_in, *host_buf_out, *host_buf_in_rev, *conspiquous_ptr = src;

	host_buf_in = CoCoMalloc(TileDim*(TileDim+1)*elemSize, -1);
	host_buf_out = CoCoMalloc(TileDim*(TileDim+1)*elemSize, -1);
	host_buf_in_rev = CoCoMalloc(TileDim*(TileDim+1)*elemSize, -1);

	fprintf(stderr, "Pointers: src=%p, dest=%p, rev_src=%p, rev_dest=%p\n\n",
		src, dest, rev_src, rev_dest);

  CoCoVecInit((double*)src, TileDim*(TileDim+1), 42, loc_src);
  CoCoVecInit((double*)rev_src, TileDim*(TileDim+1), 43, loc_dest);

  CQueue_p transfer_link = new CommandQueue(loc_src), reverse_link = new CommandQueue(loc_src);
	CoCoPeLiaSelectDevice(loc_src);
	Event_timer_p device_timer = new Event_timer(loc_src);

	double transfer_timer;

	CoCoSyncCheckErr();

	device_timer->start_point(transfer_link);
	device_timer->stop_point(transfer_link);
	CoCoSyncCheckErr();

	transfer_timer = device_timer->sync_get_time()/1000;
	fprintf(stderr, "Microbenchmark (dim1 = dim2 = %zu) complete:\t transfer_timer=%lf ms  ( %lf Gb/s)\n\n",
	 	TileDim, transfer_timer  * 1000, Gval_per_s(TileDim*TileDim*elemSize, transfer_timer));


	CoCoSyncCheckErr();
  CoCoFree(src, loc_src);
 	CoCoFree(dest, loc_dest);
  CoCoFree(rev_src, loc_dest);
  CoCoFree(rev_dest, loc_src);
  return 0;
}
