///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The start of Zawarudo
///

#include "linkmap.hpp"
#include "PARALiA.hpp"
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
  long int TileDim = 2048;

  switch (argc) {
  case (3):
 	 loc_dest = atoi(argv[ctr++]);
 	 loc_src = atoi(argv[ctr++]);
 	 break;
  default:
 	 error("Incorrect input arguments. Usage: ./correct_run loc_dest loc_src\n");
 	 }

  if (loc_src == loc_dest) error("Transfer benchmark@%s %d->%d: Same device\n",TESTBED, loc_src, loc_dest);

  fprintf(stderr,"\nTransfer benchmark@%s %d->%d : (%ld,%ld)\n", TESTBED, loc_src, loc_dest, TileDim, TileDim);

  cudaGetDeviceCount(&dev_count);

  if (TileDim < 1) error("Transfer Microbench: Bytes must be > 0");
  else if ( dev_count < loc_src + 1) error("Transfer Microbench: Src device does not exist");
  else if ( dev_count < loc_dest + 1) error("Transfer Microbench: Dest device does not exist");

  void* src, *dest, *rev_src, *rev_dest;

  //Only model pinned memory transfers loc_src host loc_dest dev and visa versa
 	if (loc_src < 0 && loc_dest < 0) error("Transfer Microbench: Both locations are in host");
  else if (loc_src == -2 || loc_dest == -2) error("Transfer Microbench: Not pinned memory (synchronous)");
	short dev_ids[LOC_NUM], num_devices = LOC_NUM;
	for(int d=0; d < LOC_NUM; d++){
		dev_ids[d] = deidxize(d);
		// Check/Enable peer access between participating GPUs
		CoCoEnableLinks(d, num_devices);
	}

	void* unit_buffs[3*LOC_NUM];
  long int ldsrc, ldest = ldsrc = TileDim + 1;
	short elemSize = sizeof(double);

	for(short dev_id_idx = 0 ; dev_id_idx < LOC_NUM; dev_id_idx++){
		double timer = csecond();
  	unit_buffs[3*dev_id_idx] = CoCoMalloc(TileDim*(TileDim+1)*elemSize, deidxize(dev_id_idx), 1);
		unit_buffs[3*dev_id_idx+1] = CoCoMalloc(TileDim*(TileDim+1)*elemSize, deidxize(dev_id_idx), 1);
		unit_buffs[3*dev_id_idx+2] = CoCoMalloc(TileDim*(TileDim+1)*elemSize, deidxize(dev_id_idx), 1);
		timer = csecond() - timer;
		//fprintf(stderr, "Allocation of 2 Tiles (dim1 = dim2 = %zu) in unit_id = %2d complete:\t alloc_timer=%lf ms\n",
		//TileDim, deidxize(dev_id_idx), timer  * 1000);

	}

	for(short dev_id_idx = 0 ; dev_id_idx < LOC_NUM; dev_id_idx++){
		double timer = csecond();
	  CoCoVecInit((double*)unit_buffs[3*dev_id_idx], TileDim*(TileDim+1), 42, deidxize(dev_id_idx));
	  CoCoVecInit((double*)unit_buffs[3*dev_id_idx+1], TileDim*(TileDim+1), 43, deidxize(dev_id_idx));
		CoCoVecInit((double*)unit_buffs[3*dev_id_idx+2], TileDim*(TileDim+1), 44, deidxize(dev_id_idx));
		timer = csecond() - timer;
		//fprintf(stderr, "Initialization of 2 Tiles in unit_id = %2d complete:\t init_timer=%lf ms\n",
		//TileDim, deidxize(dev_id_idx), timer  * 1000);
	}

  CQueue_p transfer_queue_list[LOC_NUM][LOC_NUM] = {{NULL}};
	for(short dev_id_idx = 0 ; dev_id_idx < LOC_NUM; dev_id_idx++){
		for(short dev_id_idy = 0 ; dev_id_idy < LOC_NUM; dev_id_idy++)
			if(dev_id_idy!=dev_id_idx){
					if (!transfer_queue_list[dev_id_idx][dev_id_idy]){
						//printf("dev_id = %d, dev_id_idx = %d, dev_id_idy = %d, LOC_NUM = %d\n", dev_id, dev_id_idx, dev_id_idy, LOC_NUM);
						short queue_id = (dev_id_idy == LOC_NUM - 1) ? deidxize(dev_id_idx): deidxize(dev_id_idy);
						transfer_queue_list[dev_id_idx][dev_id_idy] = new CommandQueue(queue_id, 0);
				}
			}
			else transfer_queue_list[dev_id_idx][dev_id_idy] = NULL;
	}

	CoCoPeLiaSelectDevice(loc_dest);
	Event_timer_p device_timer = new Event_timer(loc_dest);
	fprintf(stderr, "\nExploring Link(%d->%d) 1-1 Sharing..\n\n", loc_src, loc_dest);

	double transfer_timer, shared_timer;

	CoCoSyncCheckErr();

	device_timer->start_point(transfer_queue_list[idxize(loc_dest)][idxize(loc_src)]);
	CoCoMemcpy2DAsync(unit_buffs[3*idxize(loc_dest)], ldest,
								unit_buffs[3*idxize(loc_src)], ldsrc,
								TileDim, TileDim, elemSize,
								loc_dest, loc_src, transfer_queue_list[idxize(loc_dest)][idxize(loc_src)]);
	device_timer->stop_point(transfer_queue_list[idxize(loc_dest)][idxize(loc_src)]);
	CoCoSyncCheckErr();

	transfer_timer = device_timer->sync_get_time()/1000;
	fprintf(stderr, "Non-shared Link transfer complete:\t transfer_timer=%lf ms  ( %lf Gb/s)\n\n",
	 	transfer_timer  * 1000, Gval_per_s(TileDim*TileDim*elemSize, transfer_timer));

	for(short dev_id_idx = 0 ; dev_id_idx < LOC_NUM; dev_id_idx++){
		for(short dev_id_idy = 0 ; dev_id_idy < LOC_NUM; dev_id_idy++){
			if (dev_id_idx == dev_id_idy ||
				(dev_id_idx == idxize(loc_dest) && dev_id_idy == idxize(loc_src))) continue;
			for(int itt = 0 ; itt < 10; itt++) CoCoMemcpy2DAsync(unit_buffs[3*dev_id_idx+1], ldest,
										unit_buffs[3*dev_id_idy+1], ldsrc,
										TileDim, TileDim, elemSize,
										deidxize(dev_id_idx), deidxize(dev_id_idy), transfer_queue_list[dev_id_idx][dev_id_idy]);
			device_timer->start_point(transfer_queue_list[idxize(loc_dest)][idxize(loc_src)]);

			CoCoMemcpy2DAsync(unit_buffs[3*idxize(loc_dest)], ldest,
										unit_buffs[3*idxize(loc_src)], ldsrc,
										TileDim, TileDim, elemSize,
										loc_dest, loc_src, transfer_queue_list[idxize(loc_dest)][idxize(loc_src)]);
			device_timer->stop_point(transfer_queue_list[idxize(loc_dest)][idxize(loc_src)]);

			shared_timer = device_timer->sync_get_time()/1000;
			transfer_queue_list[idxize(loc_dest)][idxize(loc_src)]->sync_barrier();
			transfer_queue_list[dev_id_idx][dev_id_idy]->sync_barrier();
			//fprintf(stderr, "Shared Link (%d->%d) transfer complete:\t shared_timer=%lf ms  ( %lf Gb/s)\n\n",
			//	deidxize(dev_id_idy), deidxize(dev_id_idx), shared_timer  * 1000, Gval_per_s(TileDim*TileDim*elemSize, shared_timer));

			if (transfer_timer < shared_timer*(1-NORMALIZE_NEAR_SPLIT_LIMIT)) fprintf(stderr, "Link(%2d->%2d) & Link(%2d->%2d) partially shared: Shared_BW: %1.2lf %%\n\n",
				loc_src, loc_dest, deidxize(dev_id_idy), deidxize(dev_id_idx), 100*transfer_timer/shared_timer);
		}
	}

	for(short dev_id_idx = 0 ; dev_id_idx < LOC_NUM; dev_id_idx++)
	for(short dev_id_idy = 0 ; dev_id_idy < LOC_NUM; dev_id_idy++)
	for(short dev_id_idz = 0 ; dev_id_idz < LOC_NUM; dev_id_idz++)
	for(short dev_id_idk = 0 ; dev_id_idk < LOC_NUM; dev_id_idk++){
			if (dev_id_idx == dev_id_idy || dev_id_idz == dev_id_idk || (dev_id_idx == dev_id_idz && dev_id_idy == dev_id_idk) ||
				(dev_id_idx == idxize(loc_dest) && dev_id_idy == idxize(loc_src)) ||
				(dev_id_idz == idxize(loc_dest) && dev_id_idk == idxize(loc_src)) ) continue;
			for(int itt = 0 ; itt < 5; itt++) CoCoMemcpy2DAsync(unit_buffs[3*dev_id_idx+1], ldest,
										unit_buffs[3*dev_id_idy+1], ldsrc,
										TileDim, TileDim, elemSize,
										deidxize(dev_id_idx), deidxize(dev_id_idy), transfer_queue_list[dev_id_idx][dev_id_idy]);
			for(int itt = 0 ; itt < 5; itt++) CoCoMemcpy2DAsync(unit_buffs[3*dev_id_idz+2], ldest,
										unit_buffs[3*dev_id_idk+2], ldsrc,
										TileDim, TileDim, elemSize,
										deidxize(dev_id_idz), deidxize(dev_id_idk), transfer_queue_list[dev_id_idz][dev_id_idk]);

			device_timer->start_point(transfer_queue_list[idxize(loc_dest)][idxize(loc_src)]);

			CoCoMemcpy2DAsync(unit_buffs[3*idxize(loc_dest)], ldest,
										unit_buffs[3*idxize(loc_src)], ldsrc,
										TileDim, TileDim, elemSize,
										loc_dest, loc_src, transfer_queue_list[idxize(loc_dest)][idxize(loc_src)]);
			device_timer->stop_point(transfer_queue_list[idxize(loc_dest)][idxize(loc_src)]);

			shared_timer = device_timer->sync_get_time()/1000;
			transfer_queue_list[idxize(loc_dest)][idxize(loc_src)]->sync_barrier();
			transfer_queue_list[dev_id_idx][dev_id_idy]->sync_barrier();
			transfer_queue_list[dev_id_idz][dev_id_idk]->sync_barrier();
			CoCoSyncCheckErr();
			//fprintf(stderr, "Shared Link (%d->%d) transfer complete:\t shared_timer=%lf ms  ( %lf Gb/s)\n\n",
			//	deidxize(dev_id_idy), deidxize(dev_id_idx), shared_timer  * 1000, Gval_per_s(TileDim*TileDim*elemSize, shared_timer));

			if (transfer_timer < shared_timer*(1-NORMALIZE_NEAR_SPLIT_LIMIT)) fprintf(stderr, "Link(%2d->%2d) & Link(%2d->%2d) & Link(%2d->%2d) partially shared: Shared_BW: %1.2lf %%\n\n",
				loc_src, loc_dest, deidxize(dev_id_idy), deidxize(dev_id_idx), deidxize(dev_id_idk), deidxize(dev_id_idz), 100*transfer_timer/shared_timer);
		}




	CoCoSyncCheckErr();
	for(short dev_id_idx = 0 ; dev_id_idx < LOC_NUM; dev_id_idx++){
		double timer = csecond();
	  CoCoFree(unit_buffs[3*dev_id_idx], deidxize(dev_id_idx));
	  CoCoFree(unit_buffs[3*dev_id_idx+1], deidxize(dev_id_idx));
		CoCoFree(unit_buffs[3*dev_id_idx+2], deidxize(dev_id_idx));
		timer = csecond() - timer;
		//fprintf(stderr, "De-allocation of 2 Tiles in unit_id = %2d complete:\t init_timer=%lf ms\n",
		//TileDim, deidxize(dev_id_idx), timer  * 1000);
	}
  return 0;
}
