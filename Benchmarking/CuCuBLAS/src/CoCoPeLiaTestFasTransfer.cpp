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

	void* unit_buffs[LOC_NUM];
  long int ldsrc, ldest = ldsrc = TileDim + 1;
	short elemSize = sizeof(double);

	for(short dev_id_idx = 0 ; dev_id_idx < LOC_NUM; dev_id_idx++){
		double timer = csecond();
  	unit_buffs[dev_id_idx] = CoCoMalloc(TileDim*(TileDim+1)*elemSize, deidxize(dev_id_idx));
		timer = csecond() - timer;
		//fprintf(stderr, "Allocation of 2 Tiles (dim1 = dim2 = %zu) in unit_id = %2d complete:\t alloc_timer=%lf ms\n",
		//TileDim, deidxize(dev_id_idx), timer  * 1000);

	}

	for(short dev_id_idx = 0 ; dev_id_idx < LOC_NUM; dev_id_idx++){
		double timer = csecond();
	  CoCoVecInit((double*)unit_buffs[dev_id_idx], TileDim*(TileDim+1), 42 + dev_id_idx, deidxize(dev_id_idx));
		timer = csecond() - timer;
		//fprintf(stderr, "Initialization of 2 Tiles in unit_id = %2d complete:\t init_timer=%lf ms\n",
		//TileDim, deidxize(dev_id_idx), timer  * 1000);
	}
	double* cpu_buff = (double*) calloc (TileDim*(TileDim+1),elemSize),
					*cpu_buff_comp = (double*) calloc (TileDim*(TileDim+1),elemSize);
	CoCoMemcpy2D(cpu_buff, ldest, unit_buffs[idxize(loc_src)], ldsrc, TileDim, TileDim, elemSize,	-2, loc_src);

  CQueue_p transfer_queue_list[LOC_NUM][LOC_NUM] = {{NULL}};
	for(short dev_id_idx = 0 ; dev_id_idx < LOC_NUM; dev_id_idx++){
		short dev_id = deidxize(dev_id_idx);
		for(short dev_id_idy = 0 ; dev_id_idy < LOC_NUM; dev_id_idy++)
			if(dev_id_idy!=dev_id_idx){
					if (!transfer_queue_list[dev_id_idx][dev_id_idy]){
						//printf("dev_id = %d, dev_id_idx = %d, dev_id_idy = %d, LOC_NUM = %d\n", dev_id, dev_id_idx, dev_id_idy, LOC_NUM);
						transfer_queue_list[dev_id_idx][dev_id_idy] = new CommandQueue(dev_id);
				}
			}
			else transfer_queue_list[dev_id_idx][dev_id_idy] = NULL;
	}

	CoCoPeLiaSelectDevice(loc_src);
	Event_timer_p device_timer = new Event_timer(loc_dest);
	fprintf(stderr, "\nExploring FasT Copy(%d->%d)...\n\n", loc_src, loc_dest);

	double transfer_timer, hop_timer;

	CoCoSyncCheckErr();

	device_timer->start_point(transfer_queue_list[idxize(loc_dest)][idxize(loc_src)]);
	CoCoMemcpy2DAsync(unit_buffs[idxize(loc_dest)], ldest,
								unit_buffs[idxize(loc_src)], ldsrc,
								TileDim, TileDim, elemSize,
								loc_dest, loc_src, transfer_queue_list[idxize(loc_dest)][idxize(loc_src)]);
	device_timer->stop_point(transfer_queue_list[idxize(loc_dest)][idxize(loc_src)]);
	CoCoSyncCheckErr();

	transfer_timer = device_timer->sync_get_time()/1000;
	CoCoMemcpy2D(cpu_buff_comp, ldest, unit_buffs[idxize(loc_dest)], ldsrc, TileDim, TileDim, elemSize,	-2, loc_dest);
	Dtest_equality(cpu_buff, cpu_buff_comp, TileDim*(TileDim+1));
	memset(cpu_buff_comp, 0, TileDim*(TileDim+1)*elemSize);
	fprintf(stderr, "Direct Link transfer complete:\t transfer_timer=%lf ms  ( %lf Gb/s)\n\n",
	 	transfer_timer  * 1000, Gval_per_s(TileDim*TileDim*elemSize, transfer_timer));

	for(short dev_id_idx = 0 ; dev_id_idx < LOC_NUM; dev_id_idx++) if (dev_id_idx != idxize(loc_src) && dev_id_idx != idxize(loc_dest)){
		link_road_p test_road = (link_road_p) malloc(sizeof(struct link_road));
		test_road->hop_num = 3;
		test_road->hop_uid_list[0] = loc_src;
		test_road->hop_uid_list[1] = deidxize(dev_id_idx);
		test_road->hop_uid_list[2] = loc_dest;

		test_road->hop_buf_list[0] = unit_buffs[idxize(loc_src)];
		test_road->hop_buf_list[1] = unit_buffs[dev_id_idx];
		test_road->hop_buf_list[2] = unit_buffs[idxize(loc_dest)];

		test_road->hop_ldim_list[0] = test_road->hop_ldim_list[1] = test_road->hop_ldim_list[2] = TileDim + 1;

		test_road->hop_cqueue_list[0] = transfer_queue_list[dev_id_idx][idxize(loc_src)];
		test_road->hop_cqueue_list[1] = transfer_queue_list[idxize(loc_dest)][dev_id_idx];

		test_road->hop_event_list[0] = new Event(deidxize(dev_id_idx));
		test_road->hop_event_list[1] = new Event(loc_dest);


		hop_timer = csecond();
		FasTCoCoMemcpy2DAsync(test_road, TileDim, TileDim, elemSize);
		//transfer_queue_list[idxize(loc_dest)][dev_id_idx]->sync_barrier();
		test_road->hop_event_list[1]->sync_barrier();
		hop_timer = csecond() - hop_timer;
		CoCoMemcpy2D(cpu_buff_comp, ldest, unit_buffs[idxize(loc_dest)], ldsrc, TileDim, TileDim, elemSize,	-2, loc_dest);
		Dtest_equality(cpu_buff, cpu_buff_comp, TileDim*(TileDim+1));
		memset(cpu_buff_comp, 0, TileDim*(TileDim+1)*elemSize);
		fprintf(stderr, "1-hop Link (%d->%d->%d) transfer complete:\t hop_timer=%lf ms  ( %lf Gb/s)\n\n",
			loc_src, deidxize(dev_id_idx), loc_dest, hop_timer  * 1000, Gval_per_s(TileDim*TileDim*elemSize, hop_timer));

	}

	for(short dev_id_idx = 0 ; dev_id_idx < LOC_NUM; dev_id_idx++)
	if (dev_id_idx != idxize(loc_src) && dev_id_idx != idxize(loc_dest))
	for(short dev_id_idy = 0 ; dev_id_idy < LOC_NUM; dev_id_idy++)
	if (dev_id_idy != idxize(loc_src) && dev_id_idy != idxize(loc_dest) && dev_id_idy != dev_id_idx){
		link_road_p test_road = (link_road_p) malloc(sizeof(struct link_road));
		test_road->hop_num = 4;
		test_road->hop_uid_list[0] = loc_src;
		test_road->hop_uid_list[1] = deidxize(dev_id_idx);
		test_road->hop_uid_list[2] = deidxize(dev_id_idy);
		test_road->hop_uid_list[3] = loc_dest;

		test_road->hop_buf_list[0] = unit_buffs[idxize(loc_src)];
		test_road->hop_buf_list[1] = unit_buffs[dev_id_idx];
		test_road->hop_buf_list[2] = unit_buffs[dev_id_idy];
		test_road->hop_buf_list[3] = unit_buffs[idxize(loc_dest)];

		test_road->hop_ldim_list[0] = test_road->hop_ldim_list[1] = test_road->hop_ldim_list[2] = test_road->hop_ldim_list[3] = TileDim + 1;

		test_road->hop_cqueue_list[0] = transfer_queue_list[dev_id_idx][idxize(loc_src)];
		test_road->hop_cqueue_list[1] = transfer_queue_list[dev_id_idy][dev_id_idx];
		test_road->hop_cqueue_list[2] = transfer_queue_list[idxize(loc_dest)][dev_id_idy];

		test_road->hop_event_list[0] = new Event(deidxize(dev_id_idx));
		test_road->hop_event_list[1] = new Event(deidxize(dev_id_idy));
		test_road->hop_event_list[2] = new Event(loc_dest);

		hop_timer = csecond();
		FasTCoCoMemcpy2DAsync(test_road, TileDim, TileDim, elemSize);
		//transfer_queue_list[idxize(loc_dest)][dev_id_idx]->sync_barrier();
		test_road->hop_event_list[2]->sync_barrier();
		hop_timer = csecond() - hop_timer;
		CoCoMemcpy2D(cpu_buff_comp, ldest, unit_buffs[idxize(loc_dest)], ldsrc, TileDim, TileDim, elemSize,	-2, loc_dest);
		Dtest_equality(cpu_buff, cpu_buff_comp, TileDim*(TileDim+1));
		memset(cpu_buff_comp, 0, TileDim*(TileDim+1)*elemSize);
		fprintf(stderr, "2-hop Link (%d->%d->%d->%d) transfer complete:\t hop_timer=%lf ms  ( %lf Gb/s)\n\n",
			loc_src, deidxize(dev_id_idx), deidxize(dev_id_idy), loc_dest, hop_timer  * 1000, Gval_per_s(TileDim*TileDim*elemSize, hop_timer));

	}

	CoCoSyncCheckErr();
	for(short dev_id_idx = 0 ; dev_id_idx < LOC_NUM; dev_id_idx++){
		double timer = csecond();
	  CoCoFree(unit_buffs[dev_id_idx], deidxize(dev_id_idx));
		timer = csecond() - timer;
		//fprintf(stderr, "De-allocation of 2 Tiles in unit_id = %2d complete:\t init_timer=%lf ms\n",
		//TileDim, deidxize(dev_id_idx), timer  * 1000);
	}
  return 0;
}
