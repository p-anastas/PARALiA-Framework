///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The "Subkernel" related function implementations.
///

#include "Subkernel.hpp"
#include "unihelpers.hpp"
#include "Asset.hpp"
#include "DataCaching.hpp"
#include "backend_wrappers.hpp"

/// TODO: Works for systems with up to 128 devices, not 'completely' future-proof
CQueue_p h2d_queue[128] = {NULL}, d2h_queue[128] = {NULL}, exec_queue[128] = {NULL};
int Subkernel_num = 0, backend_init_flag[128] = {0};

#ifdef MULTIDEVICE_REDUCTION_ENABLE
void* reduce_buf[128*MAX_BUFFERING_L] = {NULL};
short reduce_buf_it[128] = {0};
long long reduce_buf_sz[128*MAX_BUFFERING_L] = {0};
#endif

Subkernel::Subkernel(short TileNum_in, const char* name){
	id = Subkernel_num;
	op_name = name;
	Subkernel_num++;
	TileNum = TileNum_in;
	TileDimlist = (short*) malloc(TileNum*sizeof(short));
	TileList = (void**) malloc(TileNum*sizeof(void*));
	work_complete = 0;
	prev = next = NULL;
}

Subkernel::~Subkernel(){
	short lvl = 5;
#ifdef DDEBUG
	lprintf(lvl, "Subkernel(dev=%d,id=%d):~Subkernel\n", run_dev_id, id);
#endif
	short run_dev_id_idx = (run_dev_id == -1)?  LOC_NUM - 1 : run_dev_id;
	Subkernel_num--;
	free(TileDimlist);
	free(TileList);
	free(operation_params);
	delete operation_complete;
	if (WR_last) delete writeback_complete;
#ifdef MULTIDEVICE_REDUCTION_ENABLE
#ifndef BUFFER_REUSE_ENABLE
		reduce_buf_it[run_dev_id_idx] = 0;
		for(int idx = 0; idx < MAX_BUFFERING_L; idx++)
			if(reduce_buf[idx*128 + run_dev_id]!= NULL){
				CoCoFree(reduce_buf[idx*128 + run_dev_id], CoCoGetPtrLoc(reduce_buf[idx*128 + run_dev_id]));
				reduce_buf[idx*128 + run_dev_id] = NULL;
#ifdef DDEBUG
				lprintf(lvl, "Subkernel(dev=%d,id=%d):~Subkernel - CoCoFreed reduce_buf[%d*128 + %d]\n",
				run_dev_id, id, idx, run_dev_id);
#endif
			}
#else
#endif
#endif
}

void Subkernel::init_events(){
	operation_complete = new Event();
	if (WR_last) writeback_complete = new Event();
}

void Subkernel::update_RunTileMaps(){
	short lvl = 4;
#ifdef DEBUG
	lprintf(lvl-1, "|-----> Subkernel(dev=%d,id=%d)::update_RunTileMap()\n", run_dev_id, id);
#endif
	short run_dev_id_idx = (run_dev_id == -1)?  LOC_NUM - 1 : run_dev_id;
	for (int j = 0; j < TileNum; j++){
		if (TileDimlist[j] == 1){
				Tile1D<VALUE_TYPE>* tmp = (Tile1D<VALUE_TYPE>*) TileList[j];
				tmp->RunTileMap[run_dev_id_idx] = 1;
		}
		else if (TileDimlist[j] == 2){
			Tile2D<VALUE_TYPE>* tmp = (Tile2D<VALUE_TYPE>*) TileList[j];
			if(!tmp->RunTileMap[run_dev_id_idx]) {
				tmp->RunTileMap[run_dev_id_idx] = 1;
			}
		}
	}
}

void Subkernel::sync_request_data(){
	short lvl = 4;
#ifdef DEBUG
	lprintf(lvl-1, "|-----> Subkernel(dev=%d,id=%d)::sync_request_data()\n", run_dev_id, id);
#endif
	short run_dev_id_idx = (run_dev_id == -1)?  LOC_NUM - 1 : run_dev_id;
	for (int j = 0; j < TileNum; j++){
		if (TileDimlist[j] == 1){
				Tile1D<VALUE_TYPE>* tmp = (Tile1D<VALUE_TYPE>*) TileList[j];
				if (tmp->R_flag && tmp->CacheLocId[run_dev_id_idx] != -1 && (!tmp->W_flag || WR_first == 42 )) tmp->available[run_dev_id_idx]->sync_barrier();
		}
		else if (TileDimlist[j] == 2){
				Tile2D<VALUE_TYPE>* tmp = (Tile2D<VALUE_TYPE>*) TileList[j];
				if (tmp->R_flag && tmp->CacheLocId[run_dev_id_idx] != -1 && (!tmp->W_flag || WR_first == 42 )) tmp->available[run_dev_id_idx]->sync_barrier();
		}
	}
}

void Subkernel::request_tile(short TileIdx){
	short lvl = 5;
	short run_dev_id_idx = (run_dev_id == -1)?  LOC_NUM - 1 : run_dev_id;
	if (TileDimlist[TileIdx] == 1){
			Tile1D<VALUE_TYPE>* tmp = (Tile1D<VALUE_TYPE>*) TileList[TileIdx];
#ifdef DEBUG
		lprintf(lvl-1, "|-----> Subkernel(dev=%d,id=%d)::request_tile(Tile(%d.[%d]))\n",
			run_dev_id, id, tmp->id, tmp->GridId);
#endif
		short FetchFromId = tmp->getClosestReadLoc(run_dev_id),
			FetchFromId_idx = (FetchFromId == -1)?  LOC_NUM - 1 : FetchFromId;
		if (FetchFromId != -1){
#ifdef CDEBUG
			lprintf(lvl, "Subkernel(dev=%d,id=%d)-Tile(%d.[%d]): Fetching Block(%d) on GPU(%d) from Block(%d) on GPU(%d)\n",
				run_dev_id, id, tmp->id,  tmp->GridId, tmp->CacheLocId[run_dev_id_idx], run_dev_id,
				tmp->CacheLocId[FetchFromId], FetchFromId);
#endif
			FetchFromId_idx = FetchFromId;
			if (tmp->CacheLocId[FetchFromId] < -1)
				error("Subkernel(dev=%d,id=%d)-Tile(%d.[%d])::request_data: Fetching from tile in GPU(%d) with loc = %d\n",
					run_dev_id, id, tmp->id,  tmp->GridId, FetchFromId, tmp->CacheLocId[FetchFromId]);
			else if (tmp->CacheLocId[FetchFromId] != -1)
				CoCacheAddPendingEvent(FetchFromId, NULL, tmp->available[run_dev_id_idx], tmp->CacheLocId[FetchFromId], R);
		}
		CoCoMemcpyAsync(tmp->adrs[run_dev_id_idx], tmp->adrs[FetchFromId_idx],
											((long long) tmp->inc[run_dev_id_idx]) * tmp->dim * tmp->dtypesize(),
											run_dev_id, FetchFromId, h2d_queue[run_dev_id_idx]);
		tmp->available[run_dev_id_idx]->record_to_queue(h2d_queue[run_dev_id_idx]);
		CoCacheAddPendingEvent(run_dev_id, tmp->available[run_dev_id_idx], NULL, tmp->CacheLocId[run_dev_id_idx], AVAILABLE);
	}
	else if (TileDimlist[TileIdx] == 2){
			Tile2D<VALUE_TYPE>* tmp = (Tile2D<VALUE_TYPE>*) TileList[TileIdx];
#ifdef DEBUG
		lprintf(lvl-1, "|-----> Subkernel(dev=%d,id=%d)::request_tile(Tile(%d.[%d,%d]))\n",
			run_dev_id, id, tmp->id, tmp->GridId1, tmp->GridId2);
#endif
	short FetchFromId = tmp->getClosestReadLoc(run_dev_id),
		FetchFromId_idx = (FetchFromId == -1)?  LOC_NUM - 1 : FetchFromId;
	if (FetchFromId != -1){
#ifdef CDEBUG
			lprintf(lvl, "Subkernel(dev=%d,id=%d)-Tile(%d.[%d,%d]): Fetching Block(%d) on GPU(%d) from Block(%d) on GPU(%d)\n",
				run_dev_id, id, tmp->id,  tmp->GridId1, tmp->GridId2, tmp->CacheLocId[run_dev_id_idx], run_dev_id,
				tmp->CacheLocId[FetchFromId], FetchFromId);
#endif
			FetchFromId_idx = FetchFromId;
			if (tmp->CacheLocId[FetchFromId] < -1)
				error("Subkernel(dev=%d,id=%d)-Tile(%d.[%d,%d])::request_data: Fetching from tile in GPU(%d) with loc = %d\n",
					run_dev_id, id, tmp->id,  tmp->GridId1, tmp->GridId2, FetchFromId, tmp->CacheLocId[FetchFromId]);
			else if (tmp->CacheLocId[FetchFromId] != -1)
				CoCacheAddPendingEvent(FetchFromId, NULL, tmp->available[run_dev_id_idx], tmp->CacheLocId[FetchFromId], R);
		}
		CoCoMemcpy2DAsync(tmp->adrs[run_dev_id_idx], tmp->ldim[run_dev_id_idx],
											tmp->adrs[FetchFromId_idx], tmp->ldim[FetchFromId_idx],
											tmp->dim1, tmp->dim2, tmp->dtypesize(),
											run_dev_id, FetchFromId, h2d_queue[run_dev_id_idx]);
		tmp->available[run_dev_id_idx]->record_to_queue(h2d_queue[run_dev_id_idx]);
		CoCacheAddPendingEvent(run_dev_id, tmp->available[run_dev_id_idx], NULL, tmp->CacheLocId[run_dev_id_idx], AVAILABLE);
	}
}

int WR_check_lock = 0;

void Subkernel::request_data(){
	short lvl = 4;
#ifdef DEBUG
	lprintf(lvl-1, "|-----> Subkernel(dev=%d,id=%d)::request_data()\n", run_dev_id, id);
#endif
	short run_dev_id_idx = (run_dev_id == -1)?  LOC_NUM - 1 : run_dev_id;
	if (prev!= NULL) prev->sync_request_data();
	for (int j = 0; j < TileNum; j++){
		if (TileDimlist[j] == 1){
			Tile1D<VALUE_TYPE>* tmp = (Tile1D<VALUE_TYPE>*) TileList[j];
			// Check for tile RW_master - first to enter OR WR Tile initial owner
			if (tmp->R_flag && tmp->W_flag && WR_first){
				if(WR_reduce){
					while(__sync_lock_test_and_set(&WR_check_lock, 1)); // highly unlikely to happen?
					if(tmp->RW_master == -42){ // First entry, decide on WR_reducer.
//#ifdef ENABLE_CPU_WORKLOAD
						for(int idx = 0; idx < LOC_NUM; idx++)
							if(tmp->CacheLocId[idx] == -1 && tmp->RunTileMap[idx] == 1) tmp->RW_master = (idx == LOC_NUM - 1)?  -1 : idx;
						if(tmp->RW_master == -42) tmp->RW_master = run_dev_id;
//#else
//						//Hack that will not work if CPU (-1) also executes split-K subkernels
//						if(tmp->CacheLocId[LOC_NUM-1] == -1) tmp->RW_master = run_dev_id;
//						else for(int idx = 0; idx < LOC_NUM -1; idx++)
//							if(tmp->CacheLocId[idx] == -1) tmp->RW_master = idx;
//#endif
						if(tmp->RW_master == -42)
							error("Subkernel(dev=%d,id=%d)-Tile(%d.[%d]): request_data failed to assign RW master\n",
								run_dev_id, id, tmp->id, tmp->GridId);
#ifdef DDEBUG
						lprintf(lvl, "Subkernel(dev=%d,id=%d)-Tile(%d.[%d]): Set RW_master = %d\n",
					run_dev_id, id, tmp->id, tmp->GridId, tmp->RW_master);
#endif
					 }
					__sync_lock_release(&WR_check_lock);
					if (tmp->RW_master == run_dev_id) WR_reduce = 0;
					else{ ;
#ifdef DDEBUG
						lprintf(lvl, "Subkernel(dev=%d,id=%d)-Tile(%d.[%d]): is a reducer.\n",
							run_dev_id, id, tmp->id, tmp->GridId);
#endif
					}
				}
				else tmp->RW_master = run_dev_id;
			 }
			if (tmp->CacheLocId[run_dev_id_idx] == -1) continue; // Skip if block native to run_dev_id
			else if(tmp->CacheLocId[run_dev_id_idx] < -1){
				tmp->adrs[run_dev_id_idx] = CoCacheAsignBlock(run_dev_id, TileList[j], TileDimlist[j]);
#ifdef CDEBUG
		lprintf(lvl, "Subkernel(dev=%d,id=%d)-Tile(%d.[%d]): Asigned buffer Block in GPU(%d)= %d\n",
					run_dev_id, id, tmp->id, tmp->GridId, run_dev_id, tmp->CacheLocId[run_dev_id_idx]);
#endif
				if (tmp->R_flag && (!tmp->W_flag || tmp->RW_master == run_dev_id )) request_tile(j);
			}
			if (tmp->W_flag) CoCacheAddPendingEvent(run_dev_id, tmp->available[run_dev_id_idx],
				writeback_complete, tmp->CacheLocId[run_dev_id_idx], W);
			else CoCacheAddPendingEvent(run_dev_id, tmp->available[run_dev_id_idx],
				operation_complete, tmp->CacheLocId[run_dev_id_idx], R);
		}
		else if (TileDimlist[j] == 2){
				Tile2D<VALUE_TYPE>* tmp = (Tile2D<VALUE_TYPE>*) TileList[j];
				// Check for tile RW_master - first to enter OR WR Tile initial owner
				if (tmp->R_flag && tmp->W_flag && WR_first){
					if(WR_reduce){
						while(__sync_lock_test_and_set(&WR_check_lock, 1)); // highly unlikely to happen?
						if(tmp->RW_master == -42){ // First entry, decide on WR_reducer.
//#ifdef ENABLE_CPU_WORKLOAD
							for(int idx = 0; idx < LOC_NUM; idx++)
								if(tmp->CacheLocId[idx] == -1 && tmp->RunTileMap[idx] == 1)
									tmp->RW_master = (idx == LOC_NUM - 1)?  -1 : idx;
							if(tmp->RW_master == -42) tmp->RW_master = run_dev_id;
//#else
////Hack that will not work if CPU (-1) also executes split-K subkernels
//if(tmp->CacheLocId[LOC_NUM-1] == -1) tmp->RW_master = run_dev_id;
//							else for(int idx = 0; idx < LOC_NUM -1; idx++)
//								if(tmp->CacheLocId[idx] == -1) tmp->RW_master = idx;
//#endif
							if(tmp->RW_master == -42)
								error("Subkernel(dev=%d,id=%d)-Tile(%d.[%d,%d]): request_data failed to assign RW master\n",
									run_dev_id, id, tmp->id, tmp->GridId1, tmp->GridId2);
#ifdef DDEBUG
							lprintf(lvl, "Subkernel(dev=%d,id=%d)-Tile(%d.[%d,%d]): Set RW_master = %d\n",
						run_dev_id, id, tmp->id, tmp->GridId1, tmp->GridId2, tmp->RW_master);
#endif
						 }
						__sync_lock_release(&WR_check_lock);
						if (tmp->RW_master == run_dev_id) WR_reduce = 0;
						else{ ;
#ifdef DDEBUG
							lprintf(lvl, "Subkernel(dev=%d,id=%d)-Tile(%d.[%d,%d]): is a reducer.\n",
								run_dev_id, id, tmp->id, tmp->GridId1, tmp->GridId2);
#endif
						}
					}
					else tmp->RW_master = run_dev_id;
				 }
				if (tmp->CacheLocId[run_dev_id_idx] == -1) continue; // Skip if block native to run_dev_id
				else if(tmp->CacheLocId[run_dev_id_idx] < -1){
					tmp->adrs[run_dev_id_idx] = CoCacheAsignBlock(run_dev_id, TileList[j], TileDimlist[j]);
#ifdef CDEBUG
			lprintf(lvl, "Subkernel(dev=%d,id=%d)-Tile(%d.[%d,%d]): Asigned buffer Block in GPU(%d)= %d\n",
						run_dev_id, id, tmp->id, tmp->GridId1, tmp->GridId2, run_dev_id, tmp->CacheLocId[run_dev_id_idx]);
#endif
					if (tmp->R_flag && (!tmp->W_flag || tmp->RW_master == run_dev_id )) request_tile(j);
				}
				if (tmp->W_flag) CoCacheAddPendingEvent(run_dev_id, tmp->available[run_dev_id_idx],
					writeback_complete, tmp->CacheLocId[run_dev_id_idx], W);
				else CoCacheAddPendingEvent(run_dev_id, tmp->available[run_dev_id_idx],
					operation_complete, tmp->CacheLocId[run_dev_id_idx], R);
			}
		else error("Subkernel(dev=%d,id=%d)::request_data: Not implemented for TileDim=%d\n",
			run_dev_id, id, TileDimlist[j]);
	}
#ifndef ASYNC_ENABLE
	CoCoSyncCheckErr();
#endif
#ifdef DEBUG
	lprintf(lvl-1, "<-----|\n");
#endif
}

void Subkernel::run_operation(){
	short lvl = 4;
#ifdef DEBUG
	lprintf(lvl-1, "|-----> Subkernel(dev=%d,id=%d)::run_operation()\n", run_dev_id, id);
#endif
	short run_dev_id_idx = (run_dev_id == -1)?  LOC_NUM - 1 : run_dev_id;
	for (int j = 0; j < TileNum; j++){
		if (TileDimlist[j] == 1){
				Tile1D<VALUE_TYPE>* tmp = (Tile1D<VALUE_TYPE>*) TileList[j];
				if (tmp->CacheLocId[run_dev_id_idx] == -1) continue;
				else if (tmp->CacheLocId[run_dev_id_idx] < -1)
					error("Subkernel(dev=%d,id=%d)-Tile(%d.[%d])::run_operation: Tile(j=%d) has loc = %d\n",
						run_dev_id, id, tmp->id, tmp->GridId, j, tmp->CacheLocId[run_dev_id_idx]);
				else if(tmp->R_flag) exec_queue[run_dev_id_idx]->wait_for_event(tmp->available[run_dev_id_idx]);
		}
		else if (TileDimlist[j] == 2){
				Tile2D<VALUE_TYPE>* tmp = (Tile2D<VALUE_TYPE>*) TileList[j];
				if (tmp->CacheLocId[run_dev_id_idx] == -1) continue;
				else if (tmp->CacheLocId[run_dev_id_idx] < -1)
					error("Subkernel(dev=%d,id=%d)-Tile(%d.[%d,%d])::run_operation: Tile(j=%d) has loc = %d\n",
						run_dev_id, id, tmp->id, tmp->GridId1, tmp->GridId2, j, tmp->CacheLocId[run_dev_id_idx]);
				else if(tmp->R_flag) exec_queue[run_dev_id_idx]->wait_for_event(tmp->available[run_dev_id_idx]);
		}
	}
	backend_run_operation(operation_params, op_name, exec_queue[run_dev_id_idx]);
	operation_complete->record_to_queue(exec_queue[run_dev_id_idx]);
	//if (prev!= NULL) prev->operation_complete->sync_barrier();
#ifndef ASYNC_ENABLE
	CoCoSyncCheckErr();
#endif
#ifdef DEBUG
	lprintf(lvl-1, "<-----|\n");
#endif
}

void Subkernel::writeback_data(){
	short lvl = 4;
#ifdef DEBUG
	lprintf(lvl-1, "|-----> Subkernel(dev=%d,id=%d)::writeback_data()\n", run_dev_id, id);
#endif
	short run_dev_id_idx = (run_dev_id == -1)?  LOC_NUM - 1 : run_dev_id;

	d2h_queue[run_dev_id_idx]->wait_for_event(operation_complete);
	for (int j = 0; j < TileNum; j++){
		if (TileDimlist[j] == 1){
				Tile1D<VALUE_TYPE>* tmp = (Tile1D<VALUE_TYPE>*) TileList[j];
				if (tmp->W_flag && tmp->RW_master == run_dev_id){
					if (tmp->CacheLocId[run_dev_id_idx] == -1);
					else if (tmp->CacheLocId[run_dev_id_idx] < -1)
						error("Subkernel(dev=%d,id=%d)-Tile(%d.[%d])::writeback_data: invoked with tile with loc = %d\n",
							run_dev_id, id, tmp->id, tmp->GridId, tmp->CacheLocId[run_dev_id_idx]);
					else{
						short WritebackIdCAdr, WritebackId = tmp->getWriteBackLoc(); //to MASTER
						if (WritebackId == -1) WritebackIdCAdr = LOC_NUM - 1;
						else WritebackIdCAdr = WritebackId;
						if(!WR_last) error("Subkernel(dev=%d,id=%d)-Tile(%d.[%d])::writeback_data:\
						Subkernel should be a WR_writer?\n", run_dev_id, id, tmp->id, tmp->GridId);
							CoCoMemcpyAsync(tmp->adrs[WritebackIdCAdr], tmp->adrs[run_dev_id_idx],
								((long long) tmp->inc[run_dev_id_idx]) * tmp->dim * tmp->dtypesize(),
								WritebackId, run_dev_id, d2h_queue[run_dev_id_idx]);
							CoCacheAddPendingEvent(run_dev_id, operation_complete, writeback_complete, tmp->CacheLocId[run_dev_id_idx], W);
					}
#ifdef MULTIDEVICE_REDUCTION_ENABLE
#ifdef DDEBUG
					lprintf(lvl, "Subkernel(dev=%d,id=%d)-Tile(%d.[%d]): Adding RW_lock(%p) in queue\n",
						run_dev_id, id, tmp->id, tmp->GridId, &tmp->RW_lock);
#endif
					d2h_queue[run_dev_id_idx]->add_host_func((void*)&CoCoQueueUnlock, (void*) &tmp->RW_lock);
				}
				else if (tmp->W_flag && tmp->RW_master != run_dev_id){
					if (tmp->CacheLocId[run_dev_id_idx] == -1) continue;
					else if (tmp->CacheLocId[run_dev_id_idx] < -1)
						error("Subkernel(dev=%d,id=%d)-Tile(%d.[%d])::writeback_reduce_data: invoked with tile with loc = %d\n",
							run_dev_id, id, tmp->id, tmp->GridId, tmp->CacheLocId[run_dev_id_idx]);
					else{
						short WritebackIdCAdr, WritebackId = tmp->getWriteBackLoc(); //to MASTER
						if (WritebackId == -1) WritebackIdCAdr = LOC_NUM - 1;
						else WritebackIdCAdr = WritebackId;
						d2h_queue[run_dev_id_idx]->wait_for_event(operation_complete);
						while(__sync_lock_test_and_set(&WR_check_lock, 1));
						long long tmp_buffsz = CoCoGetBlockSize(run_dev_id);
						if (reduce_buf[reduce_buf_it[WritebackIdCAdr]*128 + WritebackIdCAdr] == NULL){
							reduce_buf_sz[reduce_buf_it[WritebackIdCAdr]*128 + WritebackIdCAdr] = tmp_buffsz;
							reduce_buf[reduce_buf_it[WritebackIdCAdr]*128 + WritebackIdCAdr]
								= CoCoMalloc(reduce_buf_sz[reduce_buf_it[WritebackIdCAdr]*128 + WritebackIdCAdr], WritebackId);
#ifdef DDEBUG
							lprintf(lvl, "Subkernel(dev=%d,id=%d): Allocated buffer(%p) in %d\n",
							run_dev_id, id, reduce_buf[reduce_buf_it[WritebackIdCAdr]*128 + WritebackIdCAdr],
							reduce_buf_it[WritebackIdCAdr]*128 + WritebackIdCAdr);
#endif
						}
						else if(reduce_buf_sz[reduce_buf_it[WritebackIdCAdr]*128 + WritebackIdCAdr] != tmp_buffsz){
							CoCoFree(reduce_buf[reduce_buf_it[WritebackIdCAdr]*128 + WritebackIdCAdr], CoCoGetPtrLoc(
								reduce_buf[reduce_buf_it[WritebackIdCAdr]*128 + WritebackIdCAdr]));
							reduce_buf_sz[reduce_buf_it[WritebackIdCAdr]*128 + WritebackIdCAdr] = tmp_buffsz;
							reduce_buf[reduce_buf_it[WritebackIdCAdr]*128 + WritebackIdCAdr]
								= CoCoMalloc(reduce_buf_sz[reduce_buf_it[WritebackIdCAdr]*128 + WritebackIdCAdr], WritebackId);
						}
						int local_reduce_buf_it = reduce_buf_it[WritebackIdCAdr];
						if(reduce_buf_it[WritebackIdCAdr] < MAX_BUFFERING_L - 1) reduce_buf_it[WritebackIdCAdr]++;
						else reduce_buf_it[WritebackIdCAdr] = 0;
						__sync_lock_release(&WR_check_lock);
						if(!WR_last) error("Subkernel(dev=%d,id=%d)-Tile(%d.[%d,])::writeback_reduce_data:\
						Subkernel should be a WR_reduce?\n", run_dev_id, id, tmp->id, tmp->GridId);
						else{
							if (exec_queue[WritebackIdCAdr] == NULL) exec_queue[WritebackIdCAdr] = new CommandQueue(WritebackId);
							CoCoMemcpyReduceAsync(reduce_buf[local_reduce_buf_it*128 + WritebackIdCAdr], local_reduce_buf_it,
								tmp->adrs[WritebackIdCAdr], tmp->adrs[run_dev_id_idx],
								((long long) tmp->inc[run_dev_id_idx]) * tmp->dim * tmp->dtypesize(),
								WritebackId, run_dev_id, (void*)&tmp->RW_lock, d2h_queue[run_dev_id_idx], exec_queue[WritebackIdCAdr]);
							}
						//CoCacheAddPendingEvent(run_dev_id, operation_complete, tmp->available[WritebackIdCAdr], tmp->CacheLocId[run_dev_id_idx], W);
					}
#endif
				}
		}
		else if (TileDimlist[j] == 2){
				Tile2D<VALUE_TYPE>* tmp = (Tile2D<VALUE_TYPE>*) TileList[j];
				if (tmp->W_flag && tmp->RW_master == run_dev_id){
					if (tmp->CacheLocId[run_dev_id_idx] == -1);
					else if (tmp->CacheLocId[run_dev_id_idx] < -1)
						error("Subkernel(dev=%d,id=%d)-Tile(%d.[%d,%d])::writeback_data: invoked with tile with loc = %d\n",
							run_dev_id, id, tmp->id, tmp->GridId1, tmp->GridId2, tmp->CacheLocId[run_dev_id_idx]);
					else{
						short WritebackIdCAdr, WritebackId = tmp->getWriteBackLoc(); //to MASTER
						if (WritebackId == -1) WritebackIdCAdr = LOC_NUM - 1;
						else WritebackIdCAdr = WritebackId;
						if(!WR_last) error("Subkernel(dev=%d,id=%d)-Tile(%d.[%d,%d])::writeback_data:\
						Subkernel should be a WR_writer?\n", run_dev_id, id, tmp->id, tmp->GridId1, tmp->GridId2);
							CoCoMemcpy2DAsync(tmp->adrs[WritebackIdCAdr], tmp->ldim[WritebackIdCAdr],
								tmp->adrs[run_dev_id_idx], tmp->ldim[run_dev_id_idx],
								tmp->dim1, tmp->dim2, tmp->dtypesize(),
								WritebackId, run_dev_id, d2h_queue[run_dev_id_idx]);
							CoCacheAddPendingEvent(run_dev_id, operation_complete, writeback_complete, tmp->CacheLocId[run_dev_id_idx], W);
					}
#ifdef MULTIDEVICE_REDUCTION_ENABLE
#ifdef DDEBUG
					lprintf(lvl, "Subkernel(dev=%d,id=%d)-Tile(%d.[%d,%d]): Adding RW_lock(%p) in queue\n",
						run_dev_id, id, tmp->id, tmp->GridId1, tmp->GridId2, &tmp->RW_lock);
#endif
					d2h_queue[run_dev_id_idx]->add_host_func((void*)&CoCoQueueUnlock, (void*) &tmp->RW_lock);
				}
				else if (tmp->W_flag && tmp->RW_master != run_dev_id){
					if (tmp->CacheLocId[run_dev_id_idx] == -1) continue;
					else if (tmp->CacheLocId[run_dev_id_idx] < -1)
						error("Subkernel(dev=%d,id=%d)-Tile(%d.[%d,%d])::writeback_reduce_data: invoked with tile with loc = %d\n",
							run_dev_id, id, tmp->id, tmp->GridId1, tmp->GridId2, tmp->CacheLocId[run_dev_id_idx]);
					else{
						short WritebackIdCAdr, WritebackId = tmp->getWriteBackLoc(); //to MASTER
						if (WritebackId == -1) WritebackIdCAdr = LOC_NUM - 1;
						else WritebackIdCAdr = WritebackId;
						d2h_queue[run_dev_id_idx]->wait_for_event(operation_complete);
						while(__sync_lock_test_and_set(&WR_check_lock, 1));
						long long tmp_buffsz = CoCoGetBlockSize(run_dev_id);
						int buf_idx = reduce_buf_it[WritebackIdCAdr]*128 + WritebackIdCAdr;
						if (reduce_buf[buf_idx] == NULL){
							reduce_buf_sz[buf_idx] = tmp_buffsz;
							reduce_buf[buf_idx]	= CoCoMalloc(reduce_buf_sz[buf_idx], WritebackId);
#ifdef DDEBUG
							lprintf(lvl, "Subkernel(dev=%d,id=%d): Allocated buffer(%p, loc = %d, buf_sz = %lld) in reduce_buf[%d*128 + %d]\n",
							run_dev_id, id, reduce_buf[buf_idx], WritebackId, reduce_buf_sz[buf_idx],
							reduce_buf_it[WritebackIdCAdr], WritebackIdCAdr);
#endif
						}
						else if(reduce_buf_sz[buf_idx] != tmp_buffsz){
#ifdef DDEBUG
							lprintf(lvl, "Subkernel(dev=%d,id=%d): Free buffer(%p, loc = %d, buf_sz = %lld) in reduce_buf[%d*128 + %d]\
							\n Replace with new buffer(buf_sz = %lld)\n",
							run_dev_id, id, reduce_buf[buf_idx], WritebackId, reduce_buf_sz[buf_idx],
							reduce_buf_it[WritebackIdCAdr], WritebackIdCAdr, tmp_buffsz);
#endif
							CoCoFree(reduce_buf[buf_idx], CoCoGetPtrLoc(
								reduce_buf[buf_idx]));
							reduce_buf_sz[buf_idx] = tmp_buffsz;
							reduce_buf[buf_idx]
								= CoCoMalloc(reduce_buf_sz[buf_idx], WritebackId);
						}
						int local_reduce_buf_it = reduce_buf_it[WritebackIdCAdr];
						if(reduce_buf_it[WritebackIdCAdr] < MAX_BUFFERING_L - 1) reduce_buf_it[WritebackIdCAdr]++;
						else reduce_buf_it[WritebackIdCAdr] = 0;
						__sync_lock_release(&WR_check_lock);
						if(!WR_last) error("Subkernel(dev=%d,id=%d)-Tile(%d.[%d,%d])::writeback_reduce_data:\
						Subkernel should be a WR_reduce?\n", run_dev_id, id, tmp->id, tmp->GridId1, tmp->GridId2);
						else{
							if (exec_queue[WritebackIdCAdr] == NULL) exec_queue[WritebackIdCAdr] = new CommandQueue(WritebackId);
							CoCoMemcpyReduce2DAsync(reduce_buf[local_reduce_buf_it*128 + WritebackIdCAdr], local_reduce_buf_it,
								tmp->adrs[WritebackIdCAdr], tmp->ldim[WritebackIdCAdr],
								tmp->adrs[run_dev_id_idx], tmp->ldim[run_dev_id_idx],
								tmp->dim1, tmp->dim2, tmp->dtypesize(),
								WritebackId, run_dev_id, (void*)&tmp->RW_lock, d2h_queue[run_dev_id_idx], exec_queue[WritebackIdCAdr]);
							}
						//CoCacheAddPendingEvent(run_dev_id, operation_complete, tmp->available[WritebackIdCAdr], tmp->CacheLocId[run_dev_id_idx], W);
					}
#endif
				}
		}
		else error("Subkernel(dev=%d,id=%d)::writeback_data: Not implemented for TileDim=%d\n", run_dev_id, id, TileDimlist[j]);
	}
	writeback_complete->record_to_queue(d2h_queue[run_dev_id_idx]);
#ifndef ASYNC_ENABLE
	CoCoSyncCheckErr();
#endif
#ifdef DEBUG
	lprintf(lvl-1, "<-----|\n");
#endif
}

void Subkernel::writeback_reduce_data(){
	short lvl = 4;
#ifdef DEBUG
	lprintf(lvl-1, "|-----> Subkernel(dev=%d,id=%d)::writeback_reduce_data()\n", run_dev_id, id);
#endif
	for (int j = 0; j < TileNum; j++){
		if (TileDimlist[j] == 1)
		error("Subkernel(dev=%d,id=%d)::writeback_reduce_data: Tile1D not implemented\n", run_dev_id, id);
		else if (TileDimlist[j] == 2){
				Tile2D<VALUE_TYPE>* tmp = (Tile2D<VALUE_TYPE>*) TileList[j];
		}
		else error("Subkernel(dev=%d,id=%d)::writeback_data: Not implemented for TileDim=%d\n", run_dev_id, id, TileDimlist[j]);
	}
#ifndef ASYNC_ENABLE
	CoCoSyncCheckErr();
#endif
#ifdef DEBUG
	lprintf(lvl-1, "<-----|\n");
#endif
}

void 	CoCoPeLiaInitResources(short dev_id){
	short dev_id_idx = (dev_id == -1)?  LOC_NUM - 1 : dev_id;
  if (!h2d_queue[dev_id_idx]) h2d_queue[dev_id_idx] = new CommandQueue(dev_id);
  if (!d2h_queue[dev_id_idx])  d2h_queue[dev_id_idx] = new CommandQueue(dev_id);
  if (!exec_queue[dev_id_idx])  exec_queue[dev_id_idx] = new CommandQueue(dev_id);
}

void 	CoCoPeLiaFreeResources(short dev_id){
	short dev_id_idx = (dev_id == -1)?  LOC_NUM - 1 : dev_id;
  if (h2d_queue[dev_id_idx]){
		delete h2d_queue[dev_id_idx];
		h2d_queue[dev_id_idx] = NULL;
	}
  if (d2h_queue[dev_id_idx]){
		delete d2h_queue[dev_id_idx];
		d2h_queue[dev_id_idx] = NULL;
	}
  if (exec_queue[dev_id_idx]){
		delete exec_queue[dev_id_idx];
		exec_queue[dev_id_idx] = NULL;
	}
}
