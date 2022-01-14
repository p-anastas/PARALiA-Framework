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
	Subkernel_num--;
	free(TileDimlist);
	free(TileList);
	free(operation_params);
	delete operation_complete;
	if (WR_last) delete writeback_complete;
#ifdef MULTIDEVICE_REDUCTION_ENABLE
#ifndef BUFFER_REUSE_ENABLE
		reduce_buf_it[run_dev_id] = 0;
		for(int idx = 0; idx < MAX_BUFFERING_L; idx++)
			if(reduce_buf[idx*128 + run_dev_id]!= NULL){
				CoCoFree(reduce_buf[idx*128 + run_dev_id], CoCoGetPtrLoc(reduce_buf[idx*128 + run_dev_id]));
				reduce_buf[idx*128 + run_dev_id] = NULL;
			}
#else
#endif
#endif
}

void Subkernel::init_events(){
	operation_complete = new Event();
	if (WR_last) writeback_complete = new Event();
}

void Subkernel::sync_request_data(){
	short lvl = 4;
#ifdef DEBUG
	lprintf(lvl-1, "|-----> Subkernel(dev=%d,id=%d)::sync_request_data()\n", run_dev_id, id);
#endif
	for (int j = 0; j < TileNum; j++){
		if (TileDimlist[j] == 1)
			error("Subkernel(dev=%d,id=%d)::request_data: Tile1D not implemented\n", run_dev_id, id);
		else if (TileDimlist[j] == 2){
				Tile2D<VALUE_TYPE>* tmp = (Tile2D<VALUE_TYPE>*) TileList[j];
				if (tmp->R_flag && tmp->CacheLocId[run_dev_id] != -1 && (!tmp->W_flag || WR_first == 42 )) tmp->available[run_dev_id]->sync_barrier();
			}
		}
}

void Subkernel::request_tile(short TileIdx){
	short lvl = 5;
	if (TileDimlist[TileIdx] == 1)
		error("Subkernel(dev=%d,id=%d)::request_tile: Tile1D not implemented\n", run_dev_id, id);
	else if (TileDimlist[TileIdx] == 2){
			Tile2D<VALUE_TYPE>* tmp = (Tile2D<VALUE_TYPE>*) TileList[TileIdx];
#ifdef DEBUG
		lprintf(lvl-1, "|-----> Subkernel(dev=%d,id=%d)::request_tile(Tile(%d.[%d,%d]))\n",
			run_dev_id, id, tmp->id, tmp->GridId1, tmp->GridId2);
#endif
		short FetchFromIdCAdr, FetchFromId = tmp->getClosestReadLoc(run_dev_id);
		if (FetchFromId == -1) FetchFromIdCAdr = LOC_NUM - 1;
		else{
#ifdef CDEBUG
			lprintf(lvl, "Subkernel(dev=%d,id=%d)-Tile(%d.[%d,%d]): Fetching Block(%d) on GPU(%d) from Block(%d) on GPU(%d)\n",
				run_dev_id, id, tmp->id,  tmp->GridId1, tmp->GridId2, tmp->CacheLocId[run_dev_id], run_dev_id,
				tmp->CacheLocId[FetchFromId], FetchFromId);
#endif
			FetchFromIdCAdr = FetchFromId;
			if (tmp->CacheLocId[FetchFromId] < -1)
				error("Subkernel(dev=%d,id=%d)-Tile(%d.[%d,%d])::request_data: Fetching from tile in GPU(%d) with loc = %d\n",
					run_dev_id, id, tmp->id,  tmp->GridId1, tmp->GridId2, FetchFromId, tmp->CacheLocId[FetchFromId]);
			else if (tmp->CacheLocId[FetchFromId] != -1)
				CoCacheAddPendingEvent(FetchFromId, NULL, tmp->available[run_dev_id], tmp->CacheLocId[FetchFromId], R);
		}
		CoCoMemcpy2DAsync(tmp->adrs[run_dev_id], tmp->ldim[run_dev_id],
											tmp->adrs[FetchFromIdCAdr], tmp->ldim[FetchFromIdCAdr],
											tmp->dim1, tmp->dim2, tmp->dtypesize(),
											run_dev_id, FetchFromId, h2d_queue[run_dev_id]);
		tmp->available[run_dev_id]->record_to_queue(h2d_queue[run_dev_id]);
		CoCacheAddPendingEvent(run_dev_id, tmp->available[run_dev_id], NULL, tmp->CacheLocId[run_dev_id], AVAILABLE);
	}
}

int WR_check_lock = 0;

void Subkernel::request_data(){
	short lvl = 4;
#ifdef DEBUG
	lprintf(lvl-1, "|-----> Subkernel(dev=%d,id=%d)::request_data()\n", run_dev_id, id);
#endif
	if (prev!= NULL) prev->sync_request_data();
	for (int j = 0; j < TileNum; j++){
		if (TileDimlist[j] == 1)
			error("Subkernel(dev=%d,id=%d)::request_data: Tile1D not implemented\n", run_dev_id, id);
		else if (TileDimlist[j] == 2){
				Tile2D<VALUE_TYPE>* tmp = (Tile2D<VALUE_TYPE>*) TileList[j];
				// Check for tile RW_master - first to enter OR WR Tile initial owner
				if (tmp->R_flag && tmp->W_flag && WR_first){
					if(WR_reduce){
						while(__sync_lock_test_and_set(&WR_check_lock, 1)); // highly unlikely to happen?
						if(tmp->RW_master == -42){ // First entry, decide on WR_reducer.
							//Hack that will not work if CPU (-1) also executes split-K subkernels
							if(tmp->CacheLocId[LOC_NUM-1] == -1) tmp->RW_master = run_dev_id;
							else for(int idx = 0; idx < LOC_NUM -1; idx++)
								if(tmp->CacheLocId[idx] == -1) tmp->RW_master = idx;
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
				if (tmp->CacheLocId[run_dev_id] == -1) continue; // Skip if block native to run_dev_id
				else if(tmp->CacheLocId[run_dev_id] < -1){
					tmp->adrs[run_dev_id] = CoCacheAsignBlock(run_dev_id, TileList[j], TileDimlist[j]);
#ifdef CDEBUG
			lprintf(lvl, "Subkernel(dev=%d,id=%d)-Tile(%d.[%d,%d]): Asigned buffer Block in GPU(%d)= %d\n",
						run_dev_id, id, tmp->id, tmp->GridId1, tmp->GridId2, run_dev_id, tmp->CacheLocId[run_dev_id]);
#endif
					if (tmp->R_flag && (!tmp->W_flag || tmp->RW_master == run_dev_id )) request_tile(j);
				}
				if (tmp->W_flag) CoCacheAddPendingEvent(run_dev_id, tmp->available[run_dev_id],
					writeback_complete, tmp->CacheLocId[run_dev_id], W);
				else CoCacheAddPendingEvent(run_dev_id, tmp->available[run_dev_id],
					operation_complete, tmp->CacheLocId[run_dev_id], R);
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
	for (int j = 0; j < TileNum; j++){
		if (TileDimlist[j] == 1)
			error("Subkernel(dev=%d,id=%d)::run_operation: Tile1D not implemented\n", run_dev_id, id);
		else if (TileDimlist[j] == 2){
				Tile2D<VALUE_TYPE>* tmp = (Tile2D<VALUE_TYPE>*) TileList[j];
				if (tmp->CacheLocId[run_dev_id] == -1) continue;
				else if (tmp->CacheLocId[run_dev_id] < -1)
					error("Subkernel(dev=%d,id=%d)-Tile(%d.[%d,%d])::run_operation: Tile(j=%d) has loc = %d\n",
						run_dev_id, id, tmp->id, tmp->GridId1, tmp->GridId2, j, tmp->CacheLocId[run_dev_id]);
				else if(tmp->R_flag) exec_queue[run_dev_id]->wait_for_event(tmp->available[run_dev_id]);
		}
	}
	backend_run_operation(operation_params, op_name);
	operation_complete->record_to_queue(exec_queue[run_dev_id]);
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

	d2h_queue[run_dev_id]->wait_for_event(operation_complete);
	for (int j = 0; j < TileNum; j++){
		if (TileDimlist[j] == 1)
		error("Subkernel(dev=%d,id=%d)::writeback_data: Tile1D not implemented\n", run_dev_id, id);
		else if (TileDimlist[j] == 2){
				Tile2D<VALUE_TYPE>* tmp = (Tile2D<VALUE_TYPE>*) TileList[j];
				if (tmp->W_flag && tmp->RW_master == run_dev_id){
					if (tmp->CacheLocId[run_dev_id] == -1);
					else if (tmp->CacheLocId[run_dev_id] < -1)
						error("Subkernel(dev=%d,id=%d)-Tile(%d.[%d,%d])::writeback_data: invoked with tile with loc = %d\n",
							run_dev_id, id, tmp->id, tmp->GridId1, tmp->GridId2, tmp->CacheLocId[run_dev_id]);
					else{
						short WritebackIdCAdr, WritebackId = tmp->getWriteBackLoc(); //to MASTER
						if (WritebackId == -1) WritebackIdCAdr = LOC_NUM - 1;
						else WritebackIdCAdr = WritebackId;
						if(!WR_last) error("Subkernel(dev=%d,id=%d)-Tile(%d.[%d,%d])::writeback_data:\
						Subkernel should be a WR_writer?\n", run_dev_id, id, tmp->id, tmp->GridId1, tmp->GridId2);
							CoCoMemcpy2DAsync(tmp->adrs[WritebackIdCAdr], tmp->ldim[WritebackIdCAdr],
								tmp->adrs[run_dev_id], tmp->ldim[run_dev_id],
								tmp->dim1, tmp->dim2, tmp->dtypesize(),
								WritebackId, run_dev_id, d2h_queue[run_dev_id]);
							CoCacheAddPendingEvent(run_dev_id, operation_complete, writeback_complete, tmp->CacheLocId[run_dev_id], W);
					}
#ifdef MULTIDEVICE_REDUCTION_ENABLE
#ifdef DDEBUG
					lprintf(lvl, "Subkernel(dev=%d,id=%d)-Tile(%d.[%d,%d]): Adding RW_lock(%p) in queue\n",
						run_dev_id, id, tmp->id, tmp->GridId1, tmp->GridId2, &tmp->RW_lock);
#endif
					d2h_queue[run_dev_id]->add_host_func((void*)&CoCoQueueUnlock, (void*) &tmp->RW_lock);
				}
				else if (tmp->W_flag && tmp->RW_master != run_dev_id){
					if (tmp->CacheLocId[run_dev_id] == -1) continue;
					else if (tmp->CacheLocId[run_dev_id] < -1)
						error("Subkernel(dev=%d,id=%d)-Tile(%d.[%d,%d])::writeback_reduce_data: invoked with tile with loc = %d\n",
							run_dev_id, id, tmp->id, tmp->GridId1, tmp->GridId2, tmp->CacheLocId[run_dev_id]);
					else{
						short WritebackIdCAdr, WritebackId = tmp->getWriteBackLoc(); //to MASTER
						if (WritebackId == -1) WritebackIdCAdr = LOC_NUM - 1;
						else WritebackIdCAdr = WritebackId;
						d2h_queue[run_dev_id]->wait_for_event(operation_complete);
						while(__sync_lock_test_and_set(&WR_check_lock, 1));
						long long tmp_buffsz = CoCoGetBlockSize(run_dev_id);
						if (reduce_buf[reduce_buf_it[run_dev_id]*128 + run_dev_id] == NULL){
							reduce_buf_sz[reduce_buf_it[run_dev_id]*128 + run_dev_id] = tmp_buffsz;
							reduce_buf[reduce_buf_it[run_dev_id]*128 + run_dev_id]
								= CoCoMalloc(reduce_buf_sz[reduce_buf_it[run_dev_id]*128 + run_dev_id], WritebackId);
#ifdef DDEBUG
							lprintf(lvl, "Subkernel(dev=%d,id=%d): Allocated buffer(%p) in %d\n",
							run_dev_id, id, reduce_buf[reduce_buf_it[run_dev_id]*128 + run_dev_id],
							reduce_buf_it[run_dev_id]*128 + run_dev_id);
#endif
						}
						else if(reduce_buf_sz[reduce_buf_it[run_dev_id]*128 + run_dev_id] != tmp_buffsz){
							CoCoFree(reduce_buf[reduce_buf_it[run_dev_id]*128 + run_dev_id], CoCoGetPtrLoc(
								reduce_buf[reduce_buf_it[run_dev_id]*128 + run_dev_id]));
							reduce_buf_sz[reduce_buf_it[run_dev_id]*128 + run_dev_id] = tmp_buffsz;
							reduce_buf[reduce_buf_it[run_dev_id]*128 + run_dev_id]
								= CoCoMalloc(reduce_buf_sz[reduce_buf_it[run_dev_id]*128 + run_dev_id], WritebackId);
						}
						int local_reduce_buf_it = reduce_buf_it[run_dev_id];
						if(reduce_buf_it[run_dev_id] < MAX_BUFFERING_L - 1) reduce_buf_it[run_dev_id]++;
						else reduce_buf_it[run_dev_id] = 0;
						__sync_lock_release(&WR_check_lock);
						if(!WR_last) error("Subkernel(dev=%d,id=%d)-Tile(%d.[%d,%d])::writeback_reduce_data:\
						Subkernel should be a WR_reduce?\n", run_dev_id, id, tmp->id, tmp->GridId1, tmp->GridId2);
						else{
							CoCoMemcpyReduce2D(reduce_buf[local_reduce_buf_it*128 + run_dev_id], local_reduce_buf_it, tmp->adrs[WritebackIdCAdr], tmp->ldim[WritebackIdCAdr],
								tmp->adrs[run_dev_id], tmp->ldim[run_dev_id],
								tmp->dim1, tmp->dim2, tmp->dtypesize(),
								WritebackId, run_dev_id, (void*)&tmp->RW_lock, d2h_queue[run_dev_id]);
							}
						//CoCacheAddPendingEvent(run_dev_id, operation_complete, tmp->available[WritebackIdCAdr], tmp->CacheLocId[run_dev_id], W);
					}
#endif
				}
		}
		else error("Subkernel(dev=%d,id=%d)::writeback_data: Not implemented for TileDim=%d\n", run_dev_id, id, TileDimlist[j]);
	}
	writeback_complete->record_to_queue(d2h_queue[run_dev_id]);
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
  if (!h2d_queue[dev_id]) h2d_queue[dev_id] = new CommandQueue();
  if (!d2h_queue[dev_id])  d2h_queue[dev_id] = new CommandQueue();
  if (!exec_queue[dev_id])  exec_queue[dev_id] = new CommandQueue();
  if (!backend_init_flag[dev_id]){
		backend_init_flag[dev_id] = 1;
		backend_init(dev_id, h2d_queue[dev_id], d2h_queue[dev_id], exec_queue[dev_id]);
	}
}

void 	CoCoPeLiaFreeResources(short dev_id){
  if (h2d_queue[dev_id]){
		delete h2d_queue[dev_id];
		h2d_queue[dev_id] = NULL;
	}
  if (d2h_queue[dev_id]){
		delete d2h_queue[dev_id];
		d2h_queue[dev_id] = NULL;
	}
  if (exec_queue[dev_id]){
		delete exec_queue[dev_id];
		exec_queue[dev_id] = NULL;
	}
  if (backend_init_flag[dev_id]){
		backend_init_flag[dev_id] = 0;
		backend_free(dev_id);
	}
}
