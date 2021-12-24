///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The "Subkernel" related function implementations.
///

#include "Subkernel.hpp"
#include "unihelpers.hpp"
#include "Asset.hpp"
#include "DataCaching.hpp"
#include "backend_lib_wrappers.hpp"

/// TODO: Works for systems with up to 128 devices, not 'completely' future-proof
CQueue_p h2d_queue[128] = {NULL}, d2h_queue[128] = {NULL}, exec_queue[128] = {NULL};
int Subkernel_num = 0, backend_init_flag[128] = {0};

void* reduce_buf[128] = {NULL};

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
	if (WR_writer) delete writeback_complete;
}

void Subkernel::init_events(){
	operation_complete = new Event();
	if (WR_writer || WR_reducer) writeback_complete = new Event();
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
				if (tmp->R_flag && tmp->CacheLocId[run_dev_id] != -1 && (!tmp->W_flag || WR_reader )) tmp->available[run_dev_id]->sync_barrier();
			}
		}
}

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
				if (tmp->CacheLocId[run_dev_id] == -1) continue;
				short InitialId = tmp->getWriteBackLoc();
			 	if (InitialId == -1) InitialId = LOC_NUM - 1;
				if (tmp->CacheLocId[run_dev_id] < -1){
					tmp->adrs[run_dev_id] = CoCacheAsignBlock(run_dev_id, TileList[j], TileDimlist[j]);
#ifdef CDEBUG
					lprintf(lvl, "Subkernel(dev=%d,id=%d)-Tile(%d.[%d,%d]): Asigned buffer Block in GPU(%d)= %d\n",
					 run_dev_id, id, tmp->id, tmp->GridId1, tmp->GridId2, run_dev_id, tmp->CacheLocId[run_dev_id]);
#endif
					if(tmp->R_flag && (!tmp->W_flag || WR_reader )){
						short FetchFromIdCAdr, FetchFromId = tmp->getClosestReadLoc(run_dev_id);
						if (FetchFromId == -1) FetchFromIdCAdr = LOC_NUM - 1;
						else{
	#ifdef CDEBUG
							lprintf(lvl, "Subkernel(dev=%d,id=%d)-Tile(%d.[%d,%d]): Fetching Block(%d) on GPU(%d) from Block(%d) on GPU(%d)\n",
								run_dev_id, id, tmp->id,  tmp->GridId1, tmp->GridId2, tmp->CacheLocId[run_dev_id], run_dev_id, tmp->CacheLocId[FetchFromId], FetchFromId);
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
					//if (tmp->W_flag && WR_reader) tmp->available[InitialId]->reset();

					if (tmp->W_flag) CoCacheAddPendingEvent(run_dev_id, tmp->available[run_dev_id], writeback_complete, tmp->CacheLocId[run_dev_id], W);
					else CoCacheAddPendingEvent(run_dev_id, tmp->available[run_dev_id], operation_complete, tmp->CacheLocId[run_dev_id], R);

				}
				else{
					//CoCacheAddPendingEvent(run_dev_id, data_available, NULL, tmp->CacheLocId[run_dev_id], AVAILABLE);
					if (tmp->W_flag) CoCacheAddPendingEvent(run_dev_id, tmp->available[run_dev_id], writeback_complete, tmp->CacheLocId[run_dev_id], W);
					else CoCacheAddPendingEvent(run_dev_id, tmp->available[run_dev_id], operation_complete, tmp->CacheLocId[run_dev_id], R);
				}
		}
		else error("Subkernel(dev=%d,id=%d)::request_data: Not implemented for TileDim=%d\n", run_dev_id, id, TileDimlist[j]);
	}
	//data_available->record_to_queue(h2d_queue[run_dev_id]);
	//CoCoSyncCheckErr();
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
	backend_run_operation(run_dev_id, operation_params, op_name);
	operation_complete->record_to_queue(exec_queue[run_dev_id]);
	//if (prev!= NULL) prev->operation_complete->sync_barrier();
  //CoCoSyncCheckErr();
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
				if (tmp->W_flag){
					if (tmp->CacheLocId[run_dev_id] == -1) continue;
					else if (tmp->CacheLocId[run_dev_id] < -1)
						error("Subkernel(dev=%d,id=%d)-Tile(%d.[%d,%d])::writeback_data: invoked with tile with loc = %d\n",
							run_dev_id, id, tmp->id, tmp->GridId1, tmp->GridId2, tmp->CacheLocId[run_dev_id]);
					else{
						short WritebackIdCAdr, WritebackId = tmp->getWriteBackLoc(); //to MASTER
						if (WritebackId == -1) WritebackIdCAdr = LOC_NUM - 1;
						else WritebackIdCAdr = WritebackId;
						if(WR_writer==0) error("Subkernel(dev=%d,id=%d)-Tile(%d.[%d,%d])::writeback_data:\
						Subkernel should be a WR_writer?\n", run_dev_id, id, tmp->id, tmp->GridId1, tmp->GridId2);
							CoCoMemcpy2DAsync(tmp->adrs[WritebackIdCAdr], tmp->ldim[WritebackIdCAdr],
								tmp->adrs[run_dev_id], tmp->ldim[run_dev_id],
								tmp->dim1, tmp->dim2, tmp->dtypesize(),
								WritebackId, run_dev_id, d2h_queue[run_dev_id]);
							CoCacheAddPendingEvent(run_dev_id, operation_complete, writeback_complete, tmp->CacheLocId[run_dev_id], W);
					}
				}
		}
		else error("Subkernel(dev=%d,id=%d)::writeback_data: Not implemented for TileDim=%d\n", run_dev_id, id, TileDimlist[j]);
	}
	writeback_complete->record_to_queue(d2h_queue[run_dev_id]);
	//CoCoSyncCheckErr();
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
				if (tmp->W_flag){
					if (tmp->CacheLocId[run_dev_id] == -1) continue;
					else if (tmp->CacheLocId[run_dev_id] < -1)
						error("Subkernel(dev=%d,id=%d)-Tile(%d.[%d,%d])::writeback_reduce_data: invoked with tile with loc = %d\n",
							run_dev_id, id, tmp->id, tmp->GridId1, tmp->GridId2, tmp->CacheLocId[run_dev_id]);
					else{
						short WritebackIdCAdr, WritebackId = tmp->getWriteBackLoc(); //to MASTER
						if (WritebackId == -1) WritebackIdCAdr = LOC_NUM - 1;
						else WritebackIdCAdr = WritebackId;
						d2h_queue[run_dev_id]->wait_for_event(operation_complete);
						//for (int devidx = 0; devidx < LOC_NUM -1; devidx++){
						//	if(h2d_queue[devidx]!= NULL) h2d_queue[devidx]->sync_barrier();
						//}
						event_status temp_eve = tmp->available[WritebackIdCAdr]->query_status();
						while(temp_eve == UNRECORDED) temp_eve = tmp->available[WritebackIdCAdr]->query_status();
						tmp->available[WritebackIdCAdr]->sync_barrier();
						//delete tmp->available[WritebackIdCAdr];
						tmp->available[WritebackIdCAdr]->reset();
						if(WR_reducer==0) error("Subkernel(dev=%d,id=%d)-Tile(%d.[%d,%d])::writeback_reduce_data:\
						Subkernel should be a WR_reduce?\n", run_dev_id, id, tmp->id, tmp->GridId1, tmp->GridId2);
						else{
							if (reduce_buf[WritebackIdCAdr] == NULL)
								reduce_buf[WritebackIdCAdr] = CoCoMalloc(CoCoGetBlockSize(run_dev_id), WritebackId);
							CoCoMemcpyReduce2DAsync(reduce_buf[WritebackIdCAdr], tmp->adrs[WritebackIdCAdr], tmp->ldim[WritebackIdCAdr],
								tmp->adrs[run_dev_id], tmp->ldim[run_dev_id],
								tmp->dim1, tmp->dim2, tmp->dtypesize(),
								WritebackId, run_dev_id, d2h_queue);
							//tmp->available[WritebackIdCAdr]->record_to_queue(d2h_queue[run_dev_id]);
							}
						CoCacheAddPendingEvent(run_dev_id, operation_complete, tmp->available[WritebackIdCAdr], tmp->CacheLocId[run_dev_id], W);
					}
				}
		}
		else error("Subkernel(dev=%d,id=%d)::writeback_data: Not implemented for TileDim=%d\n", run_dev_id, id, TileDimlist[j]);
	}
	//CoCoSyncCheckErr();
#ifdef DEBUG
	lprintf(lvl-1, "<-----|\n");
#endif
}

void 	CoCoPeLiaInitStreams(short dev_id){
  if (!h2d_queue[dev_id]) h2d_queue[dev_id] = new CommandQueue();
  if (!d2h_queue[dev_id])  d2h_queue[dev_id] = new CommandQueue();
  if (!exec_queue[dev_id])  exec_queue[dev_id] = new CommandQueue();
  if (!backend_init_flag[dev_id]){
		backend_init_flag[dev_id] = 1;
		backend_init(dev_id, h2d_queue[dev_id], d2h_queue[dev_id], exec_queue[dev_id]);
	}
}
