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
int Subkernel_ctr = 0, backend_init_flag[128] = {0};

#ifdef MULTIDEVICE_REDUCTION_ENABLE
void* reduce_buf[128*MAX_BUFFERING_L] = {NULL};
short reduce_buf_it[128] = {0};
long long reduce_buf_sz[128*MAX_BUFFERING_L] = {0};
#endif

Subkernel::Subkernel(short TileNum_in, const char* name){
	id = Subkernel_ctr;
	run_dev_id = -42;
	op_name = name;
	Subkernel_ctr++;
	TileNum = TileNum_in;
	TileDimlist = (short*) malloc(TileNum*sizeof(short));
	TileList = (void**) malloc(TileNum*sizeof(void*));
	prev = next = NULL;
#ifdef STEST
	bytes_in = bytes_out = flops = 0;
#endif
}

Subkernel::~Subkernel(){
	short lvl = 5;
#ifdef DDEBUG
	lprintf(lvl, "Subkernel(dev=%d,id=%d):~Subkernel\n", run_dev_id, id);
#endif
	short run_dev_id_idx = (run_dev_id == -1)?  LOC_NUM - 1 : run_dev_id;
	Subkernel_ctr--;
	free(TileDimlist);
	free(TileList);
	free(operation_params);
	delete operation_complete;
	delete writeback_complete;
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
#ifdef STEST
	delete input_timer;
	delete output_timer;
	delete operation_timer;
#endif
}

void Subkernel::init_events(){
	operation_complete = new Event(run_dev_id);
	writeback_complete = new Event(run_dev_id);
#ifdef STEST
	input_timer = new Event_timer(run_dev_id);
	output_timer = new Event_timer(run_dev_id);
	operation_timer = new Event_timer(run_dev_id);
#endif
}

void Subkernel::sync_request_data(){
	short lvl = 4;
#ifdef DEBUG
	lprintf(lvl-1, "|-----> Subkernel(dev=%d,id=%d)::sync_request_data()\n", run_dev_id, id);
#endif
	short run_dev_id_idx = idxize(run_dev_id);
	for (int j = 0; j < TileNum; j++){
		if (TileDimlist[j] == 1){
				Tile1D<VALUE_TYPE>* tmp = (Tile1D<VALUE_TYPE>*) TileList[j];
				if (tmp->R_flag && tmp->CacheLocId[run_dev_id_idx] != -1) tmp->available[run_dev_id_idx]->sync_barrier();
		}
		else if (TileDimlist[j] == 2){
				Tile2D<VALUE_TYPE>* tmp = (Tile2D<VALUE_TYPE>*) TileList[j];
				if (tmp->R_flag && tmp->CacheLocId[run_dev_id_idx] != -1) tmp->available[run_dev_id_idx]->sync_barrier();
		}
	}
}

void Subkernel::request_tile(short TileIdx){
	short lvl = 5;
	short run_dev_id_idx = idxize(run_dev_id);
	if (TileDimlist[TileIdx] == 1){
			Tile1D<VALUE_TYPE>* tmp = (Tile1D<VALUE_TYPE>*) TileList[TileIdx];
#ifdef DEBUG
		lprintf(lvl-1, "|-----> Subkernel(dev=%d,id=%d)::request_tile(Tile(%d.[%d]))\n",
			run_dev_id, id, tmp->id, tmp->GridId);
#endif
		short FetchFromId = - 42;
		if(!tmp->W_flag) FetchFromId = tmp->getClosestReadLoc(run_dev_id);
		else {
			FetchFromId = tmp->RW_master;
			if (FetchFromId == run_dev_id) error("Subkernel(dev=%d,id=%d)-Tile(%d.[%d])::request_tile W_flag = %d, \
			FetchFromId == run_dev_id == %d\n", run_dev_id, id, tmp->id,  tmp->GridId, tmp->W_flag, FetchFromId);
			if(!tmp->isLocked()) error("Subkernel(dev=%d,id=%d)-Tile(%d.[%d])::request_tile W_flag = %d, \
			FetchFromId = %d, run_dev_id = %d Write tile was not already locked when requested\n",
			run_dev_id, id, tmp->id,  tmp->GridId, tmp->W_flag, FetchFromId, run_dev_id);
		}
		short FetchFromId_idx = idxize(FetchFromId);
		// TODO: why was this here? Mistake? if (FetchFromId != -1){
#ifdef CDEBUG
		lprintf(lvl, "Subkernel(dev=%d,id=%d)-Tile(%d.[%d]): Fetching Block(%d) on GPU(%d) from Block(%d) on GPU(%d)\n",
			run_dev_id, id, tmp->id,  tmp->GridId, tmp->CacheLocId[run_dev_id_idx], run_dev_id,
			tmp->CacheLocId[FetchFromId_idx], FetchFromId);
#endif
		CacheWrap_p wrap_read = NULL, wrap_fetch = NULL;
		if (tmp->CacheLocId[FetchFromId_idx] < -1)
			error("Subkernel(dev=%d,id=%d)-Tile(%d.[%d])::request_tile: Fetching from tile in GPU(%d) with loc = %d\n",
				run_dev_id, id, tmp->id,  tmp->GridId, FetchFromId, tmp->CacheLocId[FetchFromId_idx]);
		else if (tmp->CacheLocId[FetchFromId_idx] != -1){
			wrap_read = (CacheWrap_p) malloc(sizeof(struct Cache_info_wrap));
			wrap_read->dev_id = FetchFromId;
			wrap_read->BlockIdx = tmp->CacheLocId[FetchFromId_idx];
			CacheStartRead(wrap_read);
		}
		if (tmp->CacheLocId[run_dev_id_idx] != -1){
			wrap_fetch = (CacheWrap_p) malloc(sizeof(struct Cache_info_wrap));
			wrap_fetch->dev_id = run_dev_id;
			wrap_fetch->BlockIdx = tmp->CacheLocId[run_dev_id_idx];
			CacheStartFetch(wrap_fetch);
		}
		CoCoMemcpyAsync(tmp->adrs[run_dev_id_idx], tmp->adrs[FetchFromId_idx],
											((long long) tmp->inc[run_dev_id_idx]) * tmp->dim * tmp->dtypesize(),
											run_dev_id, FetchFromId, h2d_queue[run_dev_id_idx]);
		tmp->available[run_dev_id_idx]->record_to_queue(h2d_queue[run_dev_id_idx]);
		if(tmp->W_flag){
			if (tmp->CacheLocId[run_dev_id_idx] != -1)
				h2d_queue[run_dev_id_idx]->add_host_func((void*)&CacheEndFetchStartWrite, (void*) wrap_fetch);
			Ptr_and_int_p wrapped_op = (Ptr_and_int_p) malloc(sizeof(struct Ptr_and_int));
			wrapped_op->int_ptr = &tmp->RW_master;
			wrapped_op->val = run_dev_id;
			h2d_queue[run_dev_id_idx]->add_host_func((void*)&CoCoSetInt, (void*) wrapped_op);
			if (tmp->CacheLocId[FetchFromId_idx] != -1){
				h2d_queue[run_dev_id_idx]->add_host_func((void*)&CacheEndRead, (void*) wrap_read);
				//FIXME: Invalidate cache here?
				Ptr_and_int_p wrapped_op1 = (Ptr_and_int_p) malloc(sizeof(struct Ptr_and_int));
				wrapped_op1->int_ptr = &tmp->CacheLocId[FetchFromId_idx];
				wrapped_op1->val = -42;
				h2d_queue[run_dev_id_idx]->add_host_func((void*)&CoCoSetInt, (void*) wrapped_op1);
			}
		}
		else{
			if (tmp->CacheLocId[run_dev_id_idx] != -1)
				h2d_queue[run_dev_id_idx]->add_host_func((void*)&CacheEndFetchStartRead, (void*) wrap_fetch);
			if (tmp->CacheLocId[FetchFromId_idx] != -1)
				h2d_queue[run_dev_id_idx]->add_host_func((void*)&CacheEndRead, (void*) wrap_read);
		}
	}
	else if (TileDimlist[TileIdx] == 2){
			Tile2D<VALUE_TYPE>* tmp = (Tile2D<VALUE_TYPE>*) TileList[TileIdx];
#ifdef DEBUG
		lprintf(lvl-1, "|-----> Subkernel(dev=%d,id=%d)::request_tile(Tile(%d.[%d,%d]))\n",
			run_dev_id, id, tmp->id, tmp->GridId1, tmp->GridId2);
#endif
		short FetchFromId = -42;
		if(!tmp->W_flag) FetchFromId = tmp->getClosestReadLoc(run_dev_id);
		else {
			FetchFromId = tmp->RW_master;
			if (FetchFromId == run_dev_id) error("Subkernel(dev=%d,id=%d)-Tile(%d.[%d,%d])::request_tile W_flag = %d, \
			FetchFromId == run_dev_id == %d\n", run_dev_id, id, tmp->id, tmp->GridId1, tmp->GridId2, tmp->W_flag, FetchFromId);
			if(!tmp->isLocked()) error("Subkernel(dev=%d,id=%d)-Tile(%d.[%d,%d])::request_tile W_flag = %d, \
			FetchFromId = %d, run_dev_id = %d Write tile was not already locked when requested\n",
			run_dev_id, id, tmp->id, tmp->GridId1, tmp->GridId2, tmp->W_flag, FetchFromId, run_dev_id);
		}
		short FetchFromId_idx = idxize(FetchFromId);
#ifdef CDEBUG
		lprintf(lvl, "Subkernel(dev=%d,id=%d)-Tile(%d.[%d,%d]): Fetching Block(%d) on GPU(%d) from Block(%d) on GPU(%d)\n",
			run_dev_id, id, tmp->id,  tmp->GridId1, tmp->GridId2, tmp->CacheLocId[run_dev_id_idx], run_dev_id,
			tmp->CacheLocId[FetchFromId_idx], FetchFromId);
#endif
		CacheWrap_p wrap_read = NULL, wrap_fetch = NULL;
		if (tmp->CacheLocId[FetchFromId_idx] < -1)
			error("Subkernel(dev=%d,id=%d)-Tile(%d.[%d,%d])::request_data: Fetching from tile in GPU(%d) with loc = %d\n",
				run_dev_id, id, tmp->id,  tmp->GridId1, tmp->GridId2, FetchFromId, tmp->CacheLocId[FetchFromId_idx]);
		else if (tmp->CacheLocId[FetchFromId_idx] != -1){
			wrap_read = (CacheWrap_p) malloc(sizeof(struct Cache_info_wrap));
			wrap_read->dev_id = FetchFromId;
			wrap_read->BlockIdx = tmp->CacheLocId[FetchFromId_idx];
			CacheStartRead(wrap_read);
		}
		if (tmp->CacheLocId[run_dev_id_idx] != -1){
			wrap_fetch = (CacheWrap_p) malloc(sizeof(struct Cache_info_wrap));
			wrap_fetch->dev_id = run_dev_id;
			wrap_fetch->BlockIdx = tmp->CacheLocId[run_dev_id_idx];
			CacheStartFetch(wrap_fetch);
		}
		CoCoMemcpy2DAsync(tmp->adrs[run_dev_id_idx], tmp->ldim[run_dev_id_idx],
											tmp->adrs[FetchFromId_idx], tmp->ldim[FetchFromId_idx],
											tmp->dim1, tmp->dim2, tmp->dtypesize(),
											run_dev_id, FetchFromId, h2d_queue[run_dev_id_idx]);
		tmp->available[run_dev_id_idx]->record_to_queue(h2d_queue[run_dev_id_idx]);
		if(tmp->W_flag){
			if (tmp->CacheLocId[run_dev_id_idx] != -1)
				h2d_queue[run_dev_id_idx]->add_host_func((void*)&CacheEndFetchStartWrite, (void*) wrap_fetch);
			Ptr_and_int_p wrapped_op = (Ptr_and_int_p) malloc(sizeof(struct Ptr_and_int));
			wrapped_op->int_ptr = &tmp->RW_master;
			wrapped_op->val = run_dev_id;
			h2d_queue[run_dev_id_idx]->add_host_func((void*)&CoCoSetInt, (void*) wrapped_op);
			if (tmp->CacheLocId[FetchFromId_idx] != -1){
				h2d_queue[run_dev_id_idx]->add_host_func((void*)&CacheEndRead, (void*) wrap_read);
				//FIXME: Invalidate cache here?
				Ptr_and_int_p wrapped_op1 = (Ptr_and_int_p) malloc(sizeof(struct Ptr_and_int));
				wrapped_op1->int_ptr = &tmp->CacheLocId[FetchFromId_idx];
				wrapped_op1->val = -42;
				h2d_queue[run_dev_id_idx]->add_host_func((void*)&CoCoSetInt, (void*) wrapped_op1);
			}
		}
		else{
			if (tmp->CacheLocId[run_dev_id_idx] != -1)
				h2d_queue[run_dev_id_idx]->add_host_func((void*)&CacheEndFetchStartRead, (void*) wrap_fetch);
			if (tmp->CacheLocId[FetchFromId_idx] != -1)
				h2d_queue[run_dev_id_idx]->add_host_func((void*)&CacheEndRead, (void*) wrap_read);
		}
	}
}

int WR_check_lock = 0;

void Subkernel::request_data(){
	short lvl = 4;
#ifdef DEBUG
	lprintf(lvl-1, "|-----> Subkernel(dev=%d,id=%d)::request_data()\n", run_dev_id, id);
#endif
	short run_dev_id_idx = idxize(run_dev_id);
#ifdef STEST
		input_timer->start_point(h2d_queue[run_dev_id_idx]);
#endif
	CacheWrap_p wrap_read, wrap_write;
	for (int j = 0; j < TileNum; j++){
		if (TileDimlist[j] == 1){
			Tile1D<VALUE_TYPE>* tmp = (Tile1D<VALUE_TYPE>*) TileList[j];
			if (tmp->CacheLocId[run_dev_id_idx] == -1 && (!tmp->W_flag || tmp->RW_master == run_dev_id)) continue;
			else if(tmp->CacheLocId[run_dev_id_idx] < -1){
				tmp->adrs[run_dev_id_idx] = CacheAsignBlock(run_dev_id, TileList[j], TileDimlist[j]);
#ifdef CDEBUG
		lprintf(lvl, "Subkernel(dev=%d,id=%d)-Tile(%d.[%d]): Asigned buffer Block in GPU(%d)= %d\n",
					run_dev_id, id, tmp->id, tmp->GridId, run_dev_id, tmp->CacheLocId[run_dev_id_idx]);
#endif
				if (tmp->R_flag){
					request_tile(j);
#ifdef STEST
					bytes_in+= tmp->size();
#endif
				}
			}
			else{
				CacheWrap_p wrap_reuse = (CacheWrap_p) malloc(sizeof(struct Cache_info_wrap));
				wrap_reuse->dev_id = run_dev_id;
				wrap_reuse->BlockIdx = tmp->CacheLocId[run_dev_id_idx];
				if (tmp->W_flag) h2d_queue[run_dev_id_idx]->add_host_func((void*)&CacheStartWrite, (void*) wrap_reuse);
				else h2d_queue[run_dev_id_idx]->add_host_func((void*)&CacheStartRead, (void*) wrap_reuse);
			}
		}
		else if (TileDimlist[j] == 2){
				Tile2D<VALUE_TYPE>* tmp = (Tile2D<VALUE_TYPE>*) TileList[j];
				if (tmp->CacheLocId[run_dev_id_idx] == -1 && (!tmp->W_flag || tmp->RW_master == run_dev_id)) continue;
				else if(tmp->CacheLocId[run_dev_id_idx] < -1){
					tmp->adrs[run_dev_id_idx] = CacheAsignBlock(run_dev_id, TileList[j], TileDimlist[j]);
	#ifdef CDEBUG
			lprintf(lvl, "Subkernel(dev=%d,id=%d)-Tile(%d.[%d, %d]): Asigned buffer Block in GPU(%d)= %d\n",
						run_dev_id, id, tmp->id, tmp->GridId1, tmp->GridId2, run_dev_id, tmp->CacheLocId[run_dev_id_idx]);
	#endif
					if (tmp->R_flag){
						request_tile(j);
	#ifdef STEST
						bytes_in+= tmp->size();
	#endif
					}
				}
				else{
					CacheWrap_p wrap_reuse = (CacheWrap_p) malloc(sizeof(struct Cache_info_wrap));
					wrap_reuse->dev_id = run_dev_id;
					wrap_reuse->BlockIdx = tmp->CacheLocId[run_dev_id_idx];
					if (tmp->W_flag) h2d_queue[run_dev_id_idx]->add_host_func((void*)&CacheStartWrite, (void*) wrap_reuse);
					else h2d_queue[run_dev_id_idx]->add_host_func((void*)&CacheStartRead, (void*) wrap_reuse);
				}
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
#ifdef STEST
		input_timer->stop_point(h2d_queue[run_dev_id_idx]);
#endif
}

void Subkernel::run_operation(){
	short lvl = 4;
#ifdef DEBUG
	lprintf(lvl-1, "|-----> Subkernel(dev=%d,id=%d)::run_operation()\n", run_dev_id, id);
#endif
	short run_dev_id_idx = idxize(run_dev_id);
#ifdef ENABLE_PARALLEL_BACKEND
	if(WR_first) exec_queue[run_dev_id_idx]->request_parallel_backend();
#endif
	for (int j = 0; j < TileNum; j++){
		if (TileDimlist[j] == 1){
				Tile1D<VALUE_TYPE>* tmp = (Tile1D<VALUE_TYPE>*) TileList[j];
				if (tmp->CacheLocId[run_dev_id_idx] == -1 && (!tmp->W_flag || tmp->RW_master == run_dev_id)) continue;
				else if (tmp->CacheLocId[run_dev_id_idx] < -1)
					error("Subkernel(dev=%d,id=%d)-Tile(%d.[%d])::run_operation: Tile(j=%d) has loc = %d\n",
						run_dev_id, id, tmp->id, tmp->GridId, j, tmp->CacheLocId[run_dev_id_idx]);
				else if(tmp->R_flag) exec_queue[run_dev_id_idx]->wait_for_event(tmp->available[run_dev_id_idx]);
		}
		else if (TileDimlist[j] == 2){
				Tile2D<VALUE_TYPE>* tmp = (Tile2D<VALUE_TYPE>*) TileList[j];
				if (tmp->CacheLocId[run_dev_id_idx] == -1 && (!tmp->W_flag || tmp->RW_master == run_dev_id)) continue;
				else if (tmp->CacheLocId[run_dev_id_idx] < -1)
					error("Subkernel(dev=%d,id=%d)-Tile(%d.[%d,%d])::run_operation: Tile(j=%d) has loc = %d\n",
						run_dev_id, id, tmp->id, tmp->GridId1, tmp->GridId2, j, tmp->CacheLocId[run_dev_id_idx]);
				else if(tmp->R_flag) exec_queue[run_dev_id_idx]->wait_for_event(tmp->available[run_dev_id_idx]);
		}
	}
#ifdef STEST
		operation_timer->start_point(exec_queue[run_dev_id_idx]);
		if (!strcmp(op_name,"gemm")){
			gemm_backend_in_p ptr_ker_translate = (gemm_backend_in_p) operation_params;
			flops = dgemm_flops(ptr_ker_translate->M, ptr_ker_translate->N, ptr_ker_translate->K);
		}
#endif
	backend_run_operation(operation_params, op_name, exec_queue[run_dev_id_idx]);
	operation_complete->record_to_queue(exec_queue[run_dev_id_idx]);
	for (int j = 0; j < TileNum; j++){
		if (TileDimlist[j] == 1){
			Tile1D<VALUE_TYPE>* tmp = (Tile1D<VALUE_TYPE>*) TileList[j];
			if (tmp->CacheLocId[run_dev_id_idx] == -1) continue;
			else if (tmp->CacheLocId[run_dev_id_idx] < -1)
				error("Subkernel(dev=%d,id=%d)-Tile(%d.[%d])::run_operation: Tile(j=%d) has loc = %d\n",
					run_dev_id, id, tmp->id, tmp->GridId, j, tmp->CacheLocId[run_dev_id_idx]);
			else{
				CacheWrap_p wrap_oper = (CacheWrap_p) malloc(sizeof(struct Cache_info_wrap));
				wrap_oper->dev_id = run_dev_id;
				wrap_oper->BlockIdx = tmp->CacheLocId[run_dev_id_idx];
				if(tmp->W_flag){
					exec_queue[run_dev_id_idx]->add_host_func((void*)&CacheEndWrite, (void*) wrap_oper);
					exec_queue[run_dev_id_idx]->add_host_func((void*)&CoCoQueueUnlock, (void*) &tmp->RW_lock);
				}
				else exec_queue[run_dev_id_idx]->add_host_func((void*)&CacheEndRead, (void*) wrap_oper);
			}
			if(tmp->R_flag) tmp->R_flag--;
			if(tmp->W_flag){
				tmp->W_flag--;
				if(!tmp->W_flag) WR_last[j] = 1;
			}
		}
		else if (TileDimlist[j] == 2){
			Tile2D<VALUE_TYPE>* tmp = (Tile2D<VALUE_TYPE>*) TileList[j];
			if (tmp->CacheLocId[run_dev_id_idx] == -1) continue;
			else if (tmp->CacheLocId[run_dev_id_idx] < -1)
				error("Subkernel(dev=%d,id=%d)-Tile(%d.[%d,%d])::run_operation: Tile(j=%d) has loc = %d\n",
					run_dev_id, id, tmp->id, tmp->GridId1, tmp->GridId2, j, tmp->CacheLocId[run_dev_id_idx]);
			else{
				CacheWrap_p wrap_oper = (CacheWrap_p) malloc(sizeof(struct Cache_info_wrap));
				wrap_oper->dev_id = run_dev_id;
				wrap_oper->BlockIdx = tmp->CacheLocId[run_dev_id_idx];
				if(tmp->W_flag){
					exec_queue[run_dev_id_idx]->add_host_func((void*)&CacheEndWrite, (void*) wrap_oper);
					exec_queue[run_dev_id_idx]->add_host_func((void*)&CoCoQueueUnlock, (void*) &tmp->RW_lock);
				}
				else exec_queue[run_dev_id_idx]->add_host_func((void*)&CacheEndRead, (void*) wrap_oper);
			}
			if(tmp->R_flag) tmp->R_flag--;
			if(tmp->W_flag){
				tmp->W_flag--;
				if(!tmp->W_flag) WR_last[j] = 1;
			}
		}
	}
#ifndef ASYNC_ENABLE
	CoCoSyncCheckErr();
#endif
#ifdef DEBUG
	lprintf(lvl-1, "<-----|\n");
#endif
#ifdef STEST
		operation_timer->stop_point(exec_queue[run_dev_id_idx]);
#endif
}

void Subkernel::writeback_data(){
	short lvl = 4;
#ifdef DEBUG
	lprintf(lvl-1, "|-----> Subkernel(dev=%d,id=%d)::writeback_data()\n", run_dev_id, id);
#endif
	short run_dev_id_idx = idxize(run_dev_id);
#ifdef STEST
		operation_timer->start_point(d2h_queue[run_dev_id_idx]);
#endif
	for (int j = 0; j < TileNum; j++) if (WR_last[j]){
		if (TileDimlist[j] == 1){
			Tile1D<VALUE_TYPE>* tmp = (Tile1D<VALUE_TYPE>*) TileList[j];
			if (tmp->CacheLocId[run_dev_id_idx] < -1)
				error("Subkernel(dev=%d,id=%d)-Tile(%d.[%d])::writeback_data: invoked with tile with loc = %d\n",
					run_dev_id, id, tmp->id, tmp->GridId, tmp->CacheLocId[run_dev_id_idx]);
			else{
				d2h_queue[run_dev_id_idx]->wait_for_event(operation_complete);
				short WritebackIdCAdr, WritebackId = tmp->getWriteBackLoc(); //to MASTER
				WritebackIdCAdr = idxize(WritebackId);
				if (run_dev_id == WritebackId){
					;
#ifdef DEBUG
			lprintf(lvl, "Subkernel(dev=%d,id=%d)-Tile(%d.[%d])::writeback_data: run_dev_id == WritebackId == %d\n",
				run_dev_id, id, tmp->id, tmp->GridId, run_dev_id);
#endif
				}
				else{
					CoCoMemcpyAsync(tmp->adrs[WritebackIdCAdr], tmp->adrs[run_dev_id_idx],
						((long long) tmp->inc[run_dev_id_idx]) * tmp->dim * tmp->dtypesize(),
						WritebackId, run_dev_id, d2h_queue[run_dev_id_idx]);
					#ifdef STEST
					bytes_out+= tmp->size();
					#endif
				}
			}
		}
		else if (TileDimlist[j] == 2){
			Tile2D<VALUE_TYPE>* tmp = (Tile2D<VALUE_TYPE>*) TileList[j];
			if (tmp->CacheLocId[run_dev_id_idx] < -1)
				error("Subkernel(dev=%d,id=%d)-Tile(%d.[%d,%d])::writeback_data: invoked with tile with loc = %d\n",
					run_dev_id, id, tmp->id, tmp->GridId1, tmp->GridId2, tmp->CacheLocId[run_dev_id_idx]);
			else{
				d2h_queue[run_dev_id_idx]->wait_for_event(operation_complete);
				short WritebackIdCAdr, WritebackId = tmp->getWriteBackLoc(); //to MASTER
				WritebackIdCAdr = idxize(WritebackId);
				if (run_dev_id == WritebackId){
					;
#ifdef DEBUG
			lprintf(lvl, "Subkernel(dev=%d,id=%d)-Tile(%d.[%d,%d])::writeback_data: run_dev_id == WritebackId == %d\n",
				run_dev_id, id, tmp->id, tmp->GridId1, tmp->GridId2, run_dev_id);
#endif
				}
				else{
					CoCoMemcpy2DAsync(tmp->adrs[WritebackIdCAdr], tmp->ldim[WritebackIdCAdr],
						tmp->adrs[run_dev_id_idx], tmp->ldim[run_dev_id_idx],
						tmp->dim1, tmp->dim2, tmp->dtypesize(),
						WritebackId, run_dev_id, d2h_queue[run_dev_id_idx]);
					#ifdef STEST
					bytes_out+= tmp->size();
					#endif
				}
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
#ifdef STEST
		operation_timer->stop_point(d2h_queue[run_dev_id_idx]);
#endif
}

void CoCoPeLiaInitResources(short dev_id){
	short dev_id_idx = (dev_id == -1)?  LOC_NUM - 1 : dev_id;
  if (!h2d_queue[dev_id_idx]) h2d_queue[dev_id_idx] = new CommandQueue(dev_id);
  if (!d2h_queue[dev_id_idx])  d2h_queue[dev_id_idx] = new CommandQueue(dev_id);
  if (!exec_queue[dev_id_idx])  exec_queue[dev_id_idx] = new CommandQueue(dev_id);
}

void CoCoPeLiaFreeResources(short dev_id){
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

short Subkernel::is_dependency_free(){
	if (run_dev_id!= -42) return 0;
	for (int j = 0; j < TileNum; j++){
		if (TileDimlist[j] == 1){
			Tile1D<VALUE_TYPE>* tmp = (Tile1D<VALUE_TYPE>*) TileList[j];
			if(tmp->isLocked()) return 0;
		}
		else if (TileDimlist[j] == 2){
			Tile2D<VALUE_TYPE>* tmp = (Tile2D<VALUE_TYPE>*) TileList[j];
			if(tmp->isLocked()) return 0;
		}
		else error("Subkernel(dev=%d,id=%d)::writeback_data: Not implemented for TileDim=%d\n", run_dev_id, id, TileDimlist[j]);
	}
	return 1;
}

Subkernel* SubkernelSelectSimple(short dev_id, Subkernel** Subkernel_list, long Subkernel_list_len){
	Subkernel* curr_sk = NULL;
	long sk_idx;
	for (sk_idx = 0; sk_idx < Subkernel_list_len; sk_idx++){
		curr_sk = Subkernel_list[sk_idx];
		if (curr_sk->is_dependency_free()) break;

	}
	if(sk_idx==Subkernel_list_len) return NULL;
	else{
		for (int j = 0; j < curr_sk->TileNum; j++){
			if (curr_sk->TileDimlist[j] == 1){
				Tile1D<VALUE_TYPE>* tmp = (Tile1D<VALUE_TYPE>*) curr_sk->TileList[j];
				if(tmp->W_flag) {
					if (tmp->W_total == tmp->W_flag) curr_sk->WR_first = 1;
					CoCoQueueLock((void*) &tmp->RW_lock);
				}
			}
			else if (curr_sk->TileDimlist[j] == 2){
				Tile2D<VALUE_TYPE>* tmp = (Tile2D<VALUE_TYPE>*) curr_sk->TileList[j];
				if(tmp->W_flag) {
					if (tmp->W_total == tmp->W_flag) curr_sk->WR_first = 1;
					CoCoQueueLock((void*) &tmp->RW_lock);
				}
			}
		}
		return curr_sk;
	}
}
