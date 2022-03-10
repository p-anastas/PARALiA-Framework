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

int transfer_link_sharing[LOC_NUM][LOC_NUM][2] = {{{-42}}};
CQueue_p transfer_queues[LOC_NUM][LOC_NUM] = {{NULL}}, exec_queue[LOC_NUM] = {NULL};
int Subkernel_ctr = 0, backend_init_flag[LOC_NUM] = {0};

#ifdef MULTIDEVICE_REDUCTION_ENABLE
void* reduce_buf[LOC_NUM*MAX_BUFFERING_L] = {NULL};
short reduce_buf_it[LOC_NUM] = {0};
long long reduce_buf_sz[LOC_NUM*MAX_BUFFERING_L] = {0};
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
	//delete writeback_complete;
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
	for (int idx = 0; idx < TileNum; idx++){
		delete input_timer[idx];
		delete output_timer[idx];
	}
	delete operation_timer;
#endif
}

void Subkernel::init_events(){
	operation_complete = new Event(run_dev_id);
	//writeback_complete = new Event(run_dev_id);
#ifdef STEST
	for (int idx = 0; idx < TileNum; idx++){
		input_timer[idx] = new Event_timer(run_dev_id);
		output_timer[idx] = new Event_timer(run_dev_id);
	}
	operation_timer = new Event_timer(run_dev_id);
#endif
}

void Subkernel::sync_request_data_RONLY(){
	short lvl = 4;
#ifdef DEBUG
	lprintf(lvl-1, "|-----> Subkernel(dev=%d,id=%d)::sync_request_data()\n", run_dev_id, id);
#endif
	short run_dev_id_idx = idxize(run_dev_id);
	for (int j = 0; j < TileNum; j++){
		if (TileDimlist[j] == 1){
				Tile1D<VALUE_TYPE>* tmp = (Tile1D<VALUE_TYPE>*) TileList[j];
				if (!tmp->W_flag && tmp->R_flag && tmp->CacheLocId[run_dev_id_idx] != -1) tmp->available[run_dev_id_idx]->sync_barrier();
		}
		else if (TileDimlist[j] == 2){
				Tile2D<VALUE_TYPE>* tmp = (Tile2D<VALUE_TYPE>*) TileList[j];
				if (!tmp->W_flag && tmp->R_flag && tmp->CacheLocId[run_dev_id_idx] != -1) tmp->available[run_dev_id_idx]->sync_barrier();
		}
	}
#ifdef DEBUG
	lprintf(lvl-1, "<-----|\n");
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
#ifdef DEBUG
	lprintf(lvl-1, "<-----|\n");
#endif
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
		CacheGetLock(NULL);
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
			wrap_read->lock_flag = 0;
			if(tmp->W_flag){
				tmp->available[FetchFromId_idx]->reset();
				CacheInvalidate(wrap_read);
			}
			else CacheStartRead(wrap_read);
		}
		if (tmp->CacheLocId[run_dev_id_idx] != -1){
			wrap_fetch = (CacheWrap_p) malloc(sizeof(struct Cache_info_wrap));
			wrap_fetch->dev_id = run_dev_id;
			wrap_fetch->BlockIdx = tmp->CacheLocId[run_dev_id_idx];
			wrap_fetch->lock_flag = 0;
			CacheStartFetch(wrap_fetch);
		}
		CacheReleaseLock(NULL);
		if(wrap_read) wrap_read->lock_flag = 1;
		if(wrap_fetch) wrap_fetch->lock_flag = 1;
#ifdef STEST
		input_timer[TileIdx]->start_point(transfer_queues[run_dev_id_idx][FetchFromId_idx]);
#endif
		CoCoMemcpyAsync(tmp->adrs[run_dev_id_idx], tmp->adrs[FetchFromId_idx],
											((long long) tmp->inc[run_dev_id_idx]) * tmp->dim * tmp->dtypesize(),
											run_dev_id, FetchFromId, transfer_queues[run_dev_id_idx][FetchFromId_idx]);
#ifdef STEST
		input_timer[TileIdx]->stop_point(transfer_queues[run_dev_id_idx][FetchFromId_idx]);
		bytes_in+= tmp->size();
#endif
		if(tmp->W_flag){
			if (tmp->CacheLocId[run_dev_id_idx] != -1)
				transfer_queues[run_dev_id_idx][FetchFromId_idx]->add_host_func((void*)&CacheEndFetchStartWrite, (void*) wrap_fetch);
			Ptr_and_int_p wrapped_op = (Ptr_and_int_p) malloc(sizeof(struct Ptr_and_int));
			wrapped_op->int_ptr = &tmp->RW_master;
			wrapped_op->val = run_dev_id;
			transfer_queues[run_dev_id_idx][FetchFromId_idx]->add_host_func((void*)&CoCoSetInt, (void*) wrapped_op);
			if (tmp->CacheLocId[FetchFromId_idx] != -1){
				Ptr_and_int_p wrapped_op1 = (Ptr_and_int_p) malloc(sizeof(struct Ptr_and_int));
				wrapped_op1->int_ptr = &tmp->CacheLocId[FetchFromId_idx];
				wrapped_op1->val = -42;
				transfer_queues[run_dev_id_idx][FetchFromId_idx]->add_host_func((void*)&CoCoSetInt, (void*) wrapped_op1);
			}
		}
		else{
			if (tmp->CacheLocId[run_dev_id_idx] != -1)
				transfer_queues[run_dev_id_idx][FetchFromId_idx]->add_host_func((void*)&CacheEndFetchStartRead, (void*) wrap_fetch);
			if (tmp->CacheLocId[FetchFromId_idx] != -1)
				transfer_queues[run_dev_id_idx][FetchFromId_idx]->add_host_func((void*)&CacheEndRead, (void*) wrap_read);
		}
		tmp->available[run_dev_id_idx]->record_to_queue(transfer_queues[run_dev_id_idx][FetchFromId_idx]);
	}
	else if (TileDimlist[TileIdx] == 2){
			Tile2D<VALUE_TYPE>* tmp = (Tile2D<VALUE_TYPE>*) TileList[TileIdx];
#ifdef DEBUG
		lprintf(lvl-1, "|-----> Subkernel(dev=%d,id=%d)::request_tile(Tile(%d.[%d,%d]))\n",
			run_dev_id, id, tmp->id, tmp->GridId1, tmp->GridId2);
#endif
		short FetchFromId = -42;
		CacheGetLock(NULL);
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
			wrap_read->lock_flag = 0;
			if(tmp->W_flag){
				tmp->available[FetchFromId_idx]->reset();
				CacheInvalidate(wrap_read);
			}
			else CacheStartRead(wrap_read);
		}
		if (tmp->CacheLocId[run_dev_id_idx] != -1){
			wrap_fetch = (CacheWrap_p) malloc(sizeof(struct Cache_info_wrap));
			wrap_fetch->dev_id = run_dev_id;
			wrap_fetch->BlockIdx = tmp->CacheLocId[run_dev_id_idx];
			wrap_fetch->lock_flag = 0;
			CacheStartFetch(wrap_fetch);
		}
		CacheReleaseLock(NULL);
		if(wrap_read) wrap_read->lock_flag = 1;
		if(wrap_fetch) wrap_fetch->lock_flag = 1;
#ifdef STEST
		input_timer[TileIdx]->start_point(transfer_queues[run_dev_id_idx][FetchFromId_idx]);
#endif
		CoCoMemcpy2DAsync(tmp->adrs[run_dev_id_idx], tmp->ldim[run_dev_id_idx],
									tmp->adrs[FetchFromId_idx], tmp->ldim[FetchFromId_idx],
									tmp->dim1, tmp->dim2, tmp->dtypesize(),
									run_dev_id, FetchFromId, transfer_queues[run_dev_id_idx][FetchFromId_idx]);
#ifdef STEST
		input_timer[TileIdx]->stop_point(transfer_queues[run_dev_id_idx][FetchFromId_idx]);
		bytes_in+= tmp->size();
#endif
		if(tmp->W_flag){
			if (tmp->CacheLocId[run_dev_id_idx] != -1)
				transfer_queues[run_dev_id_idx][FetchFromId_idx]->add_host_func((void*)&CacheEndFetchStartWrite, (void*) wrap_fetch);
			Ptr_and_int_p wrapped_op = (Ptr_and_int_p) malloc(sizeof(struct Ptr_and_int));
			wrapped_op->int_ptr = &tmp->RW_master;
			wrapped_op->val = run_dev_id;
			transfer_queues[run_dev_id_idx][FetchFromId_idx]->add_host_func((void*)&CoCoSetInt, (void*) wrapped_op);
			if (tmp->CacheLocId[FetchFromId_idx] != -1){
				Ptr_and_int_p wrapped_op1 = (Ptr_and_int_p) malloc(sizeof(struct Ptr_and_int));
				wrapped_op1->int_ptr = &tmp->CacheLocId[FetchFromId_idx];
				wrapped_op1->val = -42;
				transfer_queues[run_dev_id_idx][FetchFromId_idx]->add_host_func((void*)&CoCoSetInt, (void*) wrapped_op1);
			}
		}
		else{
			if (tmp->CacheLocId[run_dev_id_idx] != -1)
				transfer_queues[run_dev_id_idx][FetchFromId_idx]->add_host_func((void*)&CacheEndFetchStartRead, (void*) wrap_fetch);
			if (tmp->CacheLocId[FetchFromId_idx] != -1)
				transfer_queues[run_dev_id_idx][FetchFromId_idx]->add_host_func((void*)&CacheEndRead, (void*) wrap_read);
		}
		tmp->available[run_dev_id_idx]->record_to_queue(transfer_queues[run_dev_id_idx][FetchFromId_idx]);
	}
}

void Subkernel::request_data(){
	short lvl = 4;
#ifdef DEBUG
	lprintf(lvl-1, "|-----> Subkernel(dev=%d,id=%d)::request_data()\n", run_dev_id, id);
#endif
	short run_dev_id_idx = idxize(run_dev_id);
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
				if (tmp->R_flag)request_tile(j);
			}
			else if(tmp->CacheLocId[run_dev_id_idx] == -1 && tmp->W_flag && tmp->RW_master != run_dev_id){
				if (tmp->R_flag) request_tile(j);
			}
			else{
				CacheWrap_p wrap_reuse = (CacheWrap_p) malloc(sizeof(struct Cache_info_wrap));
				wrap_reuse->dev_id = run_dev_id;
				wrap_reuse->BlockIdx = tmp->CacheLocId[run_dev_id_idx];
				wrap_reuse->lock_flag = 1;
				if (!tmp->W_flag) CacheStartRead((void*) wrap_reuse);
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
					if (tmp->R_flag) request_tile(j);
				}
				else if(tmp->CacheLocId[run_dev_id_idx] == -1 && tmp->W_flag && tmp->RW_master != run_dev_id){
					if (tmp->R_flag) request_tile(j);
				}
				else{
					CacheWrap_p wrap_reuse = (CacheWrap_p) malloc(sizeof(struct Cache_info_wrap));
					wrap_reuse->dev_id = run_dev_id;
					wrap_reuse->BlockIdx = tmp->CacheLocId[run_dev_id_idx];
					wrap_reuse->lock_flag = 1;
					if (!tmp->W_flag) CacheStartRead((void*) wrap_reuse);
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
#ifdef STEST
	operation_timer->stop_point(exec_queue[run_dev_id_idx]);
#endif
	for (int j = 0; j < TileNum; j++){
		if (TileDimlist[j] == 1){
			Tile1D<VALUE_TYPE>* tmp = (Tile1D<VALUE_TYPE>*) TileList[j];
			if (tmp->CacheLocId[run_dev_id_idx] == -1 && !tmp->W_flag) continue;
			else if (tmp->CacheLocId[run_dev_id_idx] < -1)
				error("Subkernel(dev=%d,id=%d)-Tile(%d.[%d])::run_operation: Tile(j=%d) has loc = %d\n",
					run_dev_id, id, tmp->id, tmp->GridId, j, tmp->CacheLocId[run_dev_id_idx]);
			else{
				if(tmp->W_flag)	exec_queue[run_dev_id_idx]->add_host_func((void*)&CoCoQueueUnlock, (void*) &tmp->RW_lock);
				else{
					CacheWrap_p wrap_oper = (CacheWrap_p) malloc(sizeof(struct Cache_info_wrap));
					wrap_oper->dev_id = run_dev_id;
					wrap_oper->BlockIdx = tmp->CacheLocId[run_dev_id_idx];
					wrap_oper->lock_flag = 1;
					exec_queue[run_dev_id_idx]->add_host_func((void*)&CacheEndRead, (void*) wrap_oper);
				}
			}
			if(tmp->R_flag) tmp->R_flag--;
			if(tmp->W_flag){
				tmp->W_flag--;
				if(!tmp->W_flag) WR_last[j] = 1;
			}
		}
		else if (TileDimlist[j] == 2){
			Tile2D<VALUE_TYPE>* tmp = (Tile2D<VALUE_TYPE>*) TileList[j];
			if (tmp->CacheLocId[run_dev_id_idx] == -1 && !tmp->W_flag) continue;
			else if (tmp->CacheLocId[run_dev_id_idx] < -1)
				error("Subkernel(dev=%d,id=%d)-Tile(%d.[%d,%d])::run_operation: Tile(j=%d) has loc = %d\n",
					run_dev_id, id, tmp->id, tmp->GridId1, tmp->GridId2, j, tmp->CacheLocId[run_dev_id_idx]);
			else{

				if(tmp->W_flag) exec_queue[run_dev_id_idx]->add_host_func((void*)&CoCoQueueUnlock, (void*) &tmp->RW_lock);
				else{
					CacheWrap_p wrap_oper = (CacheWrap_p) malloc(sizeof(struct Cache_info_wrap));
					wrap_oper->dev_id = run_dev_id;
					wrap_oper->BlockIdx = tmp->CacheLocId[run_dev_id_idx];
					wrap_oper->lock_flag = 1;
					exec_queue[run_dev_id_idx]->add_host_func((void*)&CacheEndRead, (void*) wrap_oper);
				}
			}
			if(tmp->R_flag) tmp->R_flag--;
			if(tmp->W_flag){
				tmp->W_flag--;
				if(!tmp->W_flag) WR_last[j] = 1;
			}
		}
	}
	operation_complete->record_to_queue(exec_queue[run_dev_id_idx]);
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
	short run_dev_id_idx = idxize(run_dev_id);
	short Writeback_id_idx, Writeback_id;
	for (int j = 0; j < TileNum; j++) if (WR_last[j]){
		if (TileDimlist[j] == 1){
			Tile1D<VALUE_TYPE>* tmp = (Tile1D<VALUE_TYPE>*) TileList[j];
			if (tmp->CacheLocId[run_dev_id_idx] < -1)
				error("Subkernel(dev=%d,id=%d)-Tile(%d.[%d])::writeback_data: invoked with tile with loc = %d\n",
					run_dev_id, id, tmp->id, tmp->GridId, tmp->CacheLocId[run_dev_id_idx]);
			else{
				Writeback_id = tmp->getWriteBackLoc(); //to MASTER
				Writeback_id_idx = idxize(Writeback_id);
				transfer_queues[Writeback_id_idx][run_dev_id_idx]->wait_for_event(operation_complete);
				if (run_dev_id == Writeback_id){
					;
#ifdef DEBUG
			lprintf(lvl, "Subkernel(dev=%d,id=%d)-Tile(%d.[%d])::writeback_data: run_dev_id == Writeback_id == %d\n",
				run_dev_id, id, tmp->id, tmp->GridId, run_dev_id);
#endif
				}
				else{
					if (tmp->CacheLocId[run_dev_id_idx] != -1){
						CacheWrap_p wrap_inval = (CacheWrap_p) malloc(sizeof(struct Cache_info_wrap));
							wrap_inval->dev_id = run_dev_id;
							wrap_inval->BlockIdx = tmp->CacheLocId[run_dev_id_idx];
							wrap_inval->lock_flag = 1;
							transfer_queues[Writeback_id_idx][run_dev_id_idx]->add_host_func((void*)&CacheInvalidate, (void*) wrap_inval);
					}
#ifdef STEST
					output_timer[j]->start_point(transfer_queues[Writeback_id_idx][run_dev_id_idx]);
#endif
					CoCoMemcpyAsync(tmp->adrs[Writeback_id_idx], tmp->adrs[run_dev_id_idx],
						((long long) tmp->inc[run_dev_id_idx]) * tmp->dim * tmp->dtypesize(),
						Writeback_id, run_dev_id, transfer_queues[Writeback_id_idx][run_dev_id_idx]);
#ifdef STEST
					output_timer[j]->stop_point(transfer_queues[Writeback_id_idx][run_dev_id_idx]);
					bytes_out+= tmp->size();
#endif
					Ptr_and_int_p wrapped_op = (Ptr_and_int_p) malloc(sizeof(struct Ptr_and_int));
					wrapped_op->int_ptr = &tmp->RW_master;
					wrapped_op->val = Writeback_id;
					transfer_queues[Writeback_id_idx][run_dev_id_idx]->add_host_func((void*)&CoCoSetInt, (void*) wrapped_op);
				}
			}
		}
		else if (TileDimlist[j] == 2){
			Tile2D<VALUE_TYPE>* tmp = (Tile2D<VALUE_TYPE>*) TileList[j];
			if (tmp->CacheLocId[run_dev_id_idx] < -1)
				error("Subkernel(dev=%d,id=%d)-Tile(%d.[%d,%d])::writeback_data: invoked with tile with loc = %d\n",
					run_dev_id, id, tmp->id, tmp->GridId1, tmp->GridId2, tmp->CacheLocId[run_dev_id_idx]);
			else{
				Writeback_id = tmp->getWriteBackLoc(); //to MASTER
				Writeback_id_idx = idxize(Writeback_id);
				transfer_queues[Writeback_id_idx][run_dev_id_idx]->wait_for_event(operation_complete);
				if (run_dev_id == Writeback_id){
					;
#ifdef DEBUG
			lprintf(lvl, "Subkernel(dev=%d,id=%d)-Tile(%d.[%d,%d])::writeback_data: run_dev_id == Writeback_id == %d\n",
				run_dev_id, id, tmp->id, tmp->GridId1, tmp->GridId2, run_dev_id);
#endif
				}
				else{
					if (tmp->CacheLocId[run_dev_id_idx] != -1){
						CacheWrap_p wrap_inval = (CacheWrap_p) malloc(sizeof(struct Cache_info_wrap));
							wrap_inval->dev_id = run_dev_id;
							wrap_inval->BlockIdx = tmp->CacheLocId[run_dev_id_idx];
							wrap_inval->lock_flag = 1;
							transfer_queues[Writeback_id_idx][run_dev_id_idx]->add_host_func((void*)&CacheInvalidate, (void*) wrap_inval);
					}
#ifdef STEST
					output_timer[j]->start_point(transfer_queues[Writeback_id_idx][run_dev_id_idx]);
#endif
					CoCoMemcpy2DAsync(tmp->adrs[Writeback_id_idx], tmp->ldim[Writeback_id_idx],
						tmp->adrs[run_dev_id_idx], tmp->ldim[run_dev_id_idx],
						tmp->dim1, tmp->dim2, tmp->dtypesize(),
						Writeback_id, run_dev_id, transfer_queues[Writeback_id_idx][run_dev_id_idx]);
#ifdef STEST
					output_timer[j]->stop_point(transfer_queues[Writeback_id_idx][run_dev_id_idx]);
					bytes_out+= tmp->size();
#endif
					Ptr_and_int_p wrapped_op = (Ptr_and_int_p) malloc(sizeof(struct Ptr_and_int));
					wrapped_op->int_ptr = &tmp->RW_master;
					wrapped_op->val = Writeback_id;
					transfer_queues[Writeback_id_idx][run_dev_id_idx]->add_host_func((void*)&CoCoSetInt, (void*) wrapped_op);
				}
			}
		}
		else error("Subkernel(dev=%d,id=%d)::writeback_data: Not implemented for TileDim=%d\n", run_dev_id, id, TileDimlist[j]);
	}
	//writeback_complete->record_to_queue(transfer_queues[Writeback_id_idx][run_dev_id_idx]);
#ifndef ASYNC_ENABLE
	CoCoSyncCheckErr();
#endif
#ifdef DEBUG
	lprintf(lvl-1, "<-----|\n");
#endif
}


int queue_d_allock = 0;

void CoCoPeLiaInitResources(short dev_id){

	while(__sync_lock_test_and_set (&queue_d_allock, 1));

	// FIXME: Handmade distribution, for testing purposes
	transfer_link_sharing[0][LOC_NUM - 1][0] = 1;
	transfer_link_sharing[0][LOC_NUM - 1][1] = LOC_NUM - 1;
	transfer_link_sharing[1][LOC_NUM - 1][0] = 0;
	transfer_link_sharing[1][LOC_NUM - 1][1] = LOC_NUM - 1;

	transfer_link_sharing[2][LOC_NUM - 1][0] = 3;
	transfer_link_sharing[2][LOC_NUM - 1][1] = LOC_NUM - 1;
	transfer_link_sharing[3][LOC_NUM - 1][0] = 2;
	transfer_link_sharing[3][LOC_NUM - 1][1] = LOC_NUM - 1;

	transfer_link_sharing[4][LOC_NUM - 1][0] = 5;
	transfer_link_sharing[4][LOC_NUM - 1][1] = LOC_NUM - 1;
	transfer_link_sharing[5][LOC_NUM - 1][0] = 4;
	transfer_link_sharing[5][LOC_NUM - 1][1] = LOC_NUM - 1;

	transfer_link_sharing[6][LOC_NUM - 1][0] = 7;
	transfer_link_sharing[6][LOC_NUM - 1][1] = LOC_NUM - 1;
	transfer_link_sharing[7][LOC_NUM - 1][0] = 6;
	transfer_link_sharing[7][LOC_NUM - 1][1] = LOC_NUM - 1;

	short dev_id_idx = idxize(dev_id);
	for(short dev_id_idy = 0 ; dev_id_idy < LOC_NUM; dev_id_idy++){
		if (!transfer_queues[dev_id_idx][dev_id_idy]){
			if(transfer_link_sharing[dev_id_idx][dev_id_idy][0] != - 42 &&
				transfer_queues[transfer_link_sharing[dev_id_idx][dev_id_idy][0]]
				[transfer_link_sharing[dev_id_idx][dev_id_idy][1]] != NULL)
				transfer_queues[dev_id_idx][dev_id_idy] =
					transfer_queues[transfer_link_sharing[dev_id_idx][dev_id_idy][0]]
				[transfer_link_sharing[dev_id_idx][dev_id_idy][1]];
			else transfer_queues[dev_id_idx][dev_id_idy] = new CommandQueue(dev_id);
			}
		if (!transfer_queues[dev_id_idy][dev_id_idx]) transfer_queues[dev_id_idy][dev_id_idx]
			= new CommandQueue(deidxize(dev_id_idy));
	}
  if (!exec_queue[dev_id_idx])  exec_queue[dev_id_idx] = new CommandQueue(dev_id);

	__sync_lock_release(&queue_d_allock);
}

void CoCoPeLiaFreeResources(short dev_id){

	while(__sync_lock_test_and_set (&queue_d_allock, 1));

	short dev_id_idx = (dev_id == -1)?  LOC_NUM - 1 : dev_id;
	for(short dev_id_idy = 0 ; dev_id_idy < LOC_NUM; dev_id_idy++){
		if (transfer_queues[dev_id_idx][dev_id_idy]) {
			CQueue_p temp = transfer_queues[dev_id_idx][dev_id_idy];
			transfer_queues[dev_id_idx][dev_id_idy] = NULL;
			delete temp;
		}
		if (transfer_queues[dev_id_idy][dev_id_idx]) {
			CQueue_p temp = transfer_queues[dev_id_idy][dev_id_idx];
			transfer_queues[dev_id_idy][dev_id_idx] = NULL;
			delete temp;
		}
	}
	if (exec_queue[dev_id_idx]){
		delete exec_queue[dev_id_idx];
		exec_queue[dev_id_idx] = NULL;
	}

	__sync_lock_release(&queue_d_allock);
}

void Subkernel::prepare_launch(){
	for (int j = 0; j < TileNum; j++){
		if (TileDimlist[j] == 1){
			Tile1D<VALUE_TYPE>* tmp = (Tile1D<VALUE_TYPE>*) TileList[j];
			if(tmp->W_flag) {
				if (tmp->W_total == tmp->W_flag) WR_first = 1;
				CoCoQueueLock((void*) &tmp->RW_lock);
			}
		}
		else if (TileDimlist[j] == 2){
			Tile2D<VALUE_TYPE>* tmp = (Tile2D<VALUE_TYPE>*) TileList[j];
			if(tmp->W_flag) {
				if (tmp->W_total == tmp->W_flag) WR_first = 1;
				CoCoQueueLock((void*) &tmp->RW_lock);
			}
		}
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

short Subkernel::is_RW_master(short dev_id){
	for (int j = 0; j < TileNum; j++){
		if (TileDimlist[j] == 1){
			Tile1D<VALUE_TYPE>* tmp = (Tile1D<VALUE_TYPE>*) TileList[j];
			if(tmp->W_flag && tmp->W_total != tmp->W_flag && tmp->RW_master!=dev_id) return 0;
		}
		else if (TileDimlist[j] == 2){
			Tile2D<VALUE_TYPE>* tmp = (Tile2D<VALUE_TYPE>*) TileList[j];
			if(tmp->W_flag && tmp->W_total != tmp->W_flag && tmp->RW_master!=dev_id) return 0;
		}
	}
	return 1;
}

double Subkernel::opt_fetch_cost_pen_multifetch(short dev_id){
	double fetch_cost = 0;
	for (int j = 0; j < TileNum; j++){
		if (TileDimlist[j] == 1){
			Tile1D<VALUE_TYPE>* tmp = (Tile1D<VALUE_TYPE>*) TileList[j];
			error("not implemented\n");
		}
		else if (TileDimlist[j] == 2){
			Tile2D<VALUE_TYPE>* tmp = (Tile2D<VALUE_TYPE>*) TileList[j];
			double temp_fetch_cost = tmp->getMinLinkCost(dev_id);
			for(int loc_idx = 0; loc_idx < LOC_NUM; loc_idx++){
				if(tmp->CacheLocId[loc_idx] > -1 && CacheGetBlockStateNoLock(deidxize(loc_idx), tmp->CacheLocId[loc_idx]) == FETCHING){
					fetch_cost+=temp_fetch_cost*MULTIFETCH_PENALTY;
					if (deidxize(loc_idx) == dev_id) warning("opt_fetch_cost_pen_multifetch(dev_id=%d, TiledIdx=%d): Already fetching in dev_id (?)\n");
				}
			}
			fetch_cost+= temp_fetch_cost;
		}
	}
	return fetch_cost;
}

double Subkernel::opt_fetch_cost(short dev_id){
	double fetch_cost = 0;
	for (int j = 0; j < TileNum; j++){
		if (TileDimlist[j] == 1){
			Tile1D<VALUE_TYPE>* tmp = (Tile1D<VALUE_TYPE>*) TileList[j];
			error("not implemented\n");
		}
		else if (TileDimlist[j] == 2){
			Tile2D<VALUE_TYPE>* tmp = (Tile2D<VALUE_TYPE>*) TileList[j];
			fetch_cost+= tmp->getMinLinkCost(dev_id);
		}
	}
	return fetch_cost;
}

Subkernel* SubkernelSelectSimple(short dev_id, Subkernel** Subkernel_list, long Subkernel_list_len){
	Subkernel* curr_sk = NULL;
	long sk_idx;
	for (sk_idx = 0; sk_idx < Subkernel_list_len; sk_idx++){
		curr_sk = Subkernel_list[sk_idx];
		if (curr_sk->is_dependency_free()) break;

	}
	if(sk_idx==Subkernel_list_len) return NULL;
	curr_sk->prepare_launch();
	return curr_sk;
}

Subkernel* SubkernelSelectNoWriteShare(short dev_id, Subkernel** Subkernel_list, long Subkernel_list_len){
	Subkernel* curr_sk = NULL;
	long sk_idx;
	for (sk_idx = 0; sk_idx < Subkernel_list_len; sk_idx++){
		curr_sk = Subkernel_list[sk_idx];
		if (curr_sk->is_dependency_free() && curr_sk->is_RW_master(dev_id)) break;
	}
	if(sk_idx==Subkernel_list_len) return NULL;
	curr_sk->prepare_launch();
	return curr_sk;
}

Subkernel* SubkernelSelectMinimizeFetch(short dev_id, Subkernel** Subkernel_list, long Subkernel_list_len){
	Subkernel* curr_sk = NULL;
	long sk_idx;
	double min_fetch_cost = 100000000;
	Subkernel* min_fetch_cost_sk = NULL;
	for (sk_idx = 0; sk_idx < Subkernel_list_len; sk_idx++){
		curr_sk = Subkernel_list[sk_idx];
		if (curr_sk->is_dependency_free()){
			double fetch_cost = curr_sk->opt_fetch_cost(dev_id);
			if(fetch_cost < min_fetch_cost){
				min_fetch_cost = fetch_cost;
				min_fetch_cost_sk = curr_sk;
			}
		}
	}
	if(!min_fetch_cost_sk) return NULL;
	min_fetch_cost_sk->prepare_launch();
	return min_fetch_cost_sk;
}

Subkernel* SubkernelSelectMinimizeFetchWritePenalty(short dev_id, Subkernel** Subkernel_list, long Subkernel_list_len){
	Subkernel* curr_sk = NULL;
	long sk_idx;
	double min_fetch_cost = 100000000;
	Subkernel* min_fetch_cost_sk = NULL;
	for (sk_idx = 0; sk_idx < Subkernel_list_len; sk_idx++){
		curr_sk = Subkernel_list[sk_idx];
		if (curr_sk->is_dependency_free()){
			double fetch_cost = curr_sk->opt_fetch_cost(dev_id);
			if(!curr_sk->is_RW_master(dev_id)) fetch_cost+=WRITE_COST_PEPENALTY*fetch_cost;
			if(fetch_cost < min_fetch_cost){
				min_fetch_cost = fetch_cost;
				min_fetch_cost_sk = curr_sk;
			}
		}
	}
	if(!min_fetch_cost_sk) return NULL;
	min_fetch_cost_sk->prepare_launch();
	return min_fetch_cost_sk;
}

Subkernel* SubkernelSelectMinimizeFetchWritePenaltyMultiFetchPenalty(short dev_id, Subkernel** Subkernel_list, long Subkernel_list_len){
	Subkernel* curr_sk = NULL;
	long sk_idx;
	double min_fetch_cost = 100000000;
	Subkernel* min_fetch_cost_sk = NULL;
	for (sk_idx = 0; sk_idx < Subkernel_list_len; sk_idx++){
		curr_sk = Subkernel_list[sk_idx];
		if (curr_sk->is_dependency_free()){
			double fetch_cost = curr_sk->opt_fetch_cost_pen_multifetch(dev_id);
			if(!curr_sk->is_RW_master(dev_id)) fetch_cost+=WRITE_COST_PEPENALTY*fetch_cost;
			if(fetch_cost < min_fetch_cost){
				min_fetch_cost = fetch_cost;
				min_fetch_cost_sk = curr_sk;
			}
		}
	}
	if(!min_fetch_cost_sk) return NULL;
	min_fetch_cost_sk->prepare_launch();
	return min_fetch_cost_sk;
}
