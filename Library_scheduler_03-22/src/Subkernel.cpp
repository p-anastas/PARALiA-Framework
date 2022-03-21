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

int transfer_link_sharing[LOC_NUM][LOC_NUM][2];
CQueue_p transfer_queues[LOC_NUM][LOC_NUM] = {{NULL}}, exec_queue[LOC_NUM] = {NULL};
int Subkernel_ctr = 0, backend_init_flag[LOC_NUM] = {0};

Subkernel::Subkernel(short TileNum_in, const char* name){
	id = Subkernel_ctr;
	run_dev_id = -42;
	op_name = name;
	Subkernel_ctr++;
	TileNum = TileNum_in;
	TileDimlist = (short*) malloc(TileNum*sizeof(short));
	TileList = (void**) malloc(TileNum*sizeof(void*));
	prev = next = NULL;
//#ifdef STEST
//	bytes_in = bytes_out = flops = 0;
//#endif
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
				if (!tmp->W_flag && tmp->R_flag) tmp->StoreBlock[run_dev_id_idx]->Available->sync_barrier();
		}
		else if (TileDimlist[j] == 2){
				Tile2D<VALUE_TYPE>* tmp = (Tile2D<VALUE_TYPE>*) TileList[j];
				if (!tmp->W_flag && tmp->R_flag) tmp->StoreBlock[run_dev_id_idx]->Available->sync_barrier();
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
				if (tmp->R_flag) tmp->StoreBlock[run_dev_id_idx]->Available->sync_barrier();
		}
		else if (TileDimlist[j] == 2){
				Tile2D<VALUE_TYPE>* tmp = (Tile2D<VALUE_TYPE>*) TileList[j];
				if (tmp->R_flag) tmp->StoreBlock[run_dev_id_idx]->Available->sync_barrier();
		}
	}
#ifdef DEBUG
	lprintf(lvl-1, "<-----|\n");
#endif
}

typedef struct tile_req{
	Subkernel* sk;
	int TileIdx;
}* tile_req_p;

void* request_tile_pthread_wrap(void* wrapped_tile_req){
	tile_req_p unwrap = (tile_req_p) wrapped_tile_req;
	unwrap->sk->request_tile(unwrap->TileIdx);
	return NULL;
}

void Subkernel::request_tile(short TileIdx){
	#ifdef STEST
			request_tile_in_ts[TileIdx] = csecond();
	#endif
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
			tmp->RW_master = run_dev_id;
			if (FetchFromId == run_dev_id) error("Subkernel(dev=%d,id=%d)-Tile(%d.[%d])::request_tile W_flag = %d, \
			FetchFromId == run_dev_id == %d\n", run_dev_id, id, tmp->id,  tmp->GridId, tmp->W_flag, FetchFromId);
		}
		short FetchFromId_idx = idxize(FetchFromId);
#ifdef CDEBUG
		lprintf(lvl, "Subkernel(dev=%d,id=%d)-Tile(%d.[%d]): Fetching Block(%d) on GPU(%d) from Block(%d) on GPU(%d)\n",
			run_dev_id, id, tmp->id,  tmp->GridId, tmp->StoreBlock[run_dev_id_idx]->id, run_dev_id,
			tmp->StoreBlock[FetchFromId_idx]->id, FetchFromId);
#endif
		CBlock_wrap_p wrap_read = NULL;
		if (tmp->StoreBlock[FetchFromId_idx]->State == INVALID)
			error("Subkernel(dev=%d,id=%d)-Tile(%d.[%d])::request_tile: Fetching from tile in GPU(%d) with INVALID state\n",
				run_dev_id, id, tmp->id,  tmp->GridId, FetchFromId);

		if(tmp->W_flag) tmp->StoreBlock[FetchFromId_idx]->reset();
		else tmp->StoreBlock[FetchFromId_idx]->add_reader();

		if(tmp->W_flag) tmp->StoreBlock[run_dev_id_idx]->add_writer();
		else tmp->StoreBlock[run_dev_id_idx]->add_reader();

#ifdef STEST
		input_timer[TileIdx]->start_point(transfer_queues[run_dev_id_idx][FetchFromId_idx]);
#endif
		CoCoMemcpyAsync(tmp->StoreBlock[run_dev_id_idx]->Adrs, tmp->StoreBlock[FetchFromId_idx]->Adrs,
											((long long) tmp->inc[run_dev_id_idx]) * tmp->dim * tmp->dtypesize(),
											run_dev_id, FetchFromId, transfer_queues[run_dev_id_idx][FetchFromId_idx]);
#ifdef STEST
		input_timer[TileIdx]->stop_point(transfer_queues[run_dev_id_idx][FetchFromId_idx]);
		bytes_in[TileIdx] = tmp->size();
#endif
		if(tmp->W_flag);
		else{
			wrap_read = (CBlock_wrap_p) malloc (sizeof(struct CBlock_wrap));
			wrap_read->CBlock = tmp->StoreBlock[FetchFromId_idx];
			wrap_read->lock_flag = 0;
			transfer_queues[run_dev_id_idx][FetchFromId_idx]->add_host_func((void*)&CBlock_RR_wrap, (void*) wrap_read);
		}
		tmp->StoreBlock[run_dev_id_idx]->Available->record_to_queue(transfer_queues[run_dev_id_idx][FetchFromId_idx]);
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
			tmp->RW_master = run_dev_id;
			if (FetchFromId == run_dev_id) error("Subkernel(dev=%d,id=%d)-Tile(%d.[%d,%d])::request_tile W_flag = %d, \
			FetchFromId == run_dev_id == %d\n", run_dev_id, id, tmp->id, tmp->GridId1, tmp->GridId2, tmp->W_flag, FetchFromId);
		}
		short FetchFromId_idx = idxize(FetchFromId);
#ifdef CDEBUG
		lprintf(lvl, "Subkernel(dev=%d,id=%d)-Tile(%d.[%d,%d]): Fetching Block(%d) on GPU(%d) from Block(%d) on GPU(%d)\n",
			run_dev_id, id, tmp->id,  tmp->GridId1, tmp->GridId2, tmp->StoreBlock[run_dev_id_idx]->id, run_dev_id,
			tmp->StoreBlock[FetchFromId_idx]->id, FetchFromId);
#endif
		CBlock_wrap_p wrap_read = NULL;
		if (tmp->StoreBlock[FetchFromId_idx]->State == INVALID)
			error("Subkernel(dev=%d,id=%d)-Tile(%d.[%d,%d])::request_tile: Fetching from tile in GPU(%d) with INVALID state\n",
				run_dev_id, id, tmp->id,  tmp->GridId1, tmp->GridId2, FetchFromId);

		if(tmp->W_flag) tmp->StoreBlock[FetchFromId_idx]->reset();
		else tmp->StoreBlock[FetchFromId_idx]->add_reader();

		if(tmp->W_flag) tmp->StoreBlock[run_dev_id_idx]->add_writer();
		else tmp->StoreBlock[run_dev_id_idx]->add_reader();

#ifdef STEST
		input_timer[TileIdx]->start_point(transfer_queues[run_dev_id_idx][FetchFromId_idx]);
#endif
		CoCoMemcpy2DAsync(tmp->StoreBlock[run_dev_id_idx]->Adrs, tmp->ldim[run_dev_id_idx],
									tmp->StoreBlock[FetchFromId_idx]->Adrs, tmp->ldim[FetchFromId_idx],
									tmp->dim1, tmp->dim2, tmp->dtypesize(),
									run_dev_id, FetchFromId, transfer_queues[run_dev_id_idx][FetchFromId_idx]);
#ifdef STEST
		input_timer[TileIdx]->stop_point(transfer_queues[run_dev_id_idx][FetchFromId_idx]);
		bytes_in[TileIdx]= tmp->size();
#endif
		if(tmp->W_flag);
		else{
			wrap_read = (CBlock_wrap_p) malloc (sizeof(struct CBlock_wrap));
			wrap_read->CBlock = tmp->StoreBlock[FetchFromId_idx];
			wrap_read->lock_flag = 0;
			transfer_queues[run_dev_id_idx][FetchFromId_idx]->add_host_func((void*)&CBlock_RR_wrap, (void*) wrap_read);
		}
		tmp->StoreBlock[run_dev_id_idx]->Available->record_to_queue(transfer_queues[run_dev_id_idx][FetchFromId_idx]);
	}
#ifdef STEST
		request_tile_out_ts[TileIdx] = csecond();
#endif
}

void Subkernel::request_data(){
	short lvl = 4;
#ifdef DEBUG
	lprintf(lvl-1, "|-----> Subkernel(dev=%d,id=%d)::request_data()\n", run_dev_id, id);
#endif
#ifdef STEST
		request_data_in_ts = csecond();
#endif
#ifdef ENABLE_PTHREAD_TILE_REQUEST
	pthread_t thread_id[TileNum];
	short requested_tiles = 0;
	pthread_attr_t attr;
	int s = pthread_attr_init(&attr);
	if (s != 0) error("Subkernel(dev=%d,id=%d)::request_data() - pthread_attr_init failed s=%d\n",
	run_dev_id, id, s);
#endif
	short run_dev_id_idx = idxize(run_dev_id);
	for (int j = 0; j < TileNum; j++){
		if (TileDimlist[j] == 1){
			Tile1D<VALUE_TYPE>* tmp = (Tile1D<VALUE_TYPE>*) TileList[j];
			if (tmp->StoreBlock[run_dev_id_idx] == NULL) {
				tmp->StoreBlock[run_dev_id_idx] = Global_Cache[run_dev_id_idx]->assign_Cblock();
				tmp->StoreBlock[run_dev_id_idx]->set_owner((void**)&tmp->StoreBlock[run_dev_id_idx]);
#ifdef CDEBUG
		lprintf(lvl, "Subkernel(dev=%d,id=%d)-Tile(%d.[%d]): Asigned buffer Block in GPU(%d)= %d\n",
					run_dev_id, id, tmp->id, tmp->GridId, run_dev_id, tmp->StoreBlock[run_dev_id_idx]->id);
#endif
#ifdef ENABLE_PTHREAD_TILE_REQUEST
					if (tmp->R_flag) {
						tile_req_p wrap_request = (tile_req_p) malloc(sizeof(struct tile_req));
						wrap_request->sk = this;
						wrap_request->TileIdx = j;
						s = pthread_create(&thread_id[requested_tiles], &attr, &request_tile_pthread_wrap,
							wrap_request);
						requested_tiles++;
					}
#else
					if (tmp->R_flag) request_tile(j);
#endif
			}
			else{
				tmp->StoreBlock[run_dev_id_idx]->Available->sync_barrier(); // Is this needed?
				if (tmp->W_flag) tmp->StoreBlock[run_dev_id_idx]->add_writer();
				else tmp->StoreBlock[run_dev_id_idx]->add_reader();
			}
		}
		else if (TileDimlist[j] == 2){
			Tile2D<VALUE_TYPE>* tmp = (Tile2D<VALUE_TYPE>*) TileList[j];
			if (tmp->StoreBlock[run_dev_id_idx] == NULL) {
				tmp->StoreBlock[run_dev_id_idx] = Global_Cache[run_dev_id_idx]->assign_Cblock();
				tmp->StoreBlock[run_dev_id_idx]->set_owner((void**)&tmp->StoreBlock[run_dev_id_idx]);
#ifdef CDEBUG
		lprintf(lvl, "Subkernel(dev=%d,id=%d)-Tile(%d.[%d,%d]): Asigned buffer Block in GPU(%d)= %d\n",
					run_dev_id, id, tmp->id, tmp->GridId1, tmp->GridId2, run_dev_id, tmp->StoreBlock[run_dev_id_idx]->id);
#endif
#ifdef ENABLE_PTHREAD_TILE_REQUEST
					if (tmp->R_flag) {
						tile_req_p wrap_request = (tile_req_p) malloc(sizeof(struct tile_req));
						wrap_request->sk = this;
						wrap_request->TileIdx = j;
						s = pthread_create(&thread_id[requested_tiles], &attr, &request_tile_pthread_wrap,
							wrap_request);
						requested_tiles++;
					}
#else
					if (tmp->R_flag) request_tile(j);
#endif
			}
			else{
				tmp->StoreBlock[run_dev_id_idx]->Available->sync_barrier(); // Is this needed?
				if (tmp->W_flag) tmp->StoreBlock[run_dev_id_idx]->add_writer();
				else tmp->StoreBlock[run_dev_id_idx]->add_reader();
			}
		}
		else error("Subkernel(dev=%d,id=%d)::request_data: Not implemented for TileDim=%d\n",
			run_dev_id, id, TileDimlist[j]);
	}
#ifndef ASYNC_ENABLE
	CoCoSyncCheckErr();
#endif
#ifdef ENABLE_PTHREAD_TILE_REQUEST
void* res;
for(int i=0; i<requested_tiles;i++){
	s = pthread_join(thread_id[i], &res);
	if (s != 0) error("Subkernel(dev=%d,id=%d)::request_data() - pthread_join failed with exit value %d\n",
	run_dev_id, id, s);
	//free(res);      /* Free memory allocated by thread */
}
#endif
#ifdef STEST
		request_data_out_ts = csecond();
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
#ifdef STEST
		run_operation_in_ts = csecond();
#endif
	short run_dev_id_idx = idxize(run_dev_id);
#ifdef ENABLE_PARALLEL_BACKEND
	short RW_parallel_backend_ctr = exec_queue[run_dev_id_idx]->request_parallel_backend();
#endif
	for (int j = 0; j < TileNum; j++){
		if (TileDimlist[j] == 1){
				Tile1D<VALUE_TYPE>* tmp = (Tile1D<VALUE_TYPE>*) TileList[j];
				if (tmp->StoreBlock[run_dev_id_idx] == NULL)
					error("Subkernel(dev=%d,id=%d)-Tile(%d.[%d])::run_operation: Tile(j=%d) Storeblock is NULL\n",
						run_dev_id, id, tmp->id, tmp->GridId, j);
				if(tmp->R_flag) exec_queue[run_dev_id_idx]->wait_for_event(tmp->StoreBlock[run_dev_id_idx]->Available);
		}
		else if (TileDimlist[j] == 2){
				Tile2D<VALUE_TYPE>* tmp = (Tile2D<VALUE_TYPE>*) TileList[j];
				if (tmp->StoreBlock[run_dev_id_idx] == NULL)
					error("Subkernel(dev=%d,id=%d)-Tile(%d.[%d,%d])::run_operation: Tile(j=%d) Storeblock is NULL\n",
						run_dev_id, id, tmp->id, tmp->GridId1, tmp->GridId2, j);
				if(tmp->R_flag) exec_queue[run_dev_id_idx]->wait_for_event(tmp->StoreBlock[run_dev_id_idx]->Available);
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
			if (tmp->StoreBlock[run_dev_id_idx] == NULL)
				error("Subkernel(dev=%d,id=%d)-Tile(%d.[%d])::run_operation: Tile(j=%d) Storeblock is NULL\n",
					run_dev_id, id, tmp->id, tmp->GridId, j);
			CBlock_wrap_p wrap_oper = NULL;
			wrap_oper = (CBlock_wrap_p) malloc (sizeof(struct CBlock_wrap));
			wrap_oper->CBlock = tmp->StoreBlock[run_dev_id_idx];
			wrap_oper->lock_flag = 0;
			if(tmp->W_flag)	exec_queue[run_dev_id_idx]->add_host_func((void*)&CBlock_RW_wrap, (void*) wrap_oper);
			else exec_queue[run_dev_id_idx]->add_host_func((void*)&CBlock_RR_wrap, (void*) wrap_oper);

			if(tmp->R_flag) tmp->R_flag--;
			if(tmp->W_flag){
				tmp->W_flag--;
				if(!tmp->W_flag) WR_last[j] = 1;
			}
		}
		else if (TileDimlist[j] == 2){
			Tile2D<VALUE_TYPE>* tmp = (Tile2D<VALUE_TYPE>*) TileList[j];
			if (tmp->StoreBlock[run_dev_id_idx] == NULL)
				error("Subkernel(dev=%d,id=%d)-Tile(%d.[%d,%d])::run_operation: Tile(j=%d) Storeblock is NULL\n",
					run_dev_id, id, tmp->id, tmp->GridId1, tmp->GridId2, j);
			CBlock_wrap_p wrap_oper = NULL;
			wrap_oper = (CBlock_wrap_p) malloc (sizeof(struct CBlock_wrap));
			wrap_oper->CBlock = tmp->StoreBlock[run_dev_id_idx];
			wrap_oper->lock_flag = 0;
			if(tmp->W_flag)	exec_queue[run_dev_id_idx]->add_host_func((void*)&CBlock_RW_wrap, (void*) wrap_oper);
			else exec_queue[run_dev_id_idx]->add_host_func((void*)&CBlock_RR_wrap, (void*) wrap_oper);

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
#ifdef STEST
		run_operation_out_ts = csecond();
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
#ifdef STEST
		writeback_data_in_ts = csecond();
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
			lprintf(lvl, "Subkernel(dev=%d,id=%d)-Tile(%d.[%d])::writeback_data: run_dev_id == Writeback_id == %d (not wrong but check)\n",
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
					bytes_out[j]= tmp->size();
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
					bytes_out[j]= tmp->size();
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
#ifdef STEST
		writeback_data_out_ts = csecond();
#endif
#ifdef DEBUG
	lprintf(lvl-1, "<-----|\n");
#endif
}


int queue_d_allock = 0;

void CoCoPeLiaInitResources(short dev_id){
short lvl = 2;
	//while(__sync_lock_test_and_set (&queue_d_allock, 1));

	for(int i = 0; i < LOC_NUM; i++)
	for(int j = 0; j < LOC_NUM; j++)
	for(int k = 0; k < 2; k++) transfer_link_sharing[i][j][k] = -42;

/*
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

	transfer_link_sharing[LOC_NUM - 1][0][0] = LOC_NUM - 1;
	transfer_link_sharing[LOC_NUM - 1][0][1] = 1;
	transfer_link_sharing[LOC_NUM - 1][1][0] = LOC_NUM - 1;
	transfer_link_sharing[LOC_NUM - 1][1][1] = 0;

	transfer_link_sharing[LOC_NUM - 1][2][0] = LOC_NUM - 1;
	transfer_link_sharing[LOC_NUM - 1][2][1] = 3;
	transfer_link_sharing[LOC_NUM - 1][3][0] = LOC_NUM - 1;
	transfer_link_sharing[LOC_NUM - 1][3][1] = 2;

	transfer_link_sharing[LOC_NUM - 1][4][0] = LOC_NUM - 1;
	transfer_link_sharing[LOC_NUM - 1][4][1] = 5;
	transfer_link_sharing[LOC_NUM - 1][5][0] = LOC_NUM - 1;
	transfer_link_sharing[LOC_NUM - 1][5][1] = 4;

	transfer_link_sharing[LOC_NUM - 1][6][0] = LOC_NUM - 1;
	transfer_link_sharing[LOC_NUM - 1][6][1] = 7;
	transfer_link_sharing[LOC_NUM - 1][7][0] = LOC_NUM - 1;
	transfer_link_sharing[LOC_NUM - 1][7][1] = 6;

	typedef struct {
	  kaapi_device_t inherited;
	  CUdevice       cu_device;
	  CUcontext      ctx;
	  uint64_t*      affinity; // of size cuda_count_perfrank -1
	  size_t         free_mem;
	  size_t         size_alloc;
	  size_t         size_free;
	  size_t         mem_limit;

	  struct {
	    bool overlap;      // if the device can concurrently copy memory between host and device while executing a kernel
	    bool integrated;   // if the device is integrated with the memory subsystem
	    bool map;          // if the device can map host memory into the CUDA address space
	    bool concurrent;   // if the device supports executing multiple kernels within the same context simultaneously
	    int async_engines; // Number of asynchronous engines
	    size_t mem_total;  //total GPU memory size in bytes
	    char name[64];     // GPU name
	  } prop;
	} kaapi_device_cuda_t;

  int pi;
	CUresult res;
	CUcontext ctx;
	CUdevice cu_device;
	res = cuDeviceGet(&cu_device, dev_id);
	CoCoASyncCheckErr();
	res = cuCtxCreate(&ctx, CU_CTX_SCHED_AUTO, cu_device);
	CoCoASyncCheckErr();

	res = cuDeviceGetAttribute (&pi, CU_DEVICE_ATTRIBUTE_GPU_OVERLAP, cu_device);
	CudaCheckError(res);
	//device->prop.overlap = pi;

	res = cuDeviceGetAttribute (&pi, CU_DEVICE_ATTRIBUTE_INTEGRATED, cu_device);
	CudaCheckError(res);
	//device->prop.integrated = pi;

	res = cuDeviceGetAttribute (&pi, CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY, cu_device);
	CudaCheckError(res);
	//device->prop.map = pi;

	res = cuDeviceGetAttribute (&pi, CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS, cu_device);
	CudaCheckError(res);
	//device->prop.concurrent = pi;

	//res = cuDeviceTotalMem(&device->prop.mem_total, cu_device);
	//CudaCheckError(res);

	//memset(device->prop.name, 0, 64*sizeof(char));
	//res = cuDeviceGetName(device->prop.name, 64, cu_device);
	//CudaCheckError(res);

	res = cuDeviceGetAttribute (&pi, CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT, cu_device);
	if(res != CUDA_SUCCESS)
		pi = 1;
	int async_engines = pi;
//#ifdef DEBUG
	lprintf(lvl, "CoCoPeLiaInitResources(dev=%d): Allows %d async engines == %d\n",
		dev_id, async_engines);
//#endif
	res = cuCtxDestroy(ctx);
*/
	short dev_id_idx = idxize(dev_id);
	for(short dev_id_idy = 0 ; dev_id_idy < LOC_NUM; dev_id_idy++)
	if(dev_id_idy!=dev_id_idx){
		if (!transfer_queues[dev_id_idx][dev_id_idy]){
			short shared_iloc0 = transfer_link_sharing[dev_id_idx][dev_id_idy][0],
				shared_iloc1 = transfer_link_sharing[dev_id_idx][dev_id_idy][1];
			if( shared_iloc0 != - 42){ // The smallest index shared link allocates the queue
				if (dev_id_idx*LOC_NUM + dev_id_idy < shared_iloc0*LOC_NUM + shared_iloc1){
					transfer_queues[dev_id_idx][dev_id_idy] = new CommandQueue(dev_id);
					transfer_queues[shared_iloc0][shared_iloc1] = transfer_queues[dev_id_idx][dev_id_idy];
				}
			}
			else transfer_queues[dev_id_idx][dev_id_idy] = new CommandQueue(dev_id);
		}
		if (!transfer_queues[dev_id_idy][dev_id_idx]){
			short shared_iloc0 = transfer_link_sharing[dev_id_idy][dev_id_idx][0],
				shared_iloc1 = transfer_link_sharing[dev_id_idy][dev_id_idx][1];
			if( shared_iloc0 != - 42){ // The smallest index shared link allocates the queue
				if (dev_id_idy*LOC_NUM + dev_id_idx < shared_iloc0*LOC_NUM + shared_iloc1){
					short writeback_queue_id = (dev_id_idy == LOC_NUM - 1)? dev_id : deidxize(dev_id_idy);
					transfer_queues[dev_id_idy][dev_id_idx] = new CommandQueue(writeback_queue_id);
					transfer_queues[shared_iloc0][shared_iloc1] = transfer_queues[dev_id_idy][dev_id_idx];
				}
			}
			else{
				short writeback_queue_id = (dev_id_idy == LOC_NUM - 1)? dev_id : deidxize(dev_id_idy);
				transfer_queues[dev_id_idy][dev_id_idx] = new CommandQueue(writeback_queue_id);
			}
		}
	}
  if (!exec_queue[dev_id_idx])  exec_queue[dev_id_idx] = new CommandQueue(dev_id);

	//__sync_lock_release(&queue_d_allock);
}

void CoCoPeLiaFreeResources(short dev_id){

	//while(__sync_lock_test_and_set (&queue_d_allock, 1));

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

	//__sync_lock_release(&queue_d_allock);
}

void Subkernel::prepare_launch(){
	for (int j = 0; j < TileNum; j++){
		if (TileDimlist[j] == 1){
			Tile1D<VALUE_TYPE>* tmp = (Tile1D<VALUE_TYPE>*) TileList[j];
			if(tmp->W_flag) {
				if (tmp->W_total == tmp->W_flag) WR_first = 1;
				//if(!tmp->isLocked())
					CoCoQueueLock((void*) &tmp->RW_lock);
			}
		}
		else if (TileDimlist[j] == 2){
			Tile2D<VALUE_TYPE>* tmp = (Tile2D<VALUE_TYPE>*) TileList[j];
			if(tmp->W_flag) {
				if (tmp->W_total == tmp->W_flag) WR_first = 1;
				//if(!tmp->isLocked())
					CoCoQueueLock((void*) &tmp->RW_lock);
			}
		}
	}
}

short Subkernel::no_locked_tiles(){
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
		if (curr_sk->no_locked_tiles()) break;

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
		if (curr_sk->no_locked_tiles() && curr_sk->is_RW_master(dev_id)) break;
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
		if (curr_sk->no_locked_tiles()){// || curr_sk->is_RW_master(dev_id)){
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
		if (curr_sk->no_locked_tiles()){// || curr_sk->is_RW_master(dev_id)){
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
		if (curr_sk->no_locked_tiles()){// || curr_sk->is_RW_master(dev_id)){
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
