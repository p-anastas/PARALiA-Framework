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

#include <cfloat>

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
#ifdef STEST
	for (int i = 0; i < 3; i++){
		dev_in_from[i] = dev_in_to[i] = dev_out_from[i] = dev_out_to[i] = -2;
	}
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
}

void Subkernel::init_events(){
	operation_complete = new Event(run_dev_id);
}

void Subkernel::sync_request_data_RONLY(){
	short lvl = 4;
#ifdef DEBUG
	lprintf(lvl-1, "|-----> Subkernel(dev=%d,id=%d)::sync_request_data_RONLY()\n", run_dev_id, id);
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

#ifdef ENABLE_TRANSFER_HOPS
void Subkernel::request_tile_hops(short TileIdx){
	#ifdef STEST
			reqT_fire_ts[TileIdx] = csecond();
	#endif
	short lvl = 5;
	short run_dev_id_idx = idxize(run_dev_id);
	if (TileDimlist[TileIdx] == 1){
		warning("Subkernel::request_tile_hops() not implemented for 1D, calling Subkernel::request_tile\n");
		request_tile(TileIdx);
	}
	else if (TileDimlist[TileIdx] == 2){
			Tile2D<VALUE_TYPE>* tmp = (Tile2D<VALUE_TYPE>*) TileList[TileIdx];
#ifdef DEBUG
		lprintf(lvl-1, "|-----> Subkernel(dev=%d,id=%d)::request_tile_hops(Tile(%d.[%d,%d]))\n",
			run_dev_id, id, tmp->id, tmp->GridId1, tmp->GridId2);
#endif
		short FetchFromId = -42;
		if(!tmp->W_flag) FetchFromId = tmp->getClosestReadLoc(run_dev_id);
		else {
			FetchFromId = tmp->RW_master;
			tmp->StoreBlock[idxize(FetchFromId)]->add_reader();
			tmp->RW_master = run_dev_id;
		}
		if (FetchFromId == run_dev_id) error("Subkernel(dev=%d,id=%d)-Tile(%d.[%d,%d])::request_tile_hops W_flag = %d, \
			FetchFromId == run_dev_id == %d, state[%d] == %s\n",  run_dev_id, id, tmp->id, tmp->GridId1, tmp->GridId2,
			tmp->W_flag, FetchFromId, FetchFromId, print_state(tmp->StoreBlock[idxize(FetchFromId)]->State));
		short FetchFromId_idx = idxize(FetchFromId);
#ifdef DEBUG
		lprintf(lvl, "Subkernel(dev=%d,id=%d)-Tile(%d.[%d,%d]): Fetching Block(%d) on GPU(%d) from Block(%d) on GPU(%d)\n",
			run_dev_id, id, tmp->id,  tmp->GridId1, tmp->GridId2, tmp->StoreBlock[run_dev_id_idx]->id, run_dev_id,
			tmp->StoreBlock[FetchFromId_idx]->id, FetchFromId);
#endif
		CBlock_wrap_p wrap_read = NULL;
		if (tmp->StoreBlock[FetchFromId_idx]->State == INVALID)
			error("Subkernel(dev=%d,id=%d)-Tile(%d.[%d,%d])::request_tile_hops: Fetching from tile in GPU(%d) with INVALID state\n",
				run_dev_id, id, tmp->id,  tmp->GridId1, tmp->GridId2, FetchFromId);

		CQueue_p used_queue;
#ifndef ENABLE_TRANSFER_W_HOPS
		if(final_estimated_linkmap->link_hop_num[run_dev_id_idx][FetchFromId_idx] == 0 || tmp->W_flag){
#else
		if(final_estimated_linkmap->link_hop_num[run_dev_id_idx][FetchFromId_idx] == 0){
#endif
			used_queue = transfer_queues[run_dev_id_idx][FetchFromId_idx];
			used_queue->wait_for_event(tmp->StoreBlock[FetchFromId_idx]->Available);
#ifdef STEST
			used_queue->add_host_func((void*)&CoCoSetTimerAsync, (void*) &reqT_start_ts[TileIdx]);
#endif
			CoCoMemcpy2DAsync(tmp->StoreBlock[run_dev_id_idx]->Adrs, tmp->ldim[run_dev_id_idx],
										tmp->StoreBlock[FetchFromId_idx]->Adrs, tmp->ldim[FetchFromId_idx],
										tmp->dim1, tmp->dim2, tmp->dtypesize(),
										run_dev_id, FetchFromId, used_queue);
#ifdef STEST
			used_queue->add_host_func((void*)&CoCoSetTimerAsync, (void*) &reqT_end_ts[TileIdx]);
			bytes_in[TileIdx]= tmp->size();
			dev_in_from[TileIdx] = FetchFromId;
			dev_in_to[TileIdx] = run_dev_id;
#endif
		}
		else{
			link_road_p test_road = (link_road_p) malloc(sizeof(struct link_road));
			int inter_hop_num = final_estimated_linkmap->link_hop_num[run_dev_id_idx][FetchFromId_idx];
			test_road->hop_num = 2 + inter_hop_num;
			test_road->hop_uid_list[0] = FetchFromId;
			test_road->hop_uid_list[1 + inter_hop_num] = run_dev_id;

			test_road->hop_ldim_list[0] = tmp->ldim[FetchFromId_idx];
			test_road->hop_ldim_list[1 + inter_hop_num] = tmp->ldim[run_dev_id_idx];

			test_road->hop_buf_list[0] = tmp->StoreBlock[FetchFromId_idx]->Adrs;
			test_road->hop_buf_list[1 + inter_hop_num] = tmp->StoreBlock[run_dev_id_idx]->Adrs;
			short selected_route = rand() % (final_estimated_linkmap->link_hop_route_num[run_dev_id_idx][FetchFromId_idx] - 0) + 0;
			CBlock_p block_ptr[inter_hop_num] = {NULL};
			for(int inter_hop = 0 ; inter_hop < inter_hop_num; inter_hop++){
				test_road->hop_uid_list[1+ inter_hop] = final_estimated_linkmap->link_hop_route[run_dev_id_idx][FetchFromId_idx][selected_route][inter_hop];
				test_road->hop_ldim_list[1+ inter_hop] = tmp->ldim[run_dev_id_idx];
				test_road->hop_cqueue_list[inter_hop] = transfer_queues[idxize(test_road->hop_uid_list[1+inter_hop])][idxize(test_road->hop_uid_list[inter_hop])];

				if (!tmp->W_flag && tmp->StoreBlock[idxize(test_road->hop_uid_list[1+inter_hop])] != NULL &&
					tmp->StoreBlock[idxize(test_road->hop_uid_list[1+inter_hop])]->State != INVALID)
					error("Subkernel(dev=%d,id=%d)-Tile(%d.[%d,%d])::request_tile_hops W_flag = %d, \
						FetchFromId = %d, run_dev_id = %d, hop_id = %d already cached in loc\n",  run_dev_id, id, tmp->id, tmp->GridId1, tmp->GridId2,
						tmp->W_flag, FetchFromId, run_dev_id, test_road->hop_uid_list[1+ inter_hop]);

				state new_block_state;
				if(!tmp->W_flag){
					new_block_state = SHARABLE;
					if(tmp->StoreBlock[idxize(test_road->hop_uid_list[1+inter_hop])] != NULL)
						tmp->StoreBlock[idxize(test_road->hop_uid_list[1+inter_hop])]->Owner_p = NULL;
					block_ptr[inter_hop] = tmp->StoreBlock[idxize(test_road->hop_uid_list[1+inter_hop])] =
						Global_Cache[idxize(test_road->hop_uid_list[1+inter_hop])]->assign_Cblock(new_block_state,false);
					block_ptr[inter_hop]->set_owner((void**)&tmp->StoreBlock[idxize(test_road->hop_uid_list[1+inter_hop])],false);
				}
				else{
					new_block_state = EXCLUSIVE;
					block_ptr[inter_hop] = Global_Cache[idxize(test_road->hop_uid_list[1+inter_hop])]->assign_Cblock(new_block_state,false);
					/// FIXME: Should never writeback hop-tiles... but what if it tries?
				  //block_ptr[inter_hop]->init_writeback_info(tmp->WriteBackBlock, &(tmp->RW_master), tmp->dim1, tmp->dim2,
					//tmp->ldim[idxize(test_road->hop_uid_list[1+inter_hop])], tmp->ldim[idxize(tmp->WriteBackLoc)], tmp->dtypesize(),
					//transfer_queues[idxize(tmp->getWriteBackLoc())][idxize(test_road->hop_uid_list[1+inter_hop])], false);
				}
				test_road->hop_buf_list[1 + inter_hop] = block_ptr[inter_hop]->Adrs;
				test_road->hop_event_list[inter_hop] = block_ptr[inter_hop]->Available;
			}

			test_road->hop_cqueue_list[0]->wait_for_event(tmp->StoreBlock[FetchFromId_idx]->Available);

			used_queue = test_road->hop_cqueue_list[inter_hop_num] =
				transfer_queues[idxize(test_road->hop_uid_list[1+inter_hop_num])][idxize(test_road->hop_uid_list[inter_hop_num])];
			test_road->hop_event_list[inter_hop_num] = tmp->StoreBlock[run_dev_id_idx]->Available;
#ifdef STEST
			used_queue->add_host_func((void*)&CoCoSetTimerAsync, (void*) &reqT_start_ts[TileIdx]);
#endif
#ifdef PDEBUG
			lprintf(1, "Subkernel(dev=%d,id=%d)-Tile(%d.[%d,%d])::request_tile_hops W_flag = %d, \
				%2d->%2d transfer sequence -> %s (route = %d)\n", run_dev_id, id, tmp->id, tmp->GridId1, tmp->GridId2,
				tmp->W_flag, FetchFromId, run_dev_id,
				printlist(final_estimated_linkmap->link_hop_route[run_dev_id_idx][FetchFromId_idx][selected_route],
					final_estimated_linkmap->link_hop_num[run_dev_id_idx][FetchFromId_idx]), selected_route);
#endif
			FasTCoCoMemcpy2DAsync(test_road, tmp->dim1, tmp->dim2, tmp->dtypesize());
#ifdef STEST
			used_queue->add_host_func((void*)&CoCoSetTimerAsync, (void*) &reqT_end_ts[TileIdx]);
			bytes_in[TileIdx]= tmp->size();
			dev_in_from[TileIdx] = FetchFromId;
			dev_in_to[TileIdx] = run_dev_id;
#endif
			if(tmp->W_flag){
				//if (tmp->StoreBlock[FetchFromId_idx]!= tmp->WriteBackBlock){
					for(int inter_hop = 0 ; inter_hop < inter_hop_num; inter_hop++){
						CBlock_wrap_p wrap_inval = NULL;
						wrap_inval = (CBlock_wrap_p) malloc (sizeof(struct CBlock_wrap));
						wrap_inval->lockfree = false;
						wrap_inval->CBlock = block_ptr[inter_hop];
						test_road->hop_cqueue_list[inter_hop+1]->add_host_func((void*)&CBlock_RW_INV_wrap, (void*) wrap_inval);
					}
				//}
			}
		}
		if(tmp->W_flag){
			CBlock_wrap_p wrap_inval = NULL;
			wrap_inval = (CBlock_wrap_p) malloc (sizeof(struct CBlock_wrap));
			wrap_inval->lockfree = false;
			wrap_inval->CBlock = tmp->StoreBlock[FetchFromId_idx];
			used_queue->add_host_func((void*)&CBlock_RR_INV_wrap, (void*) wrap_inval);
		}
		else{
			wrap_read = (CBlock_wrap_p) malloc (sizeof(struct CBlock_wrap));
			wrap_read->CBlock = tmp->StoreBlock[FetchFromId_idx];
			wrap_read->lockfree = false;
			used_queue->add_host_func((void*)&CBlock_RR_wrap, (void*) wrap_read);
		}
		tmp->StoreBlock[run_dev_id_idx]->Available->record_to_queue(used_queue);
	}
}
#endif

void Subkernel::request_tile(short TileIdx){
	#ifdef STEST
			reqT_fire_ts[TileIdx] = csecond();
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
			tmp->StoreBlock[idxize(FetchFromId)]->add_reader();
			tmp->RW_master = run_dev_id;
		}
		if (FetchFromId == run_dev_id) error("Subkernel(dev=%d,id=%d)-Tile(%d.[%d])::request_tile W_flag = %d, \
		FetchFromId == run_dev_id == %d\n", run_dev_id, id, tmp->id,  tmp->GridId, tmp->W_flag, FetchFromId);
		short FetchFromId_idx = idxize(FetchFromId);
#ifdef DEBUG
		lprintf(lvl, "Subkernel(dev=%d,id=%d)-Tile(%d.[%d]): Fetching Block(%d) on GPU(%d) from Block(%d) on GPU(%d)\n",
			run_dev_id, id, tmp->id,  tmp->GridId, tmp->StoreBlock[run_dev_id_idx]->id, run_dev_id,
			tmp->StoreBlock[FetchFromId_idx]->id, FetchFromId);
#endif
		CBlock_wrap_p wrap_read = NULL;
		if (tmp->StoreBlock[FetchFromId_idx]->State == INVALID)
			error("Subkernel(dev=%d,id=%d)-Tile(%d.[%d])::request_tile: Fetching from tile in GPU(%d) with INVALID state\n",
				run_dev_id, id, tmp->id,  tmp->GridId, FetchFromId);

		transfer_queues[run_dev_id_idx][FetchFromId_idx]->wait_for_event(tmp->StoreBlock[FetchFromId_idx]->Available);
#ifdef STEST
		transfer_queues[run_dev_id_idx][FetchFromId_idx]->add_host_func((void*)&CoCoSetTimerAsync, (void*) &reqT_start_ts[TileIdx]);
#endif
		CoCoMemcpyAsync(tmp->StoreBlock[run_dev_id_idx]->Adrs, tmp->StoreBlock[FetchFromId_idx]->Adrs,
											((long long) tmp->inc[run_dev_id_idx]) * tmp->dim * tmp->dtypesize(),
											run_dev_id, FetchFromId, transfer_queues[run_dev_id_idx][FetchFromId_idx]);
#ifdef STEST
		transfer_queues[run_dev_id_idx][FetchFromId_idx]->add_host_func((void*)&CoCoSetTimerAsync, (void*) &reqT_end_ts[TileIdx]);
		bytes_in[TileIdx] = tmp->size();
		dev_in_from[TileIdx] = FetchFromId;
		dev_in_to[TileIdx] = run_dev_id;
#endif
		if(tmp->W_flag){
			//if (tmp->StoreBlock[FetchFromId_idx]!= tmp->WriteBackBlock){
				CBlock_wrap_p wrap_inval = NULL;
				wrap_inval = (CBlock_wrap_p) malloc (sizeof(struct CBlock_wrap));
				wrap_inval->CBlock = tmp->StoreBlock[FetchFromId_idx];
				wrap_inval->lockfree = false;
				transfer_queues[run_dev_id_idx][FetchFromId_idx]->add_host_func((void*)&CBlock_RR_INV_wrap, (void*) wrap_inval);
			//}
		}
		else{
			wrap_read = (CBlock_wrap_p) malloc (sizeof(struct CBlock_wrap));
			wrap_read->CBlock = tmp->StoreBlock[FetchFromId_idx];
			wrap_read->lockfree = false;
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
			tmp->StoreBlock[idxize(FetchFromId)]->add_reader();
			tmp->RW_master = run_dev_id;
		}
		if (FetchFromId == run_dev_id) error("Subkernel(dev=%d,id=%d)-Tile(%d.[%d,%d])::request_tile W_flag = %d, \
			FetchFromId == run_dev_id == %d, state[%d] == %s\n",  run_dev_id, id, tmp->id, tmp->GridId1, tmp->GridId2,
			tmp->W_flag, FetchFromId, FetchFromId, print_state(tmp->StoreBlock[idxize(FetchFromId)]->State));
		short FetchFromId_idx = idxize(FetchFromId);
#ifdef DEBUG
		lprintf(lvl, "Subkernel(dev=%d,id=%d)-Tile(%d.[%d,%d]): Fetching Block(%d) on GPU(%d) from Block(%d) on GPU(%d)\n",
			run_dev_id, id, tmp->id,  tmp->GridId1, tmp->GridId2, tmp->StoreBlock[run_dev_id_idx]->id, run_dev_id,
			tmp->StoreBlock[FetchFromId_idx]->id, FetchFromId);
#endif
		CBlock_wrap_p wrap_read = NULL;
		if (tmp->StoreBlock[FetchFromId_idx]->State == INVALID)
			error("Subkernel(dev=%d,id=%d)-Tile(%d.[%d,%d])::request_tile: Fetching from tile in GPU(%d) with INVALID state\n",
				run_dev_id, id, tmp->id,  tmp->GridId1, tmp->GridId2, FetchFromId);

		transfer_queues[run_dev_id_idx][FetchFromId_idx]->wait_for_event(tmp->StoreBlock[FetchFromId_idx]->Available);
#ifdef STEST
		transfer_queues[run_dev_id_idx][FetchFromId_idx]->add_host_func((void*)&CoCoSetTimerAsync, (void*) &reqT_start_ts[TileIdx]);
#endif
		CoCoMemcpy2DAsync(tmp->StoreBlock[run_dev_id_idx]->Adrs, tmp->ldim[run_dev_id_idx],
									tmp->StoreBlock[FetchFromId_idx]->Adrs, tmp->ldim[FetchFromId_idx],
									tmp->dim1, tmp->dim2, tmp->dtypesize(),
									run_dev_id, FetchFromId, transfer_queues[run_dev_id_idx][FetchFromId_idx]);
#ifdef STEST
		transfer_queues[run_dev_id_idx][FetchFromId_idx]->add_host_func((void*)&CoCoSetTimerAsync, (void*) &reqT_end_ts[TileIdx]);
		bytes_in[TileIdx]= tmp->size();
		dev_in_from[TileIdx] = FetchFromId;
		dev_in_to[TileIdx] = run_dev_id;
#endif
		if(tmp->W_flag){
			//if (tmp->StoreBlock[FetchFromId_idx]!= tmp->WriteBackBlock){
				CBlock_wrap_p wrap_inval = NULL;
				wrap_inval = (CBlock_wrap_p) malloc (sizeof(struct CBlock_wrap));
				wrap_inval->CBlock = tmp->StoreBlock[FetchFromId_idx];
				wrap_inval->lockfree = false;
				transfer_queues[run_dev_id_idx][FetchFromId_idx]->add_host_func((void*)&CBlock_RR_INV_wrap, (void*) wrap_inval);
			//}
		}
		else{
			wrap_read = (CBlock_wrap_p) malloc (sizeof(struct CBlock_wrap));
			wrap_read->CBlock = tmp->StoreBlock[FetchFromId_idx];
			wrap_read->lockfree = false;
			transfer_queues[run_dev_id_idx][FetchFromId_idx]->add_host_func((void*)&CBlock_RR_wrap, (void*) wrap_read);
		}
		tmp->StoreBlock[run_dev_id_idx]->Available->record_to_queue(transfer_queues[run_dev_id_idx][FetchFromId_idx]);
	}
}

void Subkernel::request_data(){
	short lvl = 4;
#ifdef DEBUG
	lprintf(lvl-1, "|-----> Subkernel(dev=%d,id=%d)::request_data()\n", run_dev_id, id);
#endif
#ifdef STEST
		req_in_ts = csecond();
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
			if (tmp->StoreBlock[run_dev_id_idx] == NULL ||
				tmp->StoreBlock[run_dev_id_idx]->State == INVALID) {
				if(tmp->StoreBlock[run_dev_id_idx] != NULL) tmp->StoreBlock[run_dev_id_idx]->Owner_p = NULL;
				state new_block_state;
				if(tmp->W_flag) new_block_state = EXCLUSIVE;
				else new_block_state = SHARABLE;
				tmp->StoreBlock[run_dev_id_idx] = Global_Cache[run_dev_id_idx]->assign_Cblock(new_block_state,false);
				tmp->StoreBlock[run_dev_id_idx]->set_owner((void**)&tmp->StoreBlock[run_dev_id_idx],false);
				if(tmp->W_flag) tmp->StoreBlock[run_dev_id_idx]->init_writeback_info(tmp->WriteBackBlock,
					&(tmp->RW_master), 1, tmp->dim, tmp->dim, tmp->dim,
					tmp->dtypesize(), transfer_queues[idxize(tmp->getWriteBackLoc())][run_dev_id_idx], false);
#ifdef DEBUG
		lprintf(lvl, "Subkernel(dev=%d,id=%d)-Tile(%d.[%d]): Asigned buffer Block in GPU(%d)= %d\n",
					run_dev_id, id, tmp->id, tmp->GridId, run_dev_id, tmp->StoreBlock[run_dev_id_idx]->id);
#endif
#ifdef ENABLE_PTHREAD_TILE_REQUEST
					if (tmp->R_flag) {
						if(!tmp->W_flag){
							tile_req_p wrap_request = (tile_req_p) malloc(sizeof(struct tile_req));
							wrap_request->sk = this;
							wrap_request->TileIdx = j;
							s = pthread_create(&thread_id[requested_tiles], &attr, &request_tile_pthread_wrap,
								wrap_request);
							requested_tiles++;
						}
						else request_tile(j);
					}
#else
#ifdef ENABLE_TRANSFER_HOPS
					if (tmp->R_flag) request_tile_hops(j);
#else
					if (tmp->R_flag) request_tile(j);
#endif
#endif
			}
			else if(tmp->StoreBlock[run_dev_id_idx]->State == NATIVE &&
				tmp->W_flag && tmp->RW_master!= run_dev_id){
				if(tmp->W_flag);// tmp->StoreBlock[run_dev_id_idx]->add_writer();
				else tmp->StoreBlock[run_dev_id_idx]->add_reader();
#ifdef ENABLE_PTHREAD_TILE_REQUEST
				if (tmp->R_flag) {
					if(!tmp->W_flag){
						tile_req_p wrap_request = (tile_req_p) malloc(sizeof(struct tile_req));
						wrap_request->sk = this;
						wrap_request->TileIdx = j;
						s = pthread_create(&thread_id[requested_tiles], &attr, &request_tile_pthread_wrap,
							wrap_request);
						requested_tiles++;
					}
					else request_tile(j);
				}
#else
#ifdef ENABLE_TRANSFER_HOPS
					if (tmp->R_flag) request_tile_hops(j);
#else
					if (tmp->R_flag) request_tile(j);
#endif
#endif
			}
			else{
				//tmp->StoreBlock[run_dev_id_idx]->Available->sync_barrier(); // Is this needed?
				if (tmp->W_flag);// tmp->StoreBlock[run_dev_id_idx]->add_writer();
				else tmp->StoreBlock[run_dev_id_idx]->add_reader();
			}
		}
		else if (TileDimlist[j] == 2){
			Tile2D<VALUE_TYPE>* tmp = (Tile2D<VALUE_TYPE>*) TileList[j];
			if (tmp->StoreBlock[run_dev_id_idx] == NULL ||
				tmp->StoreBlock[run_dev_id_idx]->State == INVALID) {
				if(tmp->StoreBlock[run_dev_id_idx] != NULL) tmp->StoreBlock[run_dev_id_idx]->Owner_p = NULL;
				state new_block_state;
				if(tmp->W_flag) new_block_state = EXCLUSIVE;
				else new_block_state = SHARABLE;
				tmp->StoreBlock[run_dev_id_idx] = Global_Cache[run_dev_id_idx]->assign_Cblock(new_block_state,false);
				tmp->StoreBlock[run_dev_id_idx]->set_owner((void**)&tmp->StoreBlock[run_dev_id_idx],false);
				if(tmp->W_flag) tmp->StoreBlock[run_dev_id_idx]->init_writeback_info(tmp->WriteBackBlock,
					&(tmp->RW_master), tmp->dim1, tmp->dim2, tmp->ldim[run_dev_id_idx], tmp->ldim[idxize(tmp->WriteBackLoc)],
					tmp->dtypesize(), transfer_queues[idxize(tmp->getWriteBackLoc())][run_dev_id_idx], false);


#ifdef DEBUG
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
#ifdef ENABLE_TRANSFER_HOPS
					if (tmp->R_flag) request_tile_hops(j);
#else
					if (tmp->R_flag) request_tile(j);
#endif
#endif
			}
			else if(tmp->StoreBlock[run_dev_id_idx]->State == NATIVE &&
				tmp->W_flag && tmp->RW_master!= run_dev_id){
					if(tmp->W_flag);// tmp->StoreBlock[run_dev_id_idx]->add_writer();
					else tmp->StoreBlock[run_dev_id_idx]->add_reader();
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
#ifdef ENABLE_TRANSFER_HOPS
					if (tmp->R_flag) request_tile_hops(j);
#else
					if (tmp->R_flag) request_tile(j);
#endif
#endif
			}
			else{
				//tmp->StoreBlock[run_dev_id_idx]->Available->sync_barrier(); // Is this needed?
				if (tmp->W_flag);// tmp->StoreBlock[run_dev_id_idx]->add_writer();
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
		req_out_ts = csecond();
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
		op_fire_ts = csecond();
#endif
	short run_dev_id_idx = idxize(run_dev_id);

#ifdef ENABLE_PARALLEL_BACKEND
	short RW_parallel_backend_ctr = -42;
	if(is_RW_lock_master(run_dev_id) > 1){
		for (int j = 0; j < TileNum; j++){ /// Method only works for max 1 W(rite) Tile.
			if (TileDimlist[j] == 1){
					Tile1D<VALUE_TYPE>* tmp = (Tile1D<VALUE_TYPE>*) TileList[j];
					if(tmp->W_flag) RW_parallel_backend_ctr = tmp->RW_Master_backend_ctr;
			}
			else if (TileDimlist[j] == 2){
					Tile2D<VALUE_TYPE>* tmp = (Tile2D<VALUE_TYPE>*) TileList[j];
					if(tmp->W_flag) RW_parallel_backend_ctr = tmp->RW_Master_backend_ctr;
			}
		}
		exec_queue[run_dev_id_idx]->set_parallel_backend(RW_parallel_backend_ctr);
	}
	else{
		RW_parallel_backend_ctr = exec_queue[run_dev_id_idx]->request_parallel_backend();
	}
#endif
	for (int j = 0; j < TileNum; j++){
		if (TileDimlist[j] == 1){
				Tile1D<VALUE_TYPE>* tmp = (Tile1D<VALUE_TYPE>*) TileList[j];
				if (tmp->StoreBlock[run_dev_id_idx] == NULL ||
  tmp->StoreBlock[run_dev_id_idx]->State == INVALID)
					error("Subkernel(dev=%d,id=%d)-Tile(%d.[%d])::run_operation: Tile(j=%d) Storeblock is NULL\n",
						run_dev_id, id, tmp->id, tmp->GridId, j);
				if(tmp->R_flag) exec_queue[run_dev_id_idx]->wait_for_event(tmp->StoreBlock[run_dev_id_idx]->Available);
				#ifdef ENABLE_PARALLEL_BACKEND
				if(tmp->W_flag) tmp->RW_Master_backend_ctr = RW_parallel_backend_ctr;
				#endif
		}
		else if (TileDimlist[j] == 2){
				Tile2D<VALUE_TYPE>* tmp = (Tile2D<VALUE_TYPE>*) TileList[j];
				if (tmp->StoreBlock[run_dev_id_idx] == NULL ||
  tmp->StoreBlock[run_dev_id_idx]->State == INVALID)
					error("Subkernel(dev=%d,id=%d)-Tile(%d.[%d,%d])::run_operation: Tile(j=%d) Storeblock is NULL\n",
						run_dev_id, id, tmp->id, tmp->GridId1, tmp->GridId2, j);
				if(tmp->R_flag) exec_queue[run_dev_id_idx]->wait_for_event(tmp->StoreBlock[run_dev_id_idx]->Available);
				#ifdef ENABLE_PARALLEL_BACKEND
				if(tmp->W_flag) tmp->RW_Master_backend_ctr = RW_parallel_backend_ctr;
				#endif
		}
	}
#ifdef STEST
		exec_queue[run_dev_id_idx]->add_host_func(
				(void*)&CoCoSetTimerAsync, (void*) &op_start_ts);
		if (!strcmp(op_name,"gemm")){
			gemm_backend_in_p ptr_ker_translate = (gemm_backend_in_p) operation_params;
			flops = gemm_flops(ptr_ker_translate->M, ptr_ker_translate->N, ptr_ker_translate->K);
		}
		if (!strcmp(op_name,"axpy")){
			axpy_backend_in_p ptr_ker_translate = (axpy_backend_in_p) operation_params;
			flops = axpy_flops(ptr_ker_translate->N);
		}
#endif
	backend_run_operation(operation_params, op_name, exec_queue[run_dev_id_idx]);
#ifdef STEST
	exec_queue[run_dev_id_idx]->add_host_func(
			(void*)&CoCoSetTimerAsync, (void*) &op_end_ts);
#endif
	for (int j = 0; j < TileNum; j++){
		if (TileDimlist[j] == 1){
			Tile1D<VALUE_TYPE>* tmp = (Tile1D<VALUE_TYPE>*) TileList[j];
			if (tmp->StoreBlock[run_dev_id_idx] == NULL ||
  tmp->StoreBlock[run_dev_id_idx]->State == INVALID)
				error("Subkernel(dev=%d,id=%d)-Tile(%d.[%d])::run_operation: Tile(j=%d) Storeblock is NULL\n",
					run_dev_id, id, tmp->id, tmp->GridId, j);
			CBlock_wrap_p wrap_oper = NULL;
			wrap_oper = (CBlock_wrap_p) malloc (sizeof(struct CBlock_wrap));
			wrap_oper->CBlock = tmp->StoreBlock[run_dev_id_idx];
			wrap_oper->lockfree = false;

			if(tmp->R_flag) tmp->R_flag--;

			if(tmp->W_flag){
				tmp->W_flag--;
				if(!tmp->W_flag) WR_last[j] = 1;
				else{
					exec_queue[run_dev_id_idx]->add_host_func((void*)&CBlock_RW_wrap, (void*) wrap_oper);
					Ptr_atomic_int_p wrapped_op = (Ptr_atomic_int_p) malloc(sizeof(struct Ptr_atomic_int));
					wrapped_op->ato_int_ptr = &tmp->RW_lock_holders;
					exec_queue[run_dev_id_idx]->add_host_func((void*)&CoCoDecAsync, (void*) wrapped_op);
				}
			}
			else exec_queue[run_dev_id_idx]->add_host_func((void*)&CBlock_RR_wrap, (void*) wrap_oper);

		}
		else if (TileDimlist[j] == 2){
			Tile2D<VALUE_TYPE>* tmp = (Tile2D<VALUE_TYPE>*) TileList[j];
			if (tmp->StoreBlock[run_dev_id_idx] == NULL ||
  tmp->StoreBlock[run_dev_id_idx]->State == INVALID)
				error("Subkernel(dev=%d,id=%d)-Tile(%d.[%d,%d])::run_operation: Tile(j=%d) Storeblock is NULL\n",
					run_dev_id, id, tmp->id, tmp->GridId1, tmp->GridId2, j);
			CBlock_wrap_p wrap_oper = NULL;
			wrap_oper = (CBlock_wrap_p) malloc (sizeof(struct CBlock_wrap));
			wrap_oper->CBlock = tmp->StoreBlock[run_dev_id_idx];
			wrap_oper->lockfree = false;

			if(tmp->R_flag) tmp->R_flag--;

			if(tmp->W_flag){
				tmp->W_flag--;
				if(!tmp->W_flag) WR_last[j] = 1;
				else{
					exec_queue[run_dev_id_idx]->add_host_func((void*)&CBlock_RW_wrap, (void*) wrap_oper);
					Ptr_atomic_int_p wrapped_op = (Ptr_atomic_int_p) malloc(sizeof(struct Ptr_atomic_int));
					wrapped_op->ato_int_ptr = &tmp->RW_lock_holders;
					exec_queue[run_dev_id_idx]->add_host_func((void*)&CoCoDecAsync, (void*) wrapped_op);
				}
			}
			else exec_queue[run_dev_id_idx]->add_host_func((void*)&CBlock_RR_wrap, (void*) wrap_oper);

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

#ifdef ENABLE_TRANSFER_W_HOPS
void Subkernel::writeback_data_hops(){
	short lvl = 4;
#ifdef DEBUG
	lprintf(lvl-1, "|-----> Subkernel(dev=%d,id=%d)::writeback_data_hops()\n", run_dev_id, id);
#endif
#ifdef STEST
	wbT_fire_ts[0] = wbT_fire_ts[1] = wbT_fire_ts[2] = csecond();
#endif
	short run_dev_id_idx = idxize(run_dev_id);
	short Writeback_id_idx, Writeback_id;
	for (int j = 0; j < TileNum; j++) if (WR_last[j]){
		if (TileDimlist[j] == 1){
			error("Subkernel::writeback_data_hops() not implemented for 1D\n");
		}
		else if (TileDimlist[j] == 2){
			Tile2D<VALUE_TYPE>* tmp = (Tile2D<VALUE_TYPE>*) TileList[j];
			if (tmp->StoreBlock[run_dev_id_idx] == NULL ||
  tmp->StoreBlock[run_dev_id_idx]->State == INVALID)
				error("Subkernel(dev=%d,id=%d)-Tile(%d.[%d,%d])::writeback_data_hops: Tile(j=%d) Storeblock is NULL\n",
					run_dev_id, id, tmp->id, tmp->GridId1, tmp->GridId2, j);
			Writeback_id = tmp->getWriteBackLoc(); //to MASTER
			Writeback_id_idx = idxize(Writeback_id);
			if (tmp->WriteBackBlock == NULL)
				error("Subkernel(dev=%d,id=%d)-Tile(%d.[%d,%d])::writeback_data_hops: Tile(j=%d) WriteBackBlock is NULL\n",
					run_dev_id, id, tmp->id, tmp->GridId1, tmp->GridId2, j);
			if (run_dev_id == Writeback_id){
				;
#ifdef DEBUG
		lprintf(lvl, "Subkernel(dev=%d,id=%d)-Tile(%d.[%d,%d])::writeback_data_hops: run_dev_id == Writeback_id == %d (not wrong but check)\n",
			run_dev_id, id, tmp->id, tmp->GridId1, tmp->GridId2, run_dev_id);
#endif
			}
			else{
				state prev_state = tmp->WriteBackBlock->get_state();
				if (prev_state != NATIVE)
					error("Subkernel(dev=%d,id=%d)-Tile(%d.[%d,%d])::writeback_data_hops: Tile(j=%d) WriteBackBlock was %s instead of NATIVE\n",
						run_dev_id, id, tmp->id, tmp->GridId1, tmp->GridId2, j, print_state(prev_state));
				int inter_hop_num = final_estimated_linkmap->link_hop_num[Writeback_id_idx][run_dev_id_idx];
				CQueue_p used_queue;
				if(inter_hop_num == 0){
					transfer_queues[Writeback_id_idx][run_dev_id_idx]->wait_for_event(operation_complete);
#ifdef STEST
					transfer_queues[Writeback_id_idx][run_dev_id_idx]->add_host_func(
						(void*)&CoCoSetTimerAsync, (void*) &(wbT_start_ts[j]));
#endif
					used_queue = transfer_queues[Writeback_id_idx][run_dev_id_idx];
					CoCoMemcpy2DAsync(tmp->WriteBackBlock->Adrs, tmp->ldim[Writeback_id_idx],
						tmp->StoreBlock[run_dev_id_idx]->Adrs, tmp->ldim[run_dev_id_idx],
						tmp->dim1, tmp->dim2, tmp->dtypesize(),
						Writeback_id, run_dev_id, transfer_queues[Writeback_id_idx][run_dev_id_idx]);
#ifdef STEST
					transfer_queues[Writeback_id_idx][run_dev_id_idx]->add_host_func(
							(void*)&CoCoSetTimerAsync, (void*) &(wbT_end_ts[j]));
					bytes_out[j]= tmp->size();
					dev_out_from[j] = run_dev_id;
					dev_out_to[j] = Writeback_id;
#endif
				}
				else{
					link_road_p test_road = (link_road_p) malloc(sizeof(struct link_road));
					test_road->hop_num = 2 + inter_hop_num;
					test_road->hop_uid_list[0] = run_dev_id;
					test_road->hop_uid_list[1 + inter_hop_num] = Writeback_id;

					test_road->hop_ldim_list[0] = tmp->ldim[run_dev_id_idx];
					test_road->hop_ldim_list[1 + inter_hop_num] = tmp->ldim[Writeback_id_idx];

					test_road->hop_buf_list[0] = tmp->StoreBlock[run_dev_id_idx]->Adrs;
					test_road->hop_buf_list[1 + inter_hop_num] = tmp->WriteBackBlock->Adrs;
					short selected_route = rand() % (final_estimated_linkmap->link_hop_route_num[Writeback_id_idx][run_dev_id_idx] - 0) + 0;
					state new_block_state = EXCLUSIVE;
					CBlock_p block_ptr[inter_hop_num] = {NULL};
					for(int inter_hop = 0 ; inter_hop < inter_hop_num; inter_hop++){
						test_road->hop_uid_list[1+ inter_hop] = final_estimated_linkmap->link_hop_route[Writeback_id_idx][run_dev_id_idx][selected_route][inter_hop];
						test_road->hop_ldim_list[1+ inter_hop] = tmp->ldim[run_dev_id_idx];
						test_road->hop_cqueue_list[inter_hop] = transfer_queues[idxize(test_road->hop_uid_list[1+inter_hop])]
							[idxize(test_road->hop_uid_list[inter_hop])];
						block_ptr[inter_hop] = Global_Cache[idxize(test_road->hop_uid_list[1+inter_hop])]->assign_Cblock(new_block_state,false);
						//block_ptr[inter_hop]->init_writeback_info(tmp->WriteBackBlock, &(tmp->RW_master), tmp->dim1, tmp->dim2,
						//	tmp->ldim[idxize(test_road->hop_uid_list[1+inter_hop])], tmp->ldim[idxize(tmp->WriteBackLoc)],
						//	tmp->dtypesize(), transfer_queues[idxize(tmp->getWriteBackLoc())][idxize(test_road->hop_uid_list[1+inter_hop])], false);

						test_road->hop_buf_list[1 + inter_hop] = block_ptr[inter_hop]->Adrs;
						test_road->hop_event_list[inter_hop] = block_ptr[inter_hop]->Available;

						//test_road->hop_cqueue_list[inter_hop]->wait_for_event(operation_complete);
					}

					test_road->hop_cqueue_list[0]->wait_for_event(operation_complete);

					used_queue = test_road->hop_cqueue_list[inter_hop_num] =
						transfer_queues[idxize(test_road->hop_uid_list[1+inter_hop_num])][idxize(test_road->hop_uid_list[inter_hop_num])];
					test_road->hop_event_list[inter_hop_num] = tmp->WriteBackBlock->Available;
#ifdef STEST
					used_queue->add_host_func(
						(void*)&CoCoSetTimerAsync, (void*) &(wbT_start_ts[j]));
#endif
#ifdef PDEBUG
					lprintf(1, "Subkernel(dev=%d,id=%d)-Tile(%d.[%d,%d])::writeback_data_hops W_flag = %d, \
						%2d->%2d transfer sequence -> %s (route %d)\n", run_dev_id, id, tmp->id, tmp->GridId1, tmp->GridId2,
						tmp->W_flag, run_dev_id, Writeback_id,
						printlist(final_estimated_linkmap->link_hop_route[Writeback_id_idx][run_dev_id_idx][selected_route],
							final_estimated_linkmap->link_hop_num[Writeback_id_idx][run_dev_id_idx]), selected_route);
#endif
					FasTCoCoMemcpy2DAsync(test_road, tmp->dim1, tmp->dim2, tmp->dtypesize());
#ifdef STEST
				used_queue->add_host_func(
						(void*)&CoCoSetTimerAsync, (void*) &(wbT_end_ts[j]));
				bytes_out[j]= tmp->size();
				dev_out_from[j] = run_dev_id;
				dev_out_to[j] = Writeback_id;
#endif
					for(int inter_hop = 0 ; inter_hop < inter_hop_num; inter_hop++){
						CBlock_wrap_p wrap_inval = NULL;
						wrap_inval = (CBlock_wrap_p) malloc (sizeof(struct CBlock_wrap));
						wrap_inval->lockfree = false;
						wrap_inval->CBlock = block_ptr[inter_hop];
						test_road->hop_cqueue_list[inter_hop+1]->add_host_func((void*)&CBlock_RW_INV_wrap, (void*) wrap_inval);
					}
				}

				Ptr_atomic_int_p wrapped_op = (Ptr_atomic_int_p) malloc(sizeof(struct Ptr_atomic_int));
				wrapped_op->ato_int_ptr = &tmp->RW_lock_holders;
				used_queue->add_host_func((void*)&CoCoDecAsync, (void*) wrapped_op);

				CBlock_wrap_p wrap_inval = NULL;
				wrap_inval = (CBlock_wrap_p) malloc (sizeof(struct CBlock_wrap));
				wrap_inval->CBlock = tmp->StoreBlock[run_dev_id_idx];
				wrap_inval->lockfree = false;
				used_queue->add_host_func((void*)&CBlock_RW_INV_wrap, (void*) wrap_inval);
			}
		}
		else error("Subkernel(dev=%d,id=%d)::writeback_data_hops: Not implemented for TileDim=%d\n", run_dev_id, id, TileDimlist[j]);
	}
#ifndef ASYNC_ENABLE
	CoCoSyncCheckErr();
#endif
#ifdef DEBUG
	lprintf(lvl-1, "<-----|\n");
#endif
}
#endif

void Subkernel::writeback_data(){
	short lvl = 4;
#ifdef ENABLE_TRANSFER_W_HOPS
#ifdef ENABLE_PREDICT_HOP_MODE
	if(Snd_hops_and_NOSRO_enable_flag) {
		writeback_data_hops();
		return;
	}
#else
	writeback_data_hops();
	return;
#endif
#endif
#ifdef DEBUG
	lprintf(lvl-1, "|-----> Subkernel(dev=%d,id=%d)::writeback_data()\n", run_dev_id, id);
#endif
#ifdef STEST
	wbT_fire_ts[0] = wbT_fire_ts[1] = wbT_fire_ts[2] = csecond();
#endif
	short run_dev_id_idx = idxize(run_dev_id);
	short Writeback_id_idx, Writeback_id;
	for (int j = 0; j < TileNum; j++) if (WR_last[j]){
		if (TileDimlist[j] == 1){
			Tile1D<VALUE_TYPE>* tmp = (Tile1D<VALUE_TYPE>*) TileList[j];
			if (tmp->StoreBlock[run_dev_id_idx] == NULL ||
  tmp->StoreBlock[run_dev_id_idx]->State == INVALID)
				error("Subkernel(dev=%d,id=%d)-Tile(%d.[%d])::writeback_data: Tile(j=%d) Storeblock is NULL\n",
					run_dev_id, id, tmp->id, tmp->GridId, j);
			Writeback_id = tmp->getWriteBackLoc(); //to MASTER
			Writeback_id_idx = idxize(Writeback_id);
			if (tmp->WriteBackBlock == NULL)
				error("Subkernel(dev=%d,id=%d)-Tile(%d.[%d])::writeback_data: Tile(j=%d) WriteBackBlock is NULL\n",
					run_dev_id, id, tmp->id, tmp->GridId, j);
			if (run_dev_id == Writeback_id){
				;
#ifdef DEBUG
		lprintf(lvl, "Subkernel(dev=%d,id=%d)-Tile(%d.[%d])::writeback_data: run_dev_id == Writeback_id == %d (not wrong but check)\n",
			run_dev_id, id, tmp->id, tmp->GridId, run_dev_id);
#endif
			}
			else{
				transfer_queues[Writeback_id_idx][run_dev_id_idx]->wait_for_event(operation_complete);
				state prev_state = tmp->WriteBackBlock->get_state();
				if (prev_state != NATIVE)
					error("Subkernel(dev=%d,id=%d)-Tile(%d.[%d])::writeback_data: Tile(j=%d) WriteBackBlock was %s instead of NATIVE\n",
						run_dev_id, id, tmp->id, tmp->GridId, j, print_state(prev_state));
#ifdef STEST
				transfer_queues[Writeback_id_idx][run_dev_id_idx]->add_host_func(
						(void*)&CoCoSetTimerAsync, (void*) &(wbT_start_ts[j]));
#endif
				CoCoMemcpyAsync(tmp->WriteBackBlock->Adrs, tmp->StoreBlock[run_dev_id_idx]->Adrs,
					((long long) tmp->inc[run_dev_id_idx]) * tmp->dim * tmp->dtypesize(),
					Writeback_id, run_dev_id, transfer_queues[Writeback_id_idx][run_dev_id_idx]);
#ifdef STEST
				transfer_queues[Writeback_id_idx][run_dev_id_idx]->add_host_func(
						(void*)&CoCoSetTimerAsync, (void*) &(wbT_end_ts[j]));
				bytes_out[j]= tmp->size();
				dev_out_from[j] = run_dev_id;
				dev_out_to[j] = Writeback_id;
#endif
				Ptr_atomic_int_p wrapped_op = (Ptr_atomic_int_p) malloc(sizeof(struct Ptr_atomic_int));
				wrapped_op->ato_int_ptr = &tmp->RW_lock_holders;
				transfer_queues[Writeback_id_idx][run_dev_id_idx]->add_host_func((void*)&CoCoDecAsync, (void*) wrapped_op);

				CBlock_wrap_p wrap_inval = NULL;
				wrap_inval = (CBlock_wrap_p) malloc (sizeof(struct CBlock_wrap));
				wrap_inval->CBlock = tmp->StoreBlock[run_dev_id_idx];
				wrap_inval->lockfree = false;
				transfer_queues[Writeback_id_idx][run_dev_id_idx]->add_host_func((void*)&CBlock_RW_INV_wrap, (void*) wrap_inval);
			}
		}
		else if (TileDimlist[j] == 2){
			Tile2D<VALUE_TYPE>* tmp = (Tile2D<VALUE_TYPE>*) TileList[j];
			if (tmp->StoreBlock[run_dev_id_idx] == NULL ||
  tmp->StoreBlock[run_dev_id_idx]->State == INVALID)
				error("Subkernel(dev=%d,id=%d)-Tile(%d.[%d,%d])::writeback_data: Tile(j=%d) Storeblock is NULL\n",
					run_dev_id, id, tmp->id, tmp->GridId1, tmp->GridId2, j);
			Writeback_id = tmp->getWriteBackLoc(); //to MASTER
			Writeback_id_idx = idxize(Writeback_id);
			if (tmp->WriteBackBlock == NULL)
				error("Subkernel(dev=%d,id=%d)-Tile(%d.[%d,%d])::writeback_data: Tile(j=%d) WriteBackBlock is NULL\n",
					run_dev_id, id, tmp->id, tmp->GridId1, tmp->GridId2, j);
			if (run_dev_id == Writeback_id){
				;
#ifdef DEBUG
		lprintf(lvl, "Subkernel(dev=%d,id=%d)-Tile(%d.[%d,%d])::writeback_data: run_dev_id == Writeback_id == %d (not wrong but check)\n",
			run_dev_id, id, tmp->id, tmp->GridId1, tmp->GridId2, run_dev_id);
#endif
			}
			else{
				transfer_queues[Writeback_id_idx][run_dev_id_idx]->wait_for_event(operation_complete);
				state prev_state = tmp->WriteBackBlock->get_state();
				if (prev_state != NATIVE)
					error("Subkernel(dev=%d,id=%d)-Tile(%d.[%d,%d])::writeback_data: Tile(j=%d) WriteBackBlock was %s instead of NATIVE\n",
						run_dev_id, id, tmp->id, tmp->GridId1, tmp->GridId2, j, print_state(prev_state));
#ifdef STEST
				transfer_queues[Writeback_id_idx][run_dev_id_idx]->add_host_func(
						(void*)&CoCoSetTimerAsync, (void*) &(wbT_start_ts[j]));
#endif
				CoCoMemcpy2DAsync(tmp->WriteBackBlock->Adrs, tmp->ldim[Writeback_id_idx],
					tmp->StoreBlock[run_dev_id_idx]->Adrs, tmp->ldim[run_dev_id_idx],
					tmp->dim1, tmp->dim2, tmp->dtypesize(),
					Writeback_id, run_dev_id, transfer_queues[Writeback_id_idx][run_dev_id_idx]);
#ifdef STEST
				transfer_queues[Writeback_id_idx][run_dev_id_idx]->add_host_func(
						(void*)&CoCoSetTimerAsync, (void*) &(wbT_end_ts[j]));
				bytes_out[j]= tmp->size();
				dev_out_from[j] = run_dev_id;
				dev_out_to[j] = Writeback_id;
#endif

				Ptr_atomic_int_p wrapped_op = (Ptr_atomic_int_p) malloc(sizeof(struct Ptr_atomic_int));
				wrapped_op->ato_int_ptr = &tmp->RW_lock_holders;
				transfer_queues[Writeback_id_idx][run_dev_id_idx]->add_host_func((void*)&CoCoDecAsync, (void*) wrapped_op);

				CBlock_wrap_p wrap_inval = NULL;
				wrap_inval = (CBlock_wrap_p) malloc (sizeof(struct CBlock_wrap));
				wrap_inval->CBlock = tmp->StoreBlock[run_dev_id_idx];
				wrap_inval->lockfree = false;
				transfer_queues[Writeback_id_idx][run_dev_id_idx]->add_host_func((void*)&CBlock_RW_INV_wrap, (void*) wrap_inval);
			}
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


int queue_d_allock = 0;

void CoCoPeLiaInitResources(short dev_id){
	short lvl = 2;
	#ifdef DEBUG
		lprintf(lvl-1, "|-----> CoCoPeLiaInitResources(dev=%d)\n", dev_id);
	#endif
	//while(__sync_lock_test_and_set (&queue_d_allock, 1));

	for(int i = 0; i < LOC_NUM; i++)
	for(int j = 0; j < LOC_NUM; j++)
	for(int k = 0; k < 2; k++) transfer_link_sharing[i][j][k] = -42;

#ifndef ENABLE_LINK_BW_SHARING
	warning("ENABLE_LINK_BW_SHARING flag is disabled, but sharing-disabler mechanism is missing -> link bw will be shared\n")
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

/*
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
*/
#endif

	short dev_id_idx = idxize(dev_id);
	for(short dev_id_idy = 0 ; dev_id_idy < LOC_NUM; dev_id_idy++)
	if(dev_id_idy!=dev_id_idx){
		if (!transfer_queues[dev_id_idx][dev_id_idy]){
			//printf("dev_id = %d, dev_id_idx = %d, dev_id_idy = %d, LOC_NUM = %d\n", dev_id, dev_id_idx, dev_id_idy, LOC_NUM);
			short shared_iloc0 = transfer_link_sharing[dev_id_idx][dev_id_idy][0],
				shared_iloc1 = transfer_link_sharing[dev_id_idx][dev_id_idy][1];
			short queue_id = (dev_id_idy == LOC_NUM - 1)? deidxize(dev_id_idx) : deidxize(dev_id_idy);
			if( shared_iloc0 != - 42){ // The smallest index shared link allocates the queue
				if (dev_id_idx*LOC_NUM + dev_id_idy < shared_iloc0*LOC_NUM + shared_iloc1){
					transfer_queues[dev_id_idx][dev_id_idy] = new CommandQueue(queue_id);
					transfer_queues[shared_iloc0][shared_iloc1] = transfer_queues[dev_id_idx][dev_id_idy];
				}
			}
			else transfer_queues[dev_id_idx][dev_id_idy] = new CommandQueue(queue_id);
		}
		if (!transfer_queues[dev_id_idy][dev_id_idx]){
			short shared_iloc0 = transfer_link_sharing[dev_id_idy][dev_id_idx][0],
				shared_iloc1 = transfer_link_sharing[dev_id_idy][dev_id_idx][1];
			if( shared_iloc0 != - 42){ // The smallest index shared link allocates the queue
				if (dev_id_idy*LOC_NUM + dev_id_idx < shared_iloc0*LOC_NUM + shared_iloc1){
					short writeback_queue_id = (dev_id_idx == LOC_NUM - 1)? deidxize(dev_id_idy) : deidxize(dev_id_idx);
					transfer_queues[dev_id_idy][dev_id_idx] = new CommandQueue(writeback_queue_id);
					transfer_queues[shared_iloc0][shared_iloc1] = transfer_queues[dev_id_idy][dev_id_idx];
				}
			}
			else{
				short writeback_queue_id = (dev_id_idx == LOC_NUM - 1)? deidxize(dev_id_idy) : deidxize(dev_id_idx);
				transfer_queues[dev_id_idy][dev_id_idx] = new CommandQueue(writeback_queue_id);
			}
		}
	}
  if (!exec_queue[dev_id_idx])  exec_queue[dev_id_idx] = new CommandQueue(dev_id);

	//__sync_lock_release(&queue_d_allock);
	#ifdef DEBUG
		lprintf(lvl-1, "<-----|\n");
	#endif
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

void Subkernel::prepare_launch(short dev_id){
	short lvl = 3;
#ifdef DEBUG
	lprintf(lvl, "|-----> Subkernel(dev=%d, id = %d)\n", run_dev_id, id);
#endif

	for (int j = 0; j < TileNum; j++){
		if (TileDimlist[j] == 1){
			Tile1D<VALUE_TYPE>* tmp = (Tile1D<VALUE_TYPE>*) TileList[j];
			if(tmp->W_flag) {
				if(tmp->RW_lock_holders.load() > 0 && tmp->RW_lock != dev_id)
					error("Subkernel(dev=%d,id=%d)-Tile(%d.[%d])::prepare_launch: Tile(j=%d) has RW_lock = %d with RW_lock_holders = %d\n",
						run_dev_id, id, tmp->id, tmp->GridId, j, tmp->RW_lock, tmp->RW_lock_holders.load());
				if (tmp->W_total == tmp->W_flag) WR_first = 1;
				tmp->RW_lock = dev_id;
				tmp->RW_lock_holders++;
				// Put this here to avoid block being replaced by jx < j assign schedule out
				if (tmp->StoreBlock[idxize(dev_id)] &&
					tmp->StoreBlock[idxize(dev_id)]->State != INVALID) tmp->StoreBlock[idxize(dev_id)]->add_writer();
#ifdef DEBUG
				lprintf(lvl, "Subkernel(dev=%d,id=%d)-Tile(%d.[%d])::prepare_launch: Tile(j=%d) has RW_lock = %d with RW_lock_holders = %d\n",
					run_dev_id, id, tmp->id, tmp->GridId, j, tmp->RW_lock, tmp->RW_lock_holders.load());
#endif
			}
		}
		else if (TileDimlist[j] == 2){
			Tile2D<VALUE_TYPE>* tmp = (Tile2D<VALUE_TYPE>*) TileList[j];
			if(tmp->W_flag) {
				if(tmp->RW_lock_holders.load() > 0 && tmp->RW_lock != dev_id)
					error("Subkernel(dev=%d,id=%d)-Tile(%d.[%d,%d])::prepare_launch: Tile(j=%d) has RW_lock = %d with RW_lock_holders = %d\n",
						run_dev_id, id, tmp->id, tmp->GridId1, tmp->GridId2, j, tmp->RW_lock, tmp->RW_lock_holders.load());
				if (tmp->W_total == tmp->W_flag) WR_first = 1;
				tmp->RW_lock = dev_id;
				tmp->RW_lock_holders++;
				// Put this here to avoid block being replaced by jx < j assign schedule out
				if (tmp->StoreBlock[idxize(dev_id)] &&
					tmp->StoreBlock[idxize(dev_id)]->State != INVALID) tmp->StoreBlock[idxize(dev_id)]->add_writer();
#ifdef DEBUG
				lprintf(lvl, "Subkernel(dev=%d,id=%d)-Tile(%d.[%d,%d])::prepare_launch: Tile(j=%d) has RW_lock = %d with RW_lock_holders = %d\n",
					run_dev_id, id, tmp->id, tmp->GridId1, tmp->GridId2, j, tmp->RW_lock, tmp->RW_lock_holders.load());
#endif
			}
		}
	}
#ifdef DEBUG
	lprintf(lvl, "<-----|\n");
#endif
}

short Subkernel::no_locked_tiles(){
	for (int j = 0; j < TileNum; j++){
		if (TileDimlist[j] == 1){
			Tile1D<VALUE_TYPE>* tmp = (Tile1D<VALUE_TYPE>*) TileList[j];
			if(tmp->RW_lock_holders.load() > 0 ) return 0;
		}
		else if (TileDimlist[j] == 2){
			Tile2D<VALUE_TYPE>* tmp = (Tile2D<VALUE_TYPE>*) TileList[j];
			if(tmp->RW_lock_holders.load() > 0) return 0;
		}
		else error("Subkernel(dev=%d,id=%d)::no_locked_tiles: Not implemented for TileDim=%d\n", run_dev_id, id, TileDimlist[j]);
	}
	return 1;
}

/// Value returned only works for max 1 W(rite) Tile - no problem for most BLAS. Bool check holds for any.
short Subkernel::is_RW_lock_master(short dev_id){
	short RW_lock_Hos = 0;
	for (int j = 0; j < TileNum; j++){
		if (TileDimlist[j] == 1){
			Tile1D<VALUE_TYPE>* tmp = (Tile1D<VALUE_TYPE>*) TileList[j];
			if(tmp->W_flag && /*tmp->W_total != tmp->W_flag &&*/ tmp->RW_lock!=dev_id) return 0;
			else if(tmp->W_flag && tmp->RW_lock==dev_id) RW_lock_Hos = tmp->RW_lock_holders.load();
		}
		else if (TileDimlist[j] == 2){
			Tile2D<VALUE_TYPE>* tmp = (Tile2D<VALUE_TYPE>*) TileList[j];
			if(tmp->W_flag && /*tmp->W_total != tmp->W_flag &&*/ tmp->RW_lock!=dev_id) return 0;
			else if(tmp->W_flag && tmp->RW_lock==dev_id) RW_lock_Hos = tmp->RW_lock_holders.load();
		}
	}
	return RW_lock_Hos;
}

/// Value returned only works for max 1 W(rite) Tile - no problem for most BLAS. Bool check holds for any.
short Subkernel::RW_lock_initialized(){
	for (int j = 0; j < TileNum; j++){
		if (TileDimlist[j] == 1){
			Tile1D<VALUE_TYPE>* tmp = (Tile1D<VALUE_TYPE>*) TileList[j];
			if(tmp->W_flag && /*tmp->W_total != tmp->W_flag &&*/ tmp->RW_lock!=-42) return 1;
		}
		else if (TileDimlist[j] == 2){
			Tile2D<VALUE_TYPE>* tmp = (Tile2D<VALUE_TYPE>*) TileList[j];
			if(tmp->W_flag && /*tmp->W_total != tmp->W_flag &&*/ tmp->RW_lock!=-42) return 1;
		}
	}
	return 0;
}

long double Subkernel::opt_fetch_cost(short dev_id){
	long double fetch_cost = 0, inter_fetch_cost = 0;
	short prefetched = 0;
	for (int j = 0; j < TileNum; j++){
		if (TileDimlist[j] == 1){
			Tile1D<VALUE_TYPE>* tmp = (Tile1D<VALUE_TYPE>*) TileList[j];
			long double temp_fetch_cost = tmp->getMinLinkCost(dev_id);
			if(tmp->W_flag) temp_fetch_cost+= WTILE_TRANSFER_PENALTY*temp_fetch_cost;
			inter_fetch_cost = prefetched = 0;
			for(int loc_idx = 0; loc_idx < LOC_NUM; loc_idx++){
				CBlock_p curr_block = tmp->StoreBlock[loc_idx];
				if(curr_block != NULL && curr_block->Available->query_status() == RECORDED){
					if (deidxize(loc_idx) == dev_id){
						prefetched = 1;
						warning("opt_fetch_cost(dev_id=%d, TiledIdx=%d): Already fetching in dev_id (?)\n", dev_id, j);
					}
					else inter_fetch_cost+=temp_fetch_cost*MULTIFETCH_PENALTY;
				}
			}
			if(!prefetched) fetch_cost+= (inter_fetch_cost + temp_fetch_cost);
		}
		else if (TileDimlist[j] == 2){
			Tile2D<VALUE_TYPE>* tmp = (Tile2D<VALUE_TYPE>*) TileList[j];
			long double temp_fetch_cost = tmp->getMinLinkCost(dev_id);
			if(tmp->W_flag) temp_fetch_cost+= WTILE_TRANSFER_PENALTY*temp_fetch_cost;
			inter_fetch_cost = prefetched = 0;
			for(int loc_idx = 0; loc_idx < LOC_NUM; loc_idx++){
				CBlock_p curr_block = tmp->StoreBlock[loc_idx];
				if(curr_block != NULL && curr_block->Available->query_status() == RECORDED){
					if (deidxize(loc_idx) == dev_id){
						prefetched = 1;
						warning("opt_fetch_cost(dev_id=%d, TiledIdx=%d): Already fetching in dev_id (?)\n", dev_id, j);
					}
					else inter_fetch_cost+=temp_fetch_cost*MULTIFETCH_PENALTY;
				}
			}
			if(!prefetched) fetch_cost+= (inter_fetch_cost + temp_fetch_cost);
		}
	}
	return fetch_cost;
}

int SubkernelPrefetchCheapRONLYTiles(int numTiles, short dev_id, Subkernel** Subkernel_list, long Subkernel_list_len){
	short lvl = 5;
	Subkernel* curr_sk = NULL;
	long sk_idx;
	long double min_fetch_cost = DBL_MAX;
	int dev_id_idx = idxize(dev_id), tiles_prefetched = 0;
	for (sk_idx = 0; sk_idx < Subkernel_list_len; sk_idx++){
		curr_sk = Subkernel_list[sk_idx];
		if (curr_sk->run_dev_id !=- 42 || !(curr_sk->no_locked_tiles() || curr_sk->is_RW_lock_master(dev_id))) continue;
		for (int j = 0; j < curr_sk->TileNum; j++){
			if (curr_sk->TileDimlist[j] == 1){
				error("Prefetching in 1D Tiles not implemented\n");
				Tile1D<VALUE_TYPE>* tmp = (Tile1D<VALUE_TYPE>*) curr_sk->TileList[j];
			}
			else if (curr_sk->TileDimlist[j] == 2){
				Tile2D<VALUE_TYPE>* tmp = (Tile2D<VALUE_TYPE>*) curr_sk->TileList[j];
				if(!tmp->W_total){
					long double temp_fetch_cost = tmp->getMinLinkCost(dev_id);
					if(temp_fetch_cost < min_fetch_cost && temp_fetch_cost!= 0 && temp_fetch_cost < 1) min_fetch_cost = temp_fetch_cost;
				}
			}
		}
	}
	for (sk_idx = 0; sk_idx < Subkernel_list_len; sk_idx++){
		curr_sk = Subkernel_list[sk_idx];
		if(tiles_prefetched == numTiles) break;
		if (curr_sk->run_dev_id !=- 42 || !(curr_sk->no_locked_tiles() || curr_sk->is_RW_lock_master(dev_id))) continue;
		for (int j = 0; j < curr_sk->TileNum; j++){
			if (curr_sk->TileDimlist[j] == 1){
				error("Prefetching in 1D Tiles not implemented\n");
				Tile1D<VALUE_TYPE>* tmp = (Tile1D<VALUE_TYPE>*) curr_sk->TileList[j];
			}
			else if (curr_sk->TileDimlist[j] == 2){
				Tile2D<VALUE_TYPE>* tmp = (Tile2D<VALUE_TYPE>*) curr_sk->TileList[j];
				if(!tmp->W_total){
					long double temp_fetch_cost = tmp->getMinLinkCost(dev_id);
					if(temp_fetch_cost == min_fetch_cost){
						tiles_prefetched++;
						if (tmp->StoreBlock[dev_id_idx] == NULL ||
							tmp->StoreBlock[dev_id_idx]->State == INVALID) {
							if(tmp->StoreBlock[dev_id_idx] != NULL) tmp->StoreBlock[dev_id_idx]->Owner_p = NULL;
							state new_block_state;
							new_block_state = SHARABLE;
							tmp->StoreBlock[dev_id_idx] = Global_Cache[dev_id_idx]->assign_Cblock(new_block_state,false);
							tmp->StoreBlock[dev_id_idx]->set_owner((void**)&tmp->StoreBlock[dev_id_idx],false);
#ifdef DEBUG
							lprintf(lvl, "SubkernelPrefetchCheapRONLYTiles(dev=%d)-Tile(%d.[%d,%d]): Asigned buffer Block in GPU(%d)= %d\n",
									dev_id, tmp->id, tmp->GridId1, tmp->GridId2, dev_id, tmp->StoreBlock[dev_id_idx]->id);
#endif
							short FetchFromId = -42;
							if(!tmp->W_total) FetchFromId = tmp->getClosestReadLoc(dev_id);

							if (FetchFromId == dev_id) error("SubkernelPrefetchCheapRONLYTiles(dev=%d)-Tile(%d.[%d,%d])::request_tile W_flag = %d, \
								FetchFromId == dev_id == %d, state[%d] == %s\n",  dev_id, tmp->id, tmp->GridId1, tmp->GridId2,
								tmp->W_flag, FetchFromId, FetchFromId, print_state(tmp->StoreBlock[idxize(FetchFromId)]->State));
							short FetchFromId_idx = idxize(FetchFromId);
#ifdef DEBUG
							lprintf(lvl, "SubkernelPrefetchCheapRONLYTiles(dev=%d)-Tile(%d.[%d,%d]): Fetching Block(%d) on GPU(%d) from Block(%d) on GPU(%d)\n",
								dev_id, tmp->id,  tmp->GridId1, tmp->GridId2, tmp->StoreBlock[dev_id_idx]->id, dev_id,
								tmp->StoreBlock[FetchFromId_idx]->id, FetchFromId);
#endif
							CBlock_wrap_p wrap_read = NULL;
							if (tmp->StoreBlock[FetchFromId_idx]->State == INVALID)
								error("SubkernelPrefetchCheapRONLYTiles(dev=%d)-Tile(%d.[%d,%d])::request_tile: Fetching from tile in GPU(%d) with INVALID state\n",
									dev_id, tmp->id,  tmp->GridId1, tmp->GridId2, FetchFromId);
#ifdef STEST
							transfer_queues[dev_id_idx][FetchFromId_idx]->add_host_func((void*)&CoCoSetTimerAsync, (void*) &curr_sk->reqT_start_ts[j]);
#endif
							CoCoMemcpy2DAsync(tmp->StoreBlock[dev_id_idx]->Adrs, tmp->ldim[dev_id_idx],
								tmp->StoreBlock[FetchFromId_idx]->Adrs, tmp->ldim[FetchFromId_idx],
								tmp->dim1, tmp->dim2, tmp->dtypesize(),
								dev_id, FetchFromId, transfer_queues[dev_id_idx][FetchFromId_idx]);
#ifdef STEST
							transfer_queues[dev_id_idx][FetchFromId_idx]->add_host_func((void*)&CoCoSetTimerAsync, (void*) &curr_sk->reqT_end_ts[j]);
							curr_sk->bytes_in[j]= tmp->size();
							curr_sk->dev_in_from[j] = FetchFromId;
							curr_sk->dev_in_to[j] = dev_id;
#endif
							wrap_read = (CBlock_wrap_p) malloc (sizeof(struct CBlock_wrap));
							wrap_read->CBlock = tmp->StoreBlock[FetchFromId_idx];
							wrap_read->lockfree = false;
							transfer_queues[dev_id_idx][FetchFromId_idx]->add_host_func((void*)&CBlock_RR_wrap, (void*) wrap_read);
							tmp->StoreBlock[dev_id_idx]->Available->record_to_queue(transfer_queues[dev_id_idx][FetchFromId_idx]);
						}
					}
				}
			}
		}
	}
#ifdef DEBUG
	lprintf(lvl, "SubkernelPrefetchCheapRONLYTiles(dev=%d): Prefetched %d out of %d tiles with cost %llf\n",
		dev_id, tiles_prefetched, numTiles, (tiles_prefetched)? 1000*min_fetch_cost : 0);
#endif
	return tiles_prefetched;
}



long long failed_selections[LOC_NUM] = {0};

Subkernel* SubkernelSelectSerial(short dev_id, Subkernel** Subkernel_list, long Subkernel_list_len){
	//while(__sync_lock_test_and_set (&SubkernelSelectLock, 1));
	Subkernel* curr_sk = NULL;
	long sk_idx;
	//if(!Subkernel_list_len) warning("SubkernelSelectSerial: Gave 0 subkernel len with list = %p\n", Subkernel_list);
	for (sk_idx = 0; sk_idx < Subkernel_list_len; sk_idx++){
		curr_sk = Subkernel_list[sk_idx];
		//printf("SubkernelSelectSerial(dev_id=%d): curr_sk->run_dev_id = %d, curr_sk->no_locked_tiles() = %d, curr_sk->is_RW_lock_master(dev_id) = %d\n",
		//	dev_id, curr_sk->run_dev_id, curr_sk->no_locked_tiles(), curr_sk->is_RW_lock_master(dev_id));
		if (curr_sk->run_dev_id==-42 && (curr_sk->no_locked_tiles() || curr_sk->is_RW_lock_master(dev_id))) break;
	}
	if(sk_idx==Subkernel_list_len){
		failed_selections[idxize(dev_id)]++;
		curr_sk = NULL;
	}
	else curr_sk->prepare_launch(dev_id);
	//__sync_lock_release(&SubkernelSelectLock);
	return curr_sk;
}

Subkernel* SubkernelSelect(short dev_id, Subkernel** Subkernel_list, long Subkernel_list_len){
#ifdef SERIAL_SUBKERNEL_SELECTION
		return SubkernelSelectSerial(dev_id, Subkernel_list, Subkernel_list_len);
#endif
	Subkernel* curr_sk = NULL;
	long sk_idx;
	long double min_fetch_cost = DBL_MAX;
	Subkernel* min_fetch_cost_sk = NULL;
	for (sk_idx = 0; sk_idx < Subkernel_list_len; sk_idx++){
		curr_sk = Subkernel_list[sk_idx];
		if (curr_sk->run_dev_id==-42 &&
				(curr_sk->no_locked_tiles() || curr_sk->is_RW_lock_master(dev_id))) {
			long double fetch_cost = curr_sk->opt_fetch_cost(dev_id);
			if(!curr_sk->no_locked_tiles()) fetch_cost+=PARALLELBBOTLENECK_PENALTY*fetch_cost;
			else if(!curr_sk->is_RW_lock_master(dev_id)) fetch_cost+=EXSTEAL_PENALTY*fetch_cost/curr_sk->TileNum;
			if(fetch_cost < min_fetch_cost){
				min_fetch_cost = fetch_cost;
				min_fetch_cost_sk = curr_sk;
			}
			// else if(fetch_cost == min_fetch_cost && curr_sk->no_locked_tiles()){
			// 	printf("Same fetch cost\n");
			// 	int other_dev_score_winner = 0;
			// 	for (int dev = 0; dev< LOC_NUM; dev++) if(deidxize(dev)!= dev_id){
			// 		long double fetch_cost_me = curr_sk->opt_fetch_cost(deidxize(dev)),
			// 		fetch_cost_bro = min_fetch_cost_sk->opt_fetch_cost(deidxize(dev));
			// 		if(fetch_cost_me <= fetch_cost_bro) other_dev_score_winner ++;
			// 		else other_dev_score_winner--;
			// 	}
			// 	if(other_dev_score_winner < 0){
			// 		min_fetch_cost = fetch_cost;
			// 		min_fetch_cost_sk = curr_sk;
			// 	}
			// }
		}
	}
	if(min_fetch_cost_sk) min_fetch_cost_sk->prepare_launch(dev_id);
	else 	failed_selections[idxize(dev_id)]++;
	return min_fetch_cost_sk;
}

void CoCopeLiaDevCacheFree(short dev_id){
	delete Global_Cache[idxize(dev_id)];
	Global_Cache[idxize(dev_id)] = NULL;
}

#ifdef STEST
void STEST_print_SK(kernel_pthread_wrap_p* thread_dev_data_list, double routine_entry_ts, short dev_num)
{
	int Sk_num_max = 0;
	for (int d = 0; d < dev_num; d++)
		if(thread_dev_data_list[d]->SubkernelNumDev > Sk_num_max)
			Sk_num_max = thread_dev_data_list[d]->SubkernelNumDev;
#ifdef DSTEST
	lprintf(0,"Pipeline start:\n");
#endif
	int transfer_map[LOC_NUM][LOC_NUM] ={{0}}, exec_map[LOC_NUM] = {0}, total_h2d_R = 0, total_d2d_R = 0, total_reuse_R = 0, total_d2h_R = 0,
		total_h2d_W = 0, total_d2d_W = 0, total_reuse_W = 0, total_d2h_W = 0;
	double transfer_map_bw[LOC_NUM][LOC_NUM]= {{0.0}}, exec_map_perf[LOC_NUM] = {0};
#ifdef RUNTIME_SCHEDULER_VERSION
	for (int keri = 0; keri < Sk_num_max; keri++){
#else
	for (int keri = Sk_num_max - 1; keri >= 0; keri--){
#endif
		double request_t_ms[dev_num] = {0}, exec_t_ms[dev_num] = {0}, writeback_t_ms[dev_num] = {0},
			writeback_tile_ms[dev_num][3], request_tile_ms[dev_num][3];
		long long request_bytes_in[dev_num] = {0}, writeback_bytes_out[dev_num] = {0};
		for (int d = 0; d < dev_num; d++) if(keri < thread_dev_data_list[d]->SubkernelNumDev){

		exec_t_ms[d] = (thread_dev_data_list[d]->SubkernelListDev[keri]->op_end_ts -
			thread_dev_data_list[d]->SubkernelListDev[keri]->op_start_ts)*1000;
			for(int idx = 0; idx < thread_dev_data_list[d]->SubkernelListDev[keri]->TileNum; idx++){
				request_bytes_in[d]+= thread_dev_data_list[d]->SubkernelListDev[keri]->bytes_in[idx];
				request_tile_ms[d][idx] = (thread_dev_data_list[d]->SubkernelListDev[keri]->reqT_end_ts[idx] -
					thread_dev_data_list[d]->SubkernelListDev[keri]->reqT_start_ts[idx])*1000;
				request_t_ms[d]+= request_tile_ms[d][idx];
				writeback_bytes_out[d]+= thread_dev_data_list[d]->SubkernelListDev[keri]->bytes_out[idx];
				writeback_tile_ms[d][idx]= (thread_dev_data_list[d]->SubkernelListDev[keri]->wbT_end_ts[idx] -
					thread_dev_data_list[d]->SubkernelListDev[keri]->wbT_start_ts[idx])*1000;
				writeback_t_ms[d] += writeback_tile_ms[d][idx];
			}
		}
#ifdef DSTEST
		lprintf(0,"\n");
		for (int d = 0; d < dev_num ; d++) if(keri < thread_dev_data_list[d]->SubkernelNumDev)
			lprintf(0, "             Subkernel( dev =%2d, id =%6d )                                |",
				thread_dev_data_list[d]->SubkernelListDev[keri]->run_dev_id,
				thread_dev_data_list[d]->SubkernelListDev[keri]->id);
				lprintf(0,"\n");

		for (int d = 0; d < dev_num; d++) if(keri < thread_dev_data_list[d]->SubkernelNumDev)
			lprintf(0, " Req_T (%3.1lf->%3.1lf) total = %3.1lf ms (%3.1lf Gb\\s):                                |",
				(thread_dev_data_list[d]->SubkernelListDev[keri]->req_in_ts-routine_entry_ts)*1000,
				(thread_dev_data_list[d]->SubkernelListDev[keri]->req_out_ts-routine_entry_ts)*1000,
				request_t_ms[d], Gval_per_s(request_bytes_in[d], request_t_ms[d]/1000));
			lprintf(0,"\n");
#endif
		for (int tileidx = 0; tileidx < thread_dev_data_list[0]->SubkernelListDev[0]->TileNum; tileidx++){
				for (int d = 0; d < dev_num; d++) if(keri < thread_dev_data_list[d]->SubkernelNumDev){
					short Tiledim = thread_dev_data_list[d]->SubkernelListDev[keri]->TileDimlist[tileidx];
					void* TilePtr = thread_dev_data_list[d]->SubkernelListDev[keri]->TileList[tileidx];
#ifdef DSTEST
					if(d < dev_num)
					lprintf(0, " RCV_T( T(%3d.[%2d,%2d]) Dev (%2d)->(%2d) ) (%3.1lf: %3.1lf->%3.1lf) = %3.1lf ms (%3.1lf Gb\\s) |",
						(Tiledim == 2)? ((Tile2D<VALUE_TYPE>*) TilePtr)->id : ((Tile1D<VALUE_TYPE>*) TilePtr)->id,
						(Tiledim == 2)? ((Tile2D<VALUE_TYPE>*) TilePtr)->GridId1 : ((Tile1D<VALUE_TYPE>*) TilePtr)->GridId,
						(Tiledim == 2)? ((Tile2D<VALUE_TYPE>*) TilePtr)->GridId2 : -1,
						thread_dev_data_list[d]->SubkernelListDev[keri]->dev_in_from[tileidx],
						thread_dev_data_list[d]->SubkernelListDev[keri]->dev_in_to[tileidx],
						(request_tile_ms[d][tileidx])? (thread_dev_data_list[d]->SubkernelListDev[keri]->reqT_fire_ts[tileidx]-routine_entry_ts)*1000 : 0,
						(request_tile_ms[d][tileidx])? (thread_dev_data_list[d]->SubkernelListDev[keri]->reqT_start_ts[tileidx]-routine_entry_ts)*1000 : 0,
						(request_tile_ms[d][tileidx])? (thread_dev_data_list[d]->SubkernelListDev[keri]->reqT_end_ts[tileidx]-routine_entry_ts)*1000 : 0,
						request_tile_ms[d][tileidx],
						Gval_per_s(thread_dev_data_list[d]->SubkernelListDev[keri]->bytes_in[tileidx], request_tile_ms[d][tileidx]/1000));
#endif
					short dev_from = thread_dev_data_list[d]->SubkernelListDev[keri]->dev_in_from[tileidx],
						dev_to = thread_dev_data_list[d]->SubkernelListDev[keri]->dev_in_to[tileidx];
					short is_reader = (Tiledim == 2)? ((Tile2D<VALUE_TYPE>*) TilePtr)->R_flag : ((Tile1D<VALUE_TYPE>*) TilePtr)->R_flag;
					short is_writer = (Tiledim == 2)? ((Tile2D<VALUE_TYPE>*) TilePtr)->W_total : ((Tile1D<VALUE_TYPE>*) TilePtr)->W_total;
					if (dev_from != -2 && dev_to != -2){
						transfer_map[idxize(dev_to)][idxize(dev_from)]++;
						transfer_map_bw[idxize(dev_to)][idxize(dev_from)]+=
							Gval_per_s(thread_dev_data_list[d]->SubkernelListDev[keri]->bytes_in[tileidx], request_tile_ms[d][tileidx]/1000);
						if (dev_from == -1){
							if(is_writer) total_h2d_W++;
							else total_h2d_R++;
						}
						else if(dev_to == -1)
							if(is_writer) total_d2h_W++;
							else total_d2h_R++;
						else{
							if(is_writer) total_d2d_W++;
							else total_d2d_R++;
						}
					}
					else{
						if(is_writer) total_reuse_W++;
						else total_reuse_R++;
					}

				}
#ifdef DSTEST
				lprintf(0,"\n");
#endif
			}
			for (int d = 0; d < dev_num; d++) if(keri < thread_dev_data_list[d]->SubkernelNumDev){
#ifdef DSTEST
				lprintf(0, " exec_t(%s) (%3.1lf: %3.1lf->%3.1lf) = %3.1lf ms  (%3.1lf Gflops\\s)                    |",
				thread_dev_data_list[d]->SubkernelListDev[keri]->op_name,
				(thread_dev_data_list[d]->SubkernelListDev[keri]->op_fire_ts-routine_entry_ts)*1000,
				(thread_dev_data_list[d]->SubkernelListDev[keri]->op_start_ts-routine_entry_ts)*1000,
				(thread_dev_data_list[d]->SubkernelListDev[keri]->op_end_ts-routine_entry_ts)*1000,
				exec_t_ms[d], Gval_per_s(thread_dev_data_list[d]->SubkernelListDev[keri]->flops, exec_t_ms[d]/1000));
				lprintf(0,"\n");
#endif
			exec_map[idxize(thread_dev_data_list[d]->SubkernelListDev[keri]->run_dev_id)]++;
			exec_map_perf[idxize(thread_dev_data_list[d]->SubkernelListDev[keri]->run_dev_id)]+=Gval_per_s(thread_dev_data_list[d]->SubkernelListDev[keri]->flops, exec_t_ms[d]/1000);
		}
		for (int tileidx = 0; tileidx < thread_dev_data_list[0]->SubkernelListDev[0]->TileNum; tileidx++){
			short Tiledim = thread_dev_data_list[0]->SubkernelListDev[0]->TileDimlist[tileidx];
			void* TilePtr = thread_dev_data_list[0]->SubkernelListDev[0]->TileList[tileidx];
			short is_writer = (Tiledim == 2)? ((Tile2D<VALUE_TYPE>*) TilePtr)->W_total : ((Tile1D<VALUE_TYPE>*) TilePtr)->W_total;
			if(is_writer){
				for (int d = 0; d < dev_num; d++) if(keri < thread_dev_data_list[d]->SubkernelNumDev){
#ifdef DSTEST
					if(d < dev_num)
					lprintf(0, " WB_T( T(%3d.[%2d,%2d]) Dev (%2d)->(%2d) ) (%3.1lf: %3.1lf->%3.1lf) = %3.1lf ms  (%3.1lf Gb\\s) |",
						(Tiledim == 2)? ((Tile2D<VALUE_TYPE>*) TilePtr)->id : ((Tile1D<VALUE_TYPE>*) TilePtr)->id,
						(Tiledim == 2)? ((Tile2D<VALUE_TYPE>*) TilePtr)->GridId1 : ((Tile1D<VALUE_TYPE>*) TilePtr)->GridId,
						(Tiledim == 2)? ((Tile2D<VALUE_TYPE>*) TilePtr)->GridId2 : -1,
						thread_dev_data_list[d]->SubkernelListDev[keri]->dev_out_from[tileidx],
						thread_dev_data_list[d]->SubkernelListDev[keri]->dev_out_to[tileidx],
						(writeback_t_ms[d]) ? (thread_dev_data_list[d]->SubkernelListDev[keri]->wbT_fire_ts[tileidx]-routine_entry_ts)*1000 : 0,
						(writeback_t_ms[d]) ? (thread_dev_data_list[d]->SubkernelListDev[keri]->wbT_start_ts[tileidx]-routine_entry_ts)*1000 : 0,
						(writeback_t_ms[d]) ? (thread_dev_data_list[d]->SubkernelListDev[keri]->wbT_end_ts[tileidx]-routine_entry_ts)*1000 : 0,
						writeback_tile_ms[d][tileidx], Gval_per_s(thread_dev_data_list[d]->SubkernelListDev[keri]->bytes_out[tileidx], writeback_tile_ms[d][tileidx]/1000));
#endif
						short dev_from = thread_dev_data_list[d]->SubkernelListDev[keri]->dev_out_from[tileidx],
							dev_to = thread_dev_data_list[d]->SubkernelListDev[keri]->dev_out_to[tileidx];
						short is_reader = (Tiledim == 2)? ((Tile2D<VALUE_TYPE>*) TilePtr)->R_flag : ((Tile1D<VALUE_TYPE>*) TilePtr)->R_flag;
						short is_writer = (Tiledim == 2)? ((Tile2D<VALUE_TYPE>*) TilePtr)->W_total : ((Tile1D<VALUE_TYPE>*) TilePtr)->W_total;
						if (dev_from != -2 && dev_to != -2){
							transfer_map[idxize(dev_to)][idxize(dev_from)]++;
							transfer_map_bw[idxize(dev_to)][idxize(dev_from)]+=
								Gval_per_s(thread_dev_data_list[d]->SubkernelListDev[keri]->bytes_out[tileidx], writeback_tile_ms[d][tileidx]/1000);
														if (dev_from == -1){
								if(is_writer) total_h2d_W++;
								else total_h2d_R++;
							}
							else if(dev_to == -1)
								if(is_writer) total_d2h_W++;
								else total_d2h_R++;
							else{
								if(is_writer) total_d2d_W++;
								else total_d2d_R++;
							}
						}
					}
#ifdef DSTEST
					lprintf(0,"\n");
#endif
				}
			}
	}
	CoCoSyncCheckErr();
	lprintf(0,"\n Tranfer Map:\n   |");
	for (int d2 = 0; d2 < LOC_NUM; d2++)
		lprintf(0, "  %2d  |", deidxize(d2));
	lprintf(0, "\n   |");
	for (int d2 = 0; d2 < LOC_NUM; d2++)
		lprintf(0, "-------");
	lprintf(0, "\n");
	for (int d1 = 0; d1 < LOC_NUM; d1++){
		lprintf(0, "%2d | ", deidxize(d1));
		for (int d2 = 0; d2 < LOC_NUM; d2++){
			lprintf(0, "%4d | ", transfer_map[d1][d2]);
		}
		lprintf(0, "\n");
	}

	lprintf(0,"\n Tranfer Map Achieved Bandwidths (GB/s):\n   |");
	for (int d2 = 0; d2 < LOC_NUM; d2++)
		lprintf(0, "  %2d   |", deidxize(d2));
	lprintf(0, "\n   |");
	for (int d2 = 0; d2 < LOC_NUM; d2++)
		lprintf(0, "------_-");
	lprintf(0, "\n");
	for (int d1 = 0; d1 < LOC_NUM; d1++){
		lprintf(0, "%2d | ", deidxize(d1));
		for (int d2 = 0; d2 < LOC_NUM; d2++){
			if(transfer_map[d1][d2]) lprintf(0, "%5.2lf | ", transfer_map_bw[d1][d2]/transfer_map[d1][d2]);
			else lprintf(0, "  -   | ");
		}
		lprintf(0, "\n");
	}

	lprintf(0,"\n Subkernel Exec achieved Performance (GFlops/s):\n   |");
	for (int d2 = 0; d2 < LOC_NUM; d2++)
		lprintf(0, "  %2d   |", deidxize(d2));
	lprintf(0, "\n   |");
	for (int d2 = 0; d2 < LOC_NUM; d2++)
		lprintf(0, "--------");
	lprintf(0, "\n");
	lprintf(0, "   | ");
	for (int d2 = 0; d2 < LOC_NUM; d2++){
		if(exec_map[d2]) lprintf(0, "%5.0lf | ", exec_map_perf[d2]/exec_map[d2]);
		else lprintf(0, "  -   | ");
	}
	lprintf(0, "\n   |");

#ifdef TTEST
	HopMemcpyPrint();
#endif
	lprintf(0, "\n");
	lprintf(0,"\nSum-up R-Tiles:\n");
	lprintf(0,"Total H2D transfers = %d\n", total_h2d_R);
	lprintf(0,"Total D2D transfers = %d\n", total_d2d_R);
	lprintf(0,"Total Reused = %d\n", total_reuse_R);
	lprintf(0,"Total D2H = %d\n", total_d2h_R);
	lprintf(0,"\nSum-up W-Tiles:\n");
	lprintf(0,"Total H2D transfers = %d\n", total_h2d_W);
	lprintf(0,"Total D2D transfers = %d\n", total_d2d_W);
	lprintf(0,"Total Reused = %d\n", total_reuse_W);
	lprintf(0,"Total D2H = %d\n", total_d2h_W);
	lprintf(0, "\n");
}
#endif

/*

Subkernel* SubkernelSelectMinimizeFetch(short dev_id, Subkernel** Subkernel_list, long Subkernel_list_len){
	//while(__sync_lock_test_and_set (&SubkernelSelectLock, 1));
	Subkernel* curr_sk = NULL;
	long sk_idx;
	double min_fetch_cost = DBL_MAX;
	Subkernel* min_fetch_cost_sk = NULL;
	for (sk_idx = 0; sk_idx < Subkernel_list_len; sk_idx++){
		curr_sk = Subkernel_list[sk_idx];
		if (curr_sk->run_dev_id==-42 &&
				(curr_sk->no_locked_tiles() || curr_sk->is_RW_lock_master(dev_id))) {// || curr_sk->is_RW_master(dev_id)){
			double fetch_cost = curr_sk->opt_fetch_cost(dev_id);
			if(fetch_cost < min_fetch_cost){
				min_fetch_cost = fetch_cost;
				min_fetch_cost_sk = curr_sk;
			}
		}
	}
	if(min_fetch_cost_sk) min_fetch_cost_sk->prepare_launch(dev_id);
	//__sync_lock_release(&SubkernelSelectLock);
	return min_fetch_cost_sk;
}

Subkernel* SubkernelSelectMinimizeFetchParallelBackendBotleneckPenalty(short dev_id, Subkernel** Subkernel_list, long Subkernel_list_len){
	//while(__sync_lock_test_and_set (&SubkernelSelectLock, 1));
	Subkernel* curr_sk = NULL;
	long sk_idx;
	double min_fetch_cost = DBL_MAX;
	Subkernel* min_fetch_cost_sk = NULL;
	for (sk_idx = 0; sk_idx < Subkernel_list_len; sk_idx++){
		curr_sk = Subkernel_list[sk_idx];
		if (curr_sk->run_dev_id==-42 &&
				(curr_sk->no_locked_tiles() || curr_sk->is_RW_lock_master(dev_id))) {// || curr_sk->is_RW_master(dev_id)){
			double fetch_cost = curr_sk->opt_fetch_cost_pen_multifetch(dev_id);
			if(!curr_sk->no_locked_tiles()) fetch_cost+=PARALLELBBOTLENECK_PENALTY*fetch_cost;
			else if(!curr_sk->is_RW_lock_master(dev_id)) fetch_cost+=WTILE_TRANSFER_PENALTY*fetch_cost/curr_sk->TileNum;
			if(fetch_cost < min_fetch_cost){
				min_fetch_cost = fetch_cost;
				min_fetch_cost_sk = curr_sk;
			}
		}
	}
	if(min_fetch_cost_sk) min_fetch_cost_sk->prepare_launch(dev_id);
	//__sync_lock_release(&SubkernelSelectLock);
	return min_fetch_cost_sk;
}

Subkernel* SubkernelSelectMinimizeFetchWritePenaltyMultiFetchPenalty(short dev_id, Subkernel** Subkernel_list, long Subkernel_list_len){
	//while(__sync_lock_test_and_set (&SubkernelSelectLock, 1));
	Subkernel* curr_sk = NULL;
	long sk_idx;
	double min_fetch_cost = DBL_MAX;
	Subkernel* min_fetch_cost_sk = NULL;
	for (sk_idx = 0; sk_idx < Subkernel_list_len; sk_idx++){
		curr_sk = Subkernel_list[sk_idx];
		if (curr_sk->run_dev_id==-42 &&
				(curr_sk->no_locked_tiles() || curr_sk->is_RW_lock_master(dev_id))) {// || curr_sk->is_RW_master(dev_id)){
			double fetch_cost = curr_sk->opt_fetch_cost_pen_multifetch(dev_id);
			if(!curr_sk->no_locked_tiles()) fetch_cost+=PARALLELBBOTLENECK_PENALTY*fetch_cost;
			else if(!curr_sk->is_RW_lock_master(dev_id)) fetch_cost+=WTILE_TRANSFER_PENALTY*fetch_cost/curr_sk->TileNum;
			if(fetch_cost < min_fetch_cost){
				min_fetch_cost = fetch_cost;
				min_fetch_cost_sk = curr_sk;
			}
		}
	}
	if(min_fetch_cost_sk) min_fetch_cost_sk->prepare_launch(dev_id);
	//__sync_lock_release(&SubkernelSelectLock);
	return min_fetch_cost_sk;
}

Subkernel* SubkernelSelectMinimizeFetchWritePenaltyMultiFetchPenaltyMutlidevFair(short dev_id, Subkernel** Subkernel_list, long Subkernel_list_len){
	//while(__sync_lock_test_and_set (&SubkernelSelectLock, 1));
	Subkernel* curr_sk = NULL;
	long sk_idx;
	double min_fetch_cost = 100000000;
	Subkernel* min_fetch_cost_sk = NULL;
	for (sk_idx = 0; sk_idx < Subkernel_list_len; sk_idx++){
		curr_sk = Subkernel_list[sk_idx];
		if (curr_sk->run_dev_id==-42 &&
				(curr_sk->no_locked_tiles() || curr_sk->is_RW_lock_master(dev_id))) {
			double fetch_cost = curr_sk->opt_fetch_cost_pen_multifetch(dev_id);
			if(!curr_sk->no_locked_tiles()) fetch_cost+=PARALLELBBOTLENECK_PENALTY*fetch_cost;
			else if(!curr_sk->is_RW_lock_master(dev_id)) fetch_cost+=WTILE_TRANSFER_PENALTY*fetch_cost/curr_sk->TileNum;
			if(fetch_cost < min_fetch_cost){
				min_fetch_cost = fetch_cost;
				min_fetch_cost_sk = curr_sk;
			}
			else if(fetch_cost == min_fetch_cost && curr_sk->no_locked_tiles()){
				int other_dev_score_winner = 0;
				for (int dev = 0; dev< LOC_NUM; dev++) if(deidxize(dev)!= dev_id){
					double fetch_cost_me = curr_sk->opt_fetch_cost(deidxize(dev)),
					fetch_cost_bro = min_fetch_cost_sk->opt_fetch_cost(deidxize(dev));
					if(fetch_cost_me <= fetch_cost_bro) other_dev_score_winner ++;
					else other_dev_score_winner--;
				}
				if(other_dev_score_winner < 0){
					min_fetch_cost = fetch_cost;
					min_fetch_cost_sk = curr_sk;
				}
			}
		}
	}
	if(min_fetch_cost_sk) min_fetch_cost_sk->prepare_launch(dev_id);
	//__sync_lock_release(&SubkernelSelectLock);
	return min_fetch_cost_sk;
}

Subkernel* SubkernelSelectNoWriteShare(short dev_id, Subkernel** Subkernel_list, long Subkernel_list_len){
	//while(__sync_lock_test_and_set (&SubkernelSelectLock, 1));
	Subkernel* curr_sk = NULL;
	long sk_idx;
	for (sk_idx = 0; sk_idx < Subkernel_list_len; sk_idx++){
		curr_sk = Subkernel_list[sk_idx];
		if (curr_sk->run_dev_id==-42 && (!curr_sk->RW_lock_initialized() || curr_sk->is_RW_lock_master(dev_id))) break;
	}
	if(sk_idx==Subkernel_list_len) curr_sk = NULL;
	else curr_sk->prepare_launch(dev_id);
	//__sync_lock_release(&SubkernelSelectLock);
	return curr_sk;
}

Subkernel* SubkernelSelectMinimizeFetchNoWriteShareMultiFetchPenaltyMutlidevFair(short dev_id, Subkernel** Subkernel_list, long Subkernel_list_len){
	//while(__sync_lock_test_and_set (&SubkernelSelectLock, 1));
	Subkernel* curr_sk = NULL;
	long sk_idx;
	double min_fetch_cost = DBL_MAX;
	Subkernel* min_fetch_cost_sk = NULL;
	for (sk_idx = 0; sk_idx < Subkernel_list_len; sk_idx++){
		curr_sk = Subkernel_list[sk_idx];
		if (curr_sk->run_dev_id==-42 &&
				(!curr_sk->RW_lock_initialized() || curr_sk->is_RW_lock_master(dev_id))) {
			double fetch_cost = curr_sk->opt_fetch_cost_pen_multifetch(dev_id);
			if(!curr_sk->no_locked_tiles()) fetch_cost+=PARALLELBBOTLENECK_PENALTY*fetch_cost;
			if(fetch_cost < min_fetch_cost){
				min_fetch_cost = fetch_cost;
				min_fetch_cost_sk = curr_sk;
			}
			else if(fetch_cost == min_fetch_cost && curr_sk->no_locked_tiles()){
				int other_dev_score_winner = 0;
				for (int dev = 0; dev< LOC_NUM; dev++) if(deidxize(dev)!= dev_id){
					double fetch_cost_me = curr_sk->opt_fetch_cost(deidxize(dev)),
					fetch_cost_bro = min_fetch_cost_sk->opt_fetch_cost(deidxize(dev));
					if(fetch_cost_me <= fetch_cost_bro) other_dev_score_winner ++;
					else other_dev_score_winner--;
				}
				if(other_dev_score_winner < 0){
					min_fetch_cost = fetch_cost;
					min_fetch_cost_sk = curr_sk;
				}
			}
		}
	}
	if(min_fetch_cost_sk) min_fetch_cost_sk->prepare_launch(dev_id);
	//__sync_lock_release(&SubkernelSelectLock);
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
*/
