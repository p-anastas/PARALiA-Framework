///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The "Subkernel" related function implementations.
///

#include "Subkernel.hpp"
#include "linkmap.hpp"
#include "Decomposer.hpp"
#include "DataCaching.hpp"
#include "backend_wrappers.hpp"

#include <cfloat>

int Subkernel_ctr = 0, backend_init_flag[LOC_NUM] = {0};

inline int get_next_queue_ctr(int dev_id){
	exec_queue_ctr[idxize(dev_id)]++;
	if (exec_queue_ctr[idxize(dev_id)] == MAX_BACKEND_L) exec_queue_ctr[idxize(dev_id)] = 0; 
	return exec_queue_ctr[idxize(dev_id)];
}

Subkernel::Subkernel(short TileNum_in, const char* name){
	id = Subkernel_ctr;
	run_dev_id = -42;
	op_name = name;
	Subkernel_ctr++;
	TileNum = TileNum_in;
	TileDimlist = (short*) malloc(TileNum*sizeof(short));
	TileList = (DataTile_p*) malloc(TileNum*sizeof(DataTile_p));
	prev = next = NULL;
#ifdef STEST
	for (int i = 0; i < 3; i++){
		dev_in_from[i] = dev_in_to[i] = dev_out_from[i] = dev_out_to[i] = -2;
	}
#endif
}

Subkernel::~Subkernel(){
#ifdef DEBUG
	fprintf(stderr, "Subkernel(dev=%d,id=%d):~Subkernel\n", run_dev_id, id);
#endif
	short run_dev_id_idx = (run_dev_id == -1)?  LOC_NUM - 1 : run_dev_id;
	Subkernel_ctr--;
	free(TileList);
	free(operation_params);
	//delete operation_complete;
}

void Subkernel::prepare_launch(short dev_id){
#ifdef DEBUG
	fprintf(stderr, "|-----> Subkernel(dev=%d, id = %d)\n", run_dev_id, id);
#endif
	for (int j = 0; j < TileNum; j++){
		DataTile_p tmp = TileList[j];
		if(tmp->WRP == WR) {
			// Put this here to avoid block being replaced by jx < j assign schedule out
			if (tmp->StoreBlock[idxize(dev_id)] &&
				tmp->StoreBlock[idxize(dev_id)]->State != INVALID) tmp->StoreBlock[idxize(dev_id)]->add_writer();
		}
	}
#ifdef DEBUG
	fprintf(stderr, "<-----|\n");
#endif
}

void Subkernel::sync_request_data(){
	short lvl = 4;
	//return;
#ifdef DEBUG
	lprintf(lvl-1, "|-----> Subkernel(dev=%d,id=%d)::sync_request_data()\n", run_dev_id, id);
#endif
	short run_dev_id_idx = idxize(run_dev_id);
	for (int j = 0; j < TileNum; j++){
		DataTile_p tmp = TileList[j];
		if (RONLY == tmp->WRP ) tmp->StoreBlock[run_dev_id_idx]->Available->sync_barrier();
	}
#ifdef DEBUG
	lprintf(lvl-1, "<-----|\n");
#endif
}

void Subkernel::request_data(){
#ifdef DEBUG
	fprintf(stderr, "|-----> Subkernel(dev=%d,id=%d):request_data()\n", run_dev_id, id);
#endif
	short run_dev_id_idx = idxize(run_dev_id);
	CoCoPeLiaSelectDevice(run_dev_id);
	for (int j = 0; j < TileNum; j++){
		DataTile_p tmp = TileList[j];
		if (tmp->StoreBlock[run_dev_id_idx] == NULL ||
			tmp->StoreBlock[run_dev_id_idx]->State == INVALID) {
			if(tmp->StoreBlock[run_dev_id_idx] != NULL) tmp->StoreBlock[run_dev_id_idx]->Owner_p = NULL;
			state new_block_state;
			if(tmp->WRP == WR) new_block_state = EXCLUSIVE;
			else if(tmp->WRP == RONLY) new_block_state = SHARABLE;
			else error("Subkernel::request_data: Not implemented for WRP of tile %d = %s\n", j, tmp->get_WRP_string());
			tmp->StoreBlock[run_dev_id_idx] = Global_Buffer_2D[run_dev_id_idx]->assign_Cblock(new_block_state,false);
			tmp->StoreBlock[run_dev_id_idx]->set_owner((void**)&tmp->StoreBlock[run_dev_id_idx],false);
			if(tmp->WRP == WR){ 
				int WB_loc_idx = idxize(tmp->get_initial_location()); 
				tmp->StoreBlock[run_dev_id_idx]->init_writeback_info(tmp->StoreBlock[WB_loc_idx],
				&(tmp->W_master), tmp->dim1, tmp->dim2, tmp->get_chunk_size(run_dev_id_idx), tmp->get_chunk_size(WB_loc_idx),
				tmp->get_dtype_size(), recv_queues[WB_loc_idx][run_dev_id_idx], false);
			}
			else if(tmp->WRP == WREDUCE) error("Writeback not implemented for WREDUCE yet\n");
			if (tmp->WRP == WR || tmp->WRP == RONLY) tmp->fetch(run_dev_id);
		}
		else if(tmp->StoreBlock[run_dev_id_idx]->State == NATIVE && // TODO: I have no idea what this does and why it exists anymore. Maybe its useless?
			tmp->WRP == WR && tmp->W_master!= run_dev_id){
				if(tmp->WRP == WR) tmp->StoreBlock[run_dev_id_idx]->add_reader();
				if (tmp->WRP == WR || tmp->WRP == RONLY) tmp->fetch(run_dev_id);
		}
		else{
			if (tmp->WRP == RONLY) tmp->StoreBlock[run_dev_id_idx]->add_reader();
			else if (tmp->WRP == WR) tmp->StoreBlock[run_dev_id_idx]->add_writer();

		}
	}
#ifndef ASYNC_ENABLE
	CoCoSyncCheckErr();
#endif
#ifdef DEBUG
	fprintf(stderr, "<-----|\n");
#endif
}

void Subkernel::run_operation()
{
	short run_dev_id_idx = idxize(run_dev_id);
	CoCoPeLiaSelectDevice(run_dev_id);
	CQueue_p assigned_exec_queue = NULL;
	for (int j = 0; j < TileNum; j++){ // Method only works for max 1 W(rite) Tile.
		DataTile_p tmp = TileList[j];
		if(tmp->WRP == WR){
			if(tmp->W_master_backend_ctr == -42)
				tmp->W_master_backend_ctr = get_next_queue_ctr(run_dev_id);
			assigned_exec_queue = exec_queue[run_dev_id_idx][tmp->W_master_backend_ctr];
			//fprintf(stderr,"Subkernel(dev=%d,id=%d)-Tile(%d.[%d,%d])::run_operation: Tile(j=%d) RWPBC = %d\n",
			//	run_dev_id, id, tmp->id, tmp->GridId1, tmp->GridId2, j, RW_parallel_backend_ctr);
		}
	}

	for (int j = 0; j < TileNum; j++){
		DataTile_p tmp = TileList[j];
		if (tmp->StoreBlock[run_dev_id_idx] == NULL || tmp->StoreBlock[run_dev_id_idx]->State == INVALID)
			error("Subkernel(dev=%d,id=%d)-Tile(%d.[%d,%d])::run_operation: Tile(j=%d) Storeblock is NULL\n",
				run_dev_id, id, tmp->id, tmp->GridId1, tmp->GridId2, j);
		if (tmp->WRP == WR || tmp->WRP == RONLY) assigned_exec_queue->wait_for_event(tmp->StoreBlock[run_dev_id_idx]->Available);
	}

	backend_run_operation(operation_params, op_name,assigned_exec_queue);

	for (int j = 0; j < TileNum; j++){
		DataTile_p tmp = TileList[j];
		CBlock_wrap_p wrap_oper = NULL;
		wrap_oper = (CBlock_wrap_p) malloc (sizeof(struct CBlock_wrap));
		wrap_oper->CBlock = tmp->StoreBlock[run_dev_id_idx];
		wrap_oper->lockfree = false;
		if(tmp->WRP == WR){
			tmp->W_pending--;
			if(!tmp->W_pending) tmp->W_complete->record_to_queue(assigned_exec_queue);
		}
		else if(tmp->WRP == RONLY) assigned_exec_queue->add_host_func((void*)&CBlock_RR_wrap, (void*) wrap_oper);

	}
	//
#ifndef ASYNC_ENABLE
	CoCoSyncCheckErr();
#endif
}

short Subkernel::check_ready(){
	short run_dev_id_idx = idxize(run_dev_id);
	for (int j = 0; j < TileNum; j++){
		DataTile_p tmp = TileList[j];
		if (tmp->StoreBlock[run_dev_id_idx] == NULL || tmp->StoreBlock[run_dev_id_idx]->State == INVALID) return 0; 
		else if (tmp->WRP == WR || tmp->WRP == RONLY){
			event_status what = tmp->StoreBlock[run_dev_id_idx]->Available->query_status();
			if (!(COMPLETE == what || CHECKED == what)) return 0; 
		}
	}
	return 1;
}

void Subkernel::run_ready_operation(){
	CoCoPeLiaSelectDevice(run_dev_id);
	short run_dev_id_idx = idxize(run_dev_id);

	CQueue_p assigned_exec_queue = NULL;
	for (int j = 0; j < TileNum; j++){ // Method only works for max 1 W(rite) Tile.
		DataTile_p tmp = TileList[j];
		if(tmp->WRP == WR){
			if(tmp->W_master_backend_ctr == -42)
				tmp->W_master_backend_ctr = get_next_queue_ctr(run_dev_id);
			assigned_exec_queue = exec_queue[run_dev_id_idx][tmp->W_master_backend_ctr];
		}	
	}

	backend_run_operation(operation_params, op_name, assigned_exec_queue);

	for (int j = 0; j < TileNum; j++){
		DataTile_p tmp = TileList[j];
		CBlock_wrap_p wrap_oper = NULL;
		wrap_oper = (CBlock_wrap_p) malloc (sizeof(struct CBlock_wrap));
		wrap_oper->CBlock = tmp->StoreBlock[run_dev_id_idx];
		wrap_oper->lockfree = false;
		if(tmp->WRP == WR){
			tmp->W_pending--;
			if(!tmp->W_pending){
				tmp->W_complete->record_to_queue(assigned_exec_queue);
				tmp->writeback();
			}
		}
		else if(tmp->WRP == RONLY) assigned_exec_queue->add_host_func((void*)&CBlock_RR_wrap, (void*) wrap_oper);

	}
}

void CoCoPeLiaInitResources(int* dev_list, int dev_num){

	for(int i = 0; i < LOC_NUM; i++)
	for(int j = 0; j < LOC_NUM; j++)
	for(int k = 0; k < 2; k++) transfer_link_sharing[i][j][k] = -42;

#ifndef ENABLE_LINK_BW_SHARING
	///TODO: ENABLE_LINK_BW_SHARING flag is disabled, but sharing-disabler mechanism is handmade

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

	for(short dev_id_idx = 0 ; dev_id_idx < LOC_NUM; dev_id_idx++){
		for(short dev_id_idy = 0 ; dev_id_idy < LOC_NUM; dev_id_idy++)
		if(dev_id_idy!=dev_id_idx){
			if (!recv_queues[dev_id_idx][dev_id_idy]){
				//printf("dev_id = %d, dev_id_idx = %d, dev_id_idy = %d, LOC_NUM = %d\n", dev_id, dev_id_idx, dev_id_idy, LOC_NUM);
				short shared_iloc0 = transfer_link_sharing[dev_id_idx][dev_id_idy][0],
					shared_iloc1 = transfer_link_sharing[dev_id_idx][dev_id_idy][1];
				short queue_id = (dev_id_idy == LOC_NUM - 1)? deidxize(dev_id_idx) : deidxize(dev_id_idy);
				recv_queues[dev_id_idx][dev_id_idy] = new CommandQueue(queue_id, 0);
				wb_queues[dev_id_idx][dev_id_idy] = new CommandQueue(queue_id, 0);
				if( shared_iloc0 != - 42){ // The smallest index shared link allocates the queue
					if (dev_id_idx*LOC_NUM + dev_id_idy < shared_iloc0*LOC_NUM + shared_iloc1){
						recv_queues[shared_iloc0][shared_iloc1] = recv_queues[dev_id_idx][dev_id_idy];
						wb_queues[shared_iloc0][shared_iloc1] = wb_queues[dev_id_idx][dev_id_idy];
					}
				}
			}
		}
		if (!exec_queue[dev_id_idx][0]) {
			int flag_is_worker = 0; 
			for (int i = 0; i < dev_num; i++) if(dev_list[i] == deidxize(dev_id_idx)){
				flag_is_worker = 1; 
				break;
			}
			if(flag_is_worker){
				for (int i = 0; i < MAX_BACKEND_L; i++) exec_queue[dev_id_idx][i] = new CommandQueue(deidxize(dev_id_idx), 1);
			}
		}
	}

	#ifdef DEBUG
		fprintf(stderr, "<-----|\n");
	#endif
}

void CoCoPeLiaInitWS(int* dev_list, int dev_num){
	error("CoCoPeLiaInitWS: INCOMPLETE\n");
	for(short dev_id_idx = 0 ; dev_id_idx < LOC_NUM; dev_id_idx++){
		if (!exec_queue[dev_id_idx][0]) {
			int flag_is_worker = 0; 
			for (int i = 0; i < dev_num; i++) if(dev_list[i] == deidxize(dev_id_idx)){
				flag_is_worker = 1; 
				break;
			}
			if(flag_is_worker && deidxize(dev_id_idx)!= -1){
				for (int par_idx = 0; par_idx < exec_queue[dev_id_idx][0]->simultaneous_workers; par_idx++ ){
					void* local_ws = CoCoMalloc(2048, deidxize(dev_id_idx)); 
					massert(CUBLAS_STATUS_SUCCESS == cublasSetWorkspace(*((cublasHandle_t*) 
						exec_queue[dev_id_idx][0]->cqueue_backend_data[par_idx]), local_ws, 2048), 
						"CommandQueue::CommandQueue(%d): cublasSetWorkspace failed\n", deidxize(dev_id_idx));
					
				}
			}
		}
	}
}

void CoCoPeLiaFreeResources(){
	for(short dev_id_idx = 0 ; dev_id_idx < LOC_NUM; dev_id_idx++){
		for(short dev_id_idy = 0 ; dev_id_idy < LOC_NUM; dev_id_idy++)
		if(dev_id_idx!=dev_id_idy){
				delete recv_queues[dev_id_idx][dev_id_idy];
				recv_queues[dev_id_idx][dev_id_idy] = NULL;
				delete wb_queues[dev_id_idx][dev_id_idy];
				wb_queues[dev_id_idx][dev_id_idy] = NULL;
		}
		for (int i = 0; i < MAX_BACKEND_L; i++){
			delete exec_queue[dev_id_idx][i];
			exec_queue[dev_id_idx][i] = NULL;
		}
	}
}

void CoCoPeLiaCleanResources(){
	for(short dev_id_idx = 0 ; dev_id_idx < LOC_NUM; dev_id_idx++){
		for(short dev_id_idy = 0 ; dev_id_idy < LOC_NUM; dev_id_idy++)
		if(dev_id_idx!=dev_id_idy){
				recv_queues[dev_id_idx][dev_id_idy]->WT_set(0);
				wb_queues[dev_id_idx][dev_id_idy]->WT_set(0);
		}
		for (int i = 0; i < MAX_BACKEND_L; i++) exec_queue[dev_id_idx][i]->WT_set(0);
	}
}

void swap_sk(Subkernel** sk1, Subkernel** sk2){
	Subkernel* sk_tmp = *sk1;
	*sk1 = *sk2;
    *sk2 = sk_tmp; 
}

#ifdef SUBKERNEL_SELECT_MAX_FETCHED
Subkernel* SubkernelSelect(short dev_id, Subkernel** Subkernel_list, long Subkernel_list_len){
#ifdef SERIAL_SUBKERNEL_SELECTION
	Subkernel_list[0]->prepare_launch(dev_id);
	return Subkernel_list[0];
#endif
	Subkernel* curr_sk = NULL;
	int sk_idx;
	int potential_sks[Subkernel_list_len], tie_list_num = 0; 
	int max_fetches = 1;
	if(!Subkernel_list_len) error("SubkernelSelect: Gave 0 subkernel len with list = %p\n", Subkernel_list);
	for (sk_idx = 0; sk_idx < Subkernel_list_len; sk_idx++){
		curr_sk = Subkernel_list[sk_idx];
		int tmp_fetches = 0; 
		for (int j = 0; j < curr_sk->TileNum; j++){
			if(RONLY == curr_sk->TileList[j]->WRP)
				if(curr_sk->TileList[j]->loc_map[idxize(dev_id)] == 0 || 
					curr_sk->TileList[j]->loc_map[idxize(dev_id)] == 42) tmp_fetches++;
			if(WR == curr_sk->TileList[j]->WRP)
			if(curr_sk->TileList[j]->loc_map[idxize(dev_id)] == 0 || 
				curr_sk->TileList[j]->loc_map[idxize(dev_id)] == 42 ||
				curr_sk->TileList[j]->loc_map[idxize(dev_id)] == 2 ) tmp_fetches++;
		}
		if(tmp_fetches > max_fetches){
			max_fetches = tmp_fetches;
			potential_sks[0] = sk_idx;
			tie_list_num = 1; 
		}
		else if(tmp_fetches == max_fetches){
			potential_sks[tie_list_num++] = sk_idx;
		}
	}
	int selected_sk_idx = (tie_list_num)? potential_sks[int(rand() % tie_list_num)] : tie_list_num; 
	swap_sk(&(Subkernel_list[0]), &(Subkernel_list[selected_sk_idx])); 
	Subkernel_list[0]->prepare_launch(dev_id);
	return Subkernel_list[0];
}
#endif

#ifdef SUBKERNEL_SELECT_MIN_RONLY_ETA
Subkernel* SubkernelSelect(short dev_id, Subkernel** Subkernel_list, long Subkernel_list_len){
#ifdef SERIAL_SUBKERNEL_SELECTION
	Subkernel_list[0]->prepare_launch(dev_id);
	return Subkernel_list[0];
#endif
	Subkernel* curr_sk = NULL;
	long sk_idx;
	double min_ETA = DBL_MAX;
	Subkernel* min_ETA_sk = NULL;
	if(!Subkernel_list_len) error("SubkernelSelect: Gave 0 subkernel len with list = %p\n", Subkernel_list);
	for (sk_idx = 0; sk_idx < Subkernel_list_len; sk_idx++){
		curr_sk = Subkernel_list[sk_idx];
		double tmp_ETA = 0; 
		for (int j = 0; j < curr_sk->TileNum; j++) if(RONLY == curr_sk->TileList[j]->WRP){
			double block_cost = curr_sk->TileList[j]->block_ETA[idxize(dev_id)];
			if(-42 == block_cost) block_cost = curr_sk->TileList[j]->size(); 
			tmp_ETA = fmax(block_cost, tmp_ETA);
		}
		if(tmp_ETA < min_ETA){
			min_ETA = tmp_ETA;
			min_ETA_sk = curr_sk;
		}
	}
	min_ETA_sk->prepare_launch(dev_id);
	return min_ETA_sk;
}
#endif

void PARALiADevCacheFree(short dev_id){
	if(Global_Buffer_1D[idxize(dev_id)]) delete Global_Buffer_1D[idxize(dev_id)];
	Global_Buffer_1D[idxize(dev_id)] = NULL;
	if(Global_Buffer_2D[idxize(dev_id)]) delete Global_Buffer_2D[idxize(dev_id)];
	Global_Buffer_2D[idxize(dev_id)] = NULL;
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
						(Tiledim == 2)? ((Tile2D*) TilePtr)->id : ((Tile1D*) TilePtr)->id,
						(Tiledim == 2)? ((Tile2D*) TilePtr)->GridId1 : ((Tile1D*) TilePtr)->GridId,
						(Tiledim == 2)? ((Tile2D*) TilePtr)->GridId2 : -1,
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
					short is_reader = (Tiledim == 2)? ((Tile2D*) TilePtr)->R_flag : ((Tile1D*) TilePtr)->R_flag;
					short is_writer = (Tiledim == 2)? ((Tile2D*) TilePtr)->W_total : ((Tile1D*) TilePtr)->W_total;
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
			short is_writer = (Tiledim == 2)? ((Tile2D*) TilePtr)->W_total : ((Tile1D*) TilePtr)->W_total;
			if(is_writer){
				for (int d = 0; d < dev_num; d++) if(keri < thread_dev_data_list[d]->SubkernelNumDev){
#ifdef DSTEST
					if(d < dev_num)
					lprintf(0, " WB_T( T(%3d.[%2d,%2d]) Dev (%2d)->(%2d) ) (%3.1lf: %3.1lf->%3.1lf) = %3.1lf ms  (%3.1lf Gb\\s) |",
						(Tiledim == 2)? ((Tile2D*) TilePtr)->id : ((Tile1D*) TilePtr)->id,
						(Tiledim == 2)? ((Tile2D*) TilePtr)->GridId1 : ((Tile1D*) TilePtr)->GridId,
						(Tiledim == 2)? ((Tile2D*) TilePtr)->GridId2 : -1,
						thread_dev_data_list[d]->SubkernelListDev[keri]->dev_out_from[tileidx],
						thread_dev_data_list[d]->SubkernelListDev[keri]->dev_out_to[tileidx],
						(writeback_t_ms[d]) ? (thread_dev_data_list[d]->SubkernelListDev[keri]->wbT_fire_ts[tileidx]-routine_entry_ts)*1000 : 0,
						(writeback_t_ms[d]) ? (thread_dev_data_list[d]->SubkernelListDev[keri]->wbT_start_ts[tileidx]-routine_entry_ts)*1000 : 0,
						(writeback_t_ms[d]) ? (thread_dev_data_list[d]->SubkernelListDev[keri]->wbT_end_ts[tileidx]-routine_entry_ts)*1000 : 0,
						writeback_tile_ms[d][tileidx], Gval_per_s(thread_dev_data_list[d]->SubkernelListDev[keri]->bytes_out[tileidx], writeback_tile_ms[d][tileidx]/1000));
#endif
						short dev_from = thread_dev_data_list[d]->SubkernelListDev[keri]->dev_out_from[tileidx],
							dev_to = thread_dev_data_list[d]->SubkernelListDev[keri]->dev_out_to[tileidx];
						short is_reader = (Tiledim == 2)? ((Tile2D*) TilePtr)->R_flag : ((Tile1D*) TilePtr)->R_flag;
						short is_writer = (Tiledim == 2)? ((Tile2D*) TilePtr)->W_total : ((Tile1D*) TilePtr)->W_total;
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