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
#include "backend_wrappers.hpp"

/// TODO: Works for systems with up to 128 devices, not 'completely' future-proof
CQueue_p h2d_queue[128] = {NULL}, d2h_queue[128] = {NULL}, exec_queue[128] = {NULL}, d2h_reduce_queue = NULL;
cublasHandle_t handle[128] = {NULL};
int Subkernel_num = 0;

void* reduce_buf[128] = {NULL};

Subkernel::Subkernel(short TileNum_in){
	id = Subkernel_num;
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
	delete data_available;
	delete operation_complete;
	if (W_resource_writer) delete writeback_complete;
}

void Subkernel::init_events(){
	data_available = new Event();
	operation_complete = new Event();
	if (W_resource_writer) writeback_complete = new Event();
}

void Subkernel::request_data(){
	short lvl = 4;
#ifdef DEBUG
	lprintf(lvl-1, "|-----> Subkernel(%d)::request_data()\n", id);
#endif
	if (prev!= NULL) prev->data_available->sync_barrier();
	for (int j = 0; j < TileNum; j++){
		if (TileDimlist[j] == 1) error("Subkernel(%d)::request_data: Tile1D not implemented\n", id);
		else if (TileDimlist[j] == 2){
				Tile2D<VALUE_TYPE>* tmp = (Tile2D<VALUE_TYPE>*) TileList[j];
				if (tmp->CacheLocId[run_dev_id] == -1) continue;
				else if (tmp->CacheLocId[run_dev_id] == -42){
					tmp->adrs[run_dev_id] = CoCacheAsignBlock(run_dev_id, TileList[j], TileDimlist[j]);
#ifdef CDEBUG
					lprintf(lvl, "Subkernel(%d)-Tile(%d.[%d,%d]): Asigned buffer Block in GPU(%d)= %d\n",
					id, tmp->id, tmp->GridId1, tmp->GridId2, run_dev_id, tmp->CacheLocId[run_dev_id]);
#endif
					Event* prev_event = (prev==NULL) ? NULL : prev->data_available;
					//CoCacheAddPendingEvent(run_dev_id, prev_event, data_available, tmp->CacheLocId[run_dev_id], W);
					CoCacheAddPendingEvent(run_dev_id, data_available, NULL, tmp->CacheLocId[run_dev_id], AVAILABLE);
					short FetchFromIdCAdr, FetchFromId = tmp->getClosestReadLoc(run_dev_id);
					if (FetchFromId == -1) FetchFromIdCAdr = LOC_NUM - 1;
					else{
#ifdef CDEBUG
						lprintf(lvl, "Subkernel(%d)-Tile(%d.[%d,%d]): Fetching Block(%d) on GPU(%d) from Block(%d) on GPU(%d)\n",
							id, tmp->id,  tmp->GridId1, tmp->GridId2, tmp->CacheLocId[run_dev_id], run_dev_id, tmp->CacheLocId[FetchFromId], FetchFromId);
#endif
						FetchFromIdCAdr = FetchFromId;
						if (tmp->CacheLocId[FetchFromId] == -42)
							error("Subkernel(%d)-Tile(%d.[%d,%d])::request_data: Fetching from tile in GPU(%d) with loc = -42\n",
							id, tmp->id,  tmp->GridId1, tmp->GridId2, FetchFromId);
						else if (tmp->CacheLocId[FetchFromId] != -1)
							CoCacheAddPendingEvent(FetchFromId, prev_event, data_available, tmp->CacheLocId[FetchFromId], R);
					}
					CoCoMemcpy2DAsync(tmp->adrs[run_dev_id], tmp->ldim[run_dev_id],
														tmp->adrs[FetchFromIdCAdr], tmp->ldim[FetchFromIdCAdr],
														tmp->dim1, tmp->dim2, tmp->dtypesize(),
														run_dev_id, FetchFromId, h2d_queue[run_dev_id]);
				}
				else{
					//if(tmp->writeback)
					CoCacheAddPendingEvent(run_dev_id, data_available, NULL, tmp->CacheLocId[run_dev_id], AVAILABLE);
					//else CoCacheAddPendingEvent(run_dev_id, data_available, data_available, tmp->CacheLocId[run_dev_id], R);
				}
		}
		else error("Subkernel(%d)::request_data: Not implemented for TileDim=%d\n", id, TileDimlist[j]);
	}
	data_available->record_to_queue(h2d_queue[run_dev_id]);
	//CoCoSyncCheckErr();
	#ifdef DEBUG
		lprintf(lvl-1, "<-----|\n");
	#endif
}

void Subkernel::run_operation(){
	short lvl = 4;
#ifdef DEBUG
	lprintf(lvl-1, "|-----> Subkernel(%d)::run_operation()\n", id);
#endif
	for (int j = 0; j < TileNum; j++){
		if (TileDimlist[j] == 1) error("Subkernel(%d)::run_operation: Tile1D not implemented\n", id);
		else if (TileDimlist[j] == 2){
				Tile2D<VALUE_TYPE>* tmp = (Tile2D<VALUE_TYPE>*) TileList[j];
				if (tmp->CacheLocId[run_dev_id] == -1) continue;
				else if (tmp->CacheLocId[run_dev_id] == -42)
					error("Subkernel(%d)-Tile(%d.[%d,%d])::run_operation: Tile(j=%d) has loc = -42\n",
						id, tmp->id, tmp->GridId1, tmp->GridId2, j);
				else{
					//Event* prev_event = (prev==NULL) ? NULL : &prev->operation_complete;
					Event* prev_event = data_available;
					if (tmp->writeback) CoCacheAddPendingEvent(run_dev_id, prev_event, writeback_complete, tmp->CacheLocId[run_dev_id], W);
					else CoCacheAddPendingEvent(run_dev_id, prev_event, operation_complete, tmp->CacheLocId[run_dev_id], R);
				}
		}
	}
	exec_queue[run_dev_id]->wait_for_event(data_available);
	gemm_backend_in_p ptr_ker_translate = (gemm_backend_in_p) operation_params;
#ifdef DDEBUG
	lprintf(lvl, "cublasDgemm(handle[%d], TransA = %c, TransB = %c, M = %d, N = %d, K = %d, alpha = %lf, A = %p, lda = %d, \n\
	B = %p, ldb = %d, beta = %lf, C = %p, ldC = %d)\n", run_dev_id, ptr_ker_translate->TransA, ptr_ker_translate->TransB,
		ptr_ker_translate->M, ptr_ker_translate->N, ptr_ker_translate->K, ptr_ker_translate->alpha, (VALUE_TYPE*) *ptr_ker_translate->A, ptr_ker_translate->ldA,
		(VALUE_TYPE*) *ptr_ker_translate->B, ptr_ker_translate->ldB, ptr_ker_translate->beta, (VALUE_TYPE*) *ptr_ker_translate->C, ptr_ker_translate->ldC);
#endif
	massert(CUBLAS_STATUS_SUCCESS == cublasDgemm(handle[run_dev_id], OpCharToCublas(ptr_ker_translate->TransA), OpCharToCublas(ptr_ker_translate->TransB),
		ptr_ker_translate->M, ptr_ker_translate->N, ptr_ker_translate->K, &ptr_ker_translate->alpha, (VALUE_TYPE*) *ptr_ker_translate->A, ptr_ker_translate->ldA,
		(VALUE_TYPE*) *ptr_ker_translate->B, ptr_ker_translate->ldB, &ptr_ker_translate->beta, (VALUE_TYPE*) *ptr_ker_translate->C, ptr_ker_translate->ldC),
		"Subkernel(%d)::run_operation: cublasDgemm failed\n", id);
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
	lprintf(lvl-1, "|-----> Subkernel(%d)::writeback_data()\n", id);
#endif

	d2h_queue[run_dev_id]->wait_for_event(operation_complete);
	for (int j = 0; j < TileNum; j++){
		if (TileDimlist[j] == 1) error("Subkernel(%d)::writeback_data: Tile1D not implemented\n", id);
		else if (TileDimlist[j] == 2){
				Tile2D<VALUE_TYPE>* tmp = (Tile2D<VALUE_TYPE>*) TileList[j];
				if (tmp->writeback){
					if (tmp->CacheLocId[run_dev_id] == -1) continue;
					else if (tmp->CacheLocId[run_dev_id] == -42)
						error("Subkernel(%d)-Tile(%d.[%d,%d])::writeback_data: invoked with tile with loc = -42\n",
							id, tmp->id, tmp->GridId1, tmp->GridId2);
					else{
						short WritebackIdCAdr, WritebackId = tmp->getWriteBackLoc(); //to MASTER
						if (WritebackId == -1) WritebackIdCAdr = LOC_NUM - 1;
						else WritebackIdCAdr = WritebackId;
						if(W_resource_writer==0) error("Subkernel(%d)-Tile(%d.[%d,%d])::writeback_data:\
						Subkernel is a W_resource_writer\n", id, tmp->id, tmp->GridId1, tmp->GridId2);
						else if (W_resource_writer == 1)
							CoCoMemcpy2DAsync(tmp->adrs[WritebackIdCAdr], tmp->ldim[WritebackIdCAdr],
								tmp->adrs[run_dev_id], tmp->ldim[run_dev_id],
								tmp->dim1, tmp->dim2, tmp->dtypesize(),
								WritebackId, run_dev_id, d2h_queue[run_dev_id]);
						else CoCoMemcpyReduce2DAsync(reduce_buf[WritebackIdCAdr], tmp->adrs[WritebackIdCAdr],
								   tmp->ldim[WritebackIdCAdr], tmp->adrs[run_dev_id], tmp->ldim[run_dev_id],
									 tmp->dim1, tmp->dim2, tmp->dtypesize(), WritebackId, run_dev_id, d2h_reduce_queue);
						Event* prev_event = operation_complete;
						CoCacheAddPendingEvent(run_dev_id, prev_event, writeback_complete, tmp->CacheLocId[run_dev_id], W);
					}
				}
		}
		else error("Subkernel(%d)::writeback_data: Not implemented for TileDim=%d\n", id, TileDimlist[j]);
	}
	writeback_complete->record_to_queue(d2h_queue[run_dev_id]);
	//CoCoSyncCheckErr();
#ifdef DEBUG
	lprintf(lvl-1, "<-----|\n");
#endif
}

void 	CoCoPeLiaInitStreams(short dev_id){
	//if (!d2h_reduce_queue) d2h_reduce_queue = new CommandQueue();
  if (!h2d_queue[dev_id]) h2d_queue[dev_id] = new CommandQueue();
  if (!d2h_queue[dev_id])  d2h_queue[dev_id] = new CommandQueue();
  if (!exec_queue[dev_id])  exec_queue[dev_id] = new CommandQueue();
  if (!handle[dev_id]){
    massert(CUBLAS_STATUS_SUCCESS == cublasCreate(&(handle[dev_id])), "cublasCreate failed\n");
    massert(CUBLAS_STATUS_SUCCESS == cublasSetStream(handle[dev_id], *(cudaStream_t*) (exec_queue[dev_id]->cqueue_backend_ptr)), "cublasSetStream failed\n");
  }
}
