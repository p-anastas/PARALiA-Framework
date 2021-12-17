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
	delete operation_complete;
	if (WR_writer) delete writeback_complete;
}

void Subkernel::init_events(){
	operation_complete = new Event();
	if (WR_writer) writeback_complete = new Event();
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
				if (tmp->CacheLocId[run_dev_id] != -1) tmp->available[run_dev_id]->sync_barrier();
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
				else if (tmp->CacheLocId[run_dev_id] < -1){
					tmp->adrs[run_dev_id] = CoCacheAsignBlock(run_dev_id, TileList[j], TileDimlist[j]);
#ifdef CDEBUG
					lprintf(lvl, "Subkernel(dev=%d,id=%d)-Tile(%d.[%d,%d]): Asigned buffer Block in GPU(%d)= %d\n",
					 run_dev_id, id, tmp->id, tmp->GridId1, tmp->GridId2, run_dev_id, tmp->CacheLocId[run_dev_id]);
#endif

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
					if (tmp->writeback) CoCacheAddPendingEvent(run_dev_id, tmp->available[run_dev_id], writeback_complete, tmp->CacheLocId[run_dev_id], W);
					else CoCacheAddPendingEvent(run_dev_id, tmp->available[run_dev_id], operation_complete, tmp->CacheLocId[run_dev_id], R);
				}
				else{
					//CoCacheAddPendingEvent(run_dev_id, data_available, NULL, tmp->CacheLocId[run_dev_id], AVAILABLE);
					if (tmp->writeback) CoCacheAddPendingEvent(run_dev_id, tmp->available[run_dev_id], writeback_complete, tmp->CacheLocId[run_dev_id], W);
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
				else exec_queue[run_dev_id]->wait_for_event(tmp->available[run_dev_id]);
		}
	}
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
		"Subkernel(dev=%d,id=%d)::run_operation: cublasDgemm failed\n", run_dev_id, id);
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
				if (tmp->writeback){
					if (tmp->CacheLocId[run_dev_id] == -1) continue;
					else if (tmp->CacheLocId[run_dev_id] < -1)
						error("Subkernel(dev=%d,id=%d)-Tile(%d.[%d,%d])::writeback_data: invoked with tile with loc = %d\n",
							run_dev_id, id, tmp->id, tmp->GridId1, tmp->GridId2, tmp->CacheLocId[run_dev_id]);
					else{
						short WritebackIdCAdr, WritebackId = tmp->getWriteBackLoc(); //to MASTER
						if (WritebackId == -1) WritebackIdCAdr = LOC_NUM - 1;
						else WritebackIdCAdr = WritebackId;
						if(WR_writer==0) error("Subkernel(dev=%d,id=%d)-Tile(%d.[%d,%d])::writeback_data:\
						Subkernel is a WR_writer\n", run_dev_id, id, tmp->id, tmp->GridId1, tmp->GridId2);
						else if (WR_writer == 1)
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
		else error("Subkernel(dev=%d,id=%d)::writeback_data: Not implemented for TileDim=%d\n", run_dev_id, id, TileDimlist[j]);
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
