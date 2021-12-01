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
CQueue_p h2d_queue[128] = {NULL}, d2h_queue[128] = {NULL}, exec_queue[128] = {NULL};
cublasHandle_t handle[128] = {NULL};

Subkernel::Subkernel(short TileNum_in){
	TileNum = TileNum_in;
	TileDimlist = (short*) malloc(TileNum*sizeof(short));
	TileList = (void**) malloc(TileNum*sizeof(void*));
	work_complete = 0;
	prev = next = NULL;
}

void Subkernel::init_events(){
	data_available = new Event();
	operation_complete = new Event();
	writeback_complete = new Event();
}
void Subkernel::request_data(){
	short lvl = 4;
#ifdef DEBUG
	lprintf(lvl-1, "|-----> Subkernel::request_data()\n");
#endif
	int cache_used_flag = 0;
	if (prev!= NULL){
		prev->data_available->sync_barrier();
		for (int j = 0; j < prev->TileNum; j++){
			if (prev->TileDimlist[j] == 1) error("Subkernel::request_data: Tile1D not implemented\n");
			else if (prev->TileDimlist[j] == 2){
					Tile2D<VALUE_TYPE>* tmp = (Tile2D<VALUE_TYPE>*) prev->TileList[j];
					if (!tmp->writeback) CoCoPeLiaUpdateCache(run_dev_id, tmp->CacheLocId[run_dev_id], AVAILABLE);
			}
		}
	}
	for (int j = 0; j < TileNum; j++){
		if (TileDimlist[j] == 1) error("Subkernel::request_data: Tile1D not implemented\n");
		else if (TileDimlist[j] == 2){
				Tile2D<VALUE_TYPE>* tmp = (Tile2D<VALUE_TYPE>*) TileList[j];
				//tmp->PendingUsage[run_dev_id]++;
				//if (writeback_master && tmp->writeback) tmp->PendingUsage[run_dev_id]++;
				if (tmp->CacheLocId[run_dev_id] == -42){
					tmp->adrs[run_dev_id] = CoCoPeLiaAsignBuffer(run_dev_id, &(tmp->CacheLocId[run_dev_id]));
#ifdef DEBUG
					lprintf(lvl, "Asigned buffer Block = %d\n", tmp->CacheLocId[run_dev_id]);
#endif
					if (tmp->adrs[run_dev_id] == NULL){
						tmp->adrs[run_dev_id] = CoCoPeLiaReAsignBuffer(this, &(tmp->CacheLocId[run_dev_id]));
						cache_used_flag = 1;
					}
					short FetchFromIdCAdr, FetchFromId = (short) fmax(tmp->getId(MASTER), tmp->getId(AVAILABLE)); //tmp->getId(MASTER); //
					if (FetchFromId == -1) FetchFromIdCAdr = LOC_NUM - 1;
					else FetchFromIdCAdr = FetchFromId;
					CoCoMemcpy2DAsync(tmp->adrs[run_dev_id], tmp->ldim[run_dev_id],
														tmp->adrs[FetchFromIdCAdr], tmp->ldim[FetchFromIdCAdr],
														tmp->dim1, tmp->dim2, tmp->dtypesize(),
														run_dev_id, FetchFromId, h2d_queue[run_dev_id]);
				}
				else CoCoPeLiaUpdateCache(run_dev_id, tmp->CacheLocId[run_dev_id], LOCKED);
		}
		else error("Subkernel::request_data: Not implemented for TileDim=%d\n", TileDimlist[j]);
	}
	data_available->record_to_queue(h2d_queue[run_dev_id]);
	if (cache_used_flag) CoCoPeLiaUnlockCache(run_dev_id);
	//CoCoSyncCheckErr();
	#ifdef DEBUG
		lprintf(lvl-1, "<-----|\n");
	#endif
}

void Subkernel::run_operation(){
	short lvl = 4;
#ifdef DEBUG
	lprintf(lvl-1, "|-----> Subkernel::run_operation()\n");
#endif
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
		"CoCoPeLiaSubkernelFireAsync: cublasDgemm failed\n");
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
	lprintf(lvl-1, "|-----> Subkernel::writeback_data()\n");
#endif

	d2h_queue[run_dev_id]->wait_for_event(operation_complete);
	for (int j = 0; j < TileNum; j++){
		if (TileDimlist[j] == 1) error("Subkernel::writeback_data: Tile1D not implemented\n");
		else if (TileDimlist[j] == 2){
				Tile2D<VALUE_TYPE>* tmp = (Tile2D<VALUE_TYPE>*) TileList[j];
				if (tmp->writeback){
					short WritebackIdCAdr, WritebackId = tmp->getId(MASTER); //to MASTER
					if (WritebackId == -1) WritebackIdCAdr = LOC_NUM - 1;
					else WritebackIdCAdr = WritebackId;
					CoCoMemcpy2DAsync(tmp->adrs[WritebackIdCAdr], tmp->ldim[WritebackIdCAdr],
														tmp->adrs[run_dev_id], tmp->ldim[run_dev_id],
														tmp->dim1, tmp->dim2, tmp->dtypesize(),
														WritebackId, run_dev_id, d2h_queue[run_dev_id]);
				}
		}
		else error("Subkernel::writeback_data: Not implemented for TileDim=%d\n", TileDimlist[j]);
	}
	writeback_complete->record_to_queue(d2h_queue[run_dev_id]);;
	//CoCoSyncCheckErr();
#ifdef DEBUG
	lprintf(lvl-1, "<-----|\n");
#endif
}

void 	CoCoPeLiaInitStreams(short dev_id){
  if (!h2d_queue[dev_id]) h2d_queue[dev_id] = new CommandQueue();
  if (!d2h_queue[dev_id])  d2h_queue[dev_id] = new CommandQueue();
  if (!exec_queue[dev_id])  exec_queue[dev_id] = new CommandQueue();
  if (!handle[dev_id]){
    massert(CUBLAS_STATUS_SUCCESS == cublasCreate(&(handle[dev_id])), "cublasCreate failed\n");
    massert(CUBLAS_STATUS_SUCCESS == cublasSetStream(handle[dev_id], *(cudaStream_t*) (exec_queue[dev_id]->cqueue_backend_ptr)), "cublasSetStream failed\n");
  }
}
