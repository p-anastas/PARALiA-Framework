///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The "Asset" related function implementations.
///

#include "Subkernel.hpp"
#include "unihelpers.hpp"
#include "backend_lib_wrappers.hpp"
#include "backend_wrappers.hpp"

/// TODO: Works for systems with up to 128 devices, not 'completely' future-proof
CQueue_p h2d_queue[128] = {NULL}, d2h_queue[128] = {NULL}, exec_queue[128] = {NULL};
cublasHandle_t handle[128] = {NULL};

Subkernel::Subkernel(short TileNum_in){
	data_available = new Event();
	operation_complete = new Event();
	TileNum = TileNum_in;
	TileDimlist = (short*) malloc(TileNum*sizeof(short));
	TileList = (void**) malloc(TileNum*sizeof(void*));
}

void Subkernel::request_data(){
	short lvl = 4;
#ifdef DEBUG
	lprintf(lvl-1, "|-----> Subkernel::request_data()\n");
#endif
	for (int j = 0; j < TileNum; j++){
		if (TileDimlist[j] == 1) error("Subkernel::request_data: Tile1D not implemented\n");
		else if (TileDimlist[j] == 2){
				Tile2D<VALUE_TYPE>* tmp = (Tile2D<VALUE_TYPE>*) TileList[j];
				if (tmp->cachemap[run_dev_id] == INVALID){
					short FetchFromId = -1, FetchFromIdCAdr; //CoCoReturnClosestLoc(tmp->cachemap, run_dev_id);
					if (FetchFromId == -1) FetchFromIdCAdr = LOC_NUM - 1;
					else FetchFromIdCAdr = FetchFromId;
					tmp->adrs[run_dev_id] = CoCoPeLiaAsignBuffer(run_dev_id, tmp->size());
					CoCoMemcpy2DAsync(tmp->adrs[run_dev_id], tmp->ldim[run_dev_id],
														tmp->adrs[FetchFromIdCAdr], tmp->ldim[FetchFromIdCAdr],
														tmp->dim1, tmp->dim2, tmp->dtypesize(),
														run_dev_id, FetchFromId, h2d_queue[run_dev_id]);
					tmp->cachemap[run_dev_id] = AVAILABLE;
				}
		}
		else error("Subkernel::request_data: Not implemented for TileDim=%d\n", TileDimlist[j]);
		CoCoSyncCheckErr();
	}
	data_available->record_to_queue(h2d_queue[run_dev_id]);
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
	massert(CUBLAS_STATUS_SUCCESS == cublasDgemm(handle[run_dev_id], OpCharToCublas(ptr_ker_translate->TransA), OpCharToCublas(ptr_ker_translate->TransB),
		ptr_ker_translate->M, ptr_ker_translate->N, ptr_ker_translate->K, &ptr_ker_translate->alpha, ptr_ker_translate->A, ptr_ker_translate->ldA,
		ptr_ker_translate->B, ptr_ker_translate->ldB, &ptr_ker_translate->beta, ptr_ker_translate->C, ptr_ker_translate->ldC),
		"CoCoPeLiaSubkernelFireAsync: cublasDgemm failed\n");
	operation_complete->record_to_queue(exec_queue[run_dev_id]);
#ifdef DEBUG
	lprintf(lvl-1, "<-----|\n");
#endif
  CoCoSyncCheckErr();
}

void Subkernel::writeback_data(){
	d2h_queue[run_dev_id]->wait_for_event(operation_complete);
	for (int j = 0; j < TileNum; j++){
		if (TileDimlist[j] == 1) error("Subkernel::writeback_data: Tile1D not implemented\n");
		else if (TileDimlist[j] == 2){
				Tile2D<VALUE_TYPE>* tmp = (Tile2D<VALUE_TYPE>*) TileList[j];
				if (tmp->writeback){
					short WritebackId = -1, WritebackIdCAdr; //to MASTER
					if (WritebackId == -1) WritebackIdCAdr = LOC_NUM - 1;
					else WritebackIdCAdr = WritebackId;
					CoCoMemcpy2DAsync(tmp->adrs[WritebackIdCAdr], tmp->ldim[WritebackIdCAdr],
														tmp->adrs[run_dev_id], tmp->ldim[run_dev_id],
														tmp->dim1, tmp->dim2, tmp->dtypesize(),
														WritebackId, run_dev_id, d2h_queue[run_dev_id]);
				}
		}
		else error("Subkernel::writeback_data: Not implemented for TileDim=%d\n", TileDimlist[j]);
		CoCoSyncCheckErr();
	}
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

void CoCoPeLiaDevCacheInvalidate(kernel_pthread_wrap_p subkernel_data){
	for (int i = 0; i < subkernel_data->SubkernelNumDev; i++){
		Subkernel* curr = subkernel_data->SubkernelListDev[i];
		for (int j = 0; j < curr->TileNum; j++){
			if (curr->TileDimlist[j] == 1) error("CoCoPeLiaDevCacheInvalidate: Tile1D not implemented\n");
			else if (curr->TileDimlist[j] == 2){
					Tile2D<VALUE_TYPE>* tmp = (Tile2D<VALUE_TYPE>*) curr->TileList[j];
					if (tmp->cachemap[subkernel_data->devId] != MASTER) tmp->cachemap[subkernel_data->devId] = INVALID;
			}
			else error("CoCoPeLiaDevCacheInvalidate: Not implemented for TileDim=%d\n", curr->TileDimlist[j]);
		}
	}
}

long long CoCoPeLiaDevBuffSz(kernel_pthread_wrap_p subkernel_data){
	long long result = 0;
	for (int i = 0; i < subkernel_data->SubkernelNumDev; i++){
		Subkernel* curr = subkernel_data->SubkernelListDev[i];
		for (int j = 0; j < curr->TileNum; j++){
			if (curr->TileDimlist[j] == 1) error("CoCoPeLiaDevBuffSz: Tile1D not implemented\n");
			else if (curr->TileDimlist[j] == 2){
					Tile2D<VALUE_TYPE>* tmp = (Tile2D<VALUE_TYPE>*) curr->TileList[j];
					if (tmp->cachemap[subkernel_data->devId] == INVALID){
						tmp->cachemap[subkernel_data->devId] = AVAILABLE;
						//printf("Found Tile with INVALID entry, adding %d to result", tmp->size());
						result+= tmp->size();
					}
			}
			else error("CoCoPeLiaDevBuffSz: Not implemented for TileDim=%d\n", curr->TileDimlist[j]);
		}
	}
	CoCoPeLiaDevCacheInvalidate(subkernel_data);
	return result;
}

void CoCoPeLiaSubkernelFireAsync(Subkernel* subkernel){
	short lvl = 3;
#ifdef DEBUG
	lprintf(lvl-1, "|-----> CoCoPeLiaSubkernelFireAsync(subkernel)\n");
#endif

	subkernel->request_data();
	exec_queue[subkernel->run_dev_id]->wait_for_event(subkernel->data_available);
	subkernel->run_operation();
	if (subkernel->writeback_master) {
		d2h_queue[subkernel->run_dev_id]->wait_for_event(subkernel->operation_complete);
		subkernel->writeback_data();
	}
	//CoCoSyncCheckErr();
#ifdef DEBUG
	lprintf(lvl-1, "<-----|\n");
#endif
	return ;
}
