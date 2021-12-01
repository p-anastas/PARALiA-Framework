///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The CoCopeLia caching functions.
///

//#include <cassert>
//#include <atomic>

#include "unihelpers.hpp"
#include "DataCaching.hpp"
#include "Asset.hpp"
#include "backend_lib_wrappers.hpp"

DevBufPtr GloBuf[128] = {NULL};
short recursion_depth[128] = {0};

DevBufPtr CoCoPeLiaGlobufInit(kernel_pthread_wrap_p subkernel_data){
  DevBufPtr result = (DevBufPtr) malloc (sizeof(struct globuf));
  int dev_id = result->dev_id = subkernel_data->dev_id;
  if (dev_id < 0 ) error("CoCoPeLiaGlobufInit called with dev_id=%d\n", dev_id);
  result->gpu_mem_buf = NULL;
  result->gpu_mem_buf_sz = result->BlockNum = result->BlockSize = result->serialCtr = 0;
	;
	for (int i = 0; i < subkernel_data->SubkernelNumDev; i++){
		Subkernel* curr = subkernel_data->SubkernelListDev[i];
		for (int j = 0; j < curr->TileNum; j++){
			if (curr->TileDimlist[j] == 1) error("CoCoPeLiaGlobufInit: Tile1D not implemented\n");
			else if (curr->TileDimlist[j] == 2){
					Tile2D<VALUE_TYPE>* tmp = (Tile2D<VALUE_TYPE>*) curr->TileList[j];
					if (tmp->CacheLocId[dev_id] == -42){
						tmp->CacheLocId[dev_id] = 0;
						//printf("Found Tile with INVALID entry, adding %d to result", tmp->size());
						result->BlockSize = (long long) fmax(result->BlockSize, tmp->size());
						result->BlockNum++;
					}
			}
			else error("CoCoPeLiaGlobufInit: Not implemented for TileDim=%d\n", curr->TileDimlist[j]);
		}
	}

	result->gpu_mem_buf_sz= result->BlockSize*result->BlockNum;
	result->BlockStatus = (state*) calloc (result->BlockNum, sizeof(state));
	CoCoPeLiaDevCacheInvalidate(subkernel_data);
	return result;
}

void CoCoPeLiaRequestBuffer(kernel_pthread_wrap_p subkernel_data){
  short lvl = 3;
#ifdef DEBUG
  lprintf(lvl-1, "|-----> CoCoPeLiaRequestBuffer(%d)\n", subkernel_data->dev_id);
#endif
#ifdef TEST
	double cpu_timer = csecond();
#endif
  short dev_id = subkernel_data->dev_id;
  if (dev_id < 0 ) error("CoCoPeLiaRequestBuffer called with dev_id=%d\n", dev_id);
	DevBufPtr temp_globuf = CoCoPeLiaGlobufInit(subkernel_data);
  long long free_dev_mem, max_dev_mem, prev_globuf_sz = 0;
	if (GloBuf[dev_id] != NULL) prev_globuf_sz = GloBuf[dev_id]->gpu_mem_buf_sz;
  CoCoPeLiaDevGetMemInfo(&free_dev_mem, &max_dev_mem);
  long long problem_avail_mem = free_dev_mem - max_dev_mem*(1-PROBLEM_GPU_PERCENTAGE/100.0) + prev_globuf_sz;
  // For debuging large cases
  //problem_avail_mem/=120;
  #ifdef DEBUG
  	lprintf(lvl, "====================================\n");
  	lprintf(lvl, "GPU mem management:\n");
  	lprintf(lvl, " -Buffer requested for BlockSize=%zu and BlockNum=%d\n", (size_t) temp_globuf->BlockSize/1024/1024, temp_globuf->BlockNum);
  	lprintf(lvl, " -Mem required for matrices: %zu MB\n", (size_t) temp_globuf->gpu_mem_buf_sz/1024/1024);
    lprintf(lvl, " -Mem available in GPU: %zu MB\n", (size_t) problem_avail_mem/1024/1024);
  #endif

  if (temp_globuf->gpu_mem_buf_sz <= problem_avail_mem){
#ifdef DEBUG
    lprintf(lvl, " -Requested buffer fits in GPU\n");
#endif
	;}
	else{
    temp_globuf->BlockNum =  (int) problem_avail_mem/temp_globuf->BlockSize;
		temp_globuf->gpu_mem_buf_sz = temp_globuf->BlockNum*temp_globuf->BlockSize;
#ifdef DEBUG
    lprintf(lvl, " -Requested buffer does not fit in GPU\n");
#endif
	}

#ifdef DEBUG
  if (prev_globuf_sz >= temp_globuf->gpu_mem_buf_sz)
		lprintf(lvl, " -GPU buf available: %zu MB\n", (size_t) prev_globuf_sz/1024/1024);
  else if (prev_globuf_sz > 0) lprintf(lvl, " -Smaller GPU buf available -> resizing : %zu -> %zu MB\n",
		(size_t) prev_globuf_sz/1024/1024, (size_t) temp_globuf->gpu_mem_buf_sz/1024/1024);
  else lprintf(lvl, " -Initializing new buffer: %zu MB\n",(size_t) temp_globuf->gpu_mem_buf_sz/1024/1024);
#endif
  if (prev_globuf_sz >= temp_globuf->gpu_mem_buf_sz){
		temp_globuf->gpu_mem_buf_sz = GloBuf[dev_id]->gpu_mem_buf_sz;
		temp_globuf->gpu_mem_buf = GloBuf[dev_id]->gpu_mem_buf;
		free(GloBuf[dev_id]->BlockStatus);
		free(GloBuf[dev_id]);
		GloBuf[dev_id] = temp_globuf;
	}
  else{
		if (prev_globuf_sz > 0){
		  CoCoFree(GloBuf[dev_id]->gpu_mem_buf, dev_id);
			free(GloBuf[dev_id]->BlockStatus);
			free(GloBuf[dev_id]);
		}
    GloBuf[dev_id] = temp_globuf;
		GloBuf[dev_id]->gpu_mem_buf = CoCoMalloc(GloBuf[dev_id]->gpu_mem_buf_sz,
			dev_id);
  }
  CoCoSyncCheckErr();

#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "GPU(%d) Buffer allocation with sz = %zu MB: t_alloc = %lf ms\n" ,
    dev_id, (size_t) GloBuf[dev_id]->gpu_mem_buf_sz/1024/1024, cpu_timer*1000);
#endif
#ifdef DEBUG
	lprintf(lvl, "GPU(%d) Buffer allocation (Size = %zu MB, Blocksize = %zu MB, BlockNum = %d)\n" ,
  dev_id, (size_t) GloBuf[dev_id]->gpu_mem_buf_sz/1024/1024,
  (size_t) GloBuf[dev_id]->BlockSize/1024/1024,  GloBuf[dev_id]->BlockNum);
	lprintf(lvl-1, "<-----|\n");
#endif
}

void* CoCoPeLiaAsignBuffer(short dev_id, int* BlockIDptr){
  short lvl = 5;
#ifdef DEBUG
  lprintf(lvl-1, "|-----> CoCoPeLiaAsignBuffer(%d)\n", dev_id);
#endif
  if (dev_id < 0 ) error("CoCoPeLiaAsignBuffer called with dev_id=%d\n", dev_id);
	void* result = NULL;
  if (GloBuf[dev_id]->serialCtr >= GloBuf[dev_id]->BlockNum) return NULL;
		//error("CoCoPeLiaAsignBuffer: Buffer full but request for more Blocks\n");
	else{
  	result = GloBuf[dev_id]->gpu_mem_buf + GloBuf[dev_id]->serialCtr*GloBuf[dev_id]->BlockSize;
		*BlockIDptr = GloBuf[dev_id]->serialCtr;
		GloBuf[dev_id]->BlockStatus[GloBuf[dev_id]->serialCtr] = BUSY; // Cache block
		GloBuf[dev_id]->serialCtr++;
	}
#ifdef DEBUG
	lprintf(lvl-1, "<-----|\n");
#endif
  return result;
}

void* CoCoPeLiaReAsignBuffer(Subkernel* inkernel, int* BlockIDptr){
	short lvl = 5;
#ifdef DEBUG
	lprintf(lvl-1, "|-----> CoCoPeLiaReAsignBuffer(%d)\n", inkernel->run_dev_id);
#endif
  int	dev_id = inkernel->run_dev_id;
  if (dev_id < 0 ) error("CoCoPeLiaReAsignBuffer called with dev_id=%d\n", dev_id);
  if (inkernel->prev == NULL)
    error("CoCoPeLiaReAsignBuffer: Called from first subkernel in dev(%d). Given T too large for GPU.\n", dev_id);

  Subkernel* current = inkernel;
  while(current->prev!=NULL) current = current->prev;
	void* result = NULL;
	while(current!=inkernel){
		if (!current->work_complete) for (int j = 0; j < current->TileNum; j++){
			if (current->TileDimlist[j] == 1) error("CoCoPeLiaReAsignBuffer: Tile1D not implemented\n");
			else if (current->TileDimlist[j] == 2){
					Tile2D<VALUE_TYPE>* tmp = (Tile2D<VALUE_TYPE>*) current->TileList[j];
          if (tmp->CacheLocId[dev_id] == -1) error("CoCoPeLiaReAsignBuffer: tile with \
          CacheLocId[%d] = %d\n", dev_id, tmp->CacheLocId[dev_id]);
          else{
  					if (current->operation_complete->is_complete()){ //Works only because kernels are executed in order per device
              if (!tmp->writeback){
                CoCoPeLiaUpdateCache(dev_id, tmp->CacheLocId[dev_id], AVAILABLE);
#ifdef DDEBUG
                lprintf(lvl, "Operation complete - Block with id=%d available\n", tmp->CacheLocId[dev_id]);
#endif
              }
              if (!current->writeback_master) current->work_complete = 1;
            }
  					else{
              if(tmp->CacheLocId[dev_id]== -42) error("CoCoPeLiaReAsignBuffer: tile with \
              CacheLocId[%d] = %d used in incomplete operation\n", dev_id, tmp->CacheLocId[dev_id]);
              // Cache block used in operation
              CoCoPeLiaUpdateCache(dev_id, tmp->CacheLocId[dev_id], BUSY);
  #ifdef DDEBUG
            	lprintf(lvl, "Operation Pending - Block with id=%d busy\n", tmp->CacheLocId[dev_id]);
  #endif
            }
  					if (tmp->writeback && current->writeback_master && current->writeback_complete->is_complete()){
              // Writeback Master responsible for transfer bach to master
              CoCoPeLiaUpdateCache(dev_id, tmp->CacheLocId[dev_id], AVAILABLE);
              current->work_complete = 1;
  #ifdef DDEBUG
              lprintf(lvl, "Writeback complete - Block with id=%d available\n", tmp->CacheLocId[dev_id]);
  #endif
            }
        }
      }
		}
  	current = current->next;
	}

	for(int idx = 0; idx < GloBuf[dev_id]->BlockNum; idx++)
		if (GloBuf[dev_id]->BlockStatus[idx] == AVAILABLE){
			current = inkernel->prev;
			while(current!=NULL){
				for (int j = 0; j < current->TileNum; j++){
					if (current->TileDimlist[j] == 1) error("CoCoPeLiaReAsignBuffer: Tile1D not implemented\n");
					else if (current->TileDimlist[j] == 2){
							Tile2D<VALUE_TYPE>* tmp = (Tile2D<VALUE_TYPE>*) current->TileList[j];
							if (tmp->CacheLocId[dev_id] == idx){
                tmp->CacheLocId[dev_id] = -42;
              }
					}
				}
				current = current->prev;
			}
			result = GloBuf[dev_id]->gpu_mem_buf + idx*GloBuf[dev_id]->BlockSize;
			*BlockIDptr = idx;
      CoCoPeLiaUpdateCache(dev_id,  idx, BUSY);
#ifdef DEBUG
			lprintf(lvl, "Replacing old Block with new on id = %d\n", idx);
#endif
			return result;
		}
	// If this part is reached, there was supposedly no tile free to replace.
	// sync until the latest operation is complete and try again
	#ifdef DEBUG
		lprintf(lvl, "No block found available for replacement, waiting for last operation to complete\n");
	#endif
  if(recursion_depth[dev_id]==0) inkernel->prev->operation_complete->sync_barrier();
  else if(recursion_depth[dev_id]==1 && inkernel->prev->writeback_master) inkernel->writeback_complete->sync_barrier();
  else error("CoCoPeLiaReAsignBuffer: reached max recursion_depth=%d \
  while searching which Blocks to remove from Cache. Given T too large for GPU.\n", recursion_depth[dev_id]);
#ifdef DEBUG
  lprintf(lvl-1, "<-----|\n");
#endif
  recursion_depth[dev_id]++;
	return CoCoPeLiaReAsignBuffer(inkernel, BlockIDptr);
}


void* CoCoPeLiaUnlockCache(short dev_id){
	short lvl = 5;
#ifdef DEBUG
	lprintf(lvl-1, "|-----> CoCoPeLiaUnlockCache(%d)\n", dev_id);
#endif
  if (dev_id < 0 ) error("CoCoPeLiaUnlockCache called with dev_id=%d\n", dev_id);
	for(int idx = 0; idx < GloBuf[dev_id]->BlockNum; idx++)  if (GloBuf[dev_id]!= NULL)
    if (GloBuf[dev_id]->BlockStatus[idx] == LOCKED) GloBuf[dev_id]->BlockStatus[idx] = BUSY;
#ifdef DEBUG
	lprintf(lvl-1, "<-----|\n");
#endif
}



void CoCopeLiaDgemm_flush_gpu_mem_buf(short dev_id)
{
	short lvl = 3;
#ifdef DEBUG
			lprintf(lvl-1, "|-----> CoCopeLiaDgemm_flush_gpu_mem_buf(dev_id=%d)\n", dev_id);
#endif
  if (dev_id < 0 ) error("CoCopeLiaDgemm_flush_gpu_mem_buf called with dev_id=%d\n", dev_id);
#ifdef TEST
	double cpu_timer = csecond();
#endif
	if (GloBuf[dev_id]){
#ifdef DEBUG
		lprintf(lvl, "Clearing (presumably) %zu MB\n\n", (size_t) GloBuf[dev_id]->gpu_mem_buf_sz/1024/1024);
#endif
		CoCoFree(GloBuf[dev_id]->gpu_mem_buf, dev_id);
		free(GloBuf[dev_id]->BlockStatus);
		free(GloBuf[dev_id]);
		GloBuf[dev_id] = NULL;
    recursion_depth[dev_id] = 0;
		CoCoSyncCheckErr();
	}
	else{
#ifdef DEBUG
		lprintf(lvl, "Target buffer already empty\n");
#endif
		;
	}
#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Buffer de-allocation(%d): t_free = %lf ms\n" , dev_id, cpu_timer*1000);
#endif
#ifdef DEBUG
	lprintf(lvl-1, "<-----|\n");
#endif
}

void CoCoPeLiaDevCacheInvalidate(kernel_pthread_wrap_p subkernel_data){
	short lvl = 3;
#ifdef DEBUG
	lprintf(lvl-1, "|-----> CoCoPeLiaDevCacheInvalidate(subkernel_list(len=%d)\n", subkernel_data->SubkernelNumDev);
#endif
#ifdef TEST
	double cpu_timer = csecond();
#endif
  short dev_id = subkernel_data->dev_id;
  if (dev_id < 0 ) error("CoCoPeLiaDevCacheInvalidate called with dev_id=%d\n", dev_id);
	for (int i = 0; i < subkernel_data->SubkernelNumDev; i++){
		Subkernel* curr = subkernel_data->SubkernelListDev[i];
		for (int j = 0; j < curr->TileNum; j++){
			if (curr->TileDimlist[j] == 1) error("CoCoPeLiaDevCacheInvalidate: Tile1D not implemented\n");
			else if (curr->TileDimlist[j] == 2){
					Tile2D<VALUE_TYPE>* tmp = (Tile2D<VALUE_TYPE>*) curr->TileList[j];
					if (tmp->CacheLocId[dev_id] != -1) tmp->CacheLocId[dev_id] = -42;
			}
			else error("CoCoPeLiaDevCacheInvalidate: Not implemented for TileDim=%d\n", curr->TileDimlist[j]);
		}
	}
  if (GloBuf[dev_id]!= NULL) GloBuf[dev_id]->serialCtr = 0;
  recursion_depth[dev_id] = 0;
#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Cache for %d Subkernels invalidated: t_nv = %lf ms\n" , subkernel_data->SubkernelNumDev, cpu_timer*1000);
#endif
#ifdef DEBUG
	lprintf(lvl-1, "<-----|\n");
#endif
}

void CoCoPeLiaUpdateCache(short dev_id, int BlockIdx, state update_state){
  short lvl = 5;
  if (dev_id < 0 ) error("CoCoPeLiaUpdateCacheRecv called with dev_id=%d\n", dev_id);
  if (GloBuf[dev_id] == NULL)
    error("CoCoPeLiaUpdateCacheRecv(%d,%d): Called on empty buffer\n", dev_id, BlockIdx);
  if (GloBuf[dev_id]->BlockStatus[BlockIdx] != LOCKED)
    GloBuf[dev_id]->BlockStatus[BlockIdx] = update_state;
  #ifdef DDEBUG
  	lprintf(lvl, "\n");
  #endif
}

state CoCoPeLiaGetCacheState(short dev_id, int BlockIdx){
  if (dev_id < 0 ) error("CoCoPeLiaGetCacheState called with dev_id=%d\n", dev_id);
  if (GloBuf[dev_id] == NULL) error("CoCoPeLiaGetCacheState(%d) called on empty buffer\n", dev_id);
  return GloBuf[dev_id]->BlockStatus[BlockIdx];
}
