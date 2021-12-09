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
int globalock = 0;
short recursion_depth[128] = {0};

const char* print_state(state in_state){
	switch(in_state){
		case(EMPTY):
			return "EMPTY";
		case(AVAILABLE):
			return "AVAILABLE";
		case(R):
			return "R";
		case(W):
			return "W";
		default:
			error("print_state: Unknown state\n");
	}
}

int pending_events_free(pending_events_p* target){
	int num_freed = 0;
	pending_events_p current = *target, tmp;
	while(current!=NULL){
		tmp = current;
		current = current->next;
		free(current);
		num_freed++;
	}
	*target = NULL;
	return num_freed;
}

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
	result->BlockState = (state*) calloc (result->BlockNum, sizeof(state));
	result->BlockPendingEvents = (pending_events_p*) malloc (result->BlockNum* sizeof(pending_events_p));
	result->BlockCurrentTileDim = (short*) malloc (result->BlockNum* sizeof(short));
	result->BlockCurrentTilePtr = (void**) malloc (result->BlockNum* sizeof(void*));
	for (int idx = 0; idx < result->BlockNum; idx++) result->BlockPendingEvents[idx] = NULL;
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
  //problem_avail_mem/=125;
  #ifdef DEBUG
  	lprintf(lvl, "====================================\n");
  	lprintf(lvl, "GPU mem management:\n");
  	lprintf(lvl, " -Buffer requested for BlockSize=%zu MB and BlockNum=%d\n", (size_t) temp_globuf->BlockSize/1024/1024, temp_globuf->BlockNum);
  	lprintf(lvl, " -Mem required for matrices: %zu MB\n", (size_t) temp_globuf->gpu_mem_buf_sz/1024/1024);
    lprintf(lvl, " -Mem available in GPU: %zu MB\n", (size_t) problem_avail_mem/1024/1024);
  #endif

  if (temp_globuf->gpu_mem_buf_sz <= problem_avail_mem){
#ifdef DEBUG
    lprintf(lvl, " -Requested buffer fits in GPU(%d)\n", dev_id);
#endif
	;}
	else{
    temp_globuf->BlockNum =  (int) (problem_avail_mem/temp_globuf->BlockSize);
		temp_globuf->gpu_mem_buf_sz = temp_globuf->BlockNum*temp_globuf->BlockSize;
#ifdef DEBUG
    lprintf(lvl, " -Requested buffer does not fit in GPU(%d)\n", dev_id);
#endif
	}

#ifdef DEBUG
  if (prev_globuf_sz >= temp_globuf->gpu_mem_buf_sz)
		lprintf(lvl, " -GPU(%d) buf available: %zu MB\n", dev_id, (size_t) prev_globuf_sz/1024/1024);
  else if (prev_globuf_sz > 0) lprintf(lvl, " -Smaller GPU(%d) buf available -> replacing : %zu -> %zu MB\n",
		dev_id, (size_t) prev_globuf_sz/1024/1024, (size_t) temp_globuf->gpu_mem_buf_sz/1024/1024);
  else lprintf(lvl, " -Initializing new GPU(%d) buffer: %zu MB\n", dev_id, (size_t) temp_globuf->gpu_mem_buf_sz/1024/1024);
#endif
  if (prev_globuf_sz >= temp_globuf->gpu_mem_buf_sz){
		temp_globuf->gpu_mem_buf_sz = GloBuf[dev_id]->gpu_mem_buf_sz;
		temp_globuf->gpu_mem_buf = GloBuf[dev_id]->gpu_mem_buf;
		for (int ifr = 0; ifr < GloBuf[dev_id]->BlockNum; ifr++)
			pending_events_free(&GloBuf[dev_id]->BlockPendingEvents[ifr]);
		free(GloBuf[dev_id]->BlockPendingEvents);
		free(GloBuf[dev_id]->BlockState);
		free(GloBuf[dev_id]->BlockCurrentTileDim);
		free(GloBuf[dev_id]->BlockCurrentTilePtr);
		free(GloBuf[dev_id]);
		GloBuf[dev_id] = temp_globuf;
	}
  else{
		if (prev_globuf_sz > 0){
		  CoCoFree(GloBuf[dev_id]->gpu_mem_buf, dev_id);
			for (int ifr = 0; ifr < GloBuf[dev_id]->BlockNum; ifr++)
				pending_events_free(&GloBuf[dev_id]->BlockPendingEvents[ifr]);
			free(GloBuf[dev_id]->BlockPendingEvents);
			free(GloBuf[dev_id]->BlockState);
			free(GloBuf[dev_id]->BlockCurrentTileDim);
			free(GloBuf[dev_id]->BlockCurrentTilePtr);
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

void* CoCacheAsignBlock(short dev_id, void* TilePtr, short TileDim){
  short lvl = 5;
	if (TileDim != 2) error("CoCacheAsignBlock(%d): Tile%dD not implemented\n", dev_id, TileDim);
	Tile2D<VALUE_TYPE>* tmp = (Tile2D<VALUE_TYPE>*) TilePtr;
#ifdef DEBUG
  lprintf(lvl-1, "|-----> CoCacheAsignBlock(dev= %d, T = %d.[%d,%d] )\n",
		dev_id, tmp->id, tmp->GridId1, tmp->GridId2);
#endif
	if (dev_id < 0 ) error("CoCacheAsignBlock(): Invalid dev_id = %d\n", dev_id);
	if (GloBuf[dev_id] == NULL)
		error("CoCacheAsignBlock(%d): Called on empty buffer\n", dev_id);
	void* result = NULL;
  if (GloBuf[dev_id]->serialCtr >= GloBuf[dev_id]->BlockNum)
    return CoCacheUpdateAsignBlock(dev_id, TilePtr, TileDim);
		//No cache mechanism: error("CoCacheAsignBlock: Buffer full but request for more Blocks\n");
	else{
  	result = GloBuf[dev_id]->gpu_mem_buf + GloBuf[dev_id]->serialCtr*GloBuf[dev_id]->BlockSize;
		tmp->CacheLocId[dev_id] = GloBuf[dev_id]->serialCtr;
		GloBuf[dev_id]->BlockCurrentTileDim[GloBuf[dev_id]->serialCtr] = TileDim;
		GloBuf[dev_id]->BlockCurrentTilePtr[GloBuf[dev_id]->serialCtr] = TilePtr;
		GloBuf[dev_id]->serialCtr++;
	}
#ifdef DEBUG
	lprintf(lvl-1, "<-----|\n");
#endif
  return result;
}

void* CoCacheUpdateAsignBlock(short dev_id, void* TilePtr, short TileDim){
	short lvl = 5;
	if (TileDim != 2) error("CoCacheUpdateAsignBlock: Tile%dD not implemented\n", TileDim);
	Tile2D<VALUE_TYPE>* tmp = (Tile2D<VALUE_TYPE>*) TilePtr;

#ifdef DEBUG
	lprintf(lvl-1, "|-----> CoCacheUpdateAsignBlock(dev= %d, T = %d.[%d,%d] )\n",
		dev_id, tmp->id, tmp->GridId1, tmp->GridId2);
#endif
	if (dev_id < 0 ) error("CoCacheUpdateAsignBlock(): Invalid dev_id = %d\n", dev_id);
	if (GloBuf[dev_id] == NULL)
		error("CoCacheUpdateAsignBlock(%d): Called on empty buffer\n", dev_id);

	void* result = NULL;
	for (int idx = 0; idx < GloBuf[dev_id]->BlockNum; idx++){
		state tmp_state = CoCacheUpdateBlockState(dev_id, idx); // Update events etc for idx.
		while(__sync_lock_test_and_set (&globalock, 1));
		if(GloBuf[dev_id]->BlockPendingEvents[idx] == NULL){
			result = GloBuf[dev_id]->gpu_mem_buf + idx*GloBuf[dev_id]->BlockSize;
			GloBuf[dev_id]->BlockState[idx] = EMPTY;
			Tile2D<VALUE_TYPE>* prev_tile = (Tile2D<VALUE_TYPE>*) GloBuf[dev_id]->BlockCurrentTilePtr[idx];
			prev_tile->CacheLocId[dev_id] = -42;
			tmp->CacheLocId[dev_id] = idx;
			GloBuf[dev_id]->BlockCurrentTileDim[idx] = TileDim;
			GloBuf[dev_id]->BlockCurrentTilePtr[idx] = TilePtr;

#ifdef CDEBUG
			lprintf(lvl, "CoCacheUpdateAsignBlock(%d): Block(idx=%d) had no pending actions for T(%d.[%d,%d]), replacing...\n",
			dev_id, idx, prev_tile->id, prev_tile->GridId1, prev_tile->GridId2);
#endif
			break;
		}
		__sync_lock_release(&globalock);
	}
	__sync_lock_release(&globalock);
	if (result == NULL){
		if(recursion_depth[dev_id]==0){ // sync-wait for first incomplete event (not W) on each block to complete
			warning("CoCacheUpdateAsignBlock(dev= %d, T = %d.[%d,%d] )-Rec(%d): entry\n",
				dev_id, tmp->id, tmp->GridId1, tmp->GridId2, recursion_depth[dev_id]);
			for (int idx = 0; idx < GloBuf[dev_id]->BlockNum; idx++){
				//while(__sync_lock_test_and_set (&globalock, 1));
				if(GloBuf[dev_id]->BlockPendingEvents[idx] == NULL)
					error("CoCacheUpdateAsignBlock(dev= %d, T = %d.[%d,%d] )-Rec(%d): in recursion but block(%d) has no pending events\n",
					dev_id, tmp->id, tmp->GridId1, tmp->GridId2, recursion_depth[dev_id], idx);
				else if (GloBuf[dev_id]->BlockPendingEvents[idx]->effect != W){
					if (GloBuf[dev_id]->BlockPendingEvents[idx]->event_end!=NULL){
#ifdef CDEBUG
						lprintf(lvl, "CoCacheUpdateAsignBlock(dev= %d, T = %d.[%d,%d] )-Rec(%d): syncronizing event_end(%d) for block(%d)\n",
							dev_id, tmp->id, tmp->GridId1, tmp->GridId2, recursion_depth[dev_id],
							GloBuf[dev_id]->BlockPendingEvents[idx]->event_end->id, idx);
#endif
						GloBuf[dev_id]->BlockPendingEvents[idx]->event_end->sync_barrier();
					}
					else if( GloBuf[dev_id]->BlockPendingEvents[idx]->event_start!= NULL){
#ifdef CDEBUG
						lprintf(lvl, "CoCacheUpdateAsignBlock(dev= %d, T = %d.[%d,%d] )-Rec(%d): syncronizing event_start(%d) for block(%d)\n",
							dev_id, tmp->id, tmp->GridId1, tmp->GridId2, recursion_depth[dev_id],
							GloBuf[dev_id]->BlockPendingEvents[idx]->event_start->id, idx);
#endif
						GloBuf[dev_id]->BlockPendingEvents[idx]->event_start->sync_barrier();
					}
					else error("CoCacheUpdateAsignBlock(dev= %d, T = %d.[%d,%d] )-Rec(%d): First event of block(%d) is double-ghost\n",
								dev_id, tmp->id, tmp->GridId1, tmp->GridId2, recursion_depth[dev_id], idx);
				}
				//__sync_lock_release(&globalock);
			}
		}
		else if(recursion_depth[dev_id]==1){ // sync-wait for all incomplete event (not W) on each block to complete
			warning("CoCacheUpdateAsignBlock(dev= %d, T = %d.[%d,%d] ): second recursion entry\n",
				dev_id, tmp->id, tmp->GridId1, tmp->GridId2);
			for (int idx = 0; idx < GloBuf[dev_id]->BlockNum; idx++){
				pending_events_p current = GloBuf[dev_id]->BlockPendingEvents[idx];
				while(current!=NULL){
					if (current->event_end!=NULL){
#ifdef CDEBUG
					lprintf(lvl, "CoCacheUpdateAsignBlock(dev= %d, T = %d.[%d,%d] )-Rec(%d): syncronizing event_end(%d) for block(%d)\n",
						dev_id, tmp->id, tmp->GridId1, tmp->GridId2, recursion_depth[dev_id],
						current->event_end->id, idx);
#endif
						current->event_end->sync_barrier();
					}
					else if( current->event_start!= NULL){
#ifdef CDEBUG
						lprintf(lvl, "CoCacheUpdateAsignBlock(dev= %d, T = %d.[%d,%d] )-Rec(%d): syncronizing event_start(%d) for block(%d)\n",
							dev_id, tmp->id, tmp->GridId1, tmp->GridId2, recursion_depth[dev_id],
							current->event_start->id, idx);
#endif
						current->event_start->sync_barrier();
					}
					else error("CoCacheUpdateAsignBlock(dev= %d, T = %d.[%d,%d] ):\
						in recursion but some event of block(%d) is double-ghost\n",
						dev_id, tmp->id, tmp->GridId1, tmp->GridId2, idx);
					current = current->next;
				}
			}
		}
		else error("CoCacheUpdateAsignBlock: reached max recursion_depth=%d \
			while searching which Blocks to remove from Cache. Given T too large for GPU.\n", recursion_depth[dev_id]);
		recursion_depth[dev_id]++;
		return CoCacheUpdateAsignBlock(dev_id, TilePtr, TileDim);
	}
#ifdef DEBUG
	lprintf(lvl-1, "<-----|\n");
#endif
	recursion_depth[dev_id] = 0;
  return result;
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
		for (int ifr = 0; ifr < GloBuf[dev_id]->BlockNum; ifr++)
			pending_events_free(&GloBuf[dev_id]->BlockPendingEvents[ifr]);
		free(GloBuf[dev_id]->BlockPendingEvents);
		free(GloBuf[dev_id]->BlockState);
		free(GloBuf[dev_id]->BlockCurrentTileDim);
		free(GloBuf[dev_id]->BlockCurrentTilePtr);
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
    if (curr->run_dev_id != dev_id) error("CoCoPeLiaDevCacheInvalidate:\
    Subkernel(i= %d, run_dev_id = %d) in list does not belong to dev=%d", i, curr->run_dev_id, dev_id);
		for (int j = 0; j < curr->TileNum; j++){
			if (curr->TileDimlist[j] == 1) error("CoCoPeLiaDevCacheInvalidate: Tile1D not implemented\n");
			else if (curr->TileDimlist[j] == 2){
					Tile2D<VALUE_TYPE>* tmp = (Tile2D<VALUE_TYPE>*) curr->TileList[j];
					if (tmp->CacheLocId[dev_id] != -1) tmp->CacheLocId[dev_id] = -42;
			}
			else error("CoCoPeLiaDevCacheInvalidate: Not implemented for TileDim=%d\n", curr->TileDimlist[j]);
		}
	}
  if (GloBuf[dev_id]!= NULL) {
    GloBuf[dev_id]->serialCtr = 0;
    for (int ifr = 0; ifr < GloBuf[dev_id]->BlockNum; ifr++)
			pending_events_free(&GloBuf[dev_id]->BlockPendingEvents[ifr]);
  }
  recursion_depth[dev_id] = 0;
#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Cache for %d Subkernels invalidated: t_nv = %lf ms\n" , subkernel_data->SubkernelNumDev, cpu_timer*1000);
#endif
#ifdef DEBUG
	lprintf(lvl-1, "<-----|\n");
#endif
}

void CoCacheStateUpdate(state* prev, state next){
	short lvl = 6;
#ifdef DEBUG
	lprintf(lvl-1, "|-----> CoCacheStateUpdate(*%s,%s)\n", print_state(*prev), print_state(next));
#endif
	switch(*prev){
		case(EMPTY):
			if (next == R) ; //warning("CoCacheStateUpdate: Event Read EMPTY block (prev=%s, next=%s)\n",
				//print_state(*prev), print_state(next));
			else if (next == W); // warning("CoCacheStateUpdate: Event Write EMPTY block (prev=%s, next=%s)\n",
					//print_state(*prev), print_state(next));
			else *prev = next;
			break;
		case(AVAILABLE):
			*prev = next;
			break;
		case(R):
			if(next == W) ; //warning("CoCacheStateUpdate: Event Write at reading block(prev=%s, next=%s)\n",
				//print_state(*prev), print_state(next));
			else if(next == EMPTY) ;//warning("CoCacheStateUpdate: Event Emptied reading block(prev=%s, next=%s)\n",
					//print_state(*prev), print_state(next));
			else if(next == AVAILABLE) ;//warning("CoCacheStateUpdate: Event made available reading block(prev=%s, next=%s)\n",
					//print_state(*prev), print_state(next));
			break;
		case(W):
			//warning("CoCacheStateUpdate: Event change at writing block(prev=%s, next=%s)\n",
			//print_state(*prev), print_state(next));
			break;
		default:
			error("CoCacheStateUpdate: Unreachable default reached\n");
	}
	#ifdef DEBUG
		lprintf(lvl-1, "<-----|\n");
	#endif
}

state CoCacheUpdateBlockState(short dev_id, int BlockIdx){
	short lvl = 5;
#ifdef DEBUG
  lprintf(lvl-1, "|-----> CoCacheUpdateBlockState(%d,%d)\n", dev_id, BlockIdx);
#endif
	if (dev_id < 0 ) error("CoCacheUpdateBlockState(%d,%d): Invalid dev_id = %d\n", dev_id, BlockIdx, dev_id);
  if (GloBuf[dev_id] == NULL)
    error("CoCacheUpdateBlockState(%d,%d): Called on empty buffer\n", dev_id, BlockIdx);
  if (BlockIdx < 0 || BlockIdx >= GloBuf[dev_id]->BlockNum)
    error("CoCacheUpdateBlockState(%d,%d): Invalid BlockIdx = %d\n", dev_id, BlockIdx, BlockIdx);
		while(__sync_lock_test_and_set (&globalock, 1));
		state temp = (GloBuf[dev_id]->BlockState[BlockIdx] > 0) ? AVAILABLE : EMPTY;
		pending_events_p current = GloBuf[dev_id]->BlockPendingEvents[BlockIdx], prev = NULL;
		event_status start_status, end_status;
		short delete_curr_flag = 0;
	  while(current!=NULL){
			start_status = (current->event_start==NULL) ? GHOST : current->event_start->query_status();
			end_status = (current->event_end==NULL) ? GHOST : current->event_end->query_status();
#ifdef CDEBUG
			if (GloBuf[dev_id]->BlockCurrentTileDim[BlockIdx] != 2)
				error("CoCacheUpdateBlockState: Tile%dD not implemented\n", GloBuf[dev_id]->BlockCurrentTileDim[BlockIdx]);
			Tile2D<VALUE_TYPE>* tmp = (Tile2D<VALUE_TYPE>*) GloBuf[dev_id]->BlockCurrentTilePtr[BlockIdx];
			lprintf(lvl, "Found pending action for BlockIdx = %d hosting T(%d.[%d,%d])\n",
				BlockIdx, tmp->id, tmp->GridId1, tmp->GridId2);
			lprintf(lvl, "with event_start(%d) = %s, event_end(%d) = %s and effect = %s\n",
				(current->event_start==NULL) ? -1 : current->event_start->id, print_event_status(start_status),
				(current->event_end==NULL) ? -1 :  current->event_end->id, print_event_status(end_status),
				print_state(current->effect));
#endif
			if (start_status == GHOST && end_status == GHOST){
				warning("CoCacheUpdateBlockState: Double-GHOST event found, removing (bug?)\n");
				delete_curr_flag = 1;
			}
			else if(end_status == GHOST){ // Instantaneous event
				if (start_status == COMPLETE || start_status == CHECKED){
					CoCacheStateUpdate(&temp, current->effect);
					delete_curr_flag = 1;
#ifdef CDEBUG
			lprintf(lvl, "Instantaneous action complete, updating cache and removing\n");
#endif
				}
				else{;
#ifdef CDEBUG
					lprintf(lvl, "Instantaneous action still incomplete, waiting\n");
#endif
				}
			}
			else if ((start_status == COMPLETE || start_status == CHECKED) &&
				(end_status == RECORDED || end_status == UNRECORDED)){
				CoCacheStateUpdate(&temp, current->effect);
#ifdef CDEBUG
				lprintf(lvl, "Normal action still incomplete, updating cache and waiting\n");
#endif
			}
			else if ((start_status == COMPLETE || start_status == CHECKED) &&
				(end_status == COMPLETE || end_status == CHECKED)){
				delete_curr_flag = 1;
#ifdef CDEBUG
				lprintf(lvl, "Normal action complete, removing\n");
#endif
			}
			else{;
#ifdef CDEBUG
			lprintf(lvl, "Normal action not launched yet, waiting\n");
#endif
			}
			pending_events_p free_tmp = current;
			current = current->next;
			if(!delete_curr_flag) prev = free_tmp;
			else{
				delete_curr_flag = 0;
				if (prev == NULL) GloBuf[dev_id]->BlockPendingEvents[BlockIdx] = current;
				else prev->next = current;
				free(free_tmp);
			}


		}
	  __sync_lock_release(&globalock);
	#ifdef DEBUG
  	lprintf(lvl-1, "<-----|\n");
  #endif
	GloBuf[dev_id]->BlockState[BlockIdx] = temp;
  return temp;
}

void CoCacheAddPendingEvent(short dev_id, Event* e_start, Event* e_end, int BlockIdx, state effect){
	short lvl = 5;
#ifdef DEBUG
	lprintf(lvl-1, "|-----> CoCacheAddPendingEvent(dev = %d, e_start(%d), e_end(%d), BlockIdx = %d, %s)\n",
		dev_id, (e_start==NULL) ? -1 : e_start->id,
		(e_end==NULL) ? -1 : e_end->id, BlockIdx, print_state(effect));
#endif

	if (dev_id < 0 ) error("CoCacheAddPendingEvent(%d,%d): Invalid dev_id = %d\n", dev_id, BlockIdx, dev_id);
  if (GloBuf[dev_id] == NULL)
    error("CoCacheAddPendingEvent(%d,%d): Called on empty buffer\n", dev_id, BlockIdx);
  if (BlockIdx < 0 || BlockIdx >= GloBuf[dev_id]->BlockNum)
    error("CoCacheAddPendingEvent(%d,%d): Invalid BlockIdx = %d\n", dev_id, BlockIdx, BlockIdx);
	while(__sync_lock_test_and_set (&globalock, 1));

	pending_events_p current = GloBuf[dev_id]->BlockPendingEvents[BlockIdx], new_pending_event;
	new_pending_event = (pending_events_p) malloc(sizeof(struct pending_action_list));
	new_pending_event->event_start = e_start;
	new_pending_event->event_end = e_end;
	new_pending_event->effect = effect;
	new_pending_event->next = NULL;
	if (current == NULL ) GloBuf[dev_id]->BlockPendingEvents[BlockIdx] = new_pending_event;
	else {
		while(current->next!=NULL) current = current->next;
		current->next = new_pending_event;
	}
	__sync_lock_release(&globalock);
#ifdef DEBUG
  lprintf(lvl-1, "<-----|\n");
#endif
}
