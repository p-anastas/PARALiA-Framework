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
#include "backend_wrappers.hpp"

DevCachePtr DevCache[128] = {NULL};
int globalock = 0;
short recursion_depth[128] = {0};

#if CACHE_SCHEDULING_POLICY==1 || CACHE_SCHEDULING_POLICY==2 || CACHE_SCHEDULING_POLICY==3
LinkedList::LinkedList(){
	start = NULL;
	end = NULL;
	length = 0;
	iter = NULL;
}

void LinkedList::invalidate(Node_LL* node){
	if(node == NULL)
		error("LinkedList_Invalidate: Node not found.");
	else
		node->valid = false;
}

void LinkedList::push_back(int idx){
	// Pushes an element in the back of the queue.
	Node_LL* tmp = new Node_LL();
	tmp->next = NULL;
	tmp->previous = NULL;
	tmp->idx = idx;
	tmp->valid = true;
	if(start == NULL){ // Queue is empty.
		start = tmp;
		end = tmp;

	}
	else{ // Queue not empty.
		end->next = tmp;
		tmp->previous = end;
		end = tmp;
	}
	length++;
}


Node_LL* LinkedList::start_iterration(){
	short lvl = 6;
	// Returns the first valid element without removing it.
	if(start == NULL) // Queue is empty.
		error("LinkedList_start_iterration: Fifo queue is empty. Can't pop element.\n");
	else{ // Queue not empty.
		iter = start;
		while(iter != NULL && !iter->valid)
			iter = iter->next;
		if(iter == NULL){
			Node_LL* tmp_node = new Node_LL();
			tmp_node->idx = -1;
			return tmp_node;
		}
		else
			return iter;
	}
}

Node_LL* LinkedList::next_in_line(){
	// Returns next element in iterration.
	short lvl = 6;
	if(iter == NULL)
		error("LinkedList_next_in_line: Iterration not started. Call check_first.\n");
	else{
		iter = iter->next;
		while(iter != NULL && !iter->valid)
			iter = iter->next;
		if(iter == NULL){
			Node_LL* tmp_node = new Node_LL();
			tmp_node->idx = -1;
			return tmp_node;
		}
		else
			return iter;
	}
}

int LinkedList::remove(Node_LL* node){
	// Removes first element.
	int tmp_idx;
	Node_LL* tmp_node;

	if(node == NULL) // Node not found.
		error("LinkedList_remove: Input node not found. Can't pop element.\n");
	else{
		tmp_idx = node->idx;
		tmp_node = node;
		(tmp_node->previous)->next = tmp_node->next;
		free(tmp_node);
		length--;
		return tmp_idx;
	}
}

void LinkedList::put_first(Node_LL* node){
	// Puts the element first on the list.
	if(node == NULL)
		error("LinkedList_put_first: Input node not found. Can't move element.\n");
	else if(node->idx<0)
		error("LinkedList_put_first: Input node is invalid. Can't move element.\n");
	else if(node->previous==NULL){
		// Node is the first on the list.
		if(start==node)
			node->valid = true;
		// Node is not on the list because it can't be inside with no next.
		else if(node->next==NULL){
		start->previous = node;
		node->next = start;
		start = node;
		node->valid = true;
		}
		else
			error("LinkedList_put_first: Unexpected internal error.\n");
	}// Node is somewhere inside the list but not last.
	else{
		if(end==node)
			end=node->previous;

		(node->previous)->next = node->next;
		if(node->next != NULL)
			(node->next)->previous = node->previous;
		start->previous = node;
		node->next = start;
		node->previous = NULL;
		node->valid = true;
		start = node;
	}
}

void LinkedList::put_last(Node_LL* node){
	// Takes a node and puts it in the end of the list.
	if(node == NULL)
		error("LinkedList_put_last: Input node not found. Can't move element.\n");
	else if(node->idx<0)
		error("LinkedList_put_last: Input node is invalid. Can't move element.\n");
	else if(node->next==NULL){
		// Node is the last on the list.
		if(end==node)
			node->valid = true;
		// Node is not on the list because it can't be inside with no next.
		else if(node->previous==NULL){
			end->next = node;
			node->previous = end;
			end = node;
			node->valid = true;
		}
		else
			error("LinkedList_put_last: Unexpected internal error.\n");
	} // Node is somewhere inside the list but not last.
	else{
		if(node==start)
			start = node->next;

		(node->next)->previous = node->previous;
		if(node->previous != NULL)
			(node->previous)->next = node->next;
		end->next = node;
		node->previous = end;
		node->next = NULL;
		node->valid = true;
		end = node;
	}
}


LinkedList::~LinkedList(){
	Node_LL* tmp;
	while(start != NULL){
		tmp = start;
		start = tmp->next;
		free(tmp);
	}
}
#endif

#if CACHE_SCHEDULING_POLICY==1
LinkedList fifo_queues[128];
#elif CACHE_SCHEDULING_POLICY==2 || CACHE_SCHEDULING_POLICY==3
Node_LL** mru_lru_hash[128];
LinkedList mru_lru_queues[128];
#endif


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

DevCachePtr CoCoPeLiaDevCacheInit(kernel_pthread_wrap_p subkernel_data, long long block_size){
  DevCachePtr result = (DevCachePtr) malloc (sizeof(struct DevCache_str));
  int dev_id = result->dev_id = subkernel_data->dev_id;
  short dev_id_idx = (dev_id >= 0) ? dev_id : LOC_NUM - 1;
  result->gpu_mem_buf = NULL;
  result->mem_buf_sz = result->BlockNum = result->serialCtr = 0;
	result->BlockSize = block_size;
	for (int i = 0; i < subkernel_data->SubkernelNumDev; i++){
		Subkernel* curr = subkernel_data->SubkernelListDev[i];
		for (int j = 0; j < curr->TileNum; j++){
			if (curr->TileDimlist[j] == 1) {
					Tile1D<VALUE_TYPE>* tmp = (Tile1D<VALUE_TYPE>*) curr->TileList[j];
					if (tmp->CacheLocId[dev_id_idx] == -42){
						tmp->CacheLocId[dev_id_idx] = -2;
						//printf("Found Tile with INVALID entry, adding %d to result", tmp->size());
						result->BlockNum++;
					}
			}
			else if (curr->TileDimlist[j] == 2){
					Tile2D<VALUE_TYPE>* tmp = (Tile2D<VALUE_TYPE>*) curr->TileList[j];
					if (tmp->CacheLocId[dev_id_idx] == -42){
						tmp->CacheLocId[dev_id_idx] = -2;
						//printf("Found Tile with INVALID entry, adding %d to result", tmp->size());
						result->BlockNum++;
					}
			}
			else error("CoCoPeLiaDevCacheInit: Not implemented for TileDim=%d\n", curr->TileDimlist[j]);
		}
	}
	CoCoPeLiaDevCacheInvalidate(subkernel_data);
	result->mem_buf_sz= result->BlockSize*result->BlockNum;
	result->BlockState = (state*) calloc (result->BlockNum, sizeof(state));
	result->BlockPendingEvents = (pending_events_p*) malloc (result->BlockNum* sizeof(pending_events_p));
	result->BlockCurrentTileDim = (short*) malloc (result->BlockNum* sizeof(short));
	result->BlockCurrentTilePtr = (void**) malloc (result->BlockNum* sizeof(void*));
	for (int idx = 0; idx < result->BlockNum; idx++) result->BlockPendingEvents[idx] = NULL;
	return result;
}

void CoCoPeLiaRequestBuffer(kernel_pthread_wrap_p subkernel_data, long long bufsize_limit, long long block_size){
  short lvl = 3;
#ifdef DEBUG
  lprintf(lvl-1, "|-----> CoCoPeLiaRequestBuffer(%d)\n", subkernel_data->dev_id);
#endif
#ifdef TEST
	double cpu_timer = csecond();
#endif
  short dev_id = subkernel_data->dev_id, dev_id_idx = (dev_id >= 0) ? dev_id : LOC_NUM - 1;
	DevCachePtr temp_DevCache = CoCoPeLiaDevCacheInit(subkernel_data, block_size);
	if (temp_DevCache->mem_buf_sz > 0){
	  long long free_dev_mem, max_dev_mem,  prev_DevCache_sz = 0;
		if (DevCache[dev_id_idx] != NULL) prev_DevCache_sz = DevCache[dev_id_idx]->mem_buf_sz;
		CoCoPeLiaDevGetMemInfo(&free_dev_mem, &max_dev_mem);
	  long long problem_avail_mem = free_dev_mem - max_dev_mem*(1-PROBLEM_GPU_PERCENTAGE/100.0) + prev_DevCache_sz;
	  // For debuging large cases
	  if(MEM_LIMIT>=free_dev_mem)
	  	problem_avail_mem=MEM_LIMIT;
	  else
	  	problem_avail_mem=free_dev_mem;

	  #ifdef DEBUG
	  	lprintf(lvl, "====================================\n");
	  	lprintf(lvl, "Buffer mem management:\n");
	  	lprintf(lvl, " -Buffer requested for BlockSize=%zu MB and BlockNum=%d\n", (size_t) temp_DevCache->BlockSize/1024/1024, temp_DevCache->BlockNum);
	  	lprintf(lvl, " -Mem required for matrices: %zu MB\n", (size_t) temp_DevCache->mem_buf_sz/1024/1024);
	    lprintf(lvl, " -Mem available in loc(%d): %zu MB\n", dev_id, (size_t) problem_avail_mem/1024/1024);
	  #endif
		if (bufsize_limit <= 0){
		  if (temp_DevCache->mem_buf_sz <= problem_avail_mem){
#ifdef DEBUG
		    lprintf(lvl, " -Requested buffer fits in loc(%d)\n", dev_id);
#endif
			;}
			else{
		    temp_DevCache->BlockNum =  (int) (problem_avail_mem/temp_DevCache->BlockSize);
				temp_DevCache->mem_buf_sz = temp_DevCache->BlockNum*temp_DevCache->BlockSize;
#ifdef DEBUG
		    lprintf(lvl, " -Requested buffer does not fit in loc(%d)\n", dev_id);
#endif
			}
	}
	else{
		if(bufsize_limit > problem_avail_mem)
			error("CoCoPeLiaRequestBuffer(dev_id=%d): Requested cache with bufsize_limit =%zu MB bigger than problem_avail_mem = %zu MB\n",
				dev_id, (size_t) bufsize_limit/1024/1024, (size_t) problem_avail_mem/1024/1024);
		else if(bufsize_limit < temp_DevCache->BlockSize)
					error("CoCoPeLiaRequestBuffer(dev_id=%d): Requested cache with bufsize_limit =%zu MB smaller than Blocksize = %zu MB\n",
						dev_id, (size_t) bufsize_limit/1024/1024, (size_t) temp_DevCache->BlockSize/1024/1024);
		else{
			temp_DevCache->BlockNum =  (int) (bufsize_limit/temp_DevCache->BlockSize);
			temp_DevCache->mem_buf_sz = temp_DevCache->BlockNum*temp_DevCache->BlockSize;
		}
	}
#if CACHE_SCHEDULING_POLICY==2 || CACHE_SCHEDULING_POLICY==3 // Memory for hash
	mru_lru_hash[dev_id_idx] = (Node_LL**) malloc(temp_DevCache->BlockNum * sizeof(Node_LL*));
#endif

	#ifdef DEBUG
	  if (prev_DevCache_sz >= temp_DevCache->mem_buf_sz)
			lprintf(lvl, " -Loc(%d) buf available: %zu MB\n", dev_id, (size_t) prev_DevCache_sz/1024/1024);
	  else if (prev_DevCache_sz > 0) lprintf(lvl, " -Smaller Loc(%d) buf available -> replacing : %zu -> %zu MB\n",
			dev_id, (size_t) prev_DevCache_sz/1024/1024, (size_t) temp_DevCache->mem_buf_sz/1024/1024);
	  else lprintf(lvl, " -Initializing new Loc(%d) buffer: %zu MB\n", dev_id, (size_t) temp_DevCache->mem_buf_sz/1024/1024);
	#endif
	  if (prev_DevCache_sz >= temp_DevCache->mem_buf_sz){
			temp_DevCache->mem_buf_sz = DevCache[dev_id_idx]->mem_buf_sz;
			temp_DevCache->gpu_mem_buf = DevCache[dev_id_idx]->gpu_mem_buf;
			for (int ifr = 0; ifr < DevCache[dev_id_idx]->BlockNum; ifr++)
				pending_events_free(&DevCache[dev_id_idx]->BlockPendingEvents[ifr]);
			free(DevCache[dev_id_idx]->BlockPendingEvents);
			free(DevCache[dev_id_idx]->BlockState);
			free(DevCache[dev_id_idx]->BlockCurrentTileDim);
			free(DevCache[dev_id_idx]->BlockCurrentTilePtr);
			free(DevCache[dev_id_idx]);
			DevCache[dev_id_idx] = temp_DevCache;
		}
	  else{
			if (prev_DevCache_sz > 0){
			  CoCoFree(DevCache[dev_id_idx]->gpu_mem_buf, dev_id);
				for (int ifr = 0; ifr < DevCache[dev_id_idx]->BlockNum; ifr++)
					pending_events_free(&DevCache[dev_id_idx]->BlockPendingEvents[ifr]);
				free(DevCache[dev_id_idx]->BlockPendingEvents);
				free(DevCache[dev_id_idx]->BlockState);
				free(DevCache[dev_id_idx]->BlockCurrentTileDim);
				free(DevCache[dev_id_idx]->BlockCurrentTilePtr);
				free(DevCache[dev_id_idx]);
			}
	    //DevCache[dev_id_idx] = temp_DevCache;
			//DevCache[dev_id_idx]->gpu_mem_buf = CoCoMalloc(DevCache[dev_id_idx]->mem_buf_sz,
			//	dev_id);
			temp_DevCache->gpu_mem_buf = CoCoMalloc(temp_DevCache->mem_buf_sz, dev_id);
			DevCache[dev_id_idx] = temp_DevCache;
	  }
	  CoCoSyncCheckErr();

#ifdef TEST
		cpu_timer = csecond() - cpu_timer;
		lprintf(lvl, "GPU(%d) Buffer allocation with sz = %zu MB: t_alloc = %lf ms\n" ,
	    dev_id, (size_t) DevCache[dev_id_idx]->mem_buf_sz/1024/1024, cpu_timer*1000);
#endif

#ifdef DEBUG
		lprintf(lvl, "GPU(%d) Buffer allocation (Size = %zu MB, Blocksize = %zu MB, BlockNum = %d)\n" ,
	  dev_id, (size_t) DevCache[dev_id_idx]->mem_buf_sz/1024/1024,
	  (size_t) DevCache[dev_id_idx]->BlockSize/1024/1024,  DevCache[dev_id_idx]->BlockNum);
		lprintf(lvl-1, "<-----|\n");
#endif
	}
	else{;
#ifdef DEBUG
		lprintf(lvl, "GPU(%d) does not require a buffer - all data available\n" , dev_id);
		lprintf(lvl-1, "<-----|\n");
#endif
	}
}

void CoCopeLiaDevCacheFree(short dev_id)
{
	short lvl = 3;
#ifdef DEBUG
			lprintf(lvl-1, "|-----> CoCopeLiaDevCacheFree(dev_id=%d)\n", dev_id);
#endif
  short dev_id_idx = (dev_id >= 0) ? dev_id : LOC_NUM - 1;
#ifdef TEST
	double cpu_timer = csecond();
#endif
	if (DevCache[dev_id_idx]){
#ifdef DEBUG
		lprintf(lvl, "Clearing (presumably) %zu MB\n\n", (size_t) DevCache[dev_id_idx]->mem_buf_sz/1024/1024);
#endif
		CoCoFree(DevCache[dev_id_idx]->gpu_mem_buf, dev_id);
		for (int ifr = 0; ifr < DevCache[dev_id_idx]->BlockNum; ifr++)
			pending_events_free(&DevCache[dev_id_idx]->BlockPendingEvents[ifr]);
		free(DevCache[dev_id_idx]->BlockPendingEvents);
		free(DevCache[dev_id_idx]->BlockState);
		free(DevCache[dev_id_idx]->BlockCurrentTileDim);
		free(DevCache[dev_id_idx]->BlockCurrentTilePtr);
		free(DevCache[dev_id_idx]);
		DevCache[dev_id_idx] = NULL;
    recursion_depth[dev_id_idx] = 0;
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
  short dev_id = subkernel_data->dev_id, dev_id_idx = (dev_id >= 0) ? dev_id : LOC_NUM - 1;
	for (int i = 0; i < subkernel_data->SubkernelNumDev; i++){
		Subkernel* curr = subkernel_data->SubkernelListDev[i];
    if (curr->run_dev_id != dev_id) error("CoCoPeLiaDevCacheInvalidate:\
    Subkernel(i= %d, run_dev_id = %d) in list does not belong to dev=%d", i, curr->run_dev_id, dev_id);
		for (int j = 0; j < curr->TileNum; j++){
			if (curr->TileDimlist[j] == 1) {
					Tile1D<VALUE_TYPE>* tmp = (Tile1D<VALUE_TYPE>*) curr->TileList[j];
					if (tmp->CacheLocId[dev_id_idx] != -1) tmp->CacheLocId[dev_id_idx] = -42;
					tmp->available[dev_id_idx]->reset();
			}
			else if (curr->TileDimlist[j] == 2){
					Tile2D<VALUE_TYPE>* tmp = (Tile2D<VALUE_TYPE>*) curr->TileList[j];
					if (tmp->CacheLocId[dev_id_idx] != -1) tmp->CacheLocId[dev_id_idx] = -42;
					tmp->available[dev_id_idx]->reset();
			}
			else error("CoCoPeLiaDevCacheInvalidate: Not implemented for TileDim=%d\n", curr->TileDimlist[j]);
		}
	}
  if (DevCache[dev_id_idx]!= NULL) {
    DevCache[dev_id_idx]->serialCtr = 0;
    for (int ifr = 0; ifr < DevCache[dev_id_idx]->BlockNum; ifr++)
			pending_events_free(&DevCache[dev_id_idx]->BlockPendingEvents[ifr]);
  }
  recursion_depth[dev_id_idx] = 0;
#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Cache for %d Subkernels invalidated: t_nv = %lf ms\n" , subkernel_data->SubkernelNumDev, cpu_timer*1000);
#endif
#ifdef DEBUG
	lprintf(lvl-1, "<-----|\n");
#endif
}

void* CoCacheAsignBlock(short dev_id, void* TilePtr, short TileDim){
  short lvl = 5;
	int* block_id_ptr;
	short dev_id_idx = (dev_id == -1) ? LOC_NUM - 1: dev_id;
	if (TileDim == 1){
		Tile1D<VALUE_TYPE>* tmp = (Tile1D<VALUE_TYPE>*) TilePtr;
		block_id_ptr = &tmp->CacheLocId[dev_id_idx];
#ifdef DEBUG
  	lprintf(lvl-1, "|-----> CoCacheAsignBlock(dev= %d, T = %d.[%d] )\n",
			dev_id, tmp->id, tmp->GridId);
#endif
	}
	else if(TileDim == 2){
		Tile2D<VALUE_TYPE>* tmp = (Tile2D<VALUE_TYPE>*) TilePtr;
		block_id_ptr = &tmp->CacheLocId[dev_id_idx];
#ifdef DEBUG
  	lprintf(lvl-1, "|-----> CoCacheAsignBlock(dev= %d, T = %d.[%d,%d] )\n",
			dev_id, tmp->id, tmp->GridId1, tmp->GridId2);
#endif
	}
	else error("CoCacheAsignBlock(%d): Tile%dD not implemented\n", dev_id, TileDim);
	if (DevCache[dev_id_idx] == NULL)
		error("CoCacheAsignBlock(%d): Called on empty buffer\n", dev_id);
	void* result = NULL;
  if (DevCache[dev_id_idx]->serialCtr >= DevCache[dev_id_idx]->BlockNum)
    return CoCacheUpdateAsignBlock(dev_id, TilePtr, TileDim);
		//No cache mechanism: error("CoCacheAsignBlock: Buffer full but request for more Blocks\n");
	else{
 #if CACHE_SCHEDULING_POLICY==1
		fifo_queues[dev_id_idx].lock_ll.lock();
#elif CACHE_SCHEDULING_POLICY==2 || CACHE_SCHEDULING_POLICY==3
		mru_lru_queues[dev_id_idx].lock_ll.lock();
#endif
  	result = DevCache[dev_id_idx]->gpu_mem_buf + DevCache[dev_id_idx]->serialCtr*DevCache[dev_id_idx]->BlockSize;
		*block_id_ptr = DevCache[dev_id_idx]->serialCtr;
		DevCache[dev_id_idx]->BlockCurrentTileDim[DevCache[dev_id_idx]->serialCtr] = TileDim;
		DevCache[dev_id_idx]->BlockCurrentTilePtr[DevCache[dev_id_idx]->serialCtr] = TilePtr;
#if CACHE_SCHEDULING_POLICY==1
		fifo_queues[dev_id_idx].push_back(DevCache[dev_id_idx]->serialCtr);
		fifo_queues[dev_id_idx].lock_ll.unlock();
#elif CACHE_SCHEDULING_POLICY==2
		mru_lru_queues[dev_id_idx].push_back(DevCache[dev_id_idx]->serialCtr);
		mru_lru_hash[dev_id_idx][DevCache[dev_id_idx]->serialCtr] = mru_lru_queues[dev_id_idx].end;
		mru_lru_queues[dev_id_idx].put_first(mru_lru_hash[dev_id_idx][DevCache[dev_id_idx]->serialCtr]);
		mru_lru_queues[dev_id_idx].lock_ll.unlock();
#elif CACHE_SCHEDULING_POLICY==3
		mru_lru_queues[dev_id_idx].push_back(DevCache[dev_id_idx]->serialCtr);
		mru_lru_hash[dev_id_idx][DevCache[dev_id_idx]->serialCtr] = mru_lru_queues[dev_id_idx].end;
		mru_lru_queues[dev_id_idx].lock_ll.unlock();
#endif
		DevCache[dev_id_idx]->serialCtr++;
	}
#ifdef DEBUG
	lprintf(lvl-1, "<-----|\n");
#endif
  return result;
}

void* CoCacheUpdateAsignBlock(short dev_id, void* TilePtr, short TileDim){
	short lvl = 5;
	int* block_id_ptr;
	short dev_id_idx = (dev_id == -1) ? LOC_NUM - 1: dev_id;
	if (TileDim == 1){
		Tile1D<VALUE_TYPE>* tmp = (Tile1D<VALUE_TYPE>*) TilePtr;
		block_id_ptr = &tmp->CacheLocId[dev_id_idx];
#ifdef DEBUG
  	lprintf(lvl-1, "|-----> CoCacheUpdateAsignBlock(dev= %d, T = %d.[%d] )\n",
			dev_id, tmp->id, tmp->GridId);
#endif
	}
	else if(TileDim == 2){
		Tile2D<VALUE_TYPE>* tmp = (Tile2D<VALUE_TYPE>*) TilePtr;
		block_id_ptr = &tmp->CacheLocId[dev_id_idx];
#ifdef DEBUG
  	lprintf(lvl-1, "|-----> CoCacheUpdateAsignBlock(dev= %d, T = %d.[%d,%d] )\n",
			dev_id, tmp->id, tmp->GridId1, tmp->GridId2);
#endif
	}
	else error("CoCacheUpdateAsignBlock(%d): Tile%dD not implemented\n", dev_id, TileDim);
	if (DevCache[dev_id_idx] == NULL)
		error("CoCacheUpdateAsignBlock(%d): Called on empty buffer\n", dev_id);

	void* result = NULL;
#if CACHE_SCHEDULING_POLICY==0
	int remove_block_idx = CoCacheSelectBlockToRemove_naive(dev_id);
	if(remove_block_idx >= 0){
		while(__sync_lock_test_and_set (&globalock, 1));
		result = DevCache[dev_id_idx]->gpu_mem_buf + remove_block_idx*DevCache[dev_id_idx]->BlockSize;
		DevCache[dev_id_idx]->BlockState[remove_block_idx] = EMPTY;

#elif CACHE_SCHEDULING_POLICY==1
	Node_LL* remove_block;
	remove_block = CoCacheSelectBlockToRemove_fifo(dev_id);
	if(remove_block->idx >= 0){
		while(__sync_lock_test_and_set (&globalock, 1));
		fifo_queues[dev_id_idx].lock_ll.lock();
		int remove_block_idx = fifo_queues[dev_id_idx].remove(remove_block);
		fifo_queues[dev_id_idx].push_back(remove_block_idx);
		fifo_queues[dev_id_idx].lock_ll.unlock();
		result = DevCache[dev_id_idx]->gpu_mem_buf + remove_block_idx*DevCache[dev_id_idx]->BlockSize;
		DevCache[dev_id_idx]->BlockState[remove_block_idx] = EMPTY;

#elif CACHE_SCHEDULING_POLICY==2 || CACHE_SCHEDULING_POLICY==3
	Node_LL* remove_block;
	remove_block = CoCacheSelectBlockToRemove_mru_lru(dev_id);
	if(remove_block->idx >= 0){
		while(__sync_lock_test_and_set (&globalock, 1));
		mru_lru_queues[dev_id_idx].lock_ll.lock();
		int remove_block_idx = remove_block->idx;
	#if CACHE_SCHEDULING_POLICY==2
		mru_lru_queues[dev_id_idx].put_first(remove_block);
	#elif CACHE_SCHEDULING_POLICY==3
		mru_lru_queues[dev_id_idx].put_last(remove_block);
	#endif
		mru_lru_queues[dev_id_idx].lock_ll.unlock();
		result = DevCache[dev_id_idx]->gpu_mem_buf + remove_block_idx*DevCache[dev_id_idx]->BlockSize;
		DevCache[dev_id_idx]->BlockState[remove_block_idx] = EMPTY;
#endif
		if (TileDim == 1){
			Tile1D<VALUE_TYPE>* prev_tile = (Tile1D<VALUE_TYPE>*)
				DevCache[dev_id_idx]->BlockCurrentTilePtr[remove_block_idx];
			prev_tile->CacheLocId[dev_id_idx] = -42;
			prev_tile->available[dev_id_idx]->reset();
		#ifdef CDEBUG
				lprintf(lvl, "CoCacheSelectBlockToRemove_mru_lru(%d): Block(idx=%d) had no pending actions for T(%d.[%d]), replacing...\n",
				dev_id, remove_block_idx, prev_tile->id, prev_tile->GridId);
		#endif
		}
		else if(TileDim == 2){
			Tile2D<VALUE_TYPE>* prev_tile = (Tile2D<VALUE_TYPE>*)
				DevCache[dev_id_idx]->BlockCurrentTilePtr[remove_block_idx];
			prev_tile->CacheLocId[dev_id_idx] = -42;
			prev_tile->available[dev_id_idx]->reset();
		#ifdef CDEBUG
				lprintf(lvl, "CoCacheSelectBlockToRemove_mru_lru(%d): Block(idx=%d) had no pending actions for T(%d.[%d,%d]), replacing...\n",
				dev_id, remove_block_idx, prev_tile->id, prev_tile->GridId1, prev_tile->GridId2);
		#endif
		}
		else error("CoCacheUpdateAsignBlock(%d): Tile%dD not implemented\n", dev_id, TileDim);
		*block_id_ptr = remove_block_idx;
		DevCache[dev_id_idx]->BlockCurrentTileDim[remove_block_idx] = TileDim;
		DevCache[dev_id_idx]->BlockCurrentTilePtr[remove_block_idx] = TilePtr;
		__sync_lock_release(&globalock);
	}
	else{
#if CACHE_SCHEDULING_POLICY==1 || CACHE_SCHEDULING_POLICY==2 || CACHE_SCHEDULING_POLICY==3
		free(remove_block);
#endif

		if(recursion_depth[dev_id_idx]==0){ // sync-wait for first incomplete event (not W) on each block to complete
			warning("CoCacheUpdateAsignBlock(dev= %d)-Rec(%d): entry\n",
				dev_id, recursion_depth[dev_id_idx]);
	#if CACHE_SCHEDULING_POLICY==0
			for (int idx = 0; idx < DevCache[dev_id_idx]->BlockNum; idx++){
				if(DevCache[dev_id_idx]->BlockPendingEvents[idx] == NULL)
	#elif CACHE_SCHEDULING_POLICY==1
			Node_LL* iterr = fifo_queues[dev_id_idx].start;
			int idx;
			while(iterr != NULL){
				idx = iterr->idx;
				while(__sync_lock_test_and_set (&globalock, 1));
				fifo_queues[dev_id_idx].lock_ll.lock();
				if(DevCache[dev_id_idx]->BlockPendingEvents[idx] == NULL && iterr->valid)
	#elif CACHE_SCHEDULING_POLICY==2 || CACHE_SCHEDULING_POLICY==3
			Node_LL* iterr = mru_lru_queues[dev_id_idx].start;
			int idx;
			while(iterr != NULL){
				idx = iterr->idx;
				while(__sync_lock_test_and_set (&globalock, 1));
				mru_lru_queues[dev_id_idx].lock_ll.lock();
				if(DevCache[dev_id_idx]->BlockPendingEvents[idx] == NULL && iterr->valid)
	#endif
					error("CoCacheUpdateAsignBlock(dev= %d)-Rec(%d): in recursion but block(%d) has no pending events\n",
					dev_id, recursion_depth[dev_id_idx], idx);
				else if (DevCache[dev_id_idx]->BlockPendingEvents[idx]->effect != W){
					if (DevCache[dev_id_idx]->BlockPendingEvents[idx]->event_end!=NULL){
#ifdef CDEBUG
						lprintf(lvl, "CoCacheUpdateAsignBlock(dev= %d)-Rec(%d): syncronizing event_end(%d) for block(%d)\n",
							dev_id, recursion_depth[dev_id_idx],
							DevCache[dev_id_idx]->BlockPendingEvents[idx]->event_end->id, idx);
#endif
						DevCache[dev_id_idx]->BlockPendingEvents[idx]->event_end->sync_barrier();
					}
					else if( DevCache[dev_id_idx]->BlockPendingEvents[idx]->event_start!= NULL){
#ifdef CDEBUG
						lprintf(lvl, "CoCacheUpdateAsignBlock(dev= %d)-Rec(%d): syncronizing event_start(%d) for block(%d)\n",
							dev_id, recursion_depth[dev_id_idx],
							DevCache[dev_id_idx]->BlockPendingEvents[idx]->event_start->id, idx);
#endif
						DevCache[dev_id_idx]->BlockPendingEvents[idx]->event_start->sync_barrier();
					}
					else error("CoCacheUpdateAsignBlock(dev= %d)-Rec(%d): First event of block(%d) is double-ghost\n",
								dev_id, recursion_depth[dev_id_idx], idx);
				}
			#if CACHE_SCHEDULING_POLICY==1
			iterr = iterr->next;
			fifo_queues[dev_id_idx].lock_ll.unlock();
			__sync_lock_release(&globalock);
			#elif CACHE_SCHEDULING_POLICY==2 || CACHE_SCHEDULING_POLICY==3
			iterr = iterr->next;
			mru_lru_queues[dev_id_idx].lock_ll.unlock();
			__sync_lock_release(&globalock);
			#endif
			}
		}
		else if(recursion_depth[dev_id_idx]==1){ // sync-wait for all incomplete event (not W) on each block to complete
			warning("CoCacheUpdateAsignBlock(dev= %d): second recursion entry\n",
				dev_id);
			for (int idx = 0; idx < DevCache[dev_id_idx]->BlockNum; idx++){
				pending_events_p current = DevCache[dev_id_idx]->BlockPendingEvents[idx];
				while(current!=NULL){
					if (current->event_end!=NULL){
#ifdef CDEBUG
					lprintf(lvl, "CoCacheUpdateAsignBlock(dev= %d)-Rec(%d): syncronizing event_end(%d) for block(%d)\n",
						dev_id, recursion_depth[dev_id_idx],
						current->event_end->id, idx);
#endif
						current->event_end->sync_barrier();
					}
					else if( current->event_start!= NULL){
#ifdef CDEBUG
						lprintf(lvl, "CoCacheUpdateAsignBlock(dev= %d)-Rec(%d): syncronizing event_start(%d) for block(%d)\n",
							dev_id, recursion_depth[dev_id_idx],
							current->event_start->id, idx);
#endif
						current->event_start->sync_barrier();
					}
					else error("CoCacheUpdateAsignBlock(dev= %d):\
						in recursion but some event of block(%d) is double-ghost\n",
						dev_id, idx);
					current = current->next;
				}
			}
		}
		else error("CoCacheUpdateAsignBlock: reached max recursion_depth=%d \
			while searching which Blocks to remove from Cache. Given T too large for GPU.\n", recursion_depth[dev_id_idx]);
		recursion_depth[dev_id_idx]++;
		return CoCacheUpdateAsignBlock(dev_id, TilePtr, TileDim);
	}
#ifdef DEBUG
	lprintf(lvl-1, "<-----|\n");
#endif
	recursion_depth[dev_id_idx] = 0;
  return result;
}

#if CACHE_SCHEDULING_POLICY==0
int CoCacheSelectBlockToRemove_naive(short dev_id){
	short lvl = 6;
#ifdef DEBUG
	lprintf(lvl-1, "|-----> CoCacheSelectBlockToRemove_naive(dev= %d)\n",dev_id);
#endif
#ifdef CTEST
	cpu_timer = csecond();
#endif
	short dev_id_idx = (dev_id == -1) ? LOC_NUM - 1: dev_id;
	int result_idx = -1;
	if (DevCache[dev_id_idx] == NULL)
		error("CoCacheSelectBlockToRemove_naive(%d): Called on empty buffer\n", dev_id);
	for (int idx = 0; idx < DevCache[dev_id_idx]->BlockNum; idx++){ // Iterate through cache serially.
		state tmp_state = CoCacheUpdateBlockState(dev_id, idx); // Update all events etc for idx.
		while(__sync_lock_test_and_set (&globalock, 1));
		if(DevCache[dev_id_idx]->BlockPendingEvents[idx] == NULL){ 		// Indx can be removed if there are no pending events.
			result_idx =  idx;
			__sync_lock_release(&globalock);
			break;
		}
		__sync_lock_release(&globalock);
	}
#ifdef CTEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl-1, "CoCacheSelectBlockToRemove_naive(%d): t_select = %lf\n",dev_id, cpu_timer);
	cpu_timer = csecond();
#endif
#ifdef DEBUG
	lprintf(lvl-1, "<-----|\n");
#endif
	return result_idx;
}

#elif CACHE_SCHEDULING_POLICY==1
Node_LL* CoCacheSelectBlockToRemove_fifo(short dev_id){
	short lvl = 6;
#ifdef DEBUG
	lprintf(lvl-1, "|-----> CoCacheSelectBlockToRemove_fifo(dev= %d)\n",dev_id);
#endif
#ifdef CTEST
	cpu_timer = csecond();
#endif
	Node_LL* result_node = new Node_LL();

  short dev_id_idx = (dev_id >= 0) ? dev_id : LOC_NUM - 1;
	result_node->idx = -1;
	if (DevCache[dev_id_idx] == NULL)
		error("CoCacheSelectBlockToRemove_fifo(%d): Called on empty buffer\n", dev_id);
	// while(__sync_lock_test_and_set (&globalock, 1));
	fifo_queues[dev_id_idx].lock_ll.lock();
	Node_LL* node = fifo_queues[dev_id_idx].start_iterration();
	if(node->idx >= 0)
		state tmt_state = CoCacheUpdateBlockState(dev_id, node->idx);
	while(DevCache[dev_id_idx]->BlockPendingEvents[node->idx] != NULL){
		node = fifo_queues[dev_id_idx].next_in_line();
		if(node->idx >= 0)
			state tmt_state = CoCacheUpdateBlockState(dev_id, node->idx);
		else
			break;
	}
	if(node->idx >=0 && DevCache[dev_id_idx]->BlockPendingEvents[node->idx] == NULL){
		lprintf(lvl-1, "|-----> CoCacheSelectBlockToRemove_fifo: Invalidating\n");
		fifo_queues[dev_id_idx].invalidate(node);
		free(result_node);
		result_node = node;
	}
	fifo_queues[dev_id_idx].lock_ll.unlock();
	// __sync_lock_release(&globalock);

#ifdef CTEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl-1, "CoCacheSelectBlockToRemove_fifo(%d): t_select = %lf\n",dev_id, cpu_timer);
	cpu_timer = csecond();
#endif
#ifdef DEBUG
	lprintf(lvl-1, "<-----|\n");
#endif
	return result_node;
}

#elif CACHE_SCHEDULING_POLICY==2 || CACHE_SCHEDULING_POLICY==3
Node_LL* CoCacheSelectBlockToRemove_mru_lru(short dev_id){
short lvl = 6;
#ifdef DEBUG
	lprintf(lvl-1, "|-----> CoCacheSelectBlockToRemove_mru_lru(dev= %d)\n",dev_id);
#endif
#ifdef CTEST
	cpu_timer = csecond();
#endif

Node_LL* result_node = new Node_LL();

	result_node->idx = -1;
	short dev_id_idx = (dev_id == -1) ? LOC_NUM - 1: dev_id;

	if (DevCache[dev_id_idx] == NULL)
		error("CoCacheSelectBlockToRemove_mru_lru(%d): Called on empty buffer\n", dev_id);
	mru_lru_queues[dev_id_idx].lock_ll.lock();
	Node_LL* node = mru_lru_queues[dev_id_idx].start_iterration();
	if(node->idx >= 0)
		state tmt_state = CoCacheUpdateBlockState(dev_id, node->idx);
	while(DevCache[dev_id_idx]->BlockPendingEvents[node->idx] != NULL){
		node = mru_lru_queues[dev_id_idx].next_in_line();
		if(node->idx >= 0)
			state tmt_state = CoCacheUpdateBlockState(dev_id, node->idx);
		else
			break;
	}
	if(node->idx >=0 && DevCache[dev_id_idx]->BlockPendingEvents[node->idx] == NULL){
		// lprintf(lvl-1, "|-----> CoCacheSelectBlockToRemove_mru_lru: Invalidating\n");
		mru_lru_queues[dev_id_idx].invalidate(node);
		free(result_node);
		result_node = node;
	}
	mru_lru_queues[dev_id_idx].lock_ll.unlock();

#ifdef CTEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl-1, "CoCacheSelectBlockToRemove_mru(%d): t_select = %lf\n",dev_id, cpu_timer);
	cpu_timer = csecond();
#endif
#ifdef DEBUG
	lprintf(lvl-1, "<-----|\n");
#endif
	return result_node;
}
#endif

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
	short dev_id_idx = (dev_id == -1) ? LOC_NUM - 1: dev_id;
  if (DevCache[dev_id_idx] == NULL)
    error("CoCacheUpdateBlockState(%d,%d): Called on empty buffer\n", dev_id, BlockIdx);
  if (BlockIdx < 0 || BlockIdx >= DevCache[dev_id_idx]->BlockNum)
    error("CoCacheUpdateBlockState(%d,%d): Invalid BlockIdx = %d\n", dev_id, BlockIdx, BlockIdx);
		while(__sync_lock_test_and_set (&globalock, 1));
		state temp = (DevCache[dev_id_idx]->BlockState[BlockIdx] > 0) ? AVAILABLE : EMPTY;
		pending_events_p current = DevCache[dev_id_idx]->BlockPendingEvents[BlockIdx], prev = NULL;
		event_status start_status, end_status;
		short delete_curr_flag = 0;
	  while(current!=NULL){
			start_status = (current->event_start==NULL) ? GHOST : current->event_start->query_status();
			end_status = (current->event_end==NULL) ? GHOST : current->event_end->query_status();
#ifdef CDEBUG
			if (DevCache[dev_id_idx]->BlockCurrentTileDim[BlockIdx] == 1){
				Tile1D<VALUE_TYPE>* tmp = (Tile1D<VALUE_TYPE>*) DevCache[dev_id_idx]->BlockCurrentTilePtr[BlockIdx];
				lprintf(lvl, "Found pending action for BlockIdx = %d hosting T(%d.[%d])\n",
					BlockIdx, tmp->id, tmp->GridId);
				lprintf(lvl, "with event_start(%d) = %s, event_end(%d) = %s and effect = %s\n",
					(current->event_start==NULL) ? -1 : current->event_start->id, print_event_status(start_status),
					(current->event_end==NULL) ? -1 :  current->event_end->id, print_event_status(end_status),
					print_state(current->effect));
			}
			else if (DevCache[dev_id_idx]->BlockCurrentTileDim[BlockIdx] == 2){
				Tile2D<VALUE_TYPE>* tmp = (Tile2D<VALUE_TYPE>*) DevCache[dev_id_idx]->BlockCurrentTilePtr[BlockIdx];
				lprintf(lvl, "Found pending action for BlockIdx = %d hosting T(%d.[%d,%d])\n",
					BlockIdx, tmp->id, tmp->GridId1, tmp->GridId2);
				lprintf(lvl, "with event_start(%d) = %s, event_end(%d) = %s and effect = %s\n",
					(current->event_start==NULL) ? -1 : current->event_start->id, print_event_status(start_status),
					(current->event_end==NULL) ? -1 :  current->event_end->id, print_event_status(end_status),
					print_state(current->effect));
			}
			else error("CoCacheUpdateBlockState(%d): Tile%dD not implemented\n", dev_id, DevCache[dev_id_idx]->BlockCurrentTileDim[BlockIdx]);
#endif
			if (start_status == GHOST && end_status == GHOST){
				warning("CoCacheUpdateBlockState: Double-GHOST event found, removing (bug?)\n");
				delete_curr_flag = 1;
			}
			else if(end_status == GHOST){ // Instantaneous start effect event
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
			// Continuous from record to end event.
			else if(start_status == GHOST){
				if (end_status == COMPLETE || end_status == CHECKED){
					delete_curr_flag = 1;
#ifdef CDEBUG
					lprintf(lvl, "From record to end action complete, removing\n");
#endif
				}
				else{
#ifdef CDEBUG
					lprintf(lvl, "From record to end action incomplete, updating cache and waiting\n");
#endif
					CoCacheStateUpdate(&temp, current->effect);
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
				if (prev == NULL) DevCache[dev_id_idx]->BlockPendingEvents[BlockIdx] = current;
				else prev->next = current;
				free(free_tmp);
			}
		}
	#ifdef DEBUG
  	lprintf(lvl-1, "<-----|\n");
  #endif
	DevCache[dev_id_idx]->BlockState[BlockIdx] = temp;
	__sync_lock_release(&globalock);
  return temp;
}

void CoCacheAddPendingEvent(short dev_id, Event* e_start, Event* e_end, int BlockIdx, state effect){
	short lvl = 5;
#ifdef DEBUG
	lprintf(lvl-1, "|-----> CoCacheAddPendingEvent(dev = %d, e_start(%d), e_end(%d), BlockIdx = %d, %s)\n",
		dev_id, (e_start==NULL) ? -1 : e_start->id,
		(e_end==NULL) ? -1 : e_end->id, BlockIdx, print_state(effect));
#endif

	short dev_id_idx = (dev_id == -1) ? LOC_NUM - 1: dev_id;
  if (DevCache[dev_id_idx] == NULL)
    error("CoCacheAddPendingEvent(%d,%d): Called on empty buffer\n", dev_id, BlockIdx);
  if (BlockIdx < 0 || BlockIdx >= DevCache[dev_id_idx]->BlockNum)
    error("CoCacheAddPendingEvent(%d,%d): Invalid BlockIdx = %d\n", dev_id, BlockIdx, BlockIdx);
	while(__sync_lock_test_and_set (&globalock, 1));

	pending_events_p current = DevCache[dev_id_idx]->BlockPendingEvents[BlockIdx], new_pending_event;
	new_pending_event = (pending_events_p) malloc(sizeof(struct pending_action_list));
	new_pending_event->event_start = e_start;
	new_pending_event->event_end = e_end;
	new_pending_event->effect = effect;
	new_pending_event->next = NULL;
	if (current == NULL ) DevCache[dev_id_idx]->BlockPendingEvents[BlockIdx] = new_pending_event;
	else {
		while(current->next!=NULL) current = current->next;
		current->next = new_pending_event;
	}
#if CACHE_SCHEDULING_POLICY==2
	mru_lru_queues[dev_id_idx].lock_ll.lock();
	mru_lru_queues[dev_id_idx].put_first(mru_lru_hash[dev_id_idx][BlockIdx]);
	mru_lru_queues[dev_id_idx].lock_ll.unlock();
#elif CACHE_SCHEDULING_POLICY==3
	mru_lru_queues[dev_id_idx].lock_ll.lock();
	mru_lru_queues[dev_id_idx].put_last(mru_lru_hash[dev_id_idx][BlockIdx]);
	mru_lru_queues[dev_id_idx].lock_ll.unlock();
#endif
	__sync_lock_release(&globalock);
#ifdef DEBUG
  lprintf(lvl-1, "<-----|\n");
#endif
}


long long CoCoGetBlockSize(short dev_id){
	short lvl = 5;
#ifdef DEBUG
	lprintf(lvl-1, "|-----> CoCoGetBlockSize(dev = %d)\n", dev_id);
#endif

	short dev_id_idx = (dev_id == -1) ? LOC_NUM - 1: dev_id;
	if (DevCache[dev_id_idx] == NULL)
		error("CoCoGetBlockSize(%d): Called on empty buffer\n", dev_id);
	return DevCache[dev_id_idx]->BlockSize;
}
