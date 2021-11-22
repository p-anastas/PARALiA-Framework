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

DevBufPtr CoCoPeLiaBufferInit(short dev_id){
  DevBufPtr result = (DevBufPtr) malloc (sizeof(struct globuf));
  result->dev_id = dev_id;
  result->gpu_mem_buf = NULL;
  result->gpu_mem_buf_sz = result->gpu_mem_offset = 0;
  return result;
}

void CoCoPeLiaRequestBuffer(short dev_id, long long buff_req_sz){
  short lvl = 3;
#ifdef DEBUG
  lprintf(lvl-1, "|-----> CoCoPeLiaRequestBuffer(%d, %lld)\n", dev_id, buff_req_sz);
#endif
  if (GloBuf[dev_id] == NULL) GloBuf[dev_id] = CoCoPeLiaBufferInit(dev_id);
  long long free_dev_mem, max_dev_mem;
  CoCoPeLiaDevGetMemInfo(&free_dev_mem, &max_dev_mem);
  size_t problem_avail_mem = free_dev_mem + GloBuf[dev_id]->gpu_mem_buf_sz;
  // Problem Fits-in-GPU case
  if (buff_req_sz < problem_avail_mem){
#ifdef DEBUG
    lprintf(lvl, " -Requested buffer fits in GPU\n");
    if (GloBuf[dev_id]->gpu_mem_buf_sz >= buff_req_sz) lprintf(lvl, " -GPU buf available: %zu MB\n", (size_t) buff_req_sz/1024/1024);
    else if (GloBuf[dev_id]->gpu_mem_buf_sz > 0) lprintf(lvl, " -Smaller GPU buf available -> resizing : %zu -> %zu MB\n", (size_t) GloBuf[dev_id]->gpu_mem_buf_sz/1024/1024, (size_t) buff_req_sz/1024/1024);
    else if (GloBuf[dev_id]->gpu_mem_buf_sz == 0) lprintf(lvl, " -Initializing new buffer: %zu MB\n",(size_t) buff_req_sz/1024/1024);
#endif
    if (GloBuf[dev_id]->gpu_mem_buf_sz >= buff_req_sz);
    else if (GloBuf[dev_id]->gpu_mem_buf_sz > 0){
      CoCoFree(GloBuf[dev_id]->gpu_mem_buf, dev_id);
      GloBuf[dev_id]->gpu_mem_buf = CoCoMalloc(buff_req_sz, dev_id);
      GloBuf[dev_id]->gpu_mem_buf_sz = buff_req_sz;
    }
    else if (GloBuf[dev_id]->gpu_mem_buf_sz == 0){
      GloBuf[dev_id]->gpu_mem_buf = CoCoMalloc(buff_req_sz, dev_id);
      GloBuf[dev_id]->gpu_mem_buf_sz = buff_req_sz;
    }
    else error("Unknown memory case");
    CoCoSyncCheckErr();
  }
  else error("CoCoPeLia larger than dev mem not implemented\n");
#ifdef DEBUG
	lprintf(lvl-1, "<-----|\n");
#endif
}

void* CoCoPeLiaAsignBuffer(short dev_id, long long size){
  short lvl = 5;
#ifdef DEBUG
  lprintf(lvl-1, "|-----> CoCoPeLiaAsignBuffer(%d,%lld)\n", dev_id, size);
#endif

  if (GloBuf[dev_id]->gpu_mem_offset + size > GloBuf[dev_id]->gpu_mem_buf_sz) error("CoCoPeLiaAsignBuffer: Buffer \
  full but request for more offset: %lld + %lld > %lld\n",
    GloBuf[dev_id]->gpu_mem_offset, size, GloBuf[dev_id]->gpu_mem_buf_sz);
  void* result = GloBuf[dev_id]->gpu_mem_buf + GloBuf[dev_id]->gpu_mem_offset;
  GloBuf[dev_id]->gpu_mem_offset += size;
#ifdef DEBUG
	lprintf(lvl-1, "<-----|\n");
#endif
  return result;
}

void CoCopeLiaDgemm_flush_gpu_mem_buf(short dev_id)
{
	short lvl = 3;
#ifdef DEBUG
			lprintf(lvl-1, "|-----> CoCopeLiaDgemm_flush_gpu_mem_buf(dev_id=%d)\n", dev_id);
#endif
	if (GloBuf[dev_id]){
#ifdef DEBUG
		lprintf(lvl, "Clearing (presumably) %zu MB\n\n", (size_t) GloBuf[dev_id]->gpu_mem_buf_sz/1024/1024);
#endif
		CoCoFree(GloBuf[dev_id]->gpu_mem_buf, dev_id);
		free(GloBuf[dev_id]);
		GloBuf[dev_id] = NULL;
		CoCoSyncCheckErr();
	}
	else{
#ifdef DEBUG
		lprintf(lvl, "Target buffer already empty\n");
#endif
		;
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
  if (GloBuf[subkernel_data->devId]!= NULL) GloBuf[subkernel_data->devId]->gpu_mem_offset = 0;
}
