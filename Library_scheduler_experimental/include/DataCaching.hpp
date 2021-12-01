/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The header containing the caching functions for data scheduling and management in heterogeneous multi-device systems.
///

#ifndef DATACACHNING_H
#define DATACACHNING_H

#include<iostream>
#include <string>

#include "unihelpers.hpp"
#include "Subkernel.hpp"


enum state{
	INVALID = 0, /// Tile does not exist in this location.
	MASTER = 1, /// The Tile is in its initial memory location and should NEVER be deleted internally in CoCoPeLia.
	AVAILABLE = 2, /// Tile exists in this location and is available for reading.
	BUSY = 3,  /// Tile is being modified (or transefered) by somebody in this location.
	LOCKED = 4 ///  Blocked while the cache replacement is running. Not sure this state is required.
};

/* global variable declaration */
typedef struct globuf{
	short dev_id;
	void * gpu_mem_buf;
	long long gpu_mem_buf_sz;
	int BlockNum, serialCtr;
	long long BlockSize;
	state*  BlockStatus;
}* DevBufPtr;

long long CoCoPeLiaDevBuffSz(kernel_pthread_wrap_p subkernel_data);
void CoCoPeLiaRequestBuffer(kernel_pthread_wrap_p subkernel_data);

DevBufPtr CoCoPeLiaGlobufInit(short dev_id);
void* CoCoPeLiaAsignBuffer(short dev_id, int* BlockIDptr);
void* CoCoPeLiaReAsignBuffer(Subkernel* inkernel, int* BlockIDptr);
void* CoCoPeLiaUnlockCache(short dev_id);
void CoCoPeLiaUpdateCache(short dev_id, int BlockIdx, state update_state);
state CoCoPeLiaGetCacheState(short dev_id, int BlockIdx);

#endif
