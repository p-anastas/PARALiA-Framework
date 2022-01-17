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
	EMPTY = 0, /// Cache Block is empty.
	AVAILABLE = 1, /// exists in location with no (current) operations performed on it.
	R = 2,  /// is being read/used in operation.
	W = 3,  /// is being modified (or transefered).
};

const char* print_state(state in_state);

typedef struct pending_action_list{
	Event* event_start, *event_end;
	state effect;
	struct pending_action_list* next;
}* pending_events_p;

int pending_events_free(pending_events_p target);

/* Device-wise software cache struct declaration */
typedef struct DevCache_str{
	short dev_id;
	void * gpu_mem_buf;
	long long gpu_mem_buf_sz;
	int BlockNum, serialCtr;
	long long BlockSize;
	state* BlockState;
	short* BlockCurrentTileDim;
	void** BlockCurrentTilePtr;
	pending_events_p *BlockPendingEvents;
}* DevCachePtr;

long long CoCoPeLiaDevBuffSz(kernel_pthread_wrap_p subkernel_data);
DevCachePtr CoCoPeLiaGlobufInit(short dev_id);
void CoCoPeLiaRequestBuffer(kernel_pthread_wrap_p subkernel_data, long long bufsize_limit);

void* CoCacheAsignBlock(short dev_id, void* TilePtr, short TileDim);

int CoCacheSelectBlockToRemove_naive(short dev_id);

void* CoCacheUpdateAsignBlock(short dev_id, void* TilePtr, short TileDim);

void CoCoPeLiaUnlockCache(short dev_id);

state CoCacheUpdateBlockState(short dev_id, int BlockIdx);

void CoCacheAddPendingEvent(short dev_id, Event* e_start, Event* e_end, int BlockIdx, state effect);

long long CoCoGetBlockSize(short dev_id);

///Invalidates the GPU-allocated cache buffer metadata at target device
void CoCoPeLiaDevCacheInvalidate(kernel_pthread_wrap_p subkernel_data);

#endif
