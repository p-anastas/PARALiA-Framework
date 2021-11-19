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

/* global variable declaration */
typedef struct globuf{
	short dev_id;
	void * gpu_mem_buf = NULL;
	long long gpu_mem_buf_sz = 0;
	long long gpu_mem_offset = 0;
}* DevBufPtr;

DevBufPtr CoCoPeLiaBufferInit(short dev_id);
void CoCoPeLiaRequestBuffer(short dev_id, long long size);
void* CoCoPeLiaAsignBuffer(short dev_id, long long size);

#endif
