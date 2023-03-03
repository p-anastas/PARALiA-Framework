///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The subkernel distributions implemented for PARALiA
///
#ifndef SUBKERNEL_DIST_H
#define SUBKERNEL_DIST_H

#include "Autotuner.hpp"

void CoCoDistributeSubkernelsRoundRobin(ATC_p autotune_controller);
void CoCoDistributeSubkernelsNaive(ATC_p autotune_controller);
void CoCoDistributeSubkernelsRoundRobinChunk(ATC_p autotune_controller,  int Chunk_size);
void CoCoDistributeSubkernelsRoundRobinChunkReverse(ATC_p autotune_controller,  int Chunk_size);
void CoCoDistributeSubkernels2DBlockCyclic(ATC_p autotune_controller, int D1GridSz, int D2GridSz, int D3GridSz);
#endif
