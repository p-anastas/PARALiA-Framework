/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The header containing the "Operative" (== subkernel) definition for data scheduling and management in heterogeneous multi-device systems.
///

#ifndef OPERATIVE_H
#define OPERATIVE_H

#include<iostream>
#include <string>

#include "Agent.hpp"

class Operative
{
	private:
	public:
		std::string name;
		Agent* Manager;
		short TileNum, *TileDimlist;
		DTYPE_SUP *TileDtypeList;
		void** TileList;
		Event* data_available, *operation_complete;
		void* operation_params;
		void* operation;

		/// Constructors
		Operative(short TileNum);

		/// Functions
		void request_data();
		void run_operation();
		void print() { std::cout << "Operative : " << name; }

};

Operative** CoCoAsignTilesToOperativesGemm(Asset2D<double>* A_asset, Asset2D<double>* B_asset, Asset2D<double>* C_asset, int T, int* kernelNum);

/*

typedef struct subkernel3_gen {
	cublasOperation_t gpu_op_A, gpu_op_B;
	size_t Ms, Ns, Ks;
	double alpha, beta;
	/// The initial (->start) Subkernel Matrices.
	double* As, *Bs, *CsIn, *CsOut;
	size_t ldAs, ldBs, ldCsIn, ldCsOut;

	/// The GPU buffers for each matrix, if required.
	double * AdevBuf, * BdevBuf, *CdevBuf;

	/// The GPU pointers used in the actual kernel calls (Xker infered from Xs, XdevBuf).
	double * Aker, * Bker, *Cker;
	size_t ldAker, ldBker, ldCker;

	short Asloc, Bsloc, Csloc, CsOutloc;

	size_t MgridSz, NgridSz, KgridSz;
	size_t MblockSz, NblockSz, KblockSz;

	short MgridIdx, NgridIdx, KgridIdx;
	short AsMaster, BsMaster, CsMaster, CsOutMaster;

	short devId;
	cudaEvent_t data_avail, gemm_complete;



} * kernel3_p;

kernel3_p CoCopeLiaDgemmSubkernelInit(cublasOperation_t gpu_op_A, cublasOperation_t gpu_op_B, size_t M, size_t N, size_t K, double alpha, double* A, size_t ldA, double* B, size_t ldB, double beta, double* Cin, size_t ldCin, double* Cout, size_t ldCout, BLAS3GPUBufPtr GloBuf, short dev_id);
*/

#endif
