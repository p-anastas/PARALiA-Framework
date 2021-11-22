/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The header containing the "Subkernel" (== subkernel) definition for data scheduling and management in heterogeneous multi-device systems.
///

#ifndef Subkernel_H
#define Subkernel_H

#include<iostream>
#include <string>

#include "unihelpers.hpp"

class Subkernel
{
	private:
	public:
		std::string name;
		short run_dev_id, writeback_master;
		short TileNum, *TileDimlist;
		void** TileList;
		Subkernel* prev;
		Event* data_available, *operation_complete;
		void* operation_params;
		void* operation;

		/// Constructors
		Subkernel(short TileNum);

		/// Functions
		void init_events();
		void request_data();
		void run_operation();
		void writeback_data();
		void print() { std::cout << "Subkernel : " << name; }

};

typedef struct kernel_pthread_wrap{
	short devId;
	Subkernel** SubkernelListDev;
	int SubkernelNumDev;
}* kernel_pthread_wrap_p;

//Subkernel** CoCoAsignTilesToSubkernelsGemm(Asset2D<double>* A_asset, Asset2D<double>* B_asset, Asset2D<double>* C_asset, int T, int* kernelNum);

void CoCoPeLiaDevCacheInvalidate(kernel_pthread_wrap_p subkernel_data);

long long CoCoPeLiaDevBuffSz(kernel_pthread_wrap_p subkernel_data);

void 	CoCoPeLiaInitStreams(short dev_id);

void CoCoPeLiaSubkernelFireAsync(Subkernel* subkernel);

#endif
