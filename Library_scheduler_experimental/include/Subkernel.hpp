/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The header containing the "Subkernel" definition for data scheduling and management in heterogeneous multi-device systems.
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
		int id, iloc1, iloc2, iloc3;
		short run_dev_id;
		short WR_first, *WR_last;
		short TileNum, *TileDimlist;
		void** TileList;
		Subkernel* prev, * next;
#ifdef STEST
		Event_timer_p input_timer, output_timer, operation_timer;
		long long bytes_in, bytes_out, flops;
#endif
		Event* operation_complete, *writeback_complete;
		void* operation_params;
		const char* op_name;

		/// Constructors
		Subkernel(short TileNum, const char* name);

		/// Destructors
		~Subkernel();

		/// Functions
		void init_events();
		void request_data();
		void request_tile(short TileIdx);
		void sync_request_data();
		void run_operation();
		void writeback_data();

		short is_dependency_free();
};

typedef struct kernel_pthread_wrap{
	short dev_id;
	Subkernel** SubkernelListDev;
	int SubkernelNumDev;
}* kernel_pthread_wrap_p;

//Subkernel** CoCoAsignTilesToSubkernelsGemm(Asset2D<double>* A_asset, Asset2D<double>* B_asset,
//Asset2D<double>* C_asset, int T, int* kernelNum);

void 	CoCoPeLiaInitResources(short dev_id);
void 	CoCoPeLiaFreeResources(short dev_id);

Subkernel* SubkernelSelectSimple(short dev_id, Subkernel** Subkernel_list, long Subkernel_list_len);

#endif
