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
		Event_timer_p input_timer[3], output_timer[3], operation_timer;
		double request_data_in_ts, request_tile_in_ts[3], run_operation_in_ts, writeback_data_in_ts;
		double request_data_out_ts, request_tile_out_ts[3], run_operation_out_ts, writeback_data_out_ts;
		long long bytes_in[3] = {0}, bytes_out[3] = {0}, flops = 0;
#endif
		Event* operation_complete, *writeback_complete;
		void* operation_params;
		const char* op_name;

		/// Constructors
		Subkernel(short TileNum, const char* name);

		/// Destructors
		~Subkernel();

		/// Functions
		void prepare_launch();
		void init_events();
		void request_data();
		void request_tile(short TileIdx);
		void sync_request_data_RONLY();
		void sync_request_data();
		void run_operation();
		void writeback_data();

		short no_locked_tiles();
		short is_RW_master(short dev_id);
		double opt_fetch_cost(short dev_id);
		double opt_fetch_cost_pen_multifetch(short dev_id);
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
Subkernel* SubkernelSelectNoWriteShare(short dev_id, Subkernel** Subkernel_list, long Subkernel_list_len);
Subkernel* SubkernelSelectMinimizeFetch(short dev_id, Subkernel** Subkernel_list, long Subkernel_list_len);
Subkernel* SubkernelSelectMinimizeFetchWritePenalty(short dev_id, Subkernel** Subkernel_list, long Subkernel_list_len);
Subkernel* SubkernelSelectMinimizeFetchWritePenaltyMultiFetchPenalty(short dev_id, Subkernel** Subkernel_list, long Subkernel_list_len);

void sync_request_paired(short dev_id);

#endif
