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
		double req_in_ts = 0, req_out_ts = 0;
		double reqT_fire_ts[3] = {0}, reqT_start_ts[3] = {0}, reqT_end_ts[3] = {0};
		double op_fire_ts = 0, op_start_ts = 0, op_end_ts = 0;
		double wbT_fire_ts[3] = {0}, wbT_start_ts[3] = {0}, wbT_end_ts[3] = {0};
		long long bytes_in[3] = {0}, bytes_out[3] = {0}, flops = 0;
		short dev_in_from[3], dev_in_to[3], dev_out_from[3], dev_out_to[3];
#endif
		Event* operation_complete;
		void* operation_params;
		const char* op_name;

		/// Constructors
		Subkernel(short TileNum, const char* name);

		/// Destructors
		~Subkernel();

		/// Functions
		void prepare_launch(short dev_id);
		void init_events();
		void request_data();
		void request_tile(short TileIdx);
		void sync_request_data_RONLY();
		void sync_request_data();
		void run_operation();
		void writeback_data();

		short no_locked_tiles();
		short is_RW_lock_master(short dev_id);
		short RW_lock_initialized();
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
Subkernel* SubkernelSelectMinimizeFetchWritePenaltyMultiFetchPenalty(short dev_id,
	Subkernel** Subkernel_list, long Subkernel_list_len);

Subkernel* SubkernelSelectMinimizeFetchNoWriteShareMultiFetchPenaltyMutlidevFair(short dev_id,
	Subkernel** Subkernel_list, long Subkernel_list_len);
Subkernel* SubkernelSelectMinimizeFetchWritePenaltyMultiFetchPenaltyMutlidevFair(short dev_id,
	Subkernel** Subkernel_list, long Subkernel_list_len);

void sync_request_paired(short dev_id);

#ifdef STEST
void STEST_print_SK(kernel_pthread_wrap_p* thread_dev_data_list, double routine_entry_ts, short dev_num);
#endif

#endif
