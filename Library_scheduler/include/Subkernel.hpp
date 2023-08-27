/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The header containing the "Subkernel" definition for data scheduling and management in heterogeneous multi-device systems.
///

#ifndef Subkernel_H
#define Subkernel_H

#include<iostream>
#include <string>

#include "linkmap.hpp"
#include "DataTile.hpp"

class Subkernel
{
	private:
	public:
		int id, iloc1, iloc2, iloc3;
		short run_dev_id;
		short *TileDimlist; // TODO: this might be irrelevant in new version
		short TileNum;
		DataTile** TileList;
		Subkernel* prev, * next;

#ifdef STEST
		double req_in_ts = 0, req_out_ts = 0;
		double reqT_fire_ts[3] = {0}, reqT_start_ts[3] = {0}, reqT_end_ts[3] = {0};
		double op_fire_ts = 0, op_start_ts = 0, op_end_ts = 0;
		double wbT_fire_ts[3] = {0}, wbT_start_ts[3] = {0}, wbT_end_ts[3] = {0};
		long long bytes_in[3] = {0}, bytes_out[3] = {0}, flops = 0;
		short dev_in_from[3], dev_in_to[3], dev_out_from[3], dev_out_to[3];
#endif
		//Event* operation_complete;
		void* operation_params;
		const char* op_name;

		/// Constructors
		Subkernel(short TileNum, const char* name);

		/// Destructors
		~Subkernel();

		void sync_request_data();

		/// Functions
		void prepare_launch(short dev_id);
		void request_data();
		void run_operation();
		short check_ready(); 
		void run_ready_operation();

		long double opt_fetch_cost(short dev_id);
		long double opt_fetch_score(short dev_id);
};

//Subkernel** CoCoAsignTilesToSubkernelsGemm(Decom2D<double>* A_asset, Decom2D<double>* B_asset,
//Decom2D<double>* C_asset, int T, int* kernelNum);

void CoCoPeLiaInitResources(int* dev_list, int dev_num);
void CoCoPeLiaInitWS(int* dev_list, int dev_num);
void CoCoPeLiaFreeResources();
void CoCoPeLiaCleanResources();


int SubkernelPrefetchCheapRONLYTiles(int numTiles, short dev_id, Subkernel** Subkernel_list, long Subkernel_list_len);

Subkernel* SubkernelSelect(short dev_id, Subkernel** Subkernel_list, long Subkernel_list_len);

void sync_request_paired(short dev_id);

#ifdef STEST
void STEST_print_SK(kernel_pthread_wrap_p* thread_dev_data_list, double routine_entry_ts, short dev_num);
#endif

#endif
