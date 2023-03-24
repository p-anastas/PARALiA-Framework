///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
/// \author Theodoridis Aristomenis (atheodor@cslab.ece.ntua.gr)
///
/// \brief The DDOT CoCopeLia implementation using the new mission-agent-asset C++ classes.
///

#include "backend_wrappers.hpp"
#include "Autotuner.hpp"
#include "PARALiA.hpp"
#include "unihelpers.hpp"
#include "Decomposer.hpp"
#include "Subkernel.hpp"
#include "DataCaching.hpp"

#include <pthread.h>

pthread_barrier_t  SoftCache_alloc_barrier_dot;

dot_backend_in<double>* initial_dot = NULL;
ATC_p autotune_controller_dot = NULL;
ATC_p predef_controller_dot = NULL;

int NGridSz_dot = 0;

#ifdef STEST
double dot_entry_ts;
#endif

Subkernel** Subkernel_list_dot;
int Subkernel_num_dot;
int remaining_Subkernels_dot;

int Sk_select_lock_dot = 0;

Subkernel** CoCoAsignTilesToSubkernelsDdot(Decom1D* x_asset, Decom1D* y_asset,
	int T, int* kernelNum){

	short lvl = 2;

	NGridSz_dot = x_asset->GridSz;
	*kernelNum = NGridSz_dot;
#ifdef DEBUG
	lprintf(lvl-1, "|-----> CoCoAsignTilesToSubkernelsDdot(x_asset,y_asset,%d,%d)\n", T, *kernelNum);
	lprintf(lvl,"NGridSz_dot = %d\n", NGridSz_dot);
	lprintf(lvl,"Nlast = %d\n",
	x_asset->Tile_map[NGridSz_dot-1]->dim);
#endif

Subkernel** kernels = (Subkernel**) malloc(*kernelNum*sizeof(Subkernel*));
int current_ctr = 0;
		for (int ni = 0; ni < NGridSz_dot; ni++){
            current_ctr = ni;
			kernels[current_ctr] = new Subkernel(2,"Ddot");
			kernels[current_ctr]->iloc1 = ni;
			kernels[current_ctr]->TileDimlist[0] = kernels[current_ctr]->TileDimlist[1] = 1;
			kernels[current_ctr]->TileList[0] = x_asset->getTile(ni);
			kernels[current_ctr]->TileList[1] = y_asset->getTile(ni);
			((Tile1D*)kernels[current_ctr]->TileList[0])->R_flag = 1;
			((Tile1D*)kernels[current_ctr]->TileList[1])->R_flag = 1;
			//((Tile1D*)kernels[current_ctr]->TileList[1])->W_flag = 1;
			//((Tile1D*)kernels[current_ctr]->TileList[1])->W_total = 1;
			kernels[current_ctr]->operation_params = (void*) malloc(sizeof( dot_backend_in<double>));
			dot_backend_in<double>* ptr_ker_translate = (dot_backend_in<double>*) kernels[current_ctr]->operation_params;
			ptr_ker_translate->N = ((Tile1D*) kernels[current_ctr]->TileList[0])->dim;
			ptr_ker_translate->x = NULL;
			ptr_ker_translate->y = NULL;
			ptr_ker_translate->incx = initial_dot->incx;
			ptr_ker_translate->incy = initial_dot->incy;
			ptr_ker_translate->result = (double*) calloc(1, sizeof(double));
			// No interal dims for dot to reduce
			kernels[current_ctr]->WR_first = 0;
			kernels[current_ctr]->WR_last = (short*) calloc (2, sizeof(short));
		}
#ifdef DEBUG
	lprintf(lvl-1, "<-----|\n");
#endif
	return kernels;
}

void CoCoDdotUpdateDevice(Subkernel* ker, short dev_id){
	dot_backend_in<double>* ptr_ker_translate = (dot_backend_in<double>*) ker->operation_params;
	ker->run_dev_id = ptr_ker_translate->dev_id = dev_id;
	short dev_id_idx = (dev_id == -1) ? LOC_NUM - 1: dev_id;
	ptr_ker_translate->incx = ((Tile1D*) ker->TileList[0])->inc[dev_id_idx];
	ptr_ker_translate->incy = ((Tile1D*) ker->TileList[1])->inc[dev_id_idx];
}

void CoCoDdotUpdatePointers(Subkernel* ker){
	dot_backend_in<double>* ptr_ker_translate = (dot_backend_in<double>*) ker->operation_params;
	short dev_id_idx = idxize(ker->run_dev_id);
	ptr_ker_translate->x = &((Tile1D*) ker->TileList[0])->StoreBlock[dev_id_idx]->Adrs;
	ptr_ker_translate->y = &((Tile1D*) ker->TileList[1])->StoreBlock[dev_id_idx]->Adrs;
}

void* CoCopeLiaDotAgentVoid(void* kernel_pthread_wrapped){
	short lvl = 2;

	kernel_pthread_wrap_p dot_subkernel_data = (kernel_pthread_wrap_p)kernel_pthread_wrapped;
	short dev_id = dot_subkernel_data->dev_id;
#ifdef DEBUG
	lprintf(lvl-1, "|-----> CoCopeLiaDotAgentVoid(dot_subkernel_data: dev_id = %d, SubkernelNumDev = %d)\n",
		dev_id, dot_subkernel_data->SubkernelNumDev);
#endif
#ifdef TEST
		double cpu_timer = csecond();
#endif

	CoCoPeLiaSelectDevice(dev_id);
	CoCoPeLiaInitResources(dev_id);

#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Stream/Lib Handle Initialization(%d): t_resource = %lf ms\n", dev_id, cpu_timer*1000);
	cpu_timer = csecond();
#endif

	Global_Buffer_1D[idxize(dev_id)]->allocate(true);
	//CoCoSyncCheckErr();

#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Memory management(%d): t_mem = %lf ms\n", dev_id, cpu_timer*1000);
	cpu_timer = csecond();
#endif
	pthread_barrier_wait (&SoftCache_alloc_barrier_dot);

#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Wait barrier(%d): t_wb = %lf ms\n", dev_id, cpu_timer*1000);
	cpu_timer = csecond();
#endif

	Subkernel * curr = NULL, *prev = NULL;
#ifndef RUNTIME_SCHEDULER_VERSION
	/// Rename global vars, perfectly safe.
	Subkernel** Subkernel_list_dot = dot_subkernel_data->SubkernelListDev;
	int Subkernel_num_dot = dot_subkernel_data->SubkernelNumDev;
	int remaining_Subkernels_dot = Subkernel_num_dot;
#else
	dot_subkernel_data->SubkernelNumDev = 0 ;
#endif
	while (remaining_Subkernels_dot){
		prev = curr;
		if(prev) prev->sync_request_data();
		while(__sync_lock_test_and_set(&Sk_select_lock_dot, 1));
		curr = SubkernelSelect(dev_id, Subkernel_list_dot, Subkernel_num_dot);
		if (!curr){
	//#ifdef DDEBUG
	//		lprintf(lvl, "CoCoPeLiaDotAgentVoid(%d): Got curr = NULL, repeating search\n", dev_id);
	//#endif
			__sync_lock_release(&Sk_select_lock_dot);
			continue;
		}
		remaining_Subkernels_dot--;
#ifdef RUNTIME_SCHEDULER_VERSION
		dot_subkernel_data->SubkernelListDev[dot_subkernel_data->SubkernelNumDev] = curr;
		dot_subkernel_data->SubkernelNumDev++;
#else
		for (int keri = 0; keri < dot_subkernel_data->SubkernelNumDev; keri++)
			if (curr == dot_subkernel_data->SubkernelListDev[keri]){
				dot_subkernel_data->SubkernelListDev[keri] = dot_subkernel_data->SubkernelListDev[remaining_Subkernels_dot];
				dot_subkernel_data->SubkernelListDev[remaining_Subkernels_dot] = curr;
				break;
			}
		Subkernel_num_dot = remaining_Subkernels_dot;
#endif
		curr->prev = prev;
		if(prev) prev->next = curr;
		CoCoDdotUpdateDevice(curr, dev_id);
		curr->init_events();
		curr->request_data();
		CoCoDdotUpdatePointers(curr);
		__sync_lock_release(&Sk_select_lock_dot);
		curr->run_operation();
#ifdef ENABLE_SEND_RECV_OVERLAP
		curr->writeback_data();
#endif
	}

#ifndef ENABLE_SEND_RECV_OVERLAP
	for (int keri = 0; keri < dot_subkernel_data->SubkernelNumDev; keri++)
		dot_subkernel_data->SubkernelListDev[keri]->writeback_data();
#endif

	CoCoSyncCheckErr();
#ifdef TEST
	double total_cache_timer = Global_Buffer_1D[idxize(dev_id)]->timer;
	lprintf(lvl, "Cache requests total timer (%d): t_cache = %lf ms\n" , dev_id, total_cache_timer*1000);
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Subkernels complete(%d): t_comp = %lf ms\n" , dev_id, cpu_timer*1000);
#endif

#ifdef DEBUG
	lprintf(lvl-1, "<-----|\n");
#endif
	return NULL;
}

/// A dot wrapper including auto-tuning of T and cache_size, as well as device management
ATC_p PARALiADdot(long int N, double* x, long int incx, double* y, long int incy, double* result)
{
	short lvl = 1;
#ifdef DEBUG
	lprintf(lvl-1, "|-----> PARALiADdot(%zu,x=%p(%d),%zu,y=%p(%d),%zu)\n",
		N, x, CoCoGetPtrLoc(x), incx, y, CoCoGetPtrLoc(y), incy);
#endif

#ifdef TEST
	lprintf(lvl-1, "|-----> PARALiADdot\n");
	double cpu_timer = csecond();
#endif
#ifdef STEST
	dot_entry_ts = csecond();
#endif
	int prev_dev_id = CoCoPeLiaGetDevice();

	short reuse_model_flag = 1;
	if(!initial_dot){
		initial_dot = (dot_backend_in<double>*) malloc(sizeof( dot_backend_in<double>));
		reuse_model_flag = 0;
	}

	if(reuse_model_flag && initial_dot->N != N)
		reuse_model_flag = 0;
	initial_dot->N = N;

	if(reuse_model_flag && initial_dot->x!= NULL && *initial_dot->x != x)
		reuse_model_flag = 0;
	initial_dot->x = (void**) &x;

	if(reuse_model_flag && initial_dot->y!= NULL && *initial_dot->y != y)
		reuse_model_flag = 0;
	initial_dot->y = (void**) &y;

	initial_dot->incx = incx;
	initial_dot->incy = incy;
	initial_dot->result = result;
	initial_dot->dev_id = -1;

	Decom1D* x_asset, *y_asset;
	/// Prepare Assets in parallel( e.g. initialize asset classes, pin memory with pthreads)
	/// return: x_asset, y_asset initialized and pinned
	x_asset = new Decom1D( x, N, incx, DOUBLE);
	y_asset = new Decom1D( y, N, incy, DOUBLE);

	pthread_attr_t attr;
	int s = pthread_attr_init(&attr);
	if (s != 0) error("PARALiADdot: pthread_attr_init failed s=%d\n", s);

	pthread_t asset_thread_id[2];
	x_asset->prepareAsync(&asset_thread_id[0], attr);
	y_asset->prepareAsync(&asset_thread_id[1], attr);

	if (!reuse_model_flag){
		delete autotune_controller_dot;
		autotune_controller_dot = NULL;
	}
	if (autotune_controller_dot == NULL) autotune_controller_dot = new ATC();
	if (predef_controller_dot && autotune_controller_dot->diff_intialized_params_ATC(predef_controller_dot)){
		 autotune_controller_dot->mimic_ATC(predef_controller_dot);
		 reuse_model_flag = 0;
	}
	double autotune_timer = 0;
	if(!reuse_model_flag) autotune_timer = autotune_controller_dot->autotune_problem("Ddot", initial_dot);

	void* res;
	for(int i=0; i<2;i++){
		s = pthread_join(asset_thread_id[i], &res);
		if (s != 0) error("PARALiADdot: pthread_join failed with exit value %d", s);
		//free(res);      /* Free memory allocated by thread */
	}

#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Preparing assets (parallel with pthreads) -> t_prep = %lf ms\n", cpu_timer*1000);
	cpu_timer = csecond();
#endif
	long int T = autotune_controller_dot->T;

	int GPU_Block_num, Block_num = 1 + (x_asset->dim/T + ((x_asset->dim%T)? 1 : 0)) +
		 (y_asset->dim/T + ((y_asset->dim%T)? 1 : 0));
	long long Block_sz = 	T*sizeof(double);
	GPU_Block_num = Block_num;
	if(autotune_controller_dot->cache_limit > 0){
		int max_block_num = autotune_controller_dot->cache_limit/Block_sz;
		if(max_block_num < Block_num){
			lprintf(0, "CoCopeLiaDot: Input cache limit %lld forces problem to use %d blocks\
			instead of %d needed for the full problem\n", autotune_controller_dot->cache_limit, max_block_num, Block_num);
			// With Parallel backends unfortunately == SK_blocks + Max_Exclusive_Blocks - 1
			int worst_case_ex_blocks = 2;
			if(max_block_num < worst_case_ex_blocks)
				error("CoCopeLiaDot: Not able to run with < %d blocks per cache due to EX scheduling\n", worst_case_ex_blocks);
				GPU_Block_num = max_block_num;
		}
	}
	for(int cache_loc = 0; cache_loc < LOC_NUM; cache_loc++){
		int curr_block_num = Block_num;
		if(deidxize(cache_loc)!= -1) curr_block_num = GPU_Block_num;
#ifdef BUFFER_REUSE_ENABLE
		if(Global_Buffer_1D[cache_loc] == NULL) Global_Buffer_1D[cache_loc] = new Buffer(deidxize(cache_loc), curr_block_num, Block_sz);
		else if (Global_Buffer_1D[cache_loc]->BlockSize != Block_sz || Global_Buffer_1D[cache_loc]->BlockNum < curr_block_num){
#ifdef DEBUG
		lprintf(lvl, "CoCoPeLiaDot: Previous Cache smaller than requested:\
		Global_Buffer_1D[%d]->BlockSize=%lld vs Block_sz = %lld,\
		Global_Buffer_1D[%d]->BlockNum=%d vs Block_num = %d\n",
		cache_loc, Global_Buffer_1D[cache_loc]->BlockSize, Block_sz,
		cache_loc, Global_Buffer_1D[cache_loc]->BlockNum, curr_block_num);
#endif
			delete Global_Buffer_1D[cache_loc];
			Global_Buffer_1D[cache_loc] = new Buffer(deidxize(cache_loc), curr_block_num, Block_sz);
		}
		else{
			;
		}
#else
			if(Global_Buffer_1D[cache_loc]!= NULL) error("CoCoPeLiaDot: Global_Buffer_1D[%d] was not NULL with reuse disabled\n", cache_loc);
			Global_Buffer_1D[cache_loc] = new Buffer(deidxize(cache_loc), curr_block_num, Block_sz);
#endif
	}

	/// TODO: Split each asset to Tiles
	x_asset->InitTileMap(T, Global_Buffer_1D);
	y_asset->InitTileMap(T, Global_Buffer_1D);

#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Spliting assets to tiles -> t_tile_init = %lf ms\n", cpu_timer*1000);
	cpu_timer = csecond();
#endif

	Subkernel_list_dot = CoCoAsignTilesToSubkernelsDdot(x_asset, y_asset, T,
		&Subkernel_num_dot);

	if(!reuse_model_flag) autotune_controller_dot->update_sk_num(Subkernel_num_dot);

#ifdef DEBUG
	lprintf(lvl, "Subkernel_num_dot = %d {N}GridSz = {%d}, num_devices = %d\n\n",
		Subkernel_num_dot, NGridSz_dot, autotune_controller_dot->active_unit_num);
#endif
#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Subkernel init -> t_subkernel_init = %lf ms\n", cpu_timer*1000);
	cpu_timer = csecond();
#endif

	if(!reuse_model_flag)  autotune_controller_dot->distribute_subkernels(NGridSz_dot, 1, 1);

#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Subkernel Distribute -> t_subkernel_dist = %lf ms\n", cpu_timer*1000);
	cpu_timer = csecond();
#endif

	//s = pthread_attr_init(&attr);
	//if (s != 0) error("CoCopeLiaDot: pthread_attr_init failed s=%d\n", s);

	pthread_t thread_id[autotune_controller_dot->active_unit_num];
	kernel_pthread_wrap_p thread_dev_data[autotune_controller_dot->active_unit_num];

	// create a barrier object with a count of autotune_controller_dot->active_unit_num + 1
	pthread_barrier_init (&SoftCache_alloc_barrier_dot, NULL, autotune_controller_dot->active_unit_num + 1);
	for(int d=0; d < LOC_NUM; d++) CoCoEnableLinks(deidxize(d), LOC_NUM);
	for(int d=0; d < autotune_controller_dot->active_unit_num; d++){
		if(autotune_controller_dot->Subkernels_per_unit_num[d] == 0 )
			error("CoCoPeLiaDdot: Leftover autotune_controller_dot->Subkernels_per_unit_num[%d] == 0", d);

		thread_dev_data[d] = (kernel_pthread_wrap_p) malloc(sizeof(struct kernel_pthread_wrap));
		thread_dev_data[d]->dev_id = autotune_controller_dot->active_unit_id_list[d];

#ifndef RUNTIME_SCHEDULER_VERSION
		thread_dev_data[d]->SubkernelNumDev = autotune_controller_dot->Subkernels_per_unit_num[d];
#else
		thread_dev_data[d]->SubkernelNumDev = Subkernel_num_dot;
#endif
		thread_dev_data[d]->SubkernelListDev = (Subkernel**) malloc(thread_dev_data[d]->SubkernelNumDev*sizeof(Subkernel*));
#ifndef RUNTIME_SCHEDULER_VERSION
		for(int skitt = 0; skitt < autotune_controller_dot->Subkernels_per_unit_num[d]; skitt++)
			thread_dev_data[d]->SubkernelListDev[skitt] = Subkernel_list_dot[autotune_controller_dot->Subkernels_per_unit_list[d][skitt]];
#endif
		s = pthread_create(&thread_id[d], &attr,
																	&CoCopeLiaDotAgentVoid, thread_dev_data[d]);

	}
	pthread_barrier_wait (&SoftCache_alloc_barrier_dot);

	//x_asset->DrawTileMap();
	//y_asset->DrawTileMap();

	for(int d=0; d<autotune_controller_dot->active_unit_num;d++){
		s = pthread_join(thread_id[d], &res);
		if (s != 0) error("CoCopeLiaDot: pthread_join failed with exit value %d", s);
		//free(res);      /* Free memory allocated by thread */
	}
#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Fire and gather pthreads for all devices -> t_exec_full = %lf ms\n", cpu_timer*1000);
	lprintf(lvl, "t_predicted for T=%zu was %.2lf ms : %lf percentile error\n", autotune_controller_dot->T, autotune_controller_dot->pred_t*1000,
	(autotune_controller_dot->pred_t==0)? 0.0: (autotune_controller_dot->pred_t - cpu_timer )/autotune_controller_dot->pred_t*100);
	cpu_timer = csecond();
#endif
#ifdef STEST
	STEST_print_SK(thread_dev_data, dot_entry_ts, autotune_controller_dot->active_unit_num);
#endif

#ifdef DDEBUG
  x_asset->DrawTileMap();
  y_asset->DrawTileMap();
	for(int i=0; i<LOC_NUM;i++) Global_Buffer_1D[i]->draw_buffer(true,true,true);
#endif

#ifndef BUFFER_REUSE_ENABLE
	for(int i = 0 ; i < LOC_NUM; i++){
		delete Global_Buffer_1D[i];
		Global_Buffer_1D[i] = NULL;
	}
#else
	for(int i=0; i<LOC_NUM;i++) Global_Buffer_1D[i]->reset(false,true);
#endif

#ifndef BACKEND_RES_REUSE_ENABLE
	for(int i=0; i<autotune_controller_dot->active_unit_num;i++) CoCoPeLiaFreeResources(autotune_controller_dot->active_unit_id_list[i]);
#endif

#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Invalidate caches -> t_invalidate = %lf ms\n", cpu_timer*1000);
	cpu_timer = csecond();
#endif

	for(int i=0; i<Subkernel_num_dot; i++) *(initial_dot->result)+= *((dot_backend_in<double>*) Subkernel_list_dot[i]->operation_params)->result;
	for(int i=0; i<Subkernel_num_dot; i++) delete Subkernel_list_dot[i];
	//delete [] Subkernel_list_dot;

#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Freed Subkernels -> t_invalidate = %lf ms\n", cpu_timer*1000);
	cpu_timer = csecond();
#endif

	x_asset->DestroyTileMap();
	y_asset->DestroyTileMap();

#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Destroyed Tilemaps -> t_invalidate = %lf ms\n", cpu_timer*1000);
	cpu_timer = csecond();
#endif

	CoCoPeLiaSelectDevice(prev_dev_id);

  x_asset->resetProperties();
  y_asset->resetProperties();
	delete x_asset;
	delete y_asset;

#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Unregistering assets -> t_unpin = %lf ms\n", cpu_timer*1000);
#endif

#ifdef DEBUG
	lprintf(lvl-1, "<-----|\n");
#endif
#ifdef TEST
	lprintf(lvl-1, "<-----|\n");
#endif
	return autotune_controller_dot;
}

/// A modification of PARALiADdot but with given parameters (mainly for performance/debug purposes)
ATC_p PARALiADdotControled(long int N, double* x, long int incx, double* y, long int incy, double* result, ATC_p predef_controller){
	if (predef_controller == NULL){
		warning("Calling PARALiADdotControled with empty controller -> falling back to full autotune version \'PARALiADdot\'\n");
		return PARALiADdot(N, x, incx, y, incy, result);
	}

	predef_controller_dot = predef_controller;
	return PARALiADdot(N, x, incx, y, incy, result);
}
