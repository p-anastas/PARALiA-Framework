///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The axpy CoCopeLia implementation using the new mission-agent-asset C++ classes.
///

#include "backend_wrappers.hpp"
#include "Autotuner.hpp"
#include "PARALiA.hpp"
#include "unihelpers.hpp"
#include "Decomposer.hpp"
#include "Subkernel.hpp"
#include "DataCaching.hpp"

#include <pthread.h>

pthread_barrier_t  SoftCache_alloc_barrier_axpy;

axpy_backend_in<double>* initial_axpy = NULL;
ATC_p autotune_controller_axpy = NULL;
ATC_p predef_controller_axpy = NULL;

int NGridSz_axpy = 0;

#ifdef STEST
double axpy_entry_ts;
#endif

Subkernel** Subkernel_list_axpy;
int Subkernel_num_axpy;
int remaining_Subkernels_axpy;

int Sk_select_lock_axpy = 0;

Subkernel** CoCoAsignTilesToSubkernelsDaxpy(Decom1D* x_asset, Decom1D* y_asset,
	int T, int* kernelNum){

	short lvl = 2;

	NGridSz_axpy = x_asset->GridSz;
	*kernelNum = NGridSz_axpy;
#ifdef DEBUG
	lprintf(lvl-1, "|-----> CoCoAsignTilesToSubkernelsDaxpy(x_asset,y_asset,%d,%d)\n", T, *kernelNum);
	lprintf(lvl,"NGridSz_axpy = %d\n", NGridSz_axpy);
	lprintf(lvl,"Nlast = %d\n",
	x_asset->Tile_map[NGridSz_axpy-1]->dim);
#endif

Subkernel** kernels = (Subkernel**) malloc(*kernelNum*sizeof(Subkernel*));
int current_ctr = 0;
		for (int ni = 0; ni < NGridSz_axpy; ni++){
      current_ctr = ni;
			kernels[current_ctr] = new Subkernel(2,"Daxpy");
			kernels[current_ctr]->iloc1 = ni;
			kernels[current_ctr]->TileDimlist[0] = kernels[current_ctr]->TileDimlist[1] = 1;
			kernels[current_ctr]->TileList[0] = x_asset->getTile(ni);
			kernels[current_ctr]->TileList[1] = y_asset->getTile(ni);
			((Tile1D*)kernels[current_ctr]->TileList[0])->R_flag = 1;
			((Tile1D*)kernels[current_ctr]->TileList[1])->R_flag = 1;
			((Tile1D*)kernels[current_ctr]->TileList[1])->W_flag = 1;
			((Tile1D*)kernels[current_ctr]->TileList[1])->W_total = 1;
			kernels[current_ctr]->operation_params = (void*) malloc(sizeof( axpy_backend_in<double>));
			axpy_backend_in<double>* ptr_ker_translate = (axpy_backend_in<double>*) kernels[current_ctr]->operation_params;
			ptr_ker_translate->N = ((Tile1D*) kernels[current_ctr]->TileList[0])->dim;
			ptr_ker_translate->x = NULL;
			ptr_ker_translate->y = NULL;
			ptr_ker_translate->alpha = initial_axpy->alpha;
			ptr_ker_translate->incx = initial_axpy->incx;
			ptr_ker_translate->incy = initial_axpy->incy;
			// No interal dims for axpy to reduce
			kernels[current_ctr]->WR_first = 0;
			kernels[current_ctr]->WR_last = (short*) calloc (2, sizeof(short));
		}
#ifdef DEBUG
	lprintf(lvl-1, "<-----|\n");
#endif
	return kernels;
}

void CoCoDaxpyUpdateDevice(Subkernel* ker, short dev_id){
	axpy_backend_in<double>* ptr_ker_translate = (axpy_backend_in<double>*) ker->operation_params;
	ker->run_dev_id = ptr_ker_translate->dev_id = dev_id;
	short dev_id_idx = (dev_id == -1) ? LOC_NUM - 1: dev_id;
	ptr_ker_translate->incx = ((Tile1D*) ker->TileList[0])->inc[dev_id_idx];
	ptr_ker_translate->incy = ((Tile1D*) ker->TileList[1])->inc[dev_id_idx];
}

void CoCoDaxpyUpdatePointers(Subkernel* ker){
	axpy_backend_in<double>* ptr_ker_translate = (axpy_backend_in<double>*) ker->operation_params;
	short dev_id_idx = idxize(ker->run_dev_id);
	ptr_ker_translate->x = &((Tile1D*) ker->TileList[0])->StoreBlock[dev_id_idx]->Adrs;
	ptr_ker_translate->y = &((Tile1D*) ker->TileList[1])->StoreBlock[dev_id_idx]->Adrs;
}

void* CoCopeLiaAxpyAgentVoid(void* kernel_pthread_wrapped){
	short lvl = 2;

	kernel_pthread_wrap_p axpy_subkernel_data = (kernel_pthread_wrap_p)kernel_pthread_wrapped;
	short dev_id = axpy_subkernel_data->dev_id;
#ifdef DEBUG
	lprintf(lvl-1, "|-----> CoCopeLiaAxpyAgentVoid(axpy_subkernel_data: dev_id = %d, SubkernelNumDev = %d)\n",
		dev_id, axpy_subkernel_data->SubkernelNumDev);
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
	pthread_barrier_wait (&SoftCache_alloc_barrier_axpy);

#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Wait barrier(%d): t_wb = %lf ms\n", dev_id, cpu_timer*1000);
	cpu_timer = csecond();
#endif

	Subkernel * curr = NULL, *prev = NULL;
#ifndef RUNTIME_SCHEDULER_VERSION
	/// Rename global vars, perfectly safe.
	Subkernel** Subkernel_list_axpy = axpy_subkernel_data->SubkernelListDev;
	int Subkernel_num_axpy = axpy_subkernel_data->SubkernelNumDev;
	int remaining_Subkernels_axpy = Subkernel_num_axpy;
#else
	axpy_subkernel_data->SubkernelNumDev = 0 ;
#endif
	while (remaining_Subkernels_axpy){
		prev = curr;
		if(prev) prev->sync_request_data();
		while(__sync_lock_test_and_set(&Sk_select_lock_axpy, 1));
		curr = SubkernelSelect(dev_id, Subkernel_list_axpy, Subkernel_num_axpy);
		if (!curr){
	//#ifdef DDEBUG
	//		lprintf(lvl, "CoCoPeLiaAxpyAgentVoid(%d): Got curr = NULL, repeating search\n", dev_id);
	//#endif
			__sync_lock_release(&Sk_select_lock_axpy);
			continue;
		}
		remaining_Subkernels_axpy--;
#ifdef RUNTIME_SCHEDULER_VERSION
		axpy_subkernel_data->SubkernelListDev[axpy_subkernel_data->SubkernelNumDev] = curr;
		axpy_subkernel_data->SubkernelNumDev++;
#else
		for (int keri = 0; keri < axpy_subkernel_data->SubkernelNumDev; keri++)
			if (curr == axpy_subkernel_data->SubkernelListDev[keri]){
				axpy_subkernel_data->SubkernelListDev[keri] = axpy_subkernel_data->SubkernelListDev[remaining_Subkernels_axpy];
				axpy_subkernel_data->SubkernelListDev[remaining_Subkernels_axpy] = curr;
				break;
			}
		Subkernel_num_axpy = remaining_Subkernels_axpy;
#endif
		curr->prev = prev;
		if(prev) prev->next = curr;
		CoCoDaxpyUpdateDevice(curr, dev_id);
		curr->init_events();
		curr->request_data();
		CoCoDaxpyUpdatePointers(curr);
		__sync_lock_release(&Sk_select_lock_axpy);
		curr->run_operation();
#ifdef ENABLE_SEND_RECV_OVERLAP
		curr->writeback_data();
#endif
	}

#ifndef ENABLE_SEND_RECV_OVERLAP
	for (int keri = 0; keri < axpy_subkernel_data->SubkernelNumDev; keri++)
		axpy_subkernel_data->SubkernelListDev[keri]->writeback_data();
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

/// An axpy wrapper including auto-tuning of T and cache_size, as well as device management
ATC_p PARALiADaxpy(long int N, double alpha, double* x, long int incx, double* y, long int incy)
{
	short lvl = 1;
#ifdef DEBUG
	lprintf(lvl-1, "|-----> PARALiADaxpy(%zu,%lf,x=%p(%d),%zu,y=%p(%d),%zu)\n",
		N, alpha, x, CoCoGetPtrLoc(x), incx, y, CoCoGetPtrLoc(y), incy);
#endif

#ifdef TEST
	lprintf(lvl-1, "|-----> PARALiADaxpy\n");
	double cpu_timer = csecond();
#endif
#ifdef STEST
	axpy_entry_ts = csecond();
#endif
	int prev_dev_id = CoCoPeLiaGetDevice();

	short reuse_model_flag = 1;
	if(!initial_axpy){
		initial_axpy = (axpy_backend_in<double>*) malloc(sizeof( axpy_backend_in<double>));
		reuse_model_flag = 0;
	}

	if(reuse_model_flag && initial_axpy->N != N)
		reuse_model_flag = 0;
	initial_axpy->N = N;

	if(reuse_model_flag && initial_axpy->x!= NULL && *initial_axpy->x != x)
		reuse_model_flag = 0;
	initial_axpy->x = (void**) &x;

	if(reuse_model_flag && initial_axpy->y!= NULL && *initial_axpy->y != y)
		reuse_model_flag = 0;
	initial_axpy->y = (void**) &y;


	initial_axpy->alpha = alpha;
	initial_axpy->incx = incx;
	initial_axpy->incy = incy;
	initial_axpy->dev_id = -1;

	Decom1D* x_asset, *y_asset;
	/// Prepare Assets in parallel( e.g. initialize asset classes, pin memory with pthreads)
	/// return: x_asset, y_asset initialized and pinned
	x_asset = new Decom1D( x, N, incx, DOUBLE);
	y_asset = new Decom1D( y, N, incy, DOUBLE);

	pthread_attr_t attr;
	int s = pthread_attr_init(&attr);
	if (s != 0) error("PARALiADaxpy: pthread_attr_init failed s=%d\n", s);

	pthread_t asset_thread_id[2];
	x_asset->prepareAsync(&asset_thread_id[0], attr);
	y_asset->prepareAsync(&asset_thread_id[1], attr);

	if (!reuse_model_flag){
		delete autotune_controller_axpy;
		autotune_controller_axpy = NULL;
	}
	if (autotune_controller_axpy == NULL) autotune_controller_axpy = new ATC();
	if (predef_controller_axpy && autotune_controller_axpy->diff_intialized_params_ATC(predef_controller_axpy)){
		 autotune_controller_axpy->mimic_ATC(predef_controller_axpy);
		 reuse_model_flag = 0;
	}
	double autotune_timer = 0;
	if(!reuse_model_flag) autotune_timer = autotune_controller_axpy->autotune_problem("Daxpy", initial_axpy);

	void* res;
	for(int i=0; i<2;i++){
		s = pthread_join(asset_thread_id[i], &res);
		if (s != 0) error("PARALiADaxpy: pthread_join failed with exit value %d", s);
		//free(res);      /* Free memory allocated by thread */
	}

#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Preparing assets (parallel with pthreads) -> t_prep = %lf ms\n", cpu_timer*1000);
	cpu_timer = csecond();
#endif

	long int T = autotune_controller_axpy->T;

	int GPU_Block_num, Block_num = 1 + (x_asset->dim/T + ((x_asset->dim%T)? 1 : 0)) +
		 (y_asset->dim/T + ((y_asset->dim%T)? 1 : 0));
	long long Block_sz = 	T*sizeof(double);
	GPU_Block_num = Block_num;
	if(autotune_controller_axpy->cache_limit > 0){
		int max_block_num = autotune_controller_axpy->cache_limit/Block_sz;
		if(max_block_num < Block_num){
			lprintf(0, "CoCopeLiaAxpy: Input cache limit %lld forces problem to use %d blocks\
			instead of %d needed for the full problem\n", autotune_controller_axpy->cache_limit, max_block_num, Block_num);
			// With Parallel backends unfortunately == SK_blocks + Max_Exclusive_Blocks - 1
			int worst_case_ex_blocks = 2;
			if(max_block_num < worst_case_ex_blocks)
				error("CoCopeLiaAxpy: Not able to run with < %d blocks per cache due to EX scheduling\n", worst_case_ex_blocks);
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
		lprintf(lvl, "CoCoPeLiaAxpy: Previous Cache smaller than requested:\
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
			if(Global_Buffer_1D[cache_loc]!= NULL) error("CoCoPeLiaAxpy: Global_Buffer_1D[%d] was not NULL with reuse disabled\n", cache_loc);
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

	Subkernel_list_axpy = CoCoAsignTilesToSubkernelsDaxpy(x_asset, y_asset, T,
		&Subkernel_num_axpy);

	if(!reuse_model_flag) autotune_controller_axpy->update_sk_num(Subkernel_num_axpy);

#ifdef DEBUG
	lprintf(lvl, "Subkernel_num_axpy = %d {N}GridSz = {%d}, num_devices = %d\n\n",
		Subkernel_num_axpy, NGridSz_axpy, autotune_controller_axpy->active_unit_num);
#endif
#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Subkernel init -> t_subkernel_init = %lf ms\n", cpu_timer*1000);
	cpu_timer = csecond();
#endif

	if(!reuse_model_flag)  autotune_controller_axpy->distribute_subkernels(NGridSz_axpy, 1, 1);

#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Subkernel Distribute -> t_subkernel_dist = %lf ms\n", cpu_timer*1000);
	cpu_timer = csecond();
#endif

	//s = pthread_attr_init(&attr);
	//if (s != 0) error("CoCopeLiaAxpy: pthread_attr_init failed s=%d\n", s);

	pthread_t thread_id[autotune_controller_axpy->active_unit_num];
	kernel_pthread_wrap_p thread_dev_data[autotune_controller_axpy->active_unit_num];

	// create a barrier object with a count of autotune_controller_axpy->active_unit_num + 1
	pthread_barrier_init (&SoftCache_alloc_barrier_axpy, NULL, autotune_controller_axpy->active_unit_num + 1);
	for(int d=0; d < LOC_NUM; d++) CoCoEnableLinks(deidxize(d), LOC_NUM);
	for(int d=0; d < autotune_controller_axpy->active_unit_num; d++){
		if(autotune_controller_axpy->Subkernels_per_unit_num[d] == 0 )
			error("CoCoPeLiaDaxpy: Leftover autotune_controller_axpy->Subkernels_per_unit_num[%d] == 0", d);

		thread_dev_data[d] = (kernel_pthread_wrap_p) malloc(sizeof(struct kernel_pthread_wrap));
		thread_dev_data[d]->dev_id = autotune_controller_axpy->active_unit_id_list[d];

#ifndef RUNTIME_SCHEDULER_VERSION
		thread_dev_data[d]->SubkernelNumDev = autotune_controller_axpy->Subkernels_per_unit_num[d];
#else
		thread_dev_data[d]->SubkernelNumDev = Subkernel_num_axpy;
#endif
		thread_dev_data[d]->SubkernelListDev = (Subkernel**) malloc(thread_dev_data[d]->SubkernelNumDev*sizeof(Subkernel*));
#ifndef RUNTIME_SCHEDULER_VERSION
		for(int skitt = 0; skitt < autotune_controller_axpy->Subkernels_per_unit_num[d]; skitt++)
			thread_dev_data[d]->SubkernelListDev[skitt] = Subkernel_list_axpy[autotune_controller_axpy->Subkernels_per_unit_list[d][skitt]];
#endif
		s = pthread_create(&thread_id[d], &attr,
																	&CoCopeLiaAxpyAgentVoid, thread_dev_data[d]);

	}
	pthread_barrier_wait (&SoftCache_alloc_barrier_axpy);

	//x_asset->DrawTileMap();
	//y_asset->DrawTileMap();

	for(int d=0; d<autotune_controller_axpy->active_unit_num;d++){
		s = pthread_join(thread_id[d], &res);
		if (s != 0) error("CoCopeLiaAxpy: pthread_join failed with exit value %d", s);
		//free(res);      /* Free memory allocated by thread */
	}
#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Fire and gather pthreads for all devices -> t_exec_full = %lf ms\n", cpu_timer*1000);
	lprintf(lvl, "t_predicted for T=%zu was %.2lf ms : %lf percentile error\n", autotune_controller_axpy->T, autotune_controller_axpy->pred_t*1000,
	(autotune_controller_axpy->pred_t==0)? 0.0: (autotune_controller_axpy->pred_t - cpu_timer )/autotune_controller_axpy->pred_t*100);
	cpu_timer = csecond();
#endif
#ifdef STEST
	STEST_print_SK(thread_dev_data, axpy_entry_ts, autotune_controller_axpy->active_unit_num);
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
	for(int i=0; i<autotune_controller_axpy->active_unit_num;i++) CoCoPeLiaFreeResources(autotune_controller_axpy->active_unit_id_list[i]);
#endif

#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Invalidate caches -> t_invalidate = %lf ms\n", cpu_timer*1000);
	cpu_timer = csecond();
#endif

	for(int i=0; i<Subkernel_num_axpy; i++) delete Subkernel_list_axpy[i];
	//delete [] Subkernel_list_axpy;

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
	return autotune_controller_axpy;
}

/// A modification of PARALiADaxpy but with given parameters (mainly for performance/debug purposes)
ATC_p PARALiADaxpyControled(long int N, double alpha, double* x, long int incx, double* y, long int incy, ATC_p predef_controller){
	if (predef_controller == NULL){
		warning("Calling PARALiADaxpyControled with empty controller -> falling back to full autotune version \'PARALiADaxpy\'\n");
		return PARALiADaxpy(N, alpha, x, incx, y, incy);
	}

	predef_controller_axpy = predef_controller;
	return PARALiADaxpy(N, alpha, x, incx, y, incy);
}
