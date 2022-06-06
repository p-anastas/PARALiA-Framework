///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The axpy CoCopeLia implementation using the new mission-agent-asset C++ classes.
///

#include "backend_wrappers.hpp"
#include "CoCoPeLiaModel.hpp"
#include "CoCoPeLia.hpp"
#include "unihelpers.hpp"
#include "Asset.hpp"
#include "Subkernel.hpp"
#include "DataCaching.hpp"

#include <pthread.h>
pthread_barrier_t  SoftCache_alloc_barrier_axpy;

axpy_backend_in_p initial_daxpy = NULL;

CoCoModel_p glob_model_axpy[128] = {NULL};
//Cache_p Global_Cache[LOC_NUM] = {NULL};
CoControl_p predef_vals_axpy = NULL;
CoControl_p autotuned_vals_axpy = NULL;
int NGridSz_axpy = 0;

#ifdef STEST
double axpy_entry_ts;
#endif

Subkernel** Subkernel_list_axpy;
int Subkernel_num_axpy;

int Sk_select_lock = 0;

Subkernel** CoCoAsignTilesToSubkernelsDaxpy(Asset1D<VALUE_TYPE>* x_asset, Asset1D<VALUE_TYPE>* y_asset,
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
			kernels[current_ctr] = new Subkernel(2,"axpy");
			kernels[current_ctr]->iloc1 = ni;
			kernels[current_ctr]->TileDimlist[0] = kernels[current_ctr]->TileDimlist[1] = 1;
			kernels[current_ctr]->TileList[0] = x_asset->getTile(ni);
			kernels[current_ctr]->TileList[1] = y_asset->getTile(ni);
			((Tile1D<VALUE_TYPE>*)kernels[current_ctr]->TileList[0])->R_flag = 1;
			((Tile1D<VALUE_TYPE>*)kernels[current_ctr]->TileList[1])->R_flag = 1;
			((Tile1D<VALUE_TYPE>*)kernels[current_ctr]->TileList[1])->W_flag = 1;
			((Tile1D<VALUE_TYPE>*)kernels[current_ctr]->TileList[1])->W_total = 1;
			kernels[current_ctr]->operation_params = (void*) malloc(sizeof(struct axpy_backend_in));
			axpy_backend_in_p ptr_ker_translate = (axpy_backend_in_p) kernels[current_ctr]->operation_params;
			ptr_ker_translate->N = ((Tile1D<VALUE_TYPE>*) kernels[current_ctr]->TileList[0])->dim;
			ptr_ker_translate->x = NULL;
			ptr_ker_translate->y = NULL;
			ptr_ker_translate->alpha = initial_daxpy->alpha;
			ptr_ker_translate->incx = initial_daxpy->incx;
			ptr_ker_translate->incy = initial_daxpy->incy;
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
	axpy_backend_in_p ptr_ker_translate = (axpy_backend_in_p) ker->operation_params;
	ker->run_dev_id = ptr_ker_translate->dev_id = dev_id;
	short dev_id_idx = (dev_id == -1) ? LOC_NUM - 1: dev_id;
	ptr_ker_translate->incx = ((Tile1D<VALUE_TYPE>*) ker->TileList[0])->inc[dev_id_idx];
	ptr_ker_translate->incy = ((Tile1D<VALUE_TYPE>*) ker->TileList[1])->inc[dev_id_idx];
}

void CoCoDaxpyUpdatePointers(Subkernel* ker){
	axpy_backend_in_p ptr_ker_translate = (axpy_backend_in_p) ker->operation_params;
	short dev_id_idx = idxize(ker->run_dev_id);
	ptr_ker_translate->x = &((Tile1D<VALUE_TYPE>*) ker->TileList[0])->StoreBlock[dev_id_idx]->Adrs;
	ptr_ker_translate->y = &((Tile1D<VALUE_TYPE>*) ker->TileList[1])->StoreBlock[dev_id_idx]->Adrs;
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

	Global_Cache[idxize(dev_id)]->allocate(true);
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
	int remaining_Subkernels_dev = axpy_subkernel_data->SubkernelNumDev;
#ifndef RUNTIME_SCHEDULER_VERSION
	/// Rename global vars, perfectly safe.
	Subkernel** Subkernel_list_axpy = axpy_subkernel_data->SubkernelListDev;
	int Subkernel_num_axpy = axpy_subkernel_data->SubkernelNumDev;
#endif
	while (remaining_Subkernels_dev){
		prev = curr;
		if(prev) prev->sync_request_data();
		while(__sync_lock_test_and_set(&Sk_select_lock, 1));
		if (!strcmp(SELECT_HEURISTIC, "NAIVE"))
			curr = SubkernelSelectSimple(dev_id, Subkernel_list_axpy, Subkernel_num_axpy);
		else if (!strcmp(SELECT_HEURISTIC, "NAIVE-NO-WRITE-SHARE"))
			curr = SubkernelSelectNoWriteShare(dev_id, Subkernel_list_axpy, Subkernel_num_axpy);
		else if (!strcmp(SELECT_HEURISTIC, "MINIMIZE-FETCH"))
			curr = SubkernelSelectMinimizeFetch(dev_id, Subkernel_list_axpy, Subkernel_num_axpy);
		else if (!strcmp(SELECT_HEURISTIC, "MINIMIZE-FETCH-WRITE-PENALTY"))
			curr = SubkernelSelectMinimizeFetchWritePenalty(dev_id, Subkernel_list_axpy, Subkernel_num_axpy);
		else if (!strcmp(SELECT_HEURISTIC, "MINIMIZE-FETCH-WRITE-PENALTY-MULTIFETCH-PENALTY"))
			curr = SubkernelSelectMinimizeFetchWritePenaltyMultiFetchPenalty(dev_id, Subkernel_list_axpy, Subkernel_num_axpy);
		else if (!strcmp(SELECT_HEURISTIC, "MINIMIZE-FETCH-WRITE-PENALTY-MULTIFETCH-PENALTY-MULTIDEV-FAIR"))
			curr = SubkernelSelectMinimizeFetchWritePenaltyMultiFetchPenaltyMutlidevFair(dev_id, Subkernel_list_axpy, Subkernel_num_axpy);
		else error("CoCoPeLiaAxpy: Unknown Subkernel Heuristic %s\n", SELECT_HEURISTIC);
		if (!curr){
	//#ifdef DDEBUG
	//		lprintf(lvl, "CoCoPeLiaAxpyAgentVoid(%d): Got curr = NULL, repeating search\n", dev_id);
	//#endif
			__sync_lock_release(&Sk_select_lock);
			continue;
		}
		remaining_Subkernels_dev--;
#ifdef RUNTIME_SCHEDULER_VERSION
		axpy_subkernel_data->SubkernelListDev[axpy_subkernel_data->SubkernelNumDev - (remaining_Subkernels_dev + 1)] = curr;
#else
		for (int keri = 0; keri < axpy_subkernel_data->SubkernelNumDev; keri++)
			if (curr == axpy_subkernel_data->SubkernelListDev[keri]){
				axpy_subkernel_data->SubkernelListDev[keri] = axpy_subkernel_data->SubkernelListDev[remaining_Subkernels_dev];
				axpy_subkernel_data->SubkernelListDev[remaining_Subkernels_dev] = curr;
				break;
			}
		Subkernel_num_axpy = remaining_Subkernels_dev;
#endif
		curr->prev = prev;
		if(prev) prev->next = curr;
		CoCoDaxpyUpdateDevice(curr, dev_id);
		curr->init_events();
		curr->request_data();
		CoCoDaxpyUpdatePointers(curr);
		__sync_lock_release(&Sk_select_lock);
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
	double total_cache_timer = Global_Cache[idxize(dev_id)]->timer;
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
CoControl_p CoCopeLiaDaxpy(size_t N, VALUE_TYPE alpha, VALUE_TYPE* x, size_t incx, VALUE_TYPE* y, size_t incy)
{
	short lvl = 1;
#ifdef DEBUG
	lprintf(lvl-1, "|-----> CoCopeLiaDaxpy(%zu,%lf,x=%p(%d),%zu,y=%p(%d),%zu)\n",
		N, alpha, x, CoCoGetPtrLoc(x), incx, y, CoCoGetPtrLoc(y), incy);
#endif

#ifdef TEST
	lprintf(lvl-1, "|-----> CoCopeLiaDaxpy\n");
	double cpu_timer = csecond();
#endif
#ifdef STEST
	axpy_entry_ts = csecond();
#endif
	int prev_dev_id = CoCoPeLiaGetDevice();

	short reuse_model_flag = 1;
	if(!initial_daxpy){
		initial_daxpy = (axpy_backend_in_p) malloc(sizeof(struct axpy_backend_in));
		reuse_model_flag = 0;
	}

	if(reuse_model_flag && initial_daxpy->N != N)
		reuse_model_flag = 0;
	initial_daxpy->N = N;

	if(reuse_model_flag && initial_daxpy->x!= NULL && *initial_daxpy->x != x)
		reuse_model_flag = 0;
	initial_daxpy->x = (void**) &x;

	if(reuse_model_flag && initial_daxpy->y!= NULL && *initial_daxpy->y != y)
		reuse_model_flag = 0;
	initial_daxpy->y = (void**) &y;


	initial_daxpy->alpha = alpha;
	initial_daxpy->incx = incx;
	initial_daxpy->incy = incy;
	initial_daxpy->dev_id = -1;

	Asset1D<VALUE_TYPE>* x_asset, *y_asset;
	/// Prepare Assets in parallel( e.g. initialize asset classes, pin memory with pthreads)
	/// return: x_asset, y_asset initialized and pinned
	x_asset = new Asset1D<VALUE_TYPE>( x, N, incx);
	y_asset = new Asset1D<VALUE_TYPE>( y, N, incy);

	pthread_attr_t attr;
	int s = pthread_attr_init(&attr);
	if (s != 0) error("CoCopeLiaDaxpy: pthread_attr_init failed s=%d\n", s);

	pthread_t asset_thread_id[2];
	x_asset->prepareAsync(&asset_thread_id[0], attr);
	y_asset->prepareAsync(&asset_thread_id[1], attr);

	tunableParams_p best_pred_p = CoCoAutotuneParameters("Daxpy", initial_daxpy,
  &autotuned_vals_axpy, glob_model_axpy, predef_vals_axpy, reuse_model_flag);

	void* res;
	for(int i=0; i<2;i++){
		s = pthread_join(asset_thread_id[i], &res);
		if (s != 0) error("CoCopeLiaDaxpy: pthread_join failed with exit value %d", s);
		//free(res);      /* Free memory allocated by thread */
	}

#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Preparing assets (parallel with pthreads) -> t_prep = %lf ms\n", cpu_timer*1000);
	cpu_timer = csecond();
#endif

	CoCoUpdateLinkSpeed1D(autotuned_vals_axpy, glob_model_axpy);

#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Initializing link values -> t_link_init = %lf ms\n", cpu_timer*1000);
	cpu_timer = csecond();
#endif

	size_t T = autotuned_vals_axpy->T;

	int GPU_Block_num, Block_num = 1 + (x_asset->dim/T + ((x_asset->dim%T)? 1 : 0)) +
		 (y_asset->dim/T + ((y_asset->dim%T)? 1 : 0));
	long long Block_sz = 	T*sizeof(VALUE_TYPE);
	GPU_Block_num = Block_num;
	if(autotuned_vals_axpy->cache_limit > 0){
		int max_block_num = autotuned_vals_axpy->cache_limit/Block_sz;
		if(max_block_num < Block_num){
			lprintf(0, "CoCopeLiaAxpy: Input cache limit %lld forces problem to use %d blocks\
			instead of %d needed for the full problem\n", autotuned_vals_axpy->cache_limit, max_block_num, Block_num);
			// With Parallel backends unfortunately == SK_blocks + Max_Exclusive_Blocks - 1
			int worst_case_ex_blocks = 1 + (y_asset->dim/T + ((y_asset->dim%T)? 1 : 0));
			if(max_block_num < worst_case_ex_blocks)
				error("CoCopeLiaAxpy: Not able to run with < %d blocks per cache due to EX scheduling\n", worst_case_ex_blocks);
				GPU_Block_num = max_block_num;
		}
	}
	for(int cache_loc = 0; cache_loc < LOC_NUM; cache_loc++){
		int curr_block_num = Block_num;
		if(deidxize(cache_loc)!= -1) curr_block_num = GPU_Block_num;
#ifdef BUFFER_REUSE_ENABLE
		if(Global_Cache[cache_loc] == NULL) Global_Cache[cache_loc] = new Cache(deidxize(cache_loc), curr_block_num, Block_sz);
		else if (Global_Cache[cache_loc]->BlockSize != Block_sz || Global_Cache[cache_loc]->BlockNum < curr_block_num){
#ifdef DEBUG
		lprintf(lvl, "CoCoPeLiaAxpy: Previous Cache smaller than requested:\
		Global_Cache[%d]->BlockSize=%lld vs Block_sz = %lld,\
		Global_Cache[%d]->BlockNum=%d vs Block_num = %d\n",
		cache_loc, Global_Cache[cache_loc]->BlockSize, Block_sz,
		cache_loc, Global_Cache[cache_loc]->BlockNum, curr_block_num);
#endif
			delete Global_Cache[cache_loc];
			Global_Cache[cache_loc] = new Cache(deidxize(cache_loc), curr_block_num, Block_sz);
		}
		else{
			;
		}
#else
			if(Global_Cache[cache_loc]!= NULL) error("CoCoPeLiaAxpy: Global_Cache[%d] was not NULL with reuse disabled\n", cache_loc);
			Global_Cache[cache_loc] = new Cache(deidxize(cache_loc), curr_block_num, Block_sz);
#endif
	}

	/// TODO: Split each asset to Tiles
	x_asset->InitTileMap(T, Global_Cache);
	y_asset->InitTileMap(T, Global_Cache);

#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Spliting assets to tiles -> t_tile_init = %lf ms\n", cpu_timer*1000);
	cpu_timer = csecond();
#endif

	Subkernel_list_axpy = CoCoAsignTilesToSubkernelsDaxpy(x_asset, y_asset, T,
		&Subkernel_num_axpy);
#ifdef DEBUG
	lprintf(lvl, "Subkernel_num_axpy = %d {N}GridSz = {%d}, num_devices = %d\n\n",
		Subkernel_num_axpy, NGridSz_axpy, autotuned_vals_axpy->dev_num);
#endif
#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Subkernel init -> t_subkernel_init = %lf ms\n", cpu_timer*1000);
	cpu_timer = csecond();
#endif
	autotuned_vals_axpy->Subkernel_dev_id_list = (int**) malloc(autotuned_vals_axpy->dev_num*sizeof(int*));
	for (int devidx = 0; devidx < autotuned_vals_axpy->dev_num; devidx++)
		autotuned_vals_axpy->Subkernel_dev_id_list[devidx] = (int*) malloc(Subkernel_num_axpy*sizeof(int));
	if (!strcmp(DISTRIBUTION, "ROUND-ROBIN"))
		CoCoDistributeSubkernelsRoundRobin(autotuned_vals_axpy, best_pred_p, Subkernel_num_axpy);
	else if (!strcmp(DISTRIBUTION, "SPLIT-NAIVE"))
		CoCoDistributeSubkernelsNaive(autotuned_vals_axpy, best_pred_p, Subkernel_num_axpy);
	else if (!strcmp(DISTRIBUTION, "SPLIT-CHUNKS-ROBIN"))
		CoCoDistributeSubkernelsRoundRobinChunk(autotuned_vals_axpy, best_pred_p, Subkernel_num_axpy, 0);
	else if (!strcmp(DISTRIBUTION, "SPLIT-CHUNKS-ROBIN-REVERSE"))
		CoCoDistributeSubkernelsRoundRobinChunkReverse(autotuned_vals_axpy, best_pred_p, Subkernel_num_axpy, 0);
	else if (!strcmp(DISTRIBUTION, "2D-BLOCK-CYCLIC"))
		CoCoDistributeSubkernels2DBlockCyclic(autotuned_vals_axpy, best_pred_p, NGridSz_axpy, 0, 0);
	else error("CoCoPeLiaAxpy: Unknown Subkernel Distribution %s\n", DISTRIBUTION);

#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Subkernel Distribute -> t_subkernel_dist = %lf ms\n", cpu_timer*1000);
	cpu_timer = csecond();
#endif

	int used_devices = 0;
	for (int d = 0 ; d < autotuned_vals_axpy->dev_num; d++)
		if(autotuned_vals_axpy->Subkernels_per_dev[d] > 0 ) used_devices++;
		else if(autotuned_vals_axpy->Subkernels_per_dev[d] < 0 )
			error("CoCoPeLiaAxpy: autotuned_vals_axpy->Subkernels_per_dev[%d] = %d\n",
				d, autotuned_vals_axpy->Subkernels_per_dev[d]);
		else{
			free(autotuned_vals_axpy->Subkernel_dev_id_list[d]);
			for (int d_move = d; d_move < autotuned_vals_axpy->dev_num - 1; d_move++){
				autotuned_vals_axpy->Subkernels_per_dev[d_move] = autotuned_vals_axpy->Subkernels_per_dev[d_move+1];
				autotuned_vals_axpy->Subkernel_dev_id_list[d_move] = autotuned_vals_axpy->Subkernel_dev_id_list[d_move+1];
				autotuned_vals_axpy->dev_ids[d_move] = autotuned_vals_axpy->dev_ids[d_move+1];
			}
		}
	//#ifdef DEBUG
	if(!reuse_model_flag){
		lprintf(0, "used_devices=%d out of selected autotuned_vals_axpy->dev_num=%d\n", used_devices, autotuned_vals_axpy->dev_num);
		lprintf(0, "====================================\n");
	}
	//#endif
	autotuned_vals_axpy->dev_num = used_devices;

#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Subkernel Correct Split devices -> t_subkernel_split = %lf ms\n", cpu_timer*1000);
	cpu_timer = csecond();
#endif

	//s = pthread_attr_init(&attr);
	//if (s != 0) error("CoCopeLiaAxpy: pthread_attr_init failed s=%d\n", s);

	pthread_t thread_id[used_devices];
	kernel_pthread_wrap_p thread_dev_data[used_devices];

	// create a barrier object with a count of autotuned_vals_axpy->dev_num + 1
	pthread_barrier_init (&SoftCache_alloc_barrier_axpy, NULL, autotuned_vals_axpy->dev_num + 1);

	for(int d=0; d < autotuned_vals_axpy->dev_num; d++){
		if(best_pred_p->rel_dev_score[d] == 0.0)
			error("CoCopeLiaAxpy: best_pred_p->rel_dev_score[%d] == 0 in final used best_pred_p\n",d);

		// Check/Enable peer access between participating GPUs
		CoCoEnableLinks(d, autotuned_vals_axpy->dev_ids, autotuned_vals_axpy->dev_num);

		thread_dev_data[d] = (kernel_pthread_wrap_p) malloc(sizeof(struct kernel_pthread_wrap));
		thread_dev_data[d]->dev_id = autotuned_vals_axpy->dev_ids[d];

		thread_dev_data[d]->SubkernelNumDev = autotuned_vals_axpy->Subkernels_per_dev[d];
		thread_dev_data[d]->SubkernelListDev = (Subkernel**) malloc(thread_dev_data[d]->SubkernelNumDev*sizeof(Subkernel*));
	#ifndef RUNTIME_SCHEDULER_VERSION
		for(int skitt = 0; skitt < autotuned_vals_axpy->Subkernels_per_dev[d]; skitt++)
			thread_dev_data[d]->SubkernelListDev[skitt] = Subkernel_list_axpy[autotuned_vals_axpy->Subkernel_dev_id_list[d][skitt]];
	#endif
		s = pthread_create(&thread_id[d], &attr,
																	&CoCopeLiaAxpyAgentVoid, thread_dev_data[d]);

	}
	pthread_barrier_wait (&SoftCache_alloc_barrier_axpy);

	//x_asset->DrawTileMap();
	//y_asset->DrawTileMap();

	for(int d=0; d<autotuned_vals_axpy->dev_num;d++){
		s = pthread_join(thread_id[d], &res);
		if (s != 0) error("CoCopeLiaAxpy: pthread_join failed with exit value %d", s);
		//free(res);      /* Free memory allocated by thread */
	}
#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Fire and gather pthreads for all devices -> t_exec_full = %lf ms\n", cpu_timer*1000);
	lprintf(lvl, "t_predicted for T=%zu was %.2lf ms : %lf percentile error\n", T, best_pred_p->pred_t*1000,
	(best_pred_p->pred_t==0)? 0.0: (best_pred_p->pred_t - cpu_timer )/best_pred_p->pred_t*100);
	cpu_timer = csecond();
#endif
#ifdef STEST
	STEST_print_SK(thread_dev_data, axpy_entry_ts, autotuned_vals_axpy->dev_num);
#endif

#ifdef DDEBUG
  x_asset->DrawTileMap();
  y_asset->DrawTileMap();
	for(int i=0; i<LOC_NUM;i++) Global_Cache[i]->draw_cache(true,true,true);
#endif

#ifndef BUFFER_REUSE_ENABLE
	for(int i = 0 ; i < LOC_NUM; i++){
		delete Global_Cache[i];
		Global_Cache[i] = NULL;
	}
#else
	for(int i=0; i<LOC_NUM;i++) Global_Cache[i]->reset(false,true);
#endif

#ifndef BACKEND_RES_REUSE_ENABLE
	for(int i=0; i<autotuned_vals_axpy->dev_num;i++) CoCoPeLiaFreeResources(autotuned_vals_axpy->dev_ids[i]);
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
	return autotuned_vals_axpy;
}

/// A modification of CoCopeLiaDaxpy but with given parameters (mainly for performance/debug purposes)
CoControl_p CoCopeLiaDaxpyControled(size_t N, VALUE_TYPE alpha, VALUE_TYPE* x, size_t incx, VALUE_TYPE* y, size_t incy, CoControl_p predef_control_values){
	if (!predef_control_values) return CoCopeLiaDaxpy(N, alpha, x, incx, y, incy);
	if (predef_vals_axpy == NULL) predef_vals_axpy = (CoControl_p) malloc(sizeof(struct CoControl));
	predef_vals_axpy->T = predef_control_values->T;
	predef_vals_axpy->dev_num = predef_control_values->dev_num;
	for(int idx =0; idx < LOC_NUM; idx++)
		predef_vals_axpy->dev_ids[idx] = predef_control_values->dev_ids[idx];
	predef_vals_axpy->cache_limit = predef_control_values->cache_limit;
	CoControl_p return_vals = CoCopeLiaDaxpy(N, alpha, x, incx, y, incy);
	free(predef_vals_axpy);
	predef_vals_axpy = NULL;
	return return_vals;
}
