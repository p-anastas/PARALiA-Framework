///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
/// \author Theodoridis Aristomenis (atheodor@cslab.ece.ntua.gr)
///
/// \brief The DGEMV CoCopeLia implementation.
///

#include "backend_wrappers.hpp"
#include "Autotuner.hpp"
#include "PARALiA.hpp"
#include "linkmap.hpp"
#include "Decomposer.hpp"
#include "Subkernel.hpp"
#include "DataCaching.hpp"

#include <pthread.h>

pthread_barrier_t  SoftCache_alloc_barrier_dgemv;

gemv_backend_in<double>* initial_dgemv = NULL;
ATC_p autotune_controller_dgemv = NULL;
ATC_p predef_controller_dgemv = NULL;

int MGridSz_dgemv = 0, NGridSz_dgemv = 0;

#ifdef STEST
double gemv_entry_ts;
#endif

Subkernel** Subkernel_list_dgemv;
int Subkernel_num_dgemv;
int remaining_Subkernels_dgemv;

//#define GEMV_FIRE_SK_DEV_ORDER
// #ifdef GEMV_FIRE_SK_DEV_ORDER
// int curr_sk_idx_dgemv = 0;
// int curr_sk_dgemv_unit_list[LOC_NUM], curr_sk_dgemv_unit_num;
// #endif

int Sk_select_lock_dgemv = 0;

Subkernel** CoCoAsignTilesToSubkernelsDgemv(Decom2D* A_asset, Decom1D* x_asset,
	Decom1D* y_asset, int T, int* kernelNum){

	short lvl = 2;
	/// Check Assets satisfy GEMV dim criteria for N, N transpose
	/// Harder for gemv (since TransA does not effect A dims, but x, y dims instead), skip, just do it right 
	MGridSz_dgemv = A_asset->GridSz1;
	NGridSz_dgemv = A_asset->GridSz2;
	*kernelNum = MGridSz_dgemv*NGridSz_dgemv;
#ifdef DEBUG
	lprintf(lvl-1, "|-----> CoCoAsignTilesToSubkernelsDgemv(A_asset,x_asset,y_asset,%d,%d)\n", T, *kernelNum);
	lprintf(lvl,"MgridSz = %d, NgridSz = %d\n", MGridSz_dgemv, NGridSz_dgemv);
	lprintf(lvl,"Mlast = %d, Nlast = %d\n",
	A_asset->Tile_map[A_asset->GridSz1*A_asset->GridSz2-1]->dim1,
	A_asset->Tile_map[A_asset->GridSz1*A_asset->GridSz2-1]->dim2);
#endif

Subkernel** kernels = (Subkernel**) malloc(*kernelNum*sizeof(Subkernel*));
int current_ctr = 0;
	for (int mi = 0; mi < MGridSz_dgemv; mi++){
		for (int ni = 0; ni < NGridSz_dgemv; ni++){
	      	current_ctr = mi*NGridSz_dgemv + ni;
			kernels[current_ctr] = new Subkernel(3,"Dgemv");
			kernels[current_ctr]->iloc1 = mi;
			kernels[current_ctr]->iloc2 = ni;
			kernels[current_ctr]->TileDimlist[0] = 2;
			kernels[current_ctr]->TileDimlist[1]  = kernels[current_ctr]->TileDimlist[2] = 1;
			kernels[current_ctr]->TileList[0] = A_asset->getTile(mi,ni);
			((Tile2D*)kernels[current_ctr]->TileList[0])->R_flag = 1;

			kernels[current_ctr]->operation_params = (void*) malloc(sizeof(gemv_backend_in<double>));
			gemv_backend_in<double>* ptr_ker_translate = (gemv_backend_in<double>*) kernels[current_ctr]->operation_params;
			ptr_ker_translate->TransA = initial_dgemv->TransA;

			if (ptr_ker_translate->TransA == 'N'){
				kernels[current_ctr]->TileList[1] = x_asset->getTile(ni);
				kernels[current_ctr]->TileList[2] = y_asset->getTile(mi);
				((Tile1D*)kernels[current_ctr]->TileList[1])->R_flag = MGridSz_dgemv;
				((Tile1D*)kernels[current_ctr]->TileList[2])->R_flag = NGridSz_dgemv;
				((Tile1D*)kernels[current_ctr]->TileList[2])->W_flag = NGridSz_dgemv;
				((Tile1D*)kernels[current_ctr]->TileList[2])->W_total = NGridSz_dgemv;
			}
			else{
				kernels[current_ctr]->TileList[1] = x_asset->getTile(mi);
				kernels[current_ctr]->TileList[2] = y_asset->getTile(ni);
				((Tile1D*)kernels[current_ctr]->TileList[1])->R_flag = NGridSz_dgemv;
				((Tile1D*)kernels[current_ctr]->TileList[2])->R_flag = MGridSz_dgemv;
				((Tile1D*)kernels[current_ctr]->TileList[2])->W_flag = MGridSz_dgemv;
				((Tile1D*)kernels[current_ctr]->TileList[2])->W_total = MGridSz_dgemv;			
			}
			ptr_ker_translate->M = ((Tile2D*) kernels[current_ctr]->TileList[0])->dim1;
			ptr_ker_translate->N = ((Tile2D*) kernels[current_ctr]->TileList[0])->dim2;
			ptr_ker_translate->A = NULL;
			ptr_ker_translate->x = NULL;
			ptr_ker_translate->y = NULL;
			ptr_ker_translate->alpha = initial_dgemv->alpha;
			ptr_ker_translate->beta = initial_dgemv->beta;
			ptr_ker_translate->incx = initial_dgemv->incx;
			ptr_ker_translate->incy = initial_dgemv->incy;
			kernels[current_ctr]->WR_first = 0;
			kernels[current_ctr]->WR_last = (short*) calloc (3, sizeof(short));
			//if (initial_dgemv->beta == 0.0)((Tile2D*) kernels[current_ctr]->TileList[2])->R_flag = 0; TODO: Does this break anything? :()
		}
	}

#ifdef DEBUG
	lprintf(lvl-1, "<-----|\n");
#endif
	return kernels;
}

void DgemvUpdateDevice(Subkernel* ker, short dev_id){
	gemv_backend_in<double>* ptr_ker_translate = (gemv_backend_in<double>*) ker->operation_params;
	ker->run_dev_id = ptr_ker_translate->dev_id = dev_id;
	short dev_id_idx = idxize(dev_id);
	if(!ker->WR_first) ptr_ker_translate->beta = 1.0;
	ptr_ker_translate->ldA = ((Tile2D*) ker->TileList[0])->ldim[dev_id_idx];
	ptr_ker_translate->incx = ((Tile1D*) ker->TileList[1])->inc[dev_id_idx];
	ptr_ker_translate->incy = ((Tile1D*) ker->TileList[2])->inc[dev_id_idx];
}

void DgemvUpdatePointers(Subkernel* ker){
	gemv_backend_in<double>* ptr_ker_translate = (gemv_backend_in<double>*) ker->operation_params;
	short dev_id_idx = idxize(ker->run_dev_id);
	ptr_ker_translate->A = &((Tile2D*) ker->TileList[0])->StoreBlock[dev_id_idx]->Adrs;
	ptr_ker_translate->x = &((Tile1D*) ker->TileList[1])->StoreBlock[dev_id_idx]->Adrs;
	ptr_ker_translate->y = &((Tile1D*) ker->TileList[2])->StoreBlock[dev_id_idx]->Adrs;
}

void* PARALiADgemvAgentVoid(void* kernel_pthread_wrapped){
	short lvl = 2;

	kernel_pthread_wrap_p gemv_subkernel_data = (kernel_pthread_wrap_p)kernel_pthread_wrapped;
	short dev_id = gemv_subkernel_data->dev_id;
#ifdef DEBUG
	lprintf(lvl-1, "|-----> PARALiADgemvAgentVoid(gemv_subkernel_data: dev_id = %d)\n",
		dev_id);
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
	Global_Buffer_2D[idxize(dev_id)]->allocate(true);
	//CoCoSyncCheckErr();

#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Memory management(%d): t_mem = %lf ms\n", dev_id, cpu_timer*1000);
	cpu_timer = csecond();
#endif
	pthread_barrier_wait (&SoftCache_alloc_barrier_dgemv);

#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Wait barrier(%d): t_wb = %lf ms\n", dev_id, cpu_timer*1000);
	cpu_timer = csecond();
#endif

	Subkernel * curr = NULL, *prev = NULL;
#ifndef RUNTIME_SCHEDULER_VERSION
	/// Rename global vars, perfectly safe.
	Subkernel** Subkernel_list_dgemv = gemv_subkernel_data->SubkernelListDev;
	int Subkernel_num_dgemv = gemv_subkernel_data->SubkernelNumDev;
	int remaining_Subkernels_dgemv = Subkernel_num_dgemv;
#else
	gemv_subkernel_data->SubkernelNumDev = 0 ;
#endif
	while (remaining_Subkernels_dgemv){
		prev = curr;
		if(prev){
#ifdef ENABLE_TILE_PREFETCH
			if (remaining_Subkernels_dgemv < gemv_subkernel_data->SubkernelNumDev - 1){
				while(__sync_lock_test_and_set(&Sk_select_lock_dgemv, 1));
				SubkernelPrefetchCheapRONLYTiles(2, dev_id, Subkernel_list_dgemv, Subkernel_num_dgemv);
				__sync_lock_release(&Sk_select_lock_dgemv);
			}
#endif
			prev->sync_request_data();
		}
		while(__sync_lock_test_and_set(&Sk_select_lock_dgemv, 1));

// #ifdef GEMV_FIRE_SK_DEV_ORDER
// 		if(curr_sk_dgemv_unit_list[curr_sk_idx_dgemv] != dev_id){
// 		__sync_lock_release(&Sk_select_lock_dgemv);
// 	  continue;
// 		}
// #endif
		curr = SubkernelSelect(dev_id, Subkernel_list_dgemv, Subkernel_num_dgemv);
		if (!curr){
//#ifdef DDEBUG
//		lprintf(lvl, "PARALiADgemvAgentVoid(%d): Got curr = NULL, repeating search\n", dev_id);
//#endif


// #ifdef GEMV_FIRE_SK_DEV_ORDER
// 			if (curr_sk_idx_dgemv == curr_sk_dgemv_unit_num -1) curr_sk_idx_dgemv = 0;
// 			else curr_sk_idx_dgemv++; // Fire all rounds in device order
// #endif

			__sync_lock_release(&Sk_select_lock_dgemv);
			continue;
		}
		remaining_Subkernels_dgemv--;
#ifdef RUNTIME_SCHEDULER_VERSION
		gemv_subkernel_data->SubkernelListDev[gemv_subkernel_data->SubkernelNumDev] = curr;
		gemv_subkernel_data->SubkernelNumDev++;
#else
		for (int keri = 0; keri < gemv_subkernel_data->SubkernelNumDev; keri++)
			if (curr == gemv_subkernel_data->SubkernelListDev[keri]){
				gemv_subkernel_data->SubkernelListDev[keri] = gemv_subkernel_data->SubkernelListDev[remaining_Subkernels_dgemv];
				gemv_subkernel_data->SubkernelListDev[remaining_Subkernels_dgemv] = curr;
				break;
			}
		Subkernel_num_dgemv = remaining_Subkernels_dgemv;
#endif
		curr->prev = prev;
		if(prev) prev->next = curr;
		DgemvUpdateDevice(curr, dev_id);
		curr->init_events();
		curr->request_data();
		DgemvUpdatePointers(curr);

// #ifdef GEMV_FIRE_SK_DEV_ORDER
// 		if(0 == remaining_Subkernels_dgemv){
// 			int slide_flag = 0;
// 			for (int idx = 0; idx < curr_sk_dgemv_unit_num - 1; idx++){
// 				if (curr_sk_dgemv_unit_list[curr_sk_idx_dgemv] == curr_sk_dgemv_unit_list[idx]) slide_flag  = 1;
// 				if(slide_flag) curr_sk_dgemv_unit_list[idx] = curr_sk_dgemv_unit_list[idx+1];
// 			}
// 			curr_sk_dgemv_unit_num--;
// 			if (curr_sk_idx_dgemv == curr_sk_dgemv_unit_num) curr_sk_idx_dgemv = 0;
// 		}
// 		else{
// 			if (curr_sk_idx_dgemv == curr_sk_dgemv_unit_num -1) curr_sk_idx_dgemv = 0;
// 			else curr_sk_idx_dgemv++; // Fire all rounds in device order
// 		}
// #endif

		__sync_lock_release(&Sk_select_lock_dgemv);
		curr->run_operation();
#ifdef ENABLE_SEND_RECV_OVERLAP
#ifdef ENABLE_PREDICT_HOP_MODE
	if(!Snd_hops_and_NOSRO_enable_flag) curr->writeback_data();
#else
		curr->writeback_data();
#endif
#endif
	}

#ifndef ENABLE_SEND_RECV_OVERLAP
	for (int keri = 0; keri < gemv_subkernel_data->SubkernelNumDev; keri++)
		gemv_subkernel_data->SubkernelListDev[keri]->writeback_data();
#else
#ifdef ENABLE_PREDICT_HOP_MODE
	if(Snd_hops_and_NOSRO_enable_flag)
		for (int keri = 0; keri < gemv_subkernel_data->SubkernelNumDev; keri++)
		 gemv_subkernel_data->SubkernelListDev[keri]->writeback_data();
#endif
#endif

	CoCoSyncCheckErr();
#ifdef TEST
	double total_cache_timer = Global_Buffer_2D[idxize(dev_id)]->timer;
	total_cache_timer += Global_Buffer_1D[idxize(dev_id)]->timer;
	lprintf(lvl, "Cache requests total timer (%d): t_cache = %lf ms\n" , dev_id, total_cache_timer*1000);
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Subkernels complete(%d): t_comp = %lf ms\n" , dev_id, cpu_timer*1000);
#endif

#ifdef DEBUG
	lprintf(lvl-1, "<-----|\n");
#endif
	return NULL;
}

/// A dgemv wrapper including auto-tuning of T and cache_size, as well as device management
ATC_p PARALiADgemv(char TransA,  long int M, long int N, double alpha, double* A, long int ldA,
		double* x, long int incx, double beta, double* y, long int incy)
{
	short lvl = 1;
#ifdef DEBUG
	lprintf(lvl-1, "|-----> PARALiADgemv(%c,%zu,%zu,%lf,A=%p(%d),%zu,x=%p(%d),%zu,%lf,y=%p(%d),%zu)\n",
		TransA, M, N, alpha, A, CoCoGetPtrLoc(A), ldA,
		x, CoCoGetPtrLoc(x), incx, beta, y, CoCoGetPtrLoc(y), incy);
#endif
#ifdef STEST
	gemv_entry_ts = csecond();
#endif
#ifdef TEST
	lprintf(lvl-1, "|-----> PARALiADgemv\n");
	double cpu_timer = csecond();
#endif

	int prev_dev_id = CoCoPeLiaGetDevice();

	int reuse_model_flag = 1;
	if(!initial_dgemv){
		initial_dgemv = (gemv_backend_in<double>*) malloc(sizeof(gemv_backend_in<double>));
		reuse_model_flag = 0;
	}
	if(reuse_model_flag && initial_dgemv->TransA != TransA)
		reuse_model_flag = 0;
	initial_dgemv->TransA = TransA;
	if(reuse_model_flag && initial_dgemv->M != M)
		reuse_model_flag = 0;
	initial_dgemv->M = M;
	if(reuse_model_flag && initial_dgemv->N != N)
		reuse_model_flag = 0;
	initial_dgemv->N = N;
	if(reuse_model_flag && initial_dgemv->A!= NULL && *initial_dgemv->A != A)
		reuse_model_flag = 0;
	initial_dgemv->A = (void**) &A;
    if(reuse_model_flag && initial_dgemv->x!= NULL && *initial_dgemv->x != x)
		reuse_model_flag = 0;
	initial_dgemv->x = (void**) &x;
	if(reuse_model_flag && initial_dgemv->y!= NULL && *initial_dgemv->y != y)
		reuse_model_flag = 0;
	initial_dgemv->y = (void**) &y;


	initial_dgemv->alpha = alpha;
	initial_dgemv->beta = beta;
	initial_dgemv->ldA = ldA;
    initial_dgemv->incx = incx;
    initial_dgemv->incy = incy;
	initial_dgemv->dev_id = -1;

	Decom2D* A_asset;
    Decom1D* x_asset, *y_asset;
	/// Prepare Assets in parallel( e.g. initialize asset classes, pin memory with pthreads)
	/// return: A_asset, x_asset, y_asset initialized and pinned
	A_asset = new Decom2D( A, M, N, ldA, 'N', DOUBLE);
    if (TransA == 'N'){
		x_asset = new Decom1D( x, N, incx, DOUBLE);
		y_asset = new Decom1D( y, M, incy, DOUBLE);
	}
	else{
		x_asset = new Decom1D( x, M, incx, DOUBLE);
		y_asset = new Decom1D( y, N, incy, DOUBLE);	
	}

	pthread_attr_t attr;
	int s = pthread_attr_init(&attr);
	if (s != 0) error("PARALiADgemv: pthread_attr_init failed s=%d\n", s);

	pthread_t asset_thread_id[3];
	A_asset->prepareAsync(&asset_thread_id[0], attr);
    x_asset->prepareAsync(&asset_thread_id[1], attr);
	y_asset->prepareAsync(&asset_thread_id[2], attr);

	if (!reuse_model_flag){
		delete autotune_controller_dgemv;
		autotune_controller_dgemv = NULL;
	}
	if (autotune_controller_dgemv == NULL) autotune_controller_dgemv = new ATC();
	if (predef_controller_dgemv && autotune_controller_dgemv->diff_intialized_params_ATC(predef_controller_dgemv)){
		 autotune_controller_dgemv->mimic_ATC(predef_controller_dgemv);
		 reuse_model_flag = 0;
	}
	double autotune_timer = 0;
	if(!reuse_model_flag) autotune_timer = autotune_controller_dgemv->autotune_problem("Dgemv", initial_dgemv);

	void* res;
	for(int i=0; i<3;i++){
		s = pthread_join(asset_thread_id[i], &res);
		if (s != 0) error("PARALiADgemv: pthread_join failed with exit value %d", s);
		//free(res);      /* Free memory allocated by thread */
	}

#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Preparing assets (parallel with pthreads) -> t_prep = %lf ms\n", cpu_timer*1000);
	cpu_timer = csecond();
#endif

	long int T = autotune_controller_dgemv->T;

	int Block_num_A = (A_asset->dim1/T + ((A_asset->dim1%T)? 1 : 0))* (A_asset->dim2/T + ((A_asset->dim2%T)? 1 : 0)),
		Block_num_x = x_asset->dim/T + ((x_asset->dim%T)? 1 : 0),
		Block_num_y = y_asset->dim/T + ((y_asset->dim%T)? 1 : 0);

    long long Block_sz_1D = T*sizeof(double);
	long long Block_sz_2D = T*T*sizeof(double);

	for(int cache_loc = 0; cache_loc < LOC_NUM; cache_loc++){
		int Block_num_1D = 0 ,Block_num_2D = 0, Native_block_num_1D = 0, Native_block_num_2D = 0;
		if (A_asset->loc == deidxize(cache_loc)) Native_block_num_2D+=Block_num_A;
		if (x_asset->loc == deidxize(cache_loc)) Native_block_num_1D+=Block_num_x;
		if (y_asset->loc == deidxize(cache_loc)) Native_block_num_1D+=Block_num_y;

		long long max_cache_sz = 0, max_cache_sz_1D = 0, max_cache_sz_2D = 0;
		if(autotune_controller_dgemv->cache_limit > 0) {
			max_cache_sz = autotune_controller_dgemv->cache_limit;
			max_cache_sz_2D = max_cache_sz * T / (T * autotune_controller_dgemv->active_unit_num);
			max_cache_sz_1D = max_cache_sz_2D * autotune_controller_dgemv->active_unit_num / T;

			if (max_cache_sz < Block_sz_2D + 2 * Block_sz_1D) error("PARALiADgemv: Problem cannot run with less memory than %d\n", Block_sz_2D + 2 * Block_sz_1D);

			long long free_dev_mem, max_dev_mem = 0, prev_DevCache_sz = 0, prev_DevCache_sz_1D = 0, prev_DevCache_sz_2D = 0;
			if (Global_Buffer_1D[cache_loc] != NULL) prev_DevCache_sz_1D = (long long)
				Global_Buffer_1D[cache_loc]->BlockSize* Global_Buffer_1D[cache_loc]->BlockNum;
			if (Global_Buffer_2D[cache_loc] != NULL) prev_DevCache_sz_2D = (long long)
				Global_Buffer_2D[cache_loc]->BlockSize* Global_Buffer_2D[cache_loc]->BlockNum;
			int prev_dev = CoCoPeLiaGetDevice();
			CoCoPeLiaSelectDevice(deidxize(cache_loc));
			prev_DevCache_sz = prev_DevCache_sz_1D + prev_DevCache_sz_2D;

			if(deidxize(cache_loc)!=-1) {
			// if(Native_block_num==0){
				CoCoPeLiaDevGetMemInfo(&free_dev_mem, &max_dev_mem);
				max_cache_sz = (long long) fmin(max_cache_sz, free_dev_mem - ((long long) max_dev_mem*(1-PROBLEM_GPU_PERCENTAGE/100.0)) + prev_DevCache_sz);
			}
			else {
				// free_dev_mem = max_dev_mem = 2 * Native_block_num * Block_sz;
				free_dev_mem = max_dev_mem = (Native_block_num_2D + 0) * Block_sz_2D + Native_block_num_1D * Block_sz_1D;
				max_cache_sz = free_dev_mem;
			}
			max_cache_sz = fmax((Native_block_num_1D + 2) * Block_sz_1D + (Native_block_num_2D + 1) * Block_sz_2D, max_cache_sz);
			max_cache_sz_2D = max_cache_sz * T / (T * autotune_controller_dgemv->active_unit_num);
			max_cache_sz_1D = max_cache_sz_2D * autotune_controller_dgemv->active_unit_num / T;

			CoCoPeLiaSelectDevice(prev_dev);
			// max_cache_sz = (long long) fmin(max_cache_sz, free_dev_mem - ((long long) max_dev_mem*(1-PROBLEM_GPU_PERCENTAGE/100.0)) + prev_DevCache_sz);
		}
		else{
			long long free_dev_mem, max_dev_mem = 0, prev_DevCache_sz = 0, prev_DevCache_sz_1D = 0, prev_DevCache_sz_2D = 0;
			if (Global_Buffer_1D[cache_loc] != NULL) prev_DevCache_sz_1D = (long long)
				Global_Buffer_1D[cache_loc]->BlockSize* Global_Buffer_1D[cache_loc]->BlockNum;
			if (Global_Buffer_2D[cache_loc] != NULL) prev_DevCache_sz_2D = (long long)
				Global_Buffer_2D[cache_loc]->BlockSize* Global_Buffer_2D[cache_loc]->BlockNum;
			int prev_dev = CoCoPeLiaGetDevice();
			CoCoPeLiaSelectDevice(deidxize(cache_loc));

			prev_DevCache_sz = prev_DevCache_sz_1D + prev_DevCache_sz_2D;
			if(deidxize(cache_loc)!=-1) CoCoPeLiaDevGetMemInfo(&free_dev_mem, &max_dev_mem);
			else free_dev_mem = max_dev_mem = 100000000000; // TODO: hard coded value, should put something that reads it from system?
			CoCoPeLiaSelectDevice(prev_dev);
			max_cache_sz = free_dev_mem - ((long long) max_dev_mem*(1-PROBLEM_GPU_PERCENTAGE/100.0)) + prev_DevCache_sz;
			max_cache_sz_2D = max_cache_sz * T / (T * autotune_controller_dgemv->active_unit_num);
			max_cache_sz_1D = max_cache_sz_2D * autotune_controller_dgemv->active_unit_num / T;
		}

		Block_num_1D = 1 + Block_num_x + Block_num_y;
		Block_num_2D = 1 + Block_num_A;
		int max_block_num_1D = max_cache_sz_1D/Block_sz_1D;
		int max_block_num_2D = max_cache_sz_2D/Block_sz_2D;
		if(max_block_num_1D< Block_num_1D){
			//#ifdef DEBUG
			if(!reuse_model_flag){
				lprintf(0, "PARALiADgemv: Problem will use %d 1D blocks for dev_id = %d\
					instead of %d needed for the full problem\n", max_block_num_1D, deidxize(cache_loc), Block_num_1D);
				lprintf(0, "====================================\n");
			}
			Block_num_1D = max_block_num_1D;
			// With Parallel backends unfortunately == SK_blocks + Max_Exclusive_Blocks - 1
			int worst_case_ex_blocks = 3; //2 + (C_asset->dim1/T + ((C_asset->dim1%T)? 1 : 0))* (C_asset->dim2/T + ((C_asset->dim2%T)? 1 : 0));
			if(max_block_num_1D < worst_case_ex_blocks)
				error("PARALiADgemv: Not able to run with < %d 1D blocks per cache due to EX scheduling\n", worst_case_ex_blocks);
		}
		if(max_block_num_2D < Block_num_2D){
			//#ifdef DEBUG
			if(!reuse_model_flag){
				lprintf(0, "PARALiADgemv: Problem will use %d 2D blocks for dev_id = %d\
					instead of %d needed for the full problem\n", max_block_num_2D, deidxize(cache_loc), Block_num_2D);
				lprintf(0, "====================================\n");
			}
			Block_num_2D = max_block_num_2D;
			// With Parallel backends unfortunately == SK_blocks + Max_Exclusive_Blocks - 1
			int worst_case_ex_blocks = 3; //2 + (C_asset->dim1/T + ((C_asset->dim1%T)? 1 : 0))* (C_asset->dim2/T + ((C_asset->dim2%T)? 1 : 0));
			if(max_block_num_2D < worst_case_ex_blocks)
				error("PARALiADgemv: Not able to run with < %d 2D blocks per cache due to EX scheduling\n", worst_case_ex_blocks);
		}
#ifdef BUFFER_REUSE_ENABLE
		if(Global_Buffer_1D[cache_loc] == NULL) Global_Buffer_1D[cache_loc] = new Buffer(deidxize(cache_loc), Block_num_1D, Block_sz_1D);
		else if (Global_Buffer_1D[cache_loc]->BlockSize != Block_sz_1D || Global_Buffer_1D[cache_loc]->BlockNum < Block_num_1D){
#ifdef DEBUG
		lprintf(lvl, "PARALiADgemv: Previous 1D Buffer smaller than requested:\
		Global_Buffer[%d]->BlockSize=%lld vs Block_sz = %lld,\
		Global_Buffer[%d]->BlockNum=%d vs Block_num = %d\n",
		cache_loc, Global_Buffer_1D[cache_loc]->BlockSize, Block_sz_1D,
		cache_loc, Global_Buffer_1D[cache_loc]->BlockNum, Block_num_1D);
#endif
			delete Global_Buffer_1D[cache_loc];
			Global_Buffer_1D[cache_loc] = new Buffer(deidxize(cache_loc), Block_num_1D, Block_sz_1D);
		}
		else{
			;
		}
		if(Global_Buffer_2D[cache_loc] == NULL) Global_Buffer_2D[cache_loc] = new Buffer(deidxize(cache_loc), Block_num_2D, Block_sz_2D);
		else if (Global_Buffer_2D[cache_loc]->BlockSize != Block_sz_2D || Global_Buffer_2D[cache_loc]->BlockNum < Block_num_2D){
#ifdef DEBUG
		lprintf(lvl, "PARALiADgemv: Previous 2D Buffer smaller than requested:\
		Global_Buffer[%d]->BlockSize=%lld vs Block_sz = %lld,\
		Global_Buffer[%d]->BlockNum=%d vs Block_num = %d\n",
		cache_loc, Global_Buffer_2D[cache_loc]->BlockSize, Block_sz_2D,
		cache_loc, Global_Buffer_2D[cache_loc]->BlockNum, Block_num_2D);
#endif
			delete Global_Buffer_2D[cache_loc];
			Global_Buffer_2D[cache_loc] = new Buffer(deidxize(cache_loc), Block_num_2D, Block_sz_2D);
		}
		else{
			;
		}
#else
		if(Global_Buffer_1D[cache_loc]!= NULL) error("PARALiADgemv: Global_Buffer_1D[%d] was not NULL with reuse disabled\n", cache_loc);
		Global_Buffer_1D[cache_loc] = new Buffer(deidxize(cache_loc), Block_num_1D, Block_sz_1D);
		if(Global_Buffer_2D[cache_loc]!= NULL) error("PARALiADgemv: Global_Buffer_2D[%d] was not NULL with reuse disabled\n", cache_loc);
		Global_Buffer_2D[cache_loc] = new Buffer(deidxize(cache_loc), Block_num_2D, Block_sz_2D);
#endif
	}

	/// TODO: Split each asset to Tiles
	A_asset->InitTileMap(T, T, Global_Buffer_2D);
	x_asset->InitTileMap(T, Global_Buffer_1D);
	y_asset->InitTileMap(T, Global_Buffer_1D);

#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Spliting assets to tiles -> t_tile_init = %lf ms\n", cpu_timer*1000);
	cpu_timer = csecond();
#endif

	Subkernel_list_dgemv = CoCoAsignTilesToSubkernelsDgemv(A_asset, x_asset, y_asset, T,
		&Subkernel_num_dgemv);
	kernel_pthread_wrap_p SK_wrap = (kernel_pthread_wrap_p) malloc(sizeof(struct kernel_pthread_wrap));
	SK_wrap->SubkernelListDev = Subkernel_list_dgemv;
	SK_wrap->SubkernelNumDev = Subkernel_num_dgemv;
	SK_wrap->dev_id = -42;
	remaining_Subkernels_dgemv = Subkernel_num_dgemv;


	if(!reuse_model_flag) autotune_controller_dgemv->update_sk_num(Subkernel_num_dgemv);
#ifdef DEBUG
	lprintf(lvl, "Subkernel_num_dgemv = %d {M,N}GridSz = {%d, %d}, autotune_controller_dgemv->active_unit_num = %d\n\n",
		Subkernel_num_dgemv, MGridSz_dgemv, NGridSz_dgemv, autotune_controller_dgemv->active_unit_num);
#endif

#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Subkernel init -> t_subkernel_init = %lf ms\n", cpu_timer*1000);
	cpu_timer = csecond();
#endif

	if(!reuse_model_flag) autotune_controller_dgemv->distribute_subkernels(MGridSz_dgemv, NGridSz_dgemv, 1);

#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Subkernel Distribute -> t_subkernel_dist = %lf ms\n", cpu_timer*1000);
	cpu_timer = csecond();
#endif
//////////////////////////////////////////////// I AM HERE //////////////////////////////////////////////

//#endif
	//autotune_controller_dgemv->active_unit_num = used_devices;

	//s = pthread_attr_init(&attr);
	//if (s != 0) error("PARALiADgemv: pthread_attr_init failed s=%d\n", s);

#ifdef GEMV_FIRE_SK_DEV_ORDER
	// Fire all devices in other_dev_score_winner
	curr_sk_idx_dgemv = 0;
	curr_sk_dgemv_unit_num = autotune_controller_dgemv->active_unit_num;
	for(int i = 0; i < curr_sk_dgemv_unit_num; i ++) curr_sk_dgemv_unit_list[i] = autotune_controller_dgemv->active_unit_id_list[i];
#endif

	pthread_t thread_id[autotune_controller_dgemv->active_unit_num];
	kernel_pthread_wrap_p thread_dev_data[autotune_controller_dgemv->active_unit_num];

	// create a barrier object with a count of autotune_controller_dgemv->active_unit_num + 1
	pthread_barrier_init (&SoftCache_alloc_barrier_dgemv, NULL, autotune_controller_dgemv->active_unit_num + 1);
	for(int d=0; d < LOC_NUM; d++) CoCoEnableLinks(deidxize(d), LOC_NUM);
	for(int d=0; d < autotune_controller_dgemv->active_unit_num; d++){
		if(autotune_controller_dgemv->Subkernels_per_unit_num[d] == 0 )
			error("CoCoPeLiaDgemv: Leftover autotune_controller_dgemv->Subkernels_per_unit_num[%d] == 0", d);

		thread_dev_data[d] = (kernel_pthread_wrap_p) malloc(sizeof(struct kernel_pthread_wrap));
		thread_dev_data[d]->dev_id = autotune_controller_dgemv->active_unit_id_list[d];

#ifndef RUNTIME_SCHEDULER_VERSION
		thread_dev_data[d]->SubkernelNumDev = autotune_controller_dgemv->Subkernels_per_unit_num[d];
#else
		thread_dev_data[d]->SubkernelNumDev = Subkernel_num_dgemv;
#endif
		thread_dev_data[d]->SubkernelListDev = (Subkernel**) malloc(thread_dev_data[d]->SubkernelNumDev*sizeof(Subkernel*));
#ifndef RUNTIME_SCHEDULER_VERSION
		for(int skitt = 0; skitt < autotune_controller_dgemv->Subkernels_per_unit_num[d]; skitt++)
			thread_dev_data[d]->SubkernelListDev[skitt] = Subkernel_list_dgemv[autotune_controller_dgemv->Subkernels_per_unit_list[d][skitt]];
#endif

		s = pthread_create(&thread_id[d], &attr,
                                  &PARALiADgemvAgentVoid, thread_dev_data[d]);

	}
	pthread_barrier_wait (&SoftCache_alloc_barrier_dgemv);

	//A_asset->DrawTileMap();
	//B_asset->DrawTileMap();
	//C_asset->DrawTileMap();

	for(int d=0; d < autotune_controller_dgemv->active_unit_num; d++){
		s = pthread_join(thread_id[d], &res);
		if (s != 0) error("PARALiADgemv: pthread_join failed with exit value %d", s);
		//free(res);      /* Free memory allocated by thread */
	}

	int prev_dev = CoCoPeLiaGetDevice();

	/// Small fix since host functions triggered after the last WB belong to a different device,
	/// resulting in warning if reset cache was reached prior. If it leads to slowdown will be exterminated.
	for(int i=0; i<LOC_NUM;i++){
		CoCoPeLiaSelectDevice(deidxize(i));
		CoCoSyncCheckErr();
	}
	CoCoPeLiaSelectDevice(prev_dev);
	#ifdef TEST
		cpu_timer = csecond() - cpu_timer;
		lprintf(lvl, "Fire and gather pthreads for all devices -> t_exec_full = %lf ms\n", cpu_timer*1000);
		lprintf(lvl, "t_predicted for T=%zu was %.2lf ms : %lf percentile error\n", T, autotune_controller_dgemv->pred_t*1000,
		(autotune_controller_dgemv->pred_t==0)? 0.0: (autotune_controller_dgemv->pred_t - cpu_timer )/autotune_controller_dgemv->pred_t*100);
		cpu_timer = csecond();
	#endif

#ifdef STEST
	STEST_print_SK(thread_dev_data, gemv_entry_ts, autotune_controller_dgemv->active_unit_num);
#endif

#ifdef DDEBUG
  	A_asset->DrawTileMap();
  	x_asset->DrawTileMap();
	y_asset->DrawTileMap();
#endif

#ifdef CDEBUG
	for(int i=0; i<LOC_NUM;i++) Global_Buffer_1D[i]->draw_buffer(true,true,true);
	for(int i=0; i<LOC_NUM;i++) Global_Buffer_2D[i]->draw_buffer(true,true,true);
#endif

#ifndef BUFFER_REUSE_ENABLE
	for(int i = 0 ; i < LOC_NUM; i++){
		delete Global_Buffer_1D[i];
		Global_Buffer_1D[i] = NULL;
		delete Global_Buffer_2D[i];
		Global_Buffer_2D[i] = NULL;
	}
#else
	for(int i=0; i<LOC_NUM;i++) Global_Buffer_1D[i]->reset(false,true);
	for(int i=0; i<LOC_NUM;i++) Global_Buffer_2D[i]->reset(false,true);
#endif

#ifndef BACKEND_RES_REUSE_ENABLE
	for(int i=0; i<autotune_controller_dgemv->active_unit_num;i++) CoCoPeLiaFreeResources(autotune_controller_dgemv->active_unit_id_list[i]);
#endif

#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Invalidate caches -> t_invalidate = %lf ms\n", cpu_timer*1000);
	cpu_timer = csecond();
#endif

	for(int i=0; i<Subkernel_num_dgemv; i++) delete Subkernel_list_dgemv[i];
	//delete [] Subkernel_list_dgemv;

#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Freed Subkernels -> t_invalidate = %lf ms\n", cpu_timer*1000);
	cpu_timer = csecond();
#endif

	A_asset->DestroyTileMap();
	x_asset->DestroyTileMap();
	y_asset->DestroyTileMap();

#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Destroyed Tilemaps -> t_invalidate = %lf ms\n", cpu_timer*1000);
	cpu_timer = csecond();
#endif

	CoCoPeLiaSelectDevice(prev_dev_id);

  	A_asset->resetProperties();
  	x_asset->resetProperties();
  	y_asset->resetProperties();
	delete A_asset;
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

	reuse_model_flag = 1;
	predef_controller_dgemv = NULL;
	// Better not return our global to the user, he can accidentally do stuff to it.
	ATC_p result = new ATC();
	result->mimic_ATC(autotune_controller_dgemv);
	return result;
}

/// A modification of PARALiADgemv but with given parameters (mainly for performance/debug purposes)
ATC_p PARALiADgemvControled(char TransA, long int M, long int N, double alpha, double* A, long int ldA,
		double* x, long int incx, double beta, double* y, long int incy, ATC_p predef_controller){
	if (predef_controller == NULL){
		warning("Calling PARALiADgemvControled with empty controller -> falling back to full autotune version \'PARALiADgemv\'\n");
		return PARALiADgemv(TransA, M, N, alpha, A, ldA, x, incx, beta, y, incy);
	}
	predef_controller_dgemv = predef_controller;
	return PARALiADgemv(TransA, M, N, alpha, A, ldA, x, incx, beta, y, incy);
}
