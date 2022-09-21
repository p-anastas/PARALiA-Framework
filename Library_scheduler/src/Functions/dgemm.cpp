///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The DGEMM CoCopeLia implementation using the new mission-agent-asset C++ classes.
///

#include "backend_wrappers.hpp"
#include "Autotuning_runtime.hpp"
#include "CoCoPeLia.hpp"
#include "unihelpers.hpp"
#include "Asset.hpp"
#include "Subkernel.hpp"
#include "DataCaching.hpp"

#include <pthread.h>
pthread_barrier_t  SoftCache_alloc_barrier_gemm;

gemm_backend_in_p initial_gemm = NULL;
ATC_p autotune_controller_gemm = NULL;
int MGridSz = 0, NGridSz = 0, KGridSz = 0;

#ifdef STEST
double gemm_entry_ts;
#endif

Subkernel** Subkernel_list_gemm;
int Subkernel_num_gemm;
int remaining_Subkernels_gemm;
int curr_sk_idx_gemm = 0;

int Sk_select_lock_gemm = 0;

Subkernel** CoCoAsignTilesToSubkernelsGemm(Asset2D<VALUE_TYPE>* A_asset, Asset2D<VALUE_TYPE>* B_asset,
	Asset2D<VALUE_TYPE>* C_asset, int T, int* kernelNum){

	short lvl = 2;
	/// Check Assets satisfy GEMM dim criteria for N, N transpose
	if (A_asset->transpose == 'N' && B_asset->transpose == 'N'){
		massert(A_asset->GridSz1 == C_asset->GridSz1 &&
						A_asset->Tile_map[0]->dim1 == C_asset->Tile_map[0]->dim1 &&
						A_asset->Tile_map[A_asset->GridSz1*A_asset->GridSz2-1]->dim1
						== C_asset->Tile_map[C_asset->GridSz1*C_asset->GridSz2-1]->dim1,
						"M dim does not mach between assets for GEMM\n");
		massert(B_asset->GridSz2 == C_asset->GridSz2 &&
						B_asset->Tile_map[0]->dim2 == C_asset->Tile_map[0]->dim2 &&
						B_asset->Tile_map[B_asset->GridSz1*B_asset->GridSz2-1]->dim2
						== C_asset->Tile_map[C_asset->GridSz1*C_asset->GridSz2-1]->dim2,
						"N dim does not mach between assets for GEMM\n");
		massert(A_asset->GridSz2 == B_asset->GridSz1 &&
						A_asset->Tile_map[0]->dim2 == B_asset->Tile_map[0]->dim1 &&
						A_asset->Tile_map[A_asset->GridSz1*A_asset->GridSz2-1]->dim2
						== B_asset->Tile_map[B_asset->GridSz1*B_asset->GridSz2-1]->dim1,
						"K dim does not mach between assets for GEMM\n");
	}
	MGridSz = A_asset->GridSz1;
	NGridSz = B_asset->GridSz2;
	KGridSz = A_asset->GridSz2;
	*kernelNum = MGridSz*NGridSz*KGridSz;
#ifdef DEBUG
	lprintf(lvl-1, "|-----> CoCoAsignTilesToSubkernelsGemm(A_asset,B_asset,C_asset,%d,%d)\n", T, *kernelNum);
	lprintf(lvl,"MgridSz = %d, NgridSz = %d, KgridSz = %d\n", MGridSz, NGridSz, KGridSz);
	lprintf(lvl,"Mlast = %d, Nlast = %d, Klast = %d\n",
	A_asset->Tile_map[A_asset->GridSz1*A_asset->GridSz2-1]->dim1,
	B_asset->Tile_map[B_asset->GridSz1*B_asset->GridSz2-1]->dim2,
	A_asset->Tile_map[A_asset->GridSz1*A_asset->GridSz2-1]->dim2);
#endif

Subkernel** kernels = (Subkernel**) malloc(*kernelNum*sizeof(Subkernel*));
int current_ctr = 0;
	for (int mi = 0; mi < MGridSz; mi++){
		for (int ni = 0; ni < NGridSz; ni++){
			for (int ki = 0; ki < KGridSz; ki++){
	      current_ctr = mi*NGridSz*KGridSz + ni*KGridSz + ki;
				kernels[current_ctr] = new Subkernel(3,"gemm");
				kernels[current_ctr]->iloc1 = mi;
				kernels[current_ctr]->iloc2 = ni;
				kernels[current_ctr]->iloc3 = ki;
				kernels[current_ctr]->TileDimlist[0] = kernels[current_ctr]->TileDimlist[1]
				= kernels[current_ctr]->TileDimlist[2] = 2;
				kernels[current_ctr]->TileList[0] = A_asset->getTile(mi,ki);
				kernels[current_ctr]->TileList[1] = B_asset->getTile(ki,ni);
				kernels[current_ctr]->TileList[2] = C_asset->getTile(mi,ni);
				((Tile2D<VALUE_TYPE>*)kernels[current_ctr]->TileList[0])->R_flag = NGridSz;
				((Tile2D<VALUE_TYPE>*)kernels[current_ctr]->TileList[1])->R_flag = MGridSz;
				((Tile2D<VALUE_TYPE>*)kernels[current_ctr]->TileList[2])->R_flag = KGridSz;
				((Tile2D<VALUE_TYPE>*)kernels[current_ctr]->TileList[2])->W_flag = KGridSz;
				((Tile2D<VALUE_TYPE>*)kernels[current_ctr]->TileList[2])->W_total = KGridSz;
				kernels[current_ctr]->operation_params = (void*) malloc(sizeof(struct gemm_backend_in));
				gemm_backend_in_p ptr_ker_translate = (gemm_backend_in_p) kernels[current_ctr]->operation_params;
				ptr_ker_translate->TransA = initial_gemm->TransA;
				ptr_ker_translate->TransB = initial_gemm->TransB;
				ptr_ker_translate->M = ((Tile2D<VALUE_TYPE>*) kernels[current_ctr]->TileList[2])->dim1;
				ptr_ker_translate->N = ((Tile2D<VALUE_TYPE>*) kernels[current_ctr]->TileList[2])->dim2;
				if (ptr_ker_translate->TransA == 'N') ptr_ker_translate->K = ((Tile2D<VALUE_TYPE>*) kernels[current_ctr]->TileList[0])->dim2;
				else if (ptr_ker_translate->TransA == 'T') ptr_ker_translate->K = ((Tile2D<VALUE_TYPE>*) kernels[current_ctr]->TileList[0])->dim1;
				else error("CoCoAsignTilesToSubkernelsGemm: Unknown transpose type\n");
				ptr_ker_translate->A = NULL;
				ptr_ker_translate->B = NULL;
				ptr_ker_translate->C = NULL;
				ptr_ker_translate->alpha = initial_gemm->alpha;
				ptr_ker_translate->beta = initial_gemm->beta;
				kernels[current_ctr]->WR_first = 0;
				kernels[current_ctr]->WR_last = (short*) calloc (3, sizeof(short));
				//if (initial_gemm->beta == 0.0)((Tile2D<VALUE_TYPE>*) kernels[current_ctr]->TileList[2])->R_flag = 0; TODO: Does this break anything? :()
			}
		}
	}

#ifdef DEBUG
	lprintf(lvl-1, "<-----|\n");
#endif
	return kernels;
}

void CoCoGemmUpdateDevice(Subkernel* ker, short dev_id){
	gemm_backend_in_p ptr_ker_translate = (gemm_backend_in_p) ker->operation_params;
	ker->run_dev_id = ptr_ker_translate->dev_id = dev_id;
	short dev_id_idx = idxize(dev_id);
	if(!ker->WR_first) ptr_ker_translate->beta = 1.0;
	ptr_ker_translate->ldA = ((Tile2D<VALUE_TYPE>*) ker->TileList[0])->ldim[dev_id_idx];
	ptr_ker_translate->ldB = ((Tile2D<VALUE_TYPE>*) ker->TileList[1])->ldim[dev_id_idx];
	ptr_ker_translate->ldC = ((Tile2D<VALUE_TYPE>*) ker->TileList[2])->ldim[dev_id_idx];
}

void CoCoGemmUpdatePointers(Subkernel* ker){
	gemm_backend_in_p ptr_ker_translate = (gemm_backend_in_p) ker->operation_params;
	short dev_id_idx = idxize(ker->run_dev_id);
	ptr_ker_translate->A = &((Tile2D<VALUE_TYPE>*) ker->TileList[0])->StoreBlock[dev_id_idx]->Adrs;
	ptr_ker_translate->B = &((Tile2D<VALUE_TYPE>*) ker->TileList[1])->StoreBlock[dev_id_idx]->Adrs;
	ptr_ker_translate->C = &((Tile2D<VALUE_TYPE>*) ker->TileList[2])->StoreBlock[dev_id_idx]->Adrs;
}

void* CoCopeLiaDgemmAgentVoid(void* kernel_pthread_wrapped){
	short lvl = 2;

	kernel_pthread_wrap_p gemm_subkernel_data = (kernel_pthread_wrap_p)kernel_pthread_wrapped;
	short dev_id = gemm_subkernel_data->dev_id;
#ifdef DEBUG
	lprintf(lvl-1, "|-----> CoCopeLiaDgemmAgentVoid(gemm_subkernel_data: dev_id = %d)\n",
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

	Global_Cache[idxize(dev_id)]->allocate(true);
	//CoCoSyncCheckErr();

#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Memory management(%d): t_mem = %lf ms\n", dev_id, cpu_timer*1000);
	cpu_timer = csecond();
#endif
	pthread_barrier_wait (&SoftCache_alloc_barrier_gemm);

#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Wait barrier(%d): t_wb = %lf ms\n", dev_id, cpu_timer*1000);
	cpu_timer = csecond();
#endif

	Subkernel * curr = NULL, *prev = NULL;
#ifndef RUNTIME_SCHEDULER_VERSION
	/// Rename global vars, perfectly safe.
	Subkernel** Subkernel_list_gemm = gemm_subkernel_data->SubkernelListDev;
	int Subkernel_num_gemm = gemm_subkernel_data->SubkernelNumDev;
	int remaining_Subkernels_gemm = Subkernel_num_gemm;
#else
	gemm_subkernel_data->SubkernelNumDev = 0 ;
#endif
	while (remaining_Subkernels_gemm){
		prev = curr;
		if(prev){
#ifdef ENABLE_TILE_PREFETCH
			if (remaining_Subkernels_gemm < gemm_subkernel_data->SubkernelNumDev - 1){
				while(__sync_lock_test_and_set(&Sk_select_lock_gemm, 1));
				SubkernelPrefetchCheapRONLYTiles(2, dev_id, Subkernel_list_gemm, Subkernel_num_gemm);
				__sync_lock_release(&Sk_select_lock_gemm);
			}
#endif
			prev->sync_request_data();
		}
		while(__sync_lock_test_and_set(&Sk_select_lock_gemm, 1));

		// if(curr_sk_idx_gemm < autotune_controller_gemm->active_unit_num && autotune_controller_gemm->active_unit_id_list[curr_sk_idx_gemm] != dev_id){
		// 	__sync_lock_release(&Sk_select_lock_gemm);
		// 	continue;
		// } // Fire first round in device order

		// if(autotune_controller_gemm->active_unit_id_list[curr_sk_idx_gemm] != dev_id){
		// 	__sync_lock_release(&Sk_select_lock_gemm);
		// 	continue;
		// } // Fire all rounds in device order

		curr = SubkernelSelect(dev_id, Subkernel_list_gemm, Subkernel_num_gemm);
		if (!curr){
//#ifdef DDEBUG
//		lprintf(lvl, "CoCopeLiaDgemmAgentVoid(%d): Got curr = NULL, repeating search\n", dev_id);
//#endif

			// if (curr_sk_idx_gemm < autotune_controller_gemm->active_unit_num) curr_sk_idx_gemm++; // Fire first round in device order

			// if (curr_sk_idx_gemm == autotune_controller_gemm->active_unit_num -1) curr_sk_idx_gemm = 0;
			// else curr_sk_idx_gemm++; // Fire all rounds in device order

			__sync_lock_release(&Sk_select_lock_gemm);
			continue;
		}
		remaining_Subkernels_gemm--;
#ifdef RUNTIME_SCHEDULER_VERSION
		gemm_subkernel_data->SubkernelListDev[gemm_subkernel_data->SubkernelNumDev] = curr;
		gemm_subkernel_data->SubkernelNumDev++;
#else
		for (int keri = 0; keri < gemm_subkernel_data->SubkernelNumDev; keri++)
			if (curr == gemm_subkernel_data->SubkernelListDev[keri]){
				gemm_subkernel_data->SubkernelListDev[keri] = gemm_subkernel_data->SubkernelListDev[remaining_Subkernels_gemm];
				gemm_subkernel_data->SubkernelListDev[remaining_Subkernels_gemm] = curr;
				break;
			}
		Subkernel_num_gemm = remaining_Subkernels_gemm;
#endif
		curr->prev = prev;
		if(prev) prev->next = curr;
		CoCoGemmUpdateDevice(curr, dev_id);
		curr->init_events();
		curr->request_data();
		CoCoGemmUpdatePointers(curr);

		//if (curr_sk_idx_gemm < autotune_controller_gemm->active_unit_num) curr_sk_idx_gemm++;  // Fire first round in device order

		// if (curr_sk_idx_gemm == autotune_controller_gemm->active_unit_num -1) curr_sk_idx_gemm = 0;
		// else curr_sk_idx_gemm++; // Fire all rounds in device order

		__sync_lock_release(&Sk_select_lock_gemm);
		curr->run_operation();
#ifdef ENABLE_SEND_RECV_OVERLAP
		curr->writeback_data();
#endif
	}

#ifndef ENABLE_SEND_RECV_OVERLAP
	for (int keri = 0; keri < gemm_subkernel_data->SubkernelNumDev; keri++)
		gemm_subkernel_data->SubkernelListDev[keri]->writeback_data();
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

/// A dgemm wrapper including auto-tuning of T and cache_size, as well as device management
ATC_p CoCopeLiaDgemm(char TransA,  char TransB, size_t M, size_t N, size_t K, VALUE_TYPE alpha, VALUE_TYPE* A, size_t ldA, VALUE_TYPE* B, size_t ldB, VALUE_TYPE beta, VALUE_TYPE* C, size_t ldC)
{
	short lvl = 1;
#ifdef DEBUG
	lprintf(lvl-1, "|-----> CoCopeLiaDgemm(%c,%c,%zu,%zu,%zu,%lf,A=%p(%d),%zu,B=%p(%d),%zu,%lf,C=%p(%d),%zu)\n",
		TransA, TransB, M, N, K, alpha, A, CoCoGetPtrLoc(A), ldA,
		B, CoCoGetPtrLoc(B), ldB, beta, C, CoCoGetPtrLoc(C), ldC);
#endif
#ifdef STEST
	gemm_entry_ts = csecond();
#endif
#ifdef TEST
	lprintf(lvl-1, "|-----> CoCopeLiaDgemm\n");
	double cpu_timer = csecond();
#endif

	int prev_dev_id = CoCoPeLiaGetDevice();

	short reuse_model_flag = 1;
	if(!initial_gemm){
		initial_gemm = (gemm_backend_in_p) malloc(sizeof(struct gemm_backend_in));
		reuse_model_flag = 0;
	}
	if(reuse_model_flag && initial_gemm->TransA != TransA)
		reuse_model_flag = 0;
	initial_gemm->TransA = TransA;
	if(reuse_model_flag && initial_gemm->TransB != TransB)
		reuse_model_flag = 0;
	initial_gemm->TransB = TransB;
	if(reuse_model_flag && initial_gemm->M != M)
		reuse_model_flag = 0;
	initial_gemm->M = M;
	if(reuse_model_flag && initial_gemm->N != N)
		reuse_model_flag = 0;
	initial_gemm->N = N;
	if(reuse_model_flag && initial_gemm->K != K)
		reuse_model_flag = 0;
	initial_gemm->K = K;
	if(reuse_model_flag && initial_gemm->A!= NULL && *initial_gemm->A != A)
		reuse_model_flag = 0;
	initial_gemm->A = (void**) &A;

	if(reuse_model_flag && initial_gemm->B!= NULL && *initial_gemm->B != B)
		reuse_model_flag = 0;
	initial_gemm->B = (void**) &B;

	if(reuse_model_flag && initial_gemm->C!= NULL && *initial_gemm->C != C)
		reuse_model_flag = 0;
	initial_gemm->C = (void**) &C;

	initial_gemm->alpha = alpha;
	initial_gemm->beta = beta;
	initial_gemm->ldA = ldA;
	initial_gemm->ldB = ldB;
	initial_gemm->ldC = ldC;
	initial_gemm->dev_id = -1;

	Asset2D<VALUE_TYPE>* A_asset, *B_asset, *C_asset;
	/// Prepare Assets in parallel( e.g. initialize asset classes, pin memory with pthreads)
	/// return: A_asset, B_asset, C_asset initialized and pinned
	A_asset = new Asset2D<VALUE_TYPE>( A, M, K, ldA, TransA);
	B_asset = new Asset2D<VALUE_TYPE>( B, K, N, ldB, TransB);
	C_asset = new Asset2D<VALUE_TYPE>( C, M, N, ldC, 'N');

	pthread_attr_t attr;
	int s = pthread_attr_init(&attr);
	if (s != 0) error("CoCopeLiaDgemm: pthread_attr_init failed s=%d\n", s);

	pthread_t asset_thread_id[3];
	A_asset->prepareAsync(&asset_thread_id[0], attr);
	B_asset->prepareAsync(&asset_thread_id[1], attr);
	C_asset->prepareAsync(&asset_thread_id[2], attr);

	double autotune_timer = 0;
	if(!reuse_model_flag){
		if (autotune_controller_gemm != NULL) delete autotune_controller_gemm;
		autotune_controller_gemm = new ATC();
		autotune_timer = autotune_controller_gemm->autotune_problem("Dgemm", initial_gemm);
	}
	void* res;
	for(int i=0; i<3;i++){
		s = pthread_join(asset_thread_id[i], &res);
		if (s != 0) error("CoCopeLiaDgemm: pthread_join failed with exit value %d", s);
		//free(res);      /* Free memory allocated by thread */
	}

#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Preparing assets (parallel with pthreads) -> t_prep = %lf ms\n", cpu_timer*1000);
	cpu_timer = csecond();
#endif

	size_t T = autotune_controller_gemm->T;

	int GPU_Block_num, Block_num = 1 + (A_asset->dim1/T + ((A_asset->dim1%T)? 1 : 0))* (A_asset->dim2/T + ((A_asset->dim2%T)? 1 : 0)) +
				(B_asset->dim1/T + ((B_asset->dim1%T)? 1 : 0))* (B_asset->dim2/T + ((B_asset->dim2%T)? 1 : 0)) +
				(C_asset->dim1/T + ((C_asset->dim1%T)? 1 : 0))* (C_asset->dim2/T + ((C_asset->dim2%T)? 1 : 0));
	long long Block_sz = 	T*T*sizeof(VALUE_TYPE);
	GPU_Block_num = Block_num;
	if(autotune_controller_gemm->cache_limit > 0){
		int max_block_num = autotune_controller_gemm->cache_limit/Block_sz;
		if(max_block_num < Block_num){
			//#ifdef DEBUG
			if(!reuse_model_flag){
				lprintf(0, "CoCopeLiaDgemm: Input cache limit %lld forces problem to use %d blocks\
					instead of %d needed for the full problem\n", autotune_controller_gemm->cache_limit, max_block_num, Block_num);
				lprintf(0, "====================================\n");
			}
			// With Parallel backends unfortunately == SK_blocks + Max_Exclusive_Blocks - 1
			int worst_case_ex_blocks = 3; //2 + (C_asset->dim1/T + ((C_asset->dim1%T)? 1 : 0))* (C_asset->dim2/T + ((C_asset->dim2%T)? 1 : 0));
			if(max_block_num < worst_case_ex_blocks)
				error("CoCopeLiaDgemm: Not able to run with < %d blocks per cache due to EX scheduling\n", worst_case_ex_blocks);
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
		lprintf(lvl, "CoCopeLiaDgemm: Previous Cache smaller than requested:\
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
			if(Global_Cache[cache_loc]!= NULL) error("CoCopeLiaDgemm: Global_Cache[%d] was not NULL with reuse disabled\n", cache_loc);
			Global_Cache[cache_loc] = new Cache(deidxize(cache_loc), curr_block_num, Block_sz);
#endif
	}

	/// TODO: Split each asset to Tiles
	A_asset->InitTileMap(T, T, Global_Cache);
	B_asset->InitTileMap(T, T, Global_Cache);
	C_asset->InitTileMap(T, T, Global_Cache);

#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Spliting assets to tiles -> t_tile_init = %lf ms\n", cpu_timer*1000);
	cpu_timer = csecond();
#endif

	Subkernel_list_gemm = CoCoAsignTilesToSubkernelsGemm(A_asset, B_asset, C_asset, T,
		&Subkernel_num_gemm);
	kernel_pthread_wrap_p SK_wrap = (kernel_pthread_wrap_p) malloc(sizeof(struct kernel_pthread_wrap));
	SK_wrap->SubkernelListDev = Subkernel_list_gemm;
	SK_wrap->SubkernelNumDev = Subkernel_num_gemm;
	SK_wrap->dev_id = -42;
	remaining_Subkernels_gemm = Subkernel_num_gemm;


	if(!reuse_model_flag) autotune_controller_gemm->update_sk_num(Subkernel_num_gemm);
#ifdef DEBUG
	lprintf(lvl, "Subkernel_num_gemm = %d {M,N,K}GridSz = {%d, %d, %d}, autotune_controller_gemm->active_unit_num = %d\n\n",
		Subkernel_num_gemm, MGridSz, NGridSz, KGridSz, autotune_controller_gemm->active_unit_num);
#endif

#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Subkernel init -> t_subkernel_init = %lf ms\n", cpu_timer*1000);
	cpu_timer = csecond();
#endif

	if(!reuse_model_flag) autotune_controller_gemm->distribute_subkernels(MGridSz, NGridSz, KGridSz);

#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Subkernel Distribute -> t_subkernel_dist = %lf ms\n", cpu_timer*1000);
	cpu_timer = csecond();
#endif

//#endif
	//autotune_controller_gemm->active_unit_num = used_devices;

	//s = pthread_attr_init(&attr);
	//if (s != 0) error("CoCopeLiaDgemm: pthread_attr_init failed s=%d\n", s);

	pthread_t thread_id[autotune_controller_gemm->active_unit_num];
	kernel_pthread_wrap_p thread_dev_data[autotune_controller_gemm->active_unit_num];

	// create a barrier object with a count of autotune_controller_gemm->active_unit_num + 1
	pthread_barrier_init (&SoftCache_alloc_barrier_gemm, NULL, autotune_controller_gemm->active_unit_num + 1);

	for(int d=0; d < autotune_controller_gemm->active_unit_num; d++){
		if(autotune_controller_gemm->Subkernels_per_unit_num[d] == 0 )
			error("CoCoPeLiaDgemm: Leftover autotune_controller_gemm->Subkernels_per_unit_num[%d] == 0", d);

		// Check/Enable peer access between all system units
		CoCoEnableLinks(idxize(autotune_controller_gemm->active_unit_id_list[d]), LOC_NUM);

		thread_dev_data[d] = (kernel_pthread_wrap_p) malloc(sizeof(struct kernel_pthread_wrap));
		thread_dev_data[d]->dev_id = autotune_controller_gemm->active_unit_id_list[d];

#ifndef RUNTIME_SCHEDULER_VERSION
		thread_dev_data[d]->SubkernelNumDev = autotune_controller_gemm->Subkernels_per_unit_num[d];
#else
		thread_dev_data[d]->SubkernelNumDev = Subkernel_num_gemm;
#endif
		thread_dev_data[d]->SubkernelListDev = (Subkernel**) malloc(thread_dev_data[d]->SubkernelNumDev*sizeof(Subkernel*));
#ifndef RUNTIME_SCHEDULER_VERSION
		for(int skitt = 0; skitt < autotune_controller_gemm->Subkernels_per_unit_num[d]; skitt++)
			thread_dev_data[d]->SubkernelListDev[skitt] = Subkernel_list_gemm[autotune_controller_gemm->Subkernels_per_unit_list[d][skitt]];
#endif

		s = pthread_create(&thread_id[d], &attr,
                                  &CoCopeLiaDgemmAgentVoid, thread_dev_data[d]);

	}
	pthread_barrier_wait (&SoftCache_alloc_barrier_gemm);

	//A_asset->DrawTileMap();
	//B_asset->DrawTileMap();
	//C_asset->DrawTileMap();

	for(int d=0; d < autotune_controller_gemm->active_unit_num; d++){
		s = pthread_join(thread_id[d], &res);
		if (s != 0) error("CoCopeLiaDgemm: pthread_join failed with exit value %d", s);
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
		lprintf(lvl, "t_predicted for T=%zu was %.2lf ms : %lf percentile error\n", T, autotune_controller_gemm->pred_t*1000,
		(autotune_controller_gemm->pred_t==0)? 0.0: (autotune_controller_gemm->pred_t - cpu_timer )/autotune_controller_gemm->pred_t*100);
		cpu_timer = csecond();
	#endif

#ifdef STEST
	STEST_print_SK(thread_dev_data, gemm_entry_ts, autotune_controller_gemm->active_unit_num);
#endif

#ifdef DDEBUG
  A_asset->DrawTileMap();
  B_asset->DrawTileMap();
	C_asset->DrawTileMap();
#endif

#ifdef CDEBUG
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
	for(int i=0; i<autotune_controller_gemm->active_unit_num;i++) CoCoPeLiaFreeResources(autotune_controller_gemm->active_unit_id_list[i]);
#endif

#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Invalidate caches -> t_invalidate = %lf ms\n", cpu_timer*1000);
	cpu_timer = csecond();
#endif

	for(int i=0; i<Subkernel_num_gemm; i++) delete Subkernel_list_gemm[i];
	//delete [] Subkernel_list_gemm;

#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Freed Subkernels -> t_invalidate = %lf ms\n", cpu_timer*1000);
	cpu_timer = csecond();
#endif

	A_asset->DestroyTileMap();
	B_asset->DestroyTileMap();
	C_asset->DestroyTileMap();

#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Destroyed Tilemaps -> t_invalidate = %lf ms\n", cpu_timer*1000);
	cpu_timer = csecond();
#endif

	CoCoPeLiaSelectDevice(prev_dev_id);

  A_asset->resetProperties();
  B_asset->resetProperties();
  C_asset->resetProperties();
	delete A_asset;
	delete B_asset;
	delete C_asset;

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

	// Better not return our global to the user, he can accidentally do stuff to it.
	ATC_p result = new ATC();
	result->mimic_ATC(autotune_controller_gemm);
	return result;
}

/// A modification of CoCopeLiaDgemm but with given parameters (mainly for performance/debug purposes)
ATC_p CoCopeLiaDgemmControled(char TransA,  char TransB, size_t M, size_t N, size_t K, VALUE_TYPE alpha, VALUE_TYPE* A, size_t ldA, VALUE_TYPE* B, size_t ldB, VALUE_TYPE beta, VALUE_TYPE* C, size_t ldC, ATC_p predef_controller){
	if (predef_controller == NULL){
		warning("Calling CoCopeLiaDgemmControled with empty controller -> falling back to full autotune version \'CoCopeLiaDgemm\'\n");
		return CoCopeLiaDgemm(TransA, TransB,  M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);
	}
	if (autotune_controller_gemm == NULL) autotune_controller_gemm = new ATC();
	autotune_controller_gemm->T = predef_controller->T;
	autotune_controller_gemm->active_unit_num = predef_controller->active_unit_num;
	for(int idx =0; idx < LOC_NUM; idx++)
		autotune_controller_gemm->active_unit_id_list[idx] = predef_controller->active_unit_id_list[idx];
	autotune_controller_gemm->cache_limit = predef_controller->cache_limit;
	ATC_p return_vals = CoCopeLiaDgemm(TransA, TransB,  M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);
	return return_vals;
}
