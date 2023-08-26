///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The DGEMM CoCopeLia implementation using the new mission-agent-asset C++ classes.
///

#include "backend_wrappers.hpp"
#include "Autotuner.hpp"
#include "PARALiA.hpp"
#include "linkmap.hpp"
#include "Decomposer.hpp"
#include "Subkernel.hpp"
#include "DataCaching.hpp"

#include <pthread.h>

pthread_barrier_t  SoftCache_alloc_barrier_dgemm;

gemm_backend_in<double>* initial_dgemm = NULL;
ATC_p autotune_controller_dgemm = NULL;
ATC_p predef_controller_dgemm = NULL;

int MGridSz_dgemm = 0, NGridSz_dgemm = 0, KGridSz_dgemm = 0;

#ifdef STEST
double gemm_entry_ts;
#endif

Subkernel** Subkernel_list_dgemm;
int Subkernel_num_dgemm;
int remaining_Subkernels_dgemm;

#define GEMM_FIRE_SK_DEV_ORDER
#ifdef GEMM_FIRE_SK_DEV_ORDER
int curr_sk_idx_dgemm = 0;
int curr_sk_dgemm_unit_list[LOC_NUM], curr_sk_dgemm_unit_num;
#endif

int Sk_select_lock_dgemm = 0;

Subkernel** CoCoAsignTilesToSubkernelsDgemm(Decom2D* A_asset, Decom2D* B_asset,
	Decom2D* C_asset, int T, int* kernelNum){

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
	MGridSz_dgemm = A_asset->GridSz1;
	NGridSz_dgemm = B_asset->GridSz2;
	KGridSz_dgemm = A_asset->GridSz2;
	*kernelNum = MGridSz_dgemm*NGridSz_dgemm*KGridSz_dgemm;
#ifdef DEBUG
	fprintf(stderr, "|-----> CoCoAsignTilesToSubkernelsDgemm(A_asset,B_asset,C_asset,%d,%d)\n", T, *kernelNum);
	fprintf(stderr,"MgridSz = %d, NgridSz = %d, KgridSz = %d\n", MGridSz_dgemm, NGridSz_dgemm, KGridSz_dgemm);
	fprintf(stderr,"Mlast = %d, Nlast = %d, Klast = %d\n",
	A_asset->Tile_map[A_asset->GridSz1*A_asset->GridSz2-1]->dim1,
	B_asset->Tile_map[B_asset->GridSz1*B_asset->GridSz2-1]->dim2,
	A_asset->Tile_map[A_asset->GridSz1*A_asset->GridSz2-1]->dim2);
#endif

Subkernel** kernels = (Subkernel**) malloc(*kernelNum*sizeof(Subkernel*));
int current_ctr = 0;
	for (int mi = 0; mi < MGridSz_dgemm; mi++){
		for (int ni = 0; ni < NGridSz_dgemm; ni++){
			for (int ki = 0; ki < KGridSz_dgemm; ki++){
	      		current_ctr = mi*NGridSz_dgemm*KGridSz_dgemm + ni*KGridSz_dgemm + ki;
				kernels[current_ctr] = new Subkernel(3,"Dgemm");
				kernels[current_ctr]->iloc1 = mi;
				kernels[current_ctr]->iloc2 = ni;
				kernels[current_ctr]->iloc3 = ki;
				kernels[current_ctr]->TileDimlist[0] = kernels[current_ctr]->TileDimlist[1]
				= kernels[current_ctr]->TileDimlist[2] = 2;
				kernels[current_ctr]->TileList[0] = A_asset->getTile(mi,ki);
				kernels[current_ctr]->TileList[1] = B_asset->getTile(ki,ni);
				kernels[current_ctr]->TileList[2] = C_asset->getTile(mi,ni);
				kernels[current_ctr]->TileList[0]->set_WRP(RONLY);
				kernels[current_ctr]->TileList[1]->set_WRP(RONLY);
#ifdef REDUCE_OUT
				kernels[current_ctr]->TileList[2]->set_WRP(WREDUCE);
				error("Non implemented\n");
#else
				kernels[current_ctr]->TileList[2]->set_WRP(WR);
				kernels[current_ctr]->TileList[2]->W_pending = KGridSz_dgemm;

#endif
				kernels[current_ctr]->operation_params = (void*) malloc(sizeof( gemm_backend_in<double>));
				gemm_backend_in<double>*  ptr_ker_translate = (gemm_backend_in<double>*) kernels[current_ctr]->operation_params;
				ptr_ker_translate->TransA = initial_dgemm->TransA;
				ptr_ker_translate->TransB = initial_dgemm->TransB;
				ptr_ker_translate->M = kernels[current_ctr]->TileList[2]->dim1;
				ptr_ker_translate->N = kernels[current_ctr]->TileList[2]->dim2;
				if (ptr_ker_translate->TransA == 'N') ptr_ker_translate->K = kernels[current_ctr]->TileList[0]->dim2;
				else if (ptr_ker_translate->TransA == 'T') ptr_ker_translate->K = kernels[current_ctr]->TileList[0]->dim1;
				else error("CoCoAsignTilesToSubkernelsDgemm: Unknown transpose type\n");
				ptr_ker_translate->A = NULL;
				ptr_ker_translate->B = NULL;
				ptr_ker_translate->C = NULL;
				ptr_ker_translate->alpha = initial_dgemm->alpha;
				ptr_ker_translate->beta = initial_dgemm->beta;
			}
		}
	}

#ifdef DEBUG
	fprintf(stderr, "<-----|\n");
#endif
	return kernels;
}

void DgemmUpdateDevice(Subkernel* ker, short dev_id){
	gemm_backend_in<double>*  ptr_ker_translate = (gemm_backend_in<double>* ) ker->operation_params;
	ker->run_dev_id = ptr_ker_translate->dev_id = dev_id;
#ifdef DEBUG
	fprintf(stderr, "|-----> DgemmUpdateDevice - Subkernel(dev=%d, id = %d)\n", dev_id, ker->id);
#endif
	short dev_id_idx = idxize(dev_id);
	ptr_ker_translate->ldA = ker->TileList[0]->get_chunk_size(dev_id_idx);
	ptr_ker_translate->ldB = ker->TileList[1]->get_chunk_size(dev_id_idx);
	ptr_ker_translate->ldC = ker->TileList[2]->get_chunk_size(dev_id_idx);
	ker->TileList[0]->set_loc_idx(dev_id_idx, 1);
	ker->TileList[1]->set_loc_idx(dev_id_idx, 1);
	ker->TileList[2]->set_loc_idx(dev_id_idx, 1);
	ker->TileList[2]->W_master = dev_id;
	while(__sync_lock_test_and_set(&Sk_select_lock_dgemm, 1));
	if(ker->TileList[2]->W_pending == KGridSz_dgemm) ker->TileList[2]->W_complete = new Event(dev_id);
	__sync_lock_release(&Sk_select_lock_dgemm);
#ifdef DEBUG
	fprintf(stderr, "<-----|\n");
#endif
}

void DgemmPrepareLaunch(Subkernel* ker, short dev_id){
	gemm_backend_in<double>*  ptr_ker_translate = (gemm_backend_in<double>* ) ker->operation_params;
	if(!(ker->TileList[2]->W_master_backend_ctr == -42)) // Means its not the first subkernel using the WR tile
		ptr_ker_translate->beta = 1.0;

}

void DgemmUpdatePointers(Subkernel* ker){
	gemm_backend_in<double>*  ptr_ker_translate = (gemm_backend_in<double>* ) ker->operation_params;
	short dev_id_idx = idxize(ker->run_dev_id);
	ptr_ker_translate->A = &ker->TileList[0]->StoreBlock[dev_id_idx]->Adrs;
	ptr_ker_translate->B = &ker->TileList[1]->StoreBlock[dev_id_idx]->Adrs;
	ptr_ker_translate->C = &ker->TileList[2]->StoreBlock[dev_id_idx]->Adrs;
}

void* PARALiADgemmAgentVoid(void* kernel_pthread_wrapped){
	short lvl = 2;

	kernel_pthread_wrap_p gemm_subkernel_data = (kernel_pthread_wrap_p)kernel_pthread_wrapped;
	short dev_id = gemm_subkernel_data->dev_id;
#ifdef DEBUG
	fprintf(stderr, "|-----> PARALiADgemmAgentVoid(gemm_subkernel_data: dev_id = %d)\n",
		dev_id);
#endif
#ifdef TEST
		double cpu_timer = csecond();
#endif

	CoCoPeLiaSelectDevice(dev_id);
	CoCoPeLiaInitResources(dev_id);
	for(int sk_ctr = 0; sk_ctr < gemm_subkernel_data->SubkernelNumDev; sk_ctr++)
		DgemmUpdateDevice(gemm_subkernel_data->SubkernelListDev[sk_ctr], dev_id);


#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	fprintf(stderr, "Stream/Lib Handle Initialization(%d): t_resource = %lf ms\n", dev_id, cpu_timer*1000);
	cpu_timer = csecond();
#endif

	Global_Buffer_2D[idxize(dev_id)]->allocate(true);
	//CoCoSyncCheckErr();

#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	fprintf(stderr, "Memory management(%d): t_mem = %lf ms\n", dev_id, cpu_timer*1000);
	cpu_timer = csecond();
#endif
	pthread_barrier_wait (&SoftCache_alloc_barrier_dgemm);
#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	fprintf(stderr, "Wait barrier(%d): t_wb = %lf ms\n", dev_id, cpu_timer*1000);
	cpu_timer = csecond();
#endif

	Subkernel * curr = NULL;
	int remaining_Subkernels_dgemm = gemm_subkernel_data->SubkernelNumDev;
	while (remaining_Subkernels_dgemm){
		while(__sync_lock_test_and_set(&Sk_select_lock_dgemm, 1));

#ifdef GEMM_FIRE_SK_DEV_ORDER
		if(curr_sk_dgemm_unit_list[curr_sk_idx_dgemm] != dev_id){
		__sync_lock_release(&Sk_select_lock_dgemm);
	  continue;
		}
#endif
		curr = SubkernelSelect(dev_id, &(gemm_subkernel_data->SubkernelListDev[gemm_subkernel_data->SubkernelNumDev-remaining_Subkernels_dgemm]), remaining_Subkernels_dgemm);
		if (!curr){
#ifdef DEBUG
		fprintf(stderr, "PARALiADgemmAgentVoid(%d): Got curr = NULL, repeating search\n", dev_id);
#endif


#ifdef GEMM_FIRE_SK_DEV_ORDER
			if (curr_sk_idx_dgemm == curr_sk_dgemm_unit_num -1) curr_sk_idx_dgemm = 0;
			else curr_sk_idx_dgemm++; // Fire all rounds in device order
#endif

			__sync_lock_release(&Sk_select_lock_dgemm);
			continue;
		}
		remaining_Subkernels_dgemm--;
		//for (int keri = 0; keri < gemm_subkernel_data->SubkernelNumDev; keri++)
		//	if (curr == gemm_subkernel_data->SubkernelListDev[keri]){
		//		gemm_subkernel_data->SubkernelListDev[keri] = gemm_subkernel_data->SubkernelListDev[remaining_Subkernels_dgemm];
		//		gemm_subkernel_data->SubkernelListDev[remaining_Subkernels_dgemm] = curr;
		//		break;
		//	}
		curr->init_events();
		DgemmPrepareLaunch(curr, dev_id);
		curr->request_data();
		DgemmUpdatePointers(curr);

#ifdef GEMM_FIRE_SK_DEV_ORDER
		if(0 == remaining_Subkernels_dgemm){
			int slide_flag = 0;
			for (int idx = 0; idx < curr_sk_dgemm_unit_num - 1; idx++){
				if (curr_sk_dgemm_unit_list[curr_sk_idx_dgemm] == curr_sk_dgemm_unit_list[idx]) slide_flag  = 1;
				if(slide_flag) curr_sk_dgemm_unit_list[idx] = curr_sk_dgemm_unit_list[idx+1];
			}
			curr_sk_dgemm_unit_num--;
			if (curr_sk_idx_dgemm == curr_sk_dgemm_unit_num) curr_sk_idx_dgemm = 0;
		}
		else{
			if (curr_sk_idx_dgemm == curr_sk_dgemm_unit_num -1) curr_sk_idx_dgemm = 0;
			else curr_sk_idx_dgemm++; // Fire all rounds in device order
		}
#endif
	__sync_lock_release(&Sk_select_lock_dgemm);
	curr->run_operation(); // Above or bellow?


	}
#ifdef TEST
	double total_cache_timer = Global_Buffer_2D[idxize(dev_id)]->timer;
	fprintf(stderr, "Cache requests total timer (%d): t_cache = %lf ms\n" , dev_id, total_cache_timer*1000);
	cpu_timer = csecond() - cpu_timer;
	fprintf(stderr, "Subkernels complete(%d): t_comp = %lf ms\n" , dev_id, cpu_timer*1000);
#endif

	return NULL;
}

/// A dgemm wrapper including auto-tuning of T and cache_size, as well as device management
ATC_p PARALiADgemm(char TransA,  char TransB, long int M, long int N, long int K, double alpha, double* A, long int ldA,
		double* B, long int ldB, double beta, double* C, long int ldC)
{
	short lvl = 1;
#ifdef DEBUG
	fprintf(stderr, "|-----> PARALiADgemm(%c,%c,%zu,%zu,%zu,%lf,A=%p(%d),%zu,B=%p(%d),%zu,%lf,C=%p(%d),%zu)\n",
		TransA, TransB, M, N, K, alpha, A, CoCoGetPtrLoc(A), ldA,
		B, CoCoGetPtrLoc(B), ldB, beta, C, CoCoGetPtrLoc(C), ldC);
#endif
#ifdef STEST
	gemm_entry_ts = csecond();
#endif
#ifdef TEST
	fprintf(stderr, "|-----> PARALiADgemm\n");
	double cpu_timer = csecond();
#endif
  	signal(SIGSEGV, handler);   // install segfault handler

	int prev_dev_id = CoCoPeLiaGetDevice();

	int reuse_model_flag = 1;
	if(!initial_dgemm){
		initial_dgemm = (gemm_backend_in<double>* ) malloc(sizeof( gemm_backend_in<double>));
		reuse_model_flag = 0;
	}
	if(reuse_model_flag && initial_dgemm->TransA != TransA)
		reuse_model_flag = 0;
	initial_dgemm->TransA = TransA;
	if(reuse_model_flag && initial_dgemm->TransB != TransB)
		reuse_model_flag = 0;
	initial_dgemm->TransB = TransB;
	if(reuse_model_flag && initial_dgemm->M != M)
		reuse_model_flag = 0;
	initial_dgemm->M = M;
	if(reuse_model_flag && initial_dgemm->N != N)
		reuse_model_flag = 0;
	initial_dgemm->N = N;
	if(reuse_model_flag && initial_dgemm->K != K)
		reuse_model_flag = 0;
	initial_dgemm->K = K;
	if(reuse_model_flag && initial_dgemm->A!= NULL && *initial_dgemm->A != A)
		reuse_model_flag = 0;
	initial_dgemm->A = (void**) &A;

	if(reuse_model_flag && initial_dgemm->B!= NULL && *initial_dgemm->B != B)
		reuse_model_flag = 0;
	initial_dgemm->B = (void**) &B;

	if(reuse_model_flag && initial_dgemm->C!= NULL && *initial_dgemm->C != C)
		reuse_model_flag = 0;
	initial_dgemm->C = (void**) &C;

	initial_dgemm->alpha = alpha;
	initial_dgemm->beta = beta;
	initial_dgemm->ldA = ldA;
	initial_dgemm->ldB = ldB;
	initial_dgemm->ldC = ldC;
	initial_dgemm->dev_id = -1;

	Decom2D* A_asset, *B_asset, *C_asset;
	/// Prepare Assets in parallel( e.g. initialize asset classes, pin memory with pthreads)
	/// return: A_asset, B_asset, C_asset initialized and pinned
	A_asset = new Decom2D( (void*) A, M, K, ldA, TransA, DOUBLE);
	B_asset = new Decom2D( (void*) B, K, N, ldB, TransB, DOUBLE);
	C_asset = new Decom2D( (void*) C, M, N, ldC, 'N', DOUBLE);

	pthread_attr_t attr;
	int s = pthread_attr_init(&attr);
	if (s != 0) error("PARALiADgemm: pthread_attr_init failed s=%d\n", s);

	pthread_t asset_thread_id[3];
	A_asset->prepareAsync(&asset_thread_id[0], attr);
	B_asset->prepareAsync(&asset_thread_id[1], attr);
	C_asset->prepareAsync(&asset_thread_id[2], attr);

	if (!reuse_model_flag){
		delete autotune_controller_dgemm;
		autotune_controller_dgemm = NULL;
	}
	if (autotune_controller_dgemm == NULL) autotune_controller_dgemm = new ATC();
	if (predef_controller_dgemm && autotune_controller_dgemm->diff_intialized_params_ATC(predef_controller_dgemm)){
		 autotune_controller_dgemm->mimic_ATC(predef_controller_dgemm);
		 reuse_model_flag = 0;
	}
	double autotune_timer = 0;
	if(!reuse_model_flag) autotune_timer = autotune_controller_dgemm->autotune_problem("Dgemm", initial_dgemm);

	void* res;
	for(int i=0; i<3;i++){
		s = pthread_join(asset_thread_id[i], &res);
		if (s != 0) error("PARALiADgemm: pthread_join failed with exit value %d", s);
		//free(res);      /* Free memory allocated by thread */
	}

#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	fprintf(stderr, "Preparing assets (parallel with pthreads) -> t_prep = %lf ms\n", cpu_timer*1000);
	cpu_timer = csecond();
#endif

	long int T = autotune_controller_dgemm->T;

	int Block_num_A = (A_asset->dim1/T + ((A_asset->dim1%T)? 1 : 0))* (A_asset->dim2/T + ((A_asset->dim2%T)? 1 : 0)),
			Block_num_B = (B_asset->dim1/T + ((B_asset->dim1%T)? 1 : 0))* (B_asset->dim2/T + ((B_asset->dim2%T)? 1 : 0)),
			Block_num_C = (C_asset->dim1/T + ((C_asset->dim1%T)? 1 : 0))* (C_asset->dim2/T + ((C_asset->dim2%T)? 1 : 0));
	long long Block_sz = 	T*T*sizeof(double);
	for(int cache_loc = 0; cache_loc < LOC_NUM; cache_loc++){
		int Block_num = 0, Native_block_num = 0;
		if (A_asset->loc == deidxize(cache_loc)) Native_block_num+=Block_num_A;
		if (B_asset->loc == deidxize(cache_loc)) Native_block_num+=Block_num_B;
		if (C_asset->loc == deidxize(cache_loc)) Native_block_num+=Block_num_C;

		long long max_cache_sz = 0;
		if(autotune_controller_dgemm->cache_limit > 0) {
			max_cache_sz = autotune_controller_dgemm->cache_limit;
			if (max_cache_sz < 3 * Block_sz) error("PARALiADgemm: Problem cannot run with less memory than %d\n", 3 * Block_sz);

			long long free_dev_mem, max_dev_mem = 0, prev_DevCache_sz = 0;
			if (Global_Buffer_2D[cache_loc] != NULL) prev_DevCache_sz = (long long)
				Global_Buffer_2D[cache_loc]->BlockSize* Global_Buffer_2D[cache_loc]->BlockNum;
			int prev_dev = CoCoPeLiaGetDevice();
			CoCoPeLiaSelectDevice(deidxize(cache_loc));

			if(deidxize(cache_loc)!=-1) {
			// if(Native_block_num==0){
				CoCoPeLiaDevGetMemInfo(&free_dev_mem, &max_dev_mem);
				max_cache_sz = (long long) fmin(max_cache_sz, free_dev_mem - ((long long) max_dev_mem*(1-PROBLEM_GPU_PERCENTAGE/100.0)) + prev_DevCache_sz);
			}
			else {
				// free_dev_mem = max_dev_mem = 2 * Native_block_num * Block_sz;
				free_dev_mem = max_dev_mem = (Native_block_num + 3) * Block_sz;
				max_cache_sz = free_dev_mem;
			}
			max_cache_sz = fmax((Native_block_num + 3) * Block_sz, max_cache_sz);


			CoCoPeLiaSelectDevice(prev_dev);
			// max_cache_sz = (long long) fmin(max_cache_sz, free_dev_mem - ((long long) max_dev_mem*(1-PROBLEM_GPU_PERCENTAGE/100.0)) + prev_DevCache_sz);
		}
		else{
			long long free_dev_mem, max_dev_mem = 0, prev_DevCache_sz = 0;
			if (Global_Buffer_2D[cache_loc] != NULL) prev_DevCache_sz = (long long)
				Global_Buffer_2D[cache_loc]->BlockSize* Global_Buffer_2D[cache_loc]->BlockNum;
			int prev_dev = CoCoPeLiaGetDevice();
			CoCoPeLiaSelectDevice(deidxize(cache_loc));

			if(deidxize(cache_loc)!=-1) CoCoPeLiaDevGetMemInfo(&free_dev_mem, &max_dev_mem);
			else free_dev_mem = max_dev_mem = 100000000000; // TODO: hard coded value, should put something that reads it from system?
			CoCoPeLiaSelectDevice(prev_dev);
			max_cache_sz = free_dev_mem - ((long long) max_dev_mem*(1-PROBLEM_GPU_PERCENTAGE/100.0)) + prev_DevCache_sz;
		}
		Block_num = 1 + Block_num_A + Block_num_B + Block_num_C;
		int max_block_num = max_cache_sz/Block_sz;
		if(max_block_num < Block_num){
			//#ifdef DEBUG
			if(!reuse_model_flag){
				lprintf(0, "PARALiADgemm: Problem will use %d blocks for dev_id = %d\
					instead of %d needed for the full problem\n", max_block_num, deidxize(cache_loc), Block_num);
				lprintf(0, "====================================\n");
			}
			Block_num = max_block_num;
			// With Parallel backends unfortunately == SK_blocks + Max_Exclusive_Blocks - 1
			int worst_case_ex_blocks = 3; //2 + (C_asset->dim1/T + ((C_asset->dim1%T)? 1 : 0))* (C_asset->dim2/T + ((C_asset->dim2%T)? 1 : 0));
			if(max_block_num < worst_case_ex_blocks)
				error("PARALiADgemm: Not able to run with < %d blocks per cache due to EX scheduling\n", worst_case_ex_blocks);
		}
#ifdef BUFFER_REUSE_ENABLE
		if(Global_Buffer_2D[cache_loc] == NULL) Global_Buffer_2D[cache_loc] = new Buffer(deidxize(cache_loc), Block_num, Block_sz);
		else if (Global_Buffer_2D[cache_loc]->BlockSize != Block_sz || Global_Buffer_2D[cache_loc]->BlockNum < Block_num){
#ifdef DEBUG
		fprintf(stderr, "PARALiADgemm: Previous Cache smaller than requested:\
		Global_Buffer_2D[%d]->BlockSize=%lld vs Block_sz = %lld,\
		Global_Buffer_2D[%d]->BlockNum=%d vs Block_num = %d\n",
		cache_loc, Global_Buffer_2D[cache_loc]->BlockSize, Block_sz,
		cache_loc, Global_Buffer_2D[cache_loc]->BlockNum, Block_num);
#endif
			delete Global_Buffer_2D[cache_loc];
			Global_Buffer_2D[cache_loc] = new Buffer(deidxize(cache_loc), Block_num, Block_sz);
		}
		else{
			;
		}
#else
			if(Global_Buffer_2D[cache_loc]!= NULL) error("PARALiADgemm: Global_Buffer_2D[%d] was not NULL with reuse disabled\n", cache_loc);
			Global_Buffer_2D[cache_loc] = new Buffer(deidxize(cache_loc), Block_num, Block_sz);
#endif
	}

	/// TODO: Split each asset to Tiles
	A_asset->InitTileMap(T, T, Global_Buffer_2D);
	B_asset->InitTileMap(T, T, Global_Buffer_2D);
	C_asset->InitTileMap(T, T, Global_Buffer_2D);

#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	fprintf(stderr, "Spliting assets to tiles -> t_tile_init = %lf ms\n", cpu_timer*1000);
	cpu_timer = csecond();
#endif

	Subkernel_list_dgemm = CoCoAsignTilesToSubkernelsDgemm(A_asset, B_asset, C_asset, T,
		&Subkernel_num_dgemm);
	kernel_pthread_wrap_p SK_wrap = (kernel_pthread_wrap_p) malloc(sizeof(struct kernel_pthread_wrap));
	SK_wrap->SubkernelListDev = Subkernel_list_dgemm;
	SK_wrap->SubkernelNumDev = Subkernel_num_dgemm;
	SK_wrap->dev_id = -42;
	remaining_Subkernels_dgemm = Subkernel_num_dgemm;

	if(!reuse_model_flag) autotune_controller_dgemm->update_sk_num(Subkernel_num_dgemm);
#ifdef DEBUG
	fprintf(stderr, "Subkernel_num_dgemm = %d {M,N,K}GridSz = {%d, %d, %d}, autotune_controller_dgemm->active_unit_num = %d\n\n",
		Subkernel_num_dgemm, MGridSz_dgemm, NGridSz_dgemm, KGridSz_dgemm, autotune_controller_dgemm->active_unit_num);
#endif

#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	fprintf(stderr, "Subkernel init -> t_subkernel_init = %lf ms\n", cpu_timer*1000);
	cpu_timer = csecond();
#endif

	if(!reuse_model_flag) autotune_controller_dgemm->distribute_subkernels(MGridSz_dgemm, NGridSz_dgemm, KGridSz_dgemm);

#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	fprintf(stderr, "Subkernel Distribute -> t_subkernel_dist = %lf ms\n", cpu_timer*1000);
	cpu_timer = csecond();
#endif

//#endif
	//autotune_controller_dgemm->active_unit_num = used_devices;

	//s = pthread_attr_init(&attr);
	//if (s != 0) error("PARALiADgemm: pthread_attr_init failed s=%d\n", s);

#ifdef GEMM_FIRE_SK_DEV_ORDER
	// Fire all devices in other_dev_score_winner
	curr_sk_idx_dgemm = 0;
	curr_sk_dgemm_unit_num = autotune_controller_dgemm->active_unit_num;
	for(int i = 0; i < curr_sk_dgemm_unit_num; i ++) curr_sk_dgemm_unit_list[i] = autotune_controller_dgemm->active_unit_id_list[i];
	/*curr_sk_dgemm_unit_list[0] = 0;
	curr_sk_dgemm_unit_list[1] = 2;
	curr_sk_dgemm_unit_list[2] = 4;
	curr_sk_dgemm_unit_list[3] = 6;
	curr_sk_dgemm_unit_list[4] = 1;
	curr_sk_dgemm_unit_list[5] = 3;
	curr_sk_dgemm_unit_list[6] = 5;
	curr_sk_dgemm_unit_list[7] = 7;*/

#endif

	pthread_t thread_id[autotune_controller_dgemm->active_unit_num];
	kernel_pthread_wrap_p thread_dev_data[autotune_controller_dgemm->active_unit_num];

	// create a barrier object with a count of autotune_controller_dgemm->active_unit_num + 1
	pthread_barrier_init (&SoftCache_alloc_barrier_dgemm, NULL, autotune_controller_dgemm->active_unit_num + 1);
	for(int d=0; d < LOC_NUM; d++) CoCoEnableLinks(deidxize(d), LOC_NUM);
	for(int d=0; d < autotune_controller_dgemm->active_unit_num; d++){
		if(autotune_controller_dgemm->Subkernels_per_unit_num[d] == 0 )
			error("CoCoPeLiaDgemm: Leftover autotune_controller_dgemm->Subkernels_per_unit_num[%d] == 0", d);

		thread_dev_data[d] = (kernel_pthread_wrap_p) malloc(sizeof(struct kernel_pthread_wrap));
		thread_dev_data[d]->dev_id = autotune_controller_dgemm->active_unit_id_list[d];


		thread_dev_data[d]->SubkernelNumDev = autotune_controller_dgemm->Subkernels_per_unit_num[d];
		thread_dev_data[d]->SubkernelListDev = (Subkernel**) malloc(thread_dev_data[d]->SubkernelNumDev*sizeof(Subkernel*));
		
		// TODO: Check 
		for(int skitt = 0; skitt < autotune_controller_dgemm->Subkernels_per_unit_num[d]; skitt++)
			thread_dev_data[d]->SubkernelListDev[skitt] = Subkernel_list_dgemm[autotune_controller_dgemm->Subkernels_per_unit_list[d][skitt]];
		//int skitt = 0;
		//for(int offset = 0; offset < KGridSz_dgemm; offset++)
		//	for(int skitt2 = 0; skitt2 < autotune_controller_dgemm->Subkernels_per_unit_num[d]; skitt2+=KGridSz_dgemm)
		//	thread_dev_data[d]->SubkernelListDev[skitt++] = Subkernel_list_dgemm[autotune_controller_dgemm->Subkernels_per_unit_list[d][skitt2 + offset]];

		s = pthread_create(&thread_id[d], &attr,
                                  &PARALiADgemmAgentVoid, thread_dev_data[d]);

	}
	pthread_barrier_wait (&SoftCache_alloc_barrier_dgemm);

	//A_asset->DrawTileMap();
	//B_asset->DrawTileMap();
	//C_asset->DrawTileMap();

	for(int d=0; d < autotune_controller_dgemm->active_unit_num; d++){
		s = pthread_join(thread_id[d], &res);
		if (s != 0) error("PARALiADgemm: pthread_join failed with exit value %d", s);
		//free(res);      /* Free memory allocated by thread */
	}
#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	fprintf(stderr, "Fire and gather pthreads for all devices -> t_parallel_fire = %lf ms\n", cpu_timer*1000);
	cpu_timer = csecond() - cpu_timer;
#endif
	C_asset->SyncTileMap();
	int prev_dev = CoCoPeLiaGetDevice();

	for(int i=0; i<LOC_NUM;i++){
		CoCoPeLiaSelectDevice(deidxize(i));
		CoCoSyncCheckErr();
	}
	CoCoPeLiaSelectDevice(prev_dev);
#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	fprintf(stderr, "Synced result -> t_exec_full = %lf ms\n", cpu_timer*1000);
	fprintf(stderr, "t_predicted for T=%zu was %.2lf ms : %lf percentile error\n", T, autotune_controller_dgemm->pred_t*1000,
	(autotune_controller_dgemm->pred_t==0)? 0.0: (autotune_controller_dgemm->pred_t - cpu_timer )/autotune_controller_dgemm->pred_t*100);
	cpu_timer = csecond();
#endif

#ifdef STEST
	STEST_print_SK(thread_dev_data, gemm_entry_ts, autotune_controller_dgemm->active_unit_num);
#endif

#ifdef TTEST
	HopMemcpyPrint();
	//n_HopMemcpyPrint();
#endif

#ifdef DDEBUG
  A_asset->DrawTileMap();
  B_asset->DrawTileMap();
  C_asset->DrawTileMap();
#endif

#ifdef CDEBUG
	for(int i=0; i<LOC_NUM;i++) Global_Buffer_2D[i]->draw_buffer(true,true,true);
#endif

#ifndef BUFFER_REUSE_ENABLE
	for(int i = 0 ; i < LOC_NUM; i++){
		delete Global_Buffer_2D[i];
		Global_Buffer_2D[i] = NULL;
	}
#else
	for(int i=0; i<LOC_NUM;i++) Global_Buffer_2D[i]->reset(false,true);
#endif

#ifndef BACKEND_RES_REUSE_ENABLE
	for(int i=0; i<autotune_controller_dgemm->active_unit_num;i++) CoCoPeLiaFreeResources(autotune_controller_dgemm->active_unit_id_list[i]);
#endif

#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	fprintf(stderr, "Invalidate caches -> t_invalidate = %lf ms\n", cpu_timer*1000);
	cpu_timer = csecond();
#endif

	for(int i=0; i<Subkernel_num_dgemm; i++) delete Subkernel_list_dgemm[i];
	//delete [] Subkernel_list_dgemm;

#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	fprintf(stderr, "Freed Subkernels -> t_invalidate = %lf ms\n", cpu_timer*1000);
	cpu_timer = csecond();
#endif

	A_asset->DestroyTileMap();
	B_asset->DestroyTileMap();
	C_asset->DestroyTileMap();

#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	fprintf(stderr, "Destroyed Tilemaps -> t_invalidate = %lf ms\n", cpu_timer*1000);
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
	fprintf(stderr, "Unregistering assets -> t_unpin = %lf ms\n", cpu_timer*1000);
#endif

#ifdef DEBUG
	fprintf(stderr, "<-----|\n");
#endif
#ifdef TEST
	fprintf(stderr, "<-----|\n");
#endif

	reuse_model_flag = 1;
	predef_controller_dgemm = NULL;
	// Better not return our global to the user, he can accidentally do stuff to it.
	ATC_p result = new ATC();
	result->mimic_ATC(autotune_controller_dgemm);
	return result;
}

/// A modification of PARALiADgemm but with given parameters (mainly for performance/debug purposes)
ATC_p PARALiADgemmControled(char TransA,  char TransB, long int M, long int N, long int K, double alpha, double* A, long int ldA,
		double* B, long int ldB, double beta, double* C, long int ldC, ATC_p predef_controller){
	if (predef_controller == NULL){
		warning("Calling PARALiADgemmControled with empty controller -> falling back to full autotune version \'PARALiADgemm\'\n");
		return PARALiADgemm(TransA, TransB,  M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);
	}
	predef_controller_dgemm = predef_controller;
	return PARALiADgemm(TransA, TransB,  M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);
}
