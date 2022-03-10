///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The DGEMM CoCopeLia implementation using the new mission-agent-asset C++ classes.
///

#include "backend_wrappers.hpp"
#include "CoCoPeLiaModel.hpp"
#include "CoCoPeLia.hpp"
#include "unihelpers.hpp"
#include "Asset.hpp"
#include "Subkernel.hpp"
#include "DataCaching.hpp"

#include <pthread.h>
pthread_barrier_t  RunTileMap_sync_barrier;

gemm_backend_in_p initial_gemm = NULL;

CoCoModel_p glob_model_gemm[128] = {NULL};
CoControl_p predef_vals_gemm = NULL;
CoControl_p autotuned_vals = NULL;
int MGridSz = 0, NGridSz = 0, KGridSz = 0;

#include <atomic>
std::atomic<long long> remaining_Subkernels;
Subkernel** Subkernel_list;
int Subkernel_num;
short remove_dev[LOC_NUM] = {0};

int Sk_select_lock = 0;

void CoCoGemmUpdateDevice(Subkernel* ker, short dev_id){
	gemm_backend_in_p ptr_ker_translate = (gemm_backend_in_p) ker->operation_params;
	ker->run_dev_id = ptr_ker_translate->dev_id = dev_id;
	short dev_id_idx = idxize(dev_id);
	if(!ker->WR_first) ptr_ker_translate->beta = 1.0;
	ptr_ker_translate->A = &((Tile2D<VALUE_TYPE>*) ker->TileList[0])->adrs[dev_id_idx];
	ptr_ker_translate->B = &((Tile2D<VALUE_TYPE>*) ker->TileList[1])->adrs[dev_id_idx];
	ptr_ker_translate->C = &((Tile2D<VALUE_TYPE>*) ker->TileList[2])->adrs[dev_id_idx];
	ptr_ker_translate->ldA = ((Tile2D<VALUE_TYPE>*) ker->TileList[0])->ldim[dev_id_idx];
	ptr_ker_translate->ldB = ((Tile2D<VALUE_TYPE>*) ker->TileList[1])->ldim[dev_id_idx];
	ptr_ker_translate->ldC = ((Tile2D<VALUE_TYPE>*) ker->TileList[2])->ldim[dev_id_idx];
}

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

	/// TODO: writeback event logic needs change!

	/// TODO: Update RunTileMaps
	/// gemm_subkernel_data->SubkernelListDev[keri]->update_RunTileMaps();

#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Stream/Lib Handle Initialization(%d): t_resource = %lf ms\n", dev_id, cpu_timer*1000);
	cpu_timer = csecond();
#endif

	pthread_barrier_wait (&RunTileMap_sync_barrier);

#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Wait barrier(%d): t_wb = %lf ms\n", dev_id, cpu_timer*1000);
	cpu_timer = csecond();
#endif

  	CoCoPeLiaRequestMaxBuffer(dev_id, MGridSz* NGridSz + NGridSz*KGridSz + MGridSz*KGridSz,
			autotuned_vals->T*autotuned_vals->T*sizeof(VALUE_TYPE), autotuned_vals->cache_limit);

#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Memory management(%d): t_mem = %lf ms\n", dev_id, cpu_timer*1000);
	cpu_timer = csecond();
#endif

	Subkernel * curr = NULL, *prev = NULL;
#ifdef RUNTIME_SCHEDULER_VERSION
	while (remaining_Subkernels > 0 && !remove_dev[idxize(dev_id)]){
		prev = curr;
		if(prev) prev->sync_request_data();
		while(__sync_lock_test_and_set(&Sk_select_lock, 1));
		if(!remaining_Subkernels){
			__sync_lock_release(&Sk_select_lock);
			break;
		}
		if (!strcmp(DISTRIBUTION, "NAIVE"))
			curr = SubkernelSelectSimple(dev_id, Subkernel_list, Subkernel_num);
		else if (!strcmp(DISTRIBUTION, "NAIVE-NO-WRITE-SHARE"))
			curr = SubkernelSelectNoWriteShare(dev_id, Subkernel_list, Subkernel_num);
		else if (!strcmp(DISTRIBUTION, "MINIMIZE-FETCH"))
			curr = SubkernelSelectMinimizeFetch(dev_id, Subkernel_list, Subkernel_num);
		else if (!strcmp(DISTRIBUTION, "MINIMIZE-FETCH-WRITE-PENALTY"))
			curr = SubkernelSelectMinimizeFetchWritePenalty(dev_id, Subkernel_list, Subkernel_num);
		else if (!strcmp(DISTRIBUTION, "MINIMIZE-FETCH-WRITE-PENALTY-MULTIFETCH-PENALTY"))
			curr = SubkernelSelectMinimizeFetchWritePenaltyMultiFetchPenalty(dev_id, Subkernel_list, Subkernel_num);
		else error("CoCopeLiaDgemm: Unknown Subkernel Distribution %s\n", DISTRIBUTION);
		if (!curr){
			__sync_lock_release(&Sk_select_lock);
//#ifdef DDEBUG
		//lprintf(lvl, "CoCopeLiaDgemmAgentVoid(%d): Got curr = NULL, repeating search\n", dev_id);
//#endif
			continue;
		}
		remaining_Subkernels--;
		gemm_subkernel_data->SubkernelListDev[gemm_subkernel_data->SubkernelNumDev] = curr;
		gemm_subkernel_data->SubkernelNumDev++;
		curr->prev = prev;
		if(prev) prev->next = curr;
		CoCoGemmUpdateDevice(curr, dev_id);
		curr->init_events();
		curr->request_data();
		__sync_lock_release(&Sk_select_lock);
		curr->run_operation();
#ifdef ENABLE_SEND_RECV_OVERLAP
		curr->writeback_data();
#endif
	}
#else
	int scheduled[gemm_subkernel_data->SubkernelNumDev] = {0};
	int remaining_Subkernels_dev = gemm_subkernel_data->SubkernelNumDev;
	while(remaining_Subkernels_dev){
		for (int keri = 0; keri < gemm_subkernel_data->SubkernelNumDev; keri++) if (!scheduled[keri]){
			prev = curr;
			//if(prev) prev->sync_request_data();
			Subkernel * temp = gemm_subkernel_data->SubkernelListDev[keri];
			if (!temp->is_dependency_free()) continue;
			curr = temp;
			scheduled[keri] = 1;
			remaining_Subkernels_dev--;
			curr->prev = prev;
			if(prev) prev->next = curr;
			curr->prepare_launch();
			CoCoGemmUpdateDevice(curr, dev_id);
			curr->init_events();
			curr->request_data();
			curr->run_operation();
#ifdef ENABLE_SEND_RECV_OVERLAP
			curr->writeback_data();
#endif
		}
	}
#endif

#ifndef ENABLE_SEND_RECV_OVERLAP
	for (int keri = 0; keri < gemm_subkernel_data->SubkernelNumDev; keri++)
		gemm_subkernel_data->SubkernelListDev[keri]->writeback_data();
#endif

	CoCoSyncCheckErr();
#ifdef STEST
	for (int keri = 0; keri < gemm_subkernel_data->SubkernelNumDev; keri++){
		double request_t_ms = 0, writeback_t_ms = 0, exec_t_ms = 0;
		for(int idx = 0; idx < gemm_subkernel_data->SubkernelListDev[keri]->TileNum; idx++)
			request_t_ms+= gemm_subkernel_data->SubkernelListDev[keri]->input_timer[idx]->sync_get_time();
		exec_t_ms = gemm_subkernel_data->SubkernelListDev[keri]->operation_timer->sync_get_time();
		for(int idx = 0; idx < gemm_subkernel_data->SubkernelListDev[keri]->TileNum; idx++)
			writeback_t_ms+= gemm_subkernel_data->SubkernelListDev[keri]->output_timer[idx]->sync_get_time();
		lprintf(lvl, "Subkernel(dev=%d,id=%d): Request_t = %lf ms (%3.3lf Gb\s), exec_t = %lf ms  (%3.3lf Gflops\s), writeback_t = %lf ms (%3.3lf Gb\s)\n",
			gemm_subkernel_data->SubkernelListDev[keri]->run_dev_id, gemm_subkernel_data->SubkernelListDev[keri]->id,
			request_t_ms, Gval_per_s(gemm_subkernel_data->SubkernelListDev[keri]->bytes_in, request_t_ms/1000),
			exec_t_ms, Gval_per_s(gemm_subkernel_data->SubkernelListDev[keri]->flops, exec_t_ms/1000),
			writeback_t_ms, Gval_per_s(gemm_subkernel_data->SubkernelListDev[keri]->bytes_out, writeback_t_ms/1000));
	}
	double total_cache_timer = CacheGetTimer(dev_id);
	lprintf(lvl, "Cache requests total timer (%d): t_cache = %lf ms\n" , dev_id, total_cache_timer*1000);
#endif
#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Subkernels complete(%d): t_comp = %lf ms\n" , dev_id, cpu_timer*1000);
#endif

#ifdef DEBUG
	lprintf(lvl-1, "<-----|\n");
#endif
	return NULL;
}

/// A dgemm wrapper including auto-tuning of T and cache_size, as well as device management
CoControl_p CoCopeLiaDgemm(char TransA,  char TransB, size_t M, size_t N, size_t K, VALUE_TYPE alpha, VALUE_TYPE* A, size_t ldA, VALUE_TYPE* B, size_t ldB, VALUE_TYPE beta, VALUE_TYPE* C, size_t ldC)
{
	short lvl = 1;
#ifdef DEBUG
	lprintf(lvl-1, "|-----> CoCopeLiaDgemm(%c,%c,%zu,%zu,%zu,%lf,A=%p(%d),%zu,B=%p(%d),%zu,%lf,C=%p(%d),%zu)\n",
		TransA, TransB, M, N, K, alpha, A, CoCoGetPtrLoc(A), ldA,
		B, CoCoGetPtrLoc(B), ldB, beta, C, CoCoGetPtrLoc(C), ldC);
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

	tunableParams_p best_pred_p = CoCoAutotuneParameters("Dgemm", initial_gemm,
	  &autotuned_vals, glob_model_gemm, predef_vals_gemm, reuse_model_flag);

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

	CoCoUpdateLinkSpeed2D(autotuned_vals, glob_model_gemm);

#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Initializing link values -> t_link_init = %lf ms\n", cpu_timer*1000);
	cpu_timer = csecond();
#endif

	size_t T = autotuned_vals->T;
	/// TODO: Split each asset to Tiles
	A_asset->InitTileMap(T, T);
	B_asset->InitTileMap(T, T);
	C_asset->InitTileMap(T, T);

#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Spliting assets to tiles -> t_tile_init = %lf ms\n", cpu_timer*1000);
	cpu_timer = csecond();
#endif

	Subkernel_list = CoCoAsignTilesToSubkernelsGemm(A_asset, B_asset, C_asset, T,
		&Subkernel_num);
	kernel_pthread_wrap_p SK_wrap = (kernel_pthread_wrap_p) malloc(sizeof(struct kernel_pthread_wrap));
	SK_wrap->SubkernelListDev = Subkernel_list;
	SK_wrap->SubkernelNumDev = Subkernel_num;
	SK_wrap->dev_id = -42;
#ifdef DEBUG
	lprintf(lvl, "Subkernel_num = %d {M,N,K}GridSz = {%d, %d, %d}, autotuned_vals->dev_num = %d\n\n",
		Subkernel_num, MGridSz, NGridSz, KGridSz, autotuned_vals->dev_num);
#endif

#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Subkernel init -> t_subkernel_init = %lf ms\n", cpu_timer*1000);
	cpu_timer = csecond();
#endif

#ifndef RUNTIME_SCHEDULER_VERSION
	autotuned_vals->Subkernel_dev_id_list = (int**) malloc(autotuned_vals->dev_num*sizeof(int*));
	for (int devidx = 0; devidx < autotuned_vals->dev_num; devidx++)
		autotuned_vals->Subkernel_dev_id_list[devidx] = (int*) malloc(Subkernel_num*sizeof(int));
	if (!strcmp(DISTRIBUTION, "ROUND-ROBIN"))
		CoCoDistributeSubkernelsRoundRobin(autotuned_vals, best_pred_p, Subkernel_num);
	else if (!strcmp(DISTRIBUTION, "SPLIT-NAIVE"))
		CoCoDistributeSubkernelsNaive(autotuned_vals, best_pred_p, Subkernel_num);
	else if (!strcmp(DISTRIBUTION, "SPLIT-CHUNKS-ROBIN"))
		CoCoDistributeSubkernelsRoundRobinChunk(autotuned_vals, best_pred_p, Subkernel_num, KGridSz);
	else if (!strcmp(DISTRIBUTION, "SPLIT-CHUNKS-ROBIN-REVERSE"))
		CoCoDistributeSubkernelsRoundRobinChunkReverse(autotuned_vals, best_pred_p, Subkernel_num, KGridSz);
	else if (!strcmp(DISTRIBUTION, "2D-BLOCK-CYCLIC"))
		CoCoDistributeSubkernels2DBlockCyclic(autotuned_vals, best_pred_p, MGridSz, NGridSz, KGridSz);
	else error("CoCopeLiaDgemm: Unknown Subkernel Distribution %s\n", DISTRIBUTION);

#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Subkernel Distribute -> t_subkernel_dist = %lf ms\n", cpu_timer*1000);
	cpu_timer = csecond();
#endif

	int used_devices = 0;
	for (int d = 0 ; d < autotuned_vals->dev_num; d++)
		if(autotuned_vals->Subkernels_per_dev[d] > 0 ) used_devices++;
		else if(autotuned_vals->Subkernels_per_dev[d] < 0 )
			error("CoCoPeLiaDgemm: autotuned_vals->Subkernels_per_dev[%d] = %d\n",
				d, autotuned_vals->Subkernels_per_dev[d]);
		else{
			free(autotuned_vals->Subkernel_dev_id_list[d]);
			for (int d_move = d; d_move < autotuned_vals->dev_num - 1; d_move++){
				autotuned_vals->Subkernels_per_dev[d_move] = autotuned_vals->Subkernels_per_dev[d_move+1];
				autotuned_vals->Subkernel_dev_id_list[d_move] = autotuned_vals->Subkernel_dev_id_list[d_move+1];
				autotuned_vals->dev_ids[d_move] = autotuned_vals->dev_ids[d_move+1];
			}
		}
//#ifdef DEBUG
if(!reuse_model_flag){
		lprintf(0, "used_devices=%d out of selected autotuned_vals->dev_num=%d\n", used_devices, autotuned_vals->dev_num);
		lprintf(0, "====================================\n");
}
//#endif
	autotuned_vals->dev_num = used_devices;

#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Subkernel Correct Split devices -> t_subkernel_split = %lf ms\n", cpu_timer*1000);
	cpu_timer = csecond();
#endif

#else
	remaining_Subkernels = Subkernel_num;
#endif

	s = pthread_attr_init(&attr);
	if (s != 0) error("CoCopeLiaDgemm: pthread_attr_init failed s=%d\n", s);



	pthread_t thread_id[autotuned_vals->dev_num];
	kernel_pthread_wrap_p thread_dev_data[autotuned_vals->dev_num];

	// create a barrier object with a count of autotuned_vals->dev_num + 1
	pthread_barrier_init (&RunTileMap_sync_barrier, NULL, autotuned_vals->dev_num + 1);

	for(int d=0; d < autotuned_vals->dev_num; d++){
		if(best_pred_p->rel_dev_score[d] == 0.0)
			error("CoCopeLiaDgemm: best_pred_p->rel_dev_score[%d] == 0 in final used best_pred_p\n",d);

		// Check/Enable peer access between participating GPUs
		CoCoEnableLinks(d, autotuned_vals->dev_ids, autotuned_vals->dev_num);

		thread_dev_data[d] = (kernel_pthread_wrap_p) malloc(sizeof(struct kernel_pthread_wrap));
		thread_dev_data[d]->dev_id = autotuned_vals->dev_ids[d];

#ifdef RUNTIME_SCHEDULER_VERSION
		thread_dev_data[d]->SubkernelNumDev = 0;
		thread_dev_data[d]->SubkernelListDev = (Subkernel**) malloc(Subkernel_num*sizeof(Subkernel*));
#else
		thread_dev_data[d]->SubkernelListDev = (Subkernel**) malloc(autotuned_vals->Subkernels_per_dev[d]* sizeof(Subkernel*));
		for(int skitt = 0; skitt < autotuned_vals->Subkernels_per_dev[d]; skitt++)
			thread_dev_data[d]->SubkernelListDev[skitt] = Subkernel_list[autotuned_vals->Subkernel_dev_id_list[d][skitt]];

		thread_dev_data[d]->SubkernelNumDev = autotuned_vals->Subkernels_per_dev[d];
#endif

		s = pthread_create(&thread_id[d], &attr,
                                  &CoCopeLiaDgemmAgentVoid, thread_dev_data[d]);

	}
	pthread_barrier_wait (&RunTileMap_sync_barrier);

	//A_asset->DrawTileMap();
	//B_asset->DrawTileMap();
	//C_asset->DrawTileMap();

	for(int d=0; d<autotuned_vals->dev_num;d++){
		s = pthread_join(thread_id[d], &res);
		if (s != 0) error("CoCopeLiaDgemm: pthread_join failed with exit value %d", s);
		//free(res);      /* Free memory allocated by thread */
	}
#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Fire and gather pthreads for all devices -> t_exec_full = %lf ms\n", cpu_timer*1000);
	lprintf(lvl, "t_predicted for T=%zu was %.2lf ms : %lf percentile error\n", T, best_pred_p->pred_t*1000,
	(best_pred_p->pred_t==0)? 0.0: (best_pred_p->pred_t - cpu_timer )/best_pred_p->pred_t*100);
	cpu_timer = csecond();
#endif

#ifdef MULTIDEVICE_REDUCTION_ENABLE
	CoCoReduceSyncThreads();
#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Gathered reduce pthreads for all devices -> t_reduce_extra = %lf ms\n",
		cpu_timer*1000);
	cpu_timer = csecond();
#endif
#endif

#ifdef DDEBUG
  A_asset->DrawTileMap();
  B_asset->DrawTileMap();
	C_asset->DrawTileMap();
	for(int i=0; i<autotuned_vals->dev_num;i++) CachePrint(autotuned_vals->dev_ids[i]);
#endif


#ifndef BUFFER_REUSE_ENABLE
	for(int i=0; i<autotuned_vals->dev_num;i++) CoCopeLiaDevCacheFree(autotuned_vals->dev_ids[i]);
#else
	CoCoPeLiaDevCacheInvalidate(SK_wrap);
#ifdef DDEBUG
	for(int i=0; i<autotuned_vals->dev_num;i++) CachePrint(autotuned_vals->dev_ids[i]);
#endif
#endif

#ifndef BACKEND_RES_REUSE_ENABLE
	for(int i=0; i<autotuned_vals->dev_num;i++) CoCoPeLiaFreeResources(autotuned_vals->dev_ids[i]);
#endif

#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Invalidate caches -> t_invalidate = %lf ms\n", cpu_timer*1000);
	cpu_timer = csecond();
#endif

	for(int i=0; i<Subkernel_num; i++) delete Subkernel_list[i];
	//delete [] Subkernel_list;

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
	return autotuned_vals;
}

/// A modification of CoCopeLiaDgemm but with given parameters (mainly for performance/debug purposes)
CoControl_p CoCopeLiaDgemmControled(char TransA,  char TransB, size_t M, size_t N, size_t K, VALUE_TYPE alpha, VALUE_TYPE* A, size_t ldA, VALUE_TYPE* B, size_t ldB, VALUE_TYPE beta, VALUE_TYPE* C, size_t ldC, CoControl_p predef_control_values){
	if (predef_control_values == NULL) return CoCopeLiaDgemm(TransA, TransB,  M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);
	if (predef_vals_gemm == NULL) predef_vals_gemm = (CoControl_p) malloc(sizeof(struct CoControl));
	predef_vals_gemm->T = predef_control_values->T;
	predef_vals_gemm->dev_num = predef_control_values->dev_num;
	for(int idx =0; idx < LOC_NUM; idx++)
		predef_vals_gemm->dev_ids[idx] = predef_control_values->dev_ids[idx];
	predef_vals_gemm->cache_limit = predef_control_values->cache_limit;
	CoControl_p return_vals = CoCopeLiaDgemm(TransA, TransB,  M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);
	free(predef_vals_gemm);
	predef_vals_gemm = NULL;
	return return_vals;
}
