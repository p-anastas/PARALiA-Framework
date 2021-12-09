///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The DGEMM CoCopeLia implementation using the new mission-agent-asset C++ classes.
///

#include <cblas.h>

// FIXME: Must remove any calls from this since its the backend of utils (should be wrapped).
//#include "backend_wrappers.hpp"
// FIXME: This header should be "backend_wrappers.hpp" but there is a clash with temp wrappers for Deployment. Must fix when they are removed.
#include "backend_lib_wrappers.hpp"
#include "CoCoPeLiaModel.hpp"
#include "CoCoPeLia.hpp"
#include "unihelpers.hpp"
#include "Asset.hpp"
#include "Subkernel.hpp"
#include "DataCaching.hpp"

gemm_backend_in_p initial_gemm = NULL;

CoCoModel_p glob_model;
struct CoControl predef_vals;
CoControl_p used_vals = NULL;

void CoCoGemmUpdateDevice(Subkernel* ker, short dev_id){
	gemm_backend_in_p ptr_ker_translate = (gemm_backend_in_p) ker->operation_params;
	ker->run_dev_id = ptr_ker_translate->dev_id = dev_id;
	ptr_ker_translate->A = &((Tile2D<VALUE_TYPE>*) ker->TileList[0])->adrs[dev_id];
	ptr_ker_translate->B = &((Tile2D<VALUE_TYPE>*) ker->TileList[1])->adrs[dev_id];
	ptr_ker_translate->C = &((Tile2D<VALUE_TYPE>*) ker->TileList[2])->adrs[dev_id];
	ptr_ker_translate->ldA = ((Tile2D<VALUE_TYPE>*) ker->TileList[0])->ldim[dev_id];
	ptr_ker_translate->ldB = ((Tile2D<VALUE_TYPE>*) ker->TileList[1])->ldim[dev_id];
	ptr_ker_translate->ldC = ((Tile2D<VALUE_TYPE>*) ker->TileList[2])->ldim[dev_id];
}

Subkernel** CoCoAsignTilesToSubkernelsGemm(Asset2D<VALUE_TYPE>* A_asset, Asset2D<VALUE_TYPE>* B_asset, Asset2D<VALUE_TYPE>* C_asset, int T, int* kernelNum){

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
	int MGridSz = A_asset->GridSz1, NGridSz = B_asset->GridSz2, KGridSz = A_asset->GridSz2;
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
	for (int mi = 0; mi < MGridSz; mi++)
		for (int ni = 0 ; ni < NGridSz; ni++)
			for (int ki = 0; ki < KGridSz; ki++){
	      current_ctr = mi*NGridSz*KGridSz + ni*KGridSz + ki;
				kernels[current_ctr] = new Subkernel(3);
				kernels[current_ctr]->TileDimlist[0] = kernels[current_ctr]->TileDimlist[1]
				= kernels[current_ctr]->TileDimlist[2] = 2;
				kernels[current_ctr]->TileList[0] = A_asset->getTile(mi,ki);
				kernels[current_ctr]->TileList[1] = B_asset->getTile(ki,ni);
				kernels[current_ctr]->TileList[2] = C_asset->getTile(mi,ni);
				((Tile2D<VALUE_TYPE>*)kernels[current_ctr]->TileList[2])->writeback = 1;
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
				if (ki == 0) ptr_ker_translate->beta = initial_gemm->beta;
				else ptr_ker_translate->beta = 1.0;
				if (ki == KGridSz - 1) kernels[current_ctr]->writeback_master = 1;
				else kernels[current_ctr]->writeback_master = 0;
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
	lprintf(lvl-1, "|-----> CoCopeLiaDgemmAgentVoid(gemm_subkernel_data: dev_id = %d, SubkernelNumDev = %d)\n",
		dev_id, gemm_subkernel_data->SubkernelNumDev);
#endif
#ifdef TEST
		double cpu_timer = csecond();
#endif

	CoCoPeLiaSelectDevice(dev_id);

	for (int keri = 0; keri < gemm_subkernel_data->SubkernelNumDev; keri++){
		gemm_subkernel_data->SubkernelListDev[keri]->init_events();
		CoCoGemmUpdateDevice(gemm_subkernel_data->SubkernelListDev[keri], dev_id);
		if (!keri) gemm_subkernel_data->SubkernelListDev[keri]->prev = NULL;
		else gemm_subkernel_data->SubkernelListDev[keri]->prev = gemm_subkernel_data->SubkernelListDev[keri-1];
		if(keri==gemm_subkernel_data->SubkernelNumDev - 1) gemm_subkernel_data->SubkernelListDev[keri]->next = NULL;
		else gemm_subkernel_data->SubkernelListDev[keri]->next = gemm_subkernel_data->SubkernelListDev[keri+1];
	}

	/// Only warks assuming the last subkernel writes back
	Event* tmp_writeback;
	for (int keri = gemm_subkernel_data->SubkernelNumDev -1 ; keri >= 0 ; keri--){
		if (gemm_subkernel_data->SubkernelListDev[keri]->writeback_master)
			tmp_writeback = gemm_subkernel_data->SubkernelListDev[keri]->writeback_complete;
		else{
			delete gemm_subkernel_data->SubkernelListDev[keri]->writeback_complete;
			gemm_subkernel_data->SubkernelListDev[keri]->writeback_complete = tmp_writeback;
		}
	}

  CoCoPeLiaRequestBuffer(gemm_subkernel_data);
#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Memory management(%d): t_mem = %lf ms\n", dev_id, cpu_timer*1000);
	cpu_timer = csecond();
#endif
	CoCoPeLiaInitStreams(dev_id);
#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Stream/Lib Handle Initialization(%d): t_resource = %lf ms\n", dev_id, cpu_timer*1000);
	cpu_timer = csecond();
#endif
#ifdef DEBUG
	cudaEvent_t start_firing;
	cudaEventCreateWithFlags(&start_firing, cudaEventDefault);
	cudaEventRecord(start_firing);
#endif

	for (int keri = 0; keri < gemm_subkernel_data->SubkernelNumDev; keri++){
		gemm_subkernel_data->SubkernelListDev[keri]->request_data();
		gemm_subkernel_data->SubkernelListDev[keri]->run_operation();
		if (gemm_subkernel_data->SubkernelListDev[keri]->writeback_master)
			gemm_subkernel_data->SubkernelListDev[keri]->writeback_data();
	}
	CoCoSyncCheckErr();
#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Subkernels complete: t_comp = %lf ms\n" , cpu_timer*1000);
#endif
#ifdef DEBUG //TEST
	cudaEvent_t prev_data_sent = start_firing, prev_exec =  *(cudaEvent_t*) gemm_subkernel_data->SubkernelListDev[0]->data_available->event_backend_ptr;
	for (int keri = 0; keri < gemm_subkernel_data->SubkernelNumDev; keri++){
		float t_send, t_exec, temp, t_gpu_idle = 0;
		cudaEventElapsedTime(&t_send, prev_data_sent,  *(cudaEvent_t*) gemm_subkernel_data->SubkernelListDev[keri]->data_available->event_backend_ptr);
		cudaEventElapsedTime(&temp, prev_exec, *(cudaEvent_t*) gemm_subkernel_data->SubkernelListDev[keri]->operation_complete->event_backend_ptr);
		cudaEventElapsedTime(&t_exec, *(cudaEvent_t*) gemm_subkernel_data->SubkernelListDev[keri]->data_available->event_backend_ptr, *(cudaEvent_t*) gemm_subkernel_data->SubkernelListDev[keri]->operation_complete->event_backend_ptr);
		if (!keri) t_gpu_idle = t_send;
		else if (t_exec <= temp) t_gpu_idle = fmax(0.0, temp - t_exec);
		t_exec = fmin(t_exec, temp);
		lprintf(lvl, "Subkernel(%d): t_h2d = %f ms, t_exec = %f ms, t_gpu_idle = %f ms\n" , keri, t_send, t_exec, t_gpu_idle);
		//CoCoSyncCheckErr();
		prev_data_sent = *(cudaEvent_t*) gemm_subkernel_data->SubkernelListDev[keri]->data_available->event_backend_ptr;
		prev_exec = *(cudaEvent_t*) gemm_subkernel_data->SubkernelListDev[keri]->operation_complete->event_backend_ptr;
	}
#endif
	/// Do this after pthread join to enable other devices
	/// to still read cached data after a device's part is over
	//CoCoPeLiaDevCacheInvalidate(gemm_subkernel_data);
#ifdef DEBUG
	lprintf(lvl-1, "<-----|\n");
#endif
	return NULL;
}

/// A dgemm wrapper including auto-tuning of T and cpu_ratio, as well as device management
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

	int prev_dev_id;
	cudaGetDevice(&prev_dev_id);

	if(!initial_gemm) initial_gemm = (gemm_backend_in_p) malloc(sizeof(struct gemm_backend_in));
	initial_gemm->TransA = TransA;
	initial_gemm->TransB = TransB;
	initial_gemm->M = M;
	initial_gemm->N = N;
	initial_gemm->K = K;
	initial_gemm->A = (void**) &A;
	initial_gemm->B = (void**) &B;
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
	/// FIXME: high probability the M, N etc dims are reverse (cause of column major stuff)
	{
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
	}

	/// Read predefined values for device selection or use default.
	/// return: num_devices, dev_id initialized, update used_vals
	short num_devices = 0, *dev_id = NULL;
	{
		if (predef_vals.dev_num > 0){
			num_devices = predef_vals.dev_num;
			dev_id = (short*) malloc (num_devices*sizeof(short));
			for (int i =0; i < num_devices; i++) dev_id[i] = predef_vals.dev_ids[i];
#ifdef DEBUG
			lprintf(lvl, "Running on %d devices with dev_ids=[ ", num_devices);
			for (int i =0; i < num_devices; i++) fprintf(stderr, "%d ", predef_vals.dev_ids[i]);
			fprintf(stderr, "]\n");
#endif
		}
		else if (predef_vals.dev_num == 0) error("CoCopeLiaDgemm: CPU-only version not implemented (why should it?)\n");
		else{
			num_devices = DEV_NUM;
			dev_id = (short*) malloc (num_devices*sizeof(short));
			for (int i =0; i < num_devices; i++) dev_id[i] = (short) i;
		}

		if(used_vals == NULL) {
			used_vals = (CoControl_p) malloc(sizeof(struct CoControl));
			used_vals->dev_ids = NULL;
		}
		used_vals->dev_num = num_devices;
		if(used_vals->dev_ids != NULL)  free(used_vals->dev_ids);
		used_vals->dev_ids = (int*) malloc(num_devices*sizeof(int));
		for (int d = 0; d< num_devices; d++) used_vals->dev_ids[d] = dev_id[d];
	}

	/// Read predefined values for T or use Tile selection.
	/// return: T size for datum
	size_t T = 256;
	CoCoModel_p model = NULL;
	{
		if(predef_vals.T <= 0){
			/// For each asset: find datum dimension, taking into account shared dimensions for the problem (e.g. here M, N, K are shared between two matrices each)
			/// 1) Ideally we would want a method to find the optimal Tm, Tn, Tk
			/// 2) Currently only square for 2D (sufficient for CoCoPeLia and BLAS in the general case)
			/// 3) Interesting point for exploration (how to find datum, performance impact etc.)
			/// 4)  Basically its the tile selection part of CoCoPeLia, but for multiple devices.

			/// Use static tiles until implementation is complete.
			//T = fmin(1024, fmin(M, fmin(N, K)));

			/// Naive for multiple equivalent devices.
			int slowest_problem_T = fmin(1024, fmin(M, fmin(N, K)));
			double slowest_problem_t = 0;
			tunableParams_p pred_p[num_devices];
			for (int d = 0 ; d < num_devices; d++){
				model = CoCoPeLiaModelInit(dev_id[d], "Dgemm", 'X', TransA, TransB,
					M/num_devices, N, K,
					(CoCoGetPtrLoc(A) == dev_id[d])? 0 : 1, (CoCoGetPtrLoc(B) == dev_id[d])? 0 : 1,
					(CoCoGetPtrLoc(C) == dev_id[d])? 0 : 1, (CoCoGetPtrLoc(A) == dev_id[d])? 0 : 1,
					(CoCoGetPtrLoc(B) == dev_id[d])? 0 : 1, (CoCoGetPtrLoc(C) == dev_id[d])? 0 : 1,
					ldA, ldB, ldC);

#ifdef TEST
				cpu_timer = csecond() - cpu_timer;
				lprintf(lvl, "Model Initialization(dev = %d): t_mod_init = %lf ms\n", dev_id[d], cpu_timer*1000);
				cpu_timer = csecond();
#endif

				pred_p[d] = CoCoPeLiaModelOptimizeTile(model, COCOPELIA_REUSE);
				if (pred_p[d]->pred_t > slowest_problem_t){
					slowest_problem_t = pred_p[d]->pred_t;
					slowest_problem_T = pred_p[d]->T;
				}

#ifdef TEST
				cpu_timer = csecond() - cpu_timer;
				lprintf(lvl, "Model Selected T=%zu with t_predicted = %lf ms : t_mod_opt = %lf ms\n", T, pred_p[d]->pred_t*1000, cpu_timer*1000);
				cpu_timer = csecond();
#endif

			}

			T = slowest_problem_T;
#ifdef TEST
			cpu_timer = csecond() - cpu_timer;
			lprintf(lvl, "Model Selected T=%zu with t_predicted = %lf ms : t_mod_opt = %lf ms\n", T, slowest_problem_t*1000, cpu_timer*1000);
			cpu_timer = csecond();
#endif

#ifdef DEBUG
			lprintf(lvl, "Model Selected T=%zu : t_predicted = %lf ms\n", T, slowest_problem_t*1000);
			lprintf(lvl, "====================================\n");
#endif
		}
		else{
			T = predef_vals.T;
#ifdef DEBUG
			lprintf(lvl, "====================================\n");
			lprintf(lvl, "Using predefined T=%zu\n", T);
			lprintf(lvl, "====================================\n");
#endif
		}
		if(used_vals == NULL) {
			used_vals = (CoControl_p) malloc(sizeof(struct CoControl));
			used_vals->dev_ids = NULL;
		}
		used_vals->T = T;
	}

#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Device/T selection -> t_configure = %lf ms\n", cpu_timer*1000);
	cpu_timer = csecond();
#endif

	/// TODO: Split each asset to Tiles
	A_asset->InitTileMap(T, T);
	B_asset->InitTileMap(T, T);
	C_asset->InitTileMap(T, T);

#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Spliting assets to tiles -> t_tile_init = %lf ms\n", cpu_timer*1000);
	cpu_timer = csecond();
#endif

	// Here would be the place for Agent distribution to devices.
	// Working implementation similar to old : Each agent works on part of the problem, asigns to Subkernels subproblems
	// TODO: Idea 1 - Simulate the full execution step to step using thew expected times (instead of runtime scheduler), and asign Subkernels TO agents respectively to minimize communication.
	int Subkernel_num;
	Subkernel** Subkernel_list = CoCoAsignTilesToSubkernelsGemm(A_asset, B_asset, C_asset, T, &Subkernel_num);
#ifdef DEBUG
	lprintf(lvl, "Subkernel_num = %d, num_devices = %d\n\n", Subkernel_num, num_devices);
#endif
#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Subkernel init -> t_subkernel_init = %lf ms\n", cpu_timer*1000);
	cpu_timer = csecond();
#endif

	/// TODO: Split Subkernels equally on devices. For test purposes only.
	/// FIXME: Never split K between devices, otherwise buffer needed for adding results
	/// and a more complex mechanism
	int Subkernels_per_dev[num_devices], remaining_subkernels = Subkernel_num;
	if (Subkernel_num > num_devices){
		for (int d = 0 ; d < num_devices; d++){
			int remaining_devices = num_devices - d;
			if (d>0) remaining_subkernels = remaining_subkernels - Subkernels_per_dev[d-1];
			Subkernels_per_dev[d] = remaining_subkernels/remaining_devices;
			while (Subkernel_list[Subkernel_num-remaining_subkernels + Subkernels_per_dev[d]-1]-> writeback_master!= 1)
				Subkernels_per_dev[d]++;
			if (Subkernels_per_dev[d] > remaining_subkernels) error("CoCoPeLiaDgemm: Split to devices went terribly wrong\n");
		}
	}
	else{
				num_devices = Subkernel_num;
				for (int d = 0 ; d < num_devices; d++) Subkernels_per_dev[d] = 1;
	}
	pthread_attr_t attr;
	int s = pthread_attr_init(&attr);
	if (s != 0) error("CoCopeLiaDgemm: pthread_attr_init failed s=%d\n", s);
	void* res;
	int used_devices = 0;
	for (int d = 0 ; d < num_devices; d++) if(Subkernels_per_dev[d] > 0) used_devices++;
	pthread_t thread_id[used_devices];
	kernel_pthread_wrap_p thread_dev_data[used_devices];

#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Subkernel Split devices -> t_subkernel_split = %lf ms\n", cpu_timer*1000);
	cpu_timer = csecond();
#endif

	for(int i=0; i<used_devices;i++){

		// Check/Enable peer access between participating GPUs
		CoCoEnableLinks(i, dev_id, num_devices);

		thread_dev_data[i] = (kernel_pthread_wrap_p) malloc(sizeof(struct kernel_pthread_wrap));
		thread_dev_data[i]->dev_id = dev_id[i];

		if (i>0) thread_dev_data[i]->SubkernelListDev = &(Subkernel_list[i*Subkernels_per_dev[i-1]]);
		else thread_dev_data[i]->SubkernelListDev = Subkernel_list;

		thread_dev_data[i]->SubkernelNumDev = Subkernels_per_dev[i];

		s = pthread_create(&thread_id[i], &attr,
                                  &CoCopeLiaDgemmAgentVoid, thread_dev_data[i]);

	}
	for(int i=0; i<used_devices;i++){
		s = pthread_join(thread_id[i], &res);
		if (s != 0) error("CoCopeLiaDgemm: pthread_join failed with exit value %d", s);
		//free(res);      /* Free memory allocated by thread */
	}
#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Fire and gather pthreads for all devices -> t_exec_full = %lf ms\n", cpu_timer*1000);
	cpu_timer = csecond();
#endif

	for(int i=0; i<used_devices;i++) CoCoPeLiaDevCacheInvalidate(thread_dev_data[i]);

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

	cudaSetDevice(prev_dev_id);

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
	return used_vals;
}

/// A modification of CoCopeLiaDgemm but with given parameters (mainly for performance/debug purposes)
CoControl_p CoCopeLiaDgemmControled(char TransA,  char TransB, size_t M, size_t N, size_t K, VALUE_TYPE alpha, VALUE_TYPE* A, size_t ldA, VALUE_TYPE* B, size_t ldB, VALUE_TYPE beta, VALUE_TYPE* C, size_t ldC, CoControl_p predef_control_values){
	if (predef_control_values == NULL) return CoCopeLiaDgemm(TransA, TransB,  M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);
	predef_vals.T = predef_control_values->T;
	predef_vals.dev_ids = predef_control_values->dev_ids;
	predef_vals.dev_num = predef_control_values->dev_num;
	predef_vals.cpu_ratio = predef_control_values->cpu_ratio;
	return CoCopeLiaDgemm(TransA, TransB,  M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);
}
