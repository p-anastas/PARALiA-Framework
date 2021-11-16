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
#include "Operative.hpp"

gemm_backend_in_p initial_gemm = NULL;
cublasHandle_t handle[128] = {NULL};

/// TODO: Works for systems with up to 128 devices, not 'completely' future-proof
BLAS3GPUBufPtr GloBuf[128];
CoCoModel_p glob_model;
struct CoControl predef_vals;
CoControl_p used_vals = NULL;

typedef struct gemm_pthread_wrap{
	short devId;
	Operative** OperativeListDev;
	int operativeNumDev;
}* gemm_pthread_wrap_p;

void CoCopeLiaDgemm_flush_gpu_mem_buf(short dev_id)
{}

void* CoCopeLiaDgemmAgentVoid(void* gemm_pthread_wrapped){
}

Operative** CoCoAsignTilesToOperativesGemm(Asset2D<double>* A_asset, Asset2D<double>* B_asset, Asset2D<double>* C_asset, int T, int* kernelNum){

	short lvl = 4;
	/// Check Assets satisfy GEMM dim criteria
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
	int MGridSz = A_asset->GridSz1, NGridSz = B_asset->GridSz2, KGridSz = A_asset->GridSz2;
#ifdef DEBUG
	lprintf(lvl-1, "|-----> CoCoAsignTilesToOperativesGemm(A_asset,B_asset,C_asset,%d,%d)\n", T, *kernelNum);
	lprintf(lvl,"MgridSz = %d, NgridSz = %d, KgridSz = %d\n", MGridSz, NGridSz, KGridSz);
	lprintf(lvl,"Mlast = %d, Nlast = %d, Klast = %d\n",
	A_asset->Tile_map[A_asset->GridSz1*A_asset->GridSz2-1]->dim1,
	B_asset->Tile_map[B_asset->GridSz1*B_asset->GridSz2-1]->dim2,
	A_asset->Tile_map[A_asset->GridSz1*A_asset->GridSz2-1]->dim2);
#endif

*kernelNum = MGridSz*NGridSz*KGridSz;
Operative** kernels = (Operative**) malloc(*kernelNum*sizeof(Operative*));
int current_ctr = 0;
	for (int mi = 0; mi < MGridSz; mi++)
		for (int ni = 0 ; ni < NGridSz; ni++)
			for (int ki = 0; ki < KGridSz; ki++){
	      current_ctr = mi*NGridSz*KGridSz + ni*KGridSz + ki;
				kernels[current_ctr] = new Operative(3);
				kernels[current_ctr]->TileDimlist[0] = kernels[current_ctr]->TileDimlist[1]
				= kernels[current_ctr]->TileDimlist[2] = 2;
				kernels[current_ctr]->TileDtypeList[0] = kernels[current_ctr]->TileDtypeList[1]
				= kernels[current_ctr]->TileDtypeList[2] = DOUBLE;
				kernels[current_ctr]->TileList[0] = A_asset->getTile(mi,ki);
				kernels[current_ctr]->TileList[1] = B_asset->getTile(ki,ni);
				kernels[current_ctr]->TileList[2] = C_asset->getTile(mi,ni);
				kernels[current_ctr]->operation_params = (void*) malloc(sizeof(struct gemm_backend_in));
				gemm_backend_in_p ptr_ker_translate = (gemm_backend_in_p) kernels[current_ctr]->operation_params;
				ptr_ker_translate->TransA = initial_gemm->TransA;
				ptr_ker_translate->TransB = initial_gemm->TransB;
				ptr_ker_translate->M = ((Tile2D<double>*) kernels[current_ctr]->TileList[0])->dim1;
				ptr_ker_translate->N = ((Tile2D<double>*) kernels[current_ctr]->TileList[1])->dim2;
				ptr_ker_translate->K = ((Tile2D<double>*) kernels[current_ctr]->TileList[0])->dim2;
				ptr_ker_translate->A = NULL;
				ptr_ker_translate->B = NULL;
				ptr_ker_translate->C = NULL;
				ptr_ker_translate->alpha = initial_gemm->alpha;
				if (ki == 0) ptr_ker_translate->beta = initial_gemm->beta;
				else ptr_ker_translate->beta = 1.0;
				ptr_ker_translate->ldA = ((Tile2D<double>*) kernels[current_ctr]->TileList[0])->dim2;
				ptr_ker_translate->ldB = ((Tile2D<double>*) kernels[current_ctr]->TileList[1])->dim2;
				ptr_ker_translate->ldC = ((Tile2D<double>*) kernels[current_ctr]->TileList[2])->dim2;
				ptr_ker_translate->dev_id = initial_gemm->dev_id;
			}

#ifdef DEBUG
	lprintf(lvl-1, "<-----|\n");
#endif
	return kernels;
}

/// A dgemm wrapper including auto-tuning of T and cpu_ratio, as well as device management
CoControl_p CoCopeLiaDgemm(char TransA,  char TransB, size_t M, size_t N, size_t K, double alpha, double* A, size_t ldA, double* B, size_t ldB, double beta, double* C, size_t ldC)
{
	short lvl = 1;
#ifdef DEBUG
	lprintf(lvl-1, "|-----> CoCopeLiaDgemm(%c,%c,%zu,%zu,%zu,%lf,A(%d),%zu,B(%d),%zu,%lf,C(%d),%zu)\n",
		TransA, TransB, M, N, K, alpha, CoCoGetPtrLoc(A), ldA,
		CoCoGetPtrLoc(B), ldB, beta, CoCoGetPtrLoc(C), ldC);
#endif

#ifdef TEST
	lprintf(lvl-1, "|-----> CoCopeLiaDgemm\n");
	double cpu_timer = csecond();
#endif

	int prev_devID;
	cudaGetDevice(&prev_devID);

	if(!initial_gemm) initial_gemm = (gemm_backend_in_p) malloc(sizeof(struct gemm_backend_in));
	initial_gemm->TransA = TransA;
	initial_gemm->TransB = TransB;
	initial_gemm->M = M;
	initial_gemm->N = N;
	initial_gemm->K = K;
	initial_gemm->A = A;
	initial_gemm->B = B;
	initial_gemm->C = C;
	initial_gemm->alpha = alpha;
	initial_gemm->beta = beta;
	initial_gemm->ldA = ldA;
	initial_gemm->ldB = ldB;
	initial_gemm->ldC = ldC;
	initial_gemm->dev_id = -1;

	Asset2D<double>* A_asset, *B_asset, *C_asset;
	/// Prepare Assets in parallel( e.g. initialize asset classes, pin memory with pthreads)
	/// return: A_asset, B_asset, C_asset initialized and pinned
	/// FIXME: high probability the M, N etc dims are reverse (cause of column major stuff)
	{
		A_asset = new Asset2D<double>( A, M, K, ldA);
		B_asset = new Asset2D<double>( B, K, N, ldB);
		C_asset = new Asset2D<double>( B, M, N, ldC);

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
			free(res);      /* Free memory allocated by thread */
		}

#ifdef TEST
		cpu_timer = csecond() - cpu_timer;
		lprintf(lvl, "Preparing assets (parallel with pthreads) -> t_prep = %lf ms\n", cpu_timer*1000);
		cpu_timer = csecond();
#endif
	}

	/// Read predefined values for device selection or use default.
	/// return: num_devices, dev_id initialized, update used_vals
	short num_devices, *dev_id = NULL;
	{
		if (predef_vals.dev_num > 0){
			num_devices = predef_vals.dev_num;
			dev_id = (short*) malloc (num_devices*sizeof(short));
			for (int i =0; i < num_devices; i++) dev_id[i] = predef_vals.dev_ids[i];
#ifdef TEST
			lprintf(lvl, "Running on %d devices with dev_ids=[ ", num_devices);
			for (int i =0; i < num_devices; i++) fprintf(stderr, "%d ", predef_vals.dev_ids[i]);
			fprintf(stderr, "]\n");
#endif
		}
		else if (predef_vals.dev_num == 0) error("CoCopeLiaDgemm: CPU-only version not implemented (why should it?)\n");
		else{
			num_devices = DEV_NUM;
			dev_id = (short*) malloc (num_devices*sizeof(short));
			for (int i =0; i < num_devices; i++) dev_id[i] = i;
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
	{
		if(predef_vals.T <= 0){
			/// For each asset: find datum dimension, taking into account shared dimensions for the problem (e.g. here M, N, K are shared between two matrices each)
			/// 1) Ideally we would want a method to find the optimal Tm, Tn, Tk
			/// 2) Currently only square for 2D (sufficient for CoCoPeLia and BLAS in the general case)
			/// 3) Interesting point for exploration (how to find datum, performance impact etc.)
			/// 4)  Basically its the tile selection part of CoCoPeLia, but for multiple devices.
			/// Use static tiles until implementation is complete.
			T = fmin(1024, fmin(M, fmin(N, K)));
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

	/// TODO: Split each asset to Tiles
	A_asset->InitTileMap(T, T);
	B_asset->InitTileMap(T, T);
	C_asset->InitTileMap(T, T);

	// Here would be the place for Agent distribution to devices.
	// Working implementation similar to old : Each agent works on part of the problem, asigns to operatives subproblems
	// TODO: Idea 1 - Simulate the full execution step to step using thew expected times (instead of runtime scheduler), and asign operatives TO agents respectively to minimize communication.
	int operative_num;
	Operative** Operative_list = CoCoAsignTilesToOperativesGemm(A_asset, B_asset, C_asset, T, &operative_num);

	/// TODO: Split operatives equally on devices. Naive, questinable and kinda pointless, for test purposes only.
	int operatives_per_dev = operative_num/num_devices;

	pthread_attr_t attr;
	int s = pthread_attr_init(&attr);
	if (s != 0) error("CoCopeLiaDgemm: pthread_attr_init failed s=%d\n", s);
	void* res;

	pthread_t thread_id[num_devices];
	gemm_pthread_wrap_p thread_dev_data[num_devices];
	Operative** Operative_list_dev[num_devices];

	for(int i=0; i<num_devices;i++){

		// Check/Enable peer access between participating GPUs
		CoCoEnableLinks(i, dev_id, num_devices);

		thread_dev_data[i] = (gemm_pthread_wrap_p) malloc(sizeof(gemm_pthread_wrap_p*));
		thread_dev_data[i]->devId = dev_id[i];
		thread_dev_data[i]->OperativeListDev = &(Operative_list[i*operatives_per_dev]);
		thread_dev_data[i]->operativeNumDev = operatives_per_dev;
		if (i = num_devices - 1) thread_dev_data[i]->operativeNumDev+= operative_num%num_devices;

		s = pthread_create(&thread_id[i], &attr,
                                  &CoCopeLiaDgemmAgentVoid, thread_dev_data[i]);

	}
	for(int i=0; i<num_devices;i++){
		s = pthread_join(thread_id[i], &res);
		if (s != 0) error("CoCopeLiaDgemm: pthread_join failed with exit value %d", s);
		free(res);      /* Free memory allocated by thread */
	}
	exit(0);

	cudaSetDevice(prev_devID);
#ifdef TEST
	cpu_timer = csecond();
#endif
        A_asset->resetProperties();
        B_asset->resetProperties();
        C_asset->resetProperties();
#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Unregistering matrices -> t_unpin = %lf ms\n", cpu_timer*1000);
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
CoControl_p CoCopeLiaDgemmControled(char TransA,  char TransB, size_t M, size_t N, size_t K, double alpha, double* A, size_t ldA, double* B, size_t ldB, double beta, double* C, size_t ldC, CoControl_p predef_control_values){
	if (predef_control_values == NULL) return CoCopeLiaDgemm(TransA, TransB,  M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);
	predef_vals.T = predef_control_values->T;
	predef_vals.dev_ids = predef_control_values->dev_ids;
	predef_vals.dev_num = predef_control_values->dev_num;
	predef_vals.cpu_ratio = predef_control_values->cpu_ratio;
	return CoCopeLiaDgemm(TransA, TransB,  M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);
}
