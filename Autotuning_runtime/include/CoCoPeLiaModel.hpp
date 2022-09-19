///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief Some CUDA function calls with added error-checking
///
#ifndef COCOPELIA_MODEL_H
#define COCOPELIA_MODEL_H

#include "CoCoPeLiaCoModel.hpp"
#include "backend_wrappers.hpp"

// TODO: To avoid mallocs, define a set vec size of 4 (No BLAS has that many data arguments anyway)
typedef struct V_struct{
	// Routine specific
	short numT;
	short dtype_sz;
	short in[4]; // Possibly modified from scalar_nz
	short out[4]; // Possibly modified from scalar_nz

	// Problem specific
	long int *Dim1[4];
	long int *Dim2[4];
	short loc[4];
	short out_loc[4];

}* Vdata_p;

enum ModelType{
	WERKHOVEN = 0,
	WERKHOVEN_DATALOC = 1,
	WERKHOVEN_LOOKUP_EXEC_TILES = 2,
	COCOPELIA_BASELINE = 3,
	COCOPELIA_DATALOC = 4,
	COCOPELIA_BIDIRECTIONAL = 5,
	COCOPELIA_REUSE = 6,
	COCOPELIA_PIPELINE_EMULATE = 7,
	// PARALia Model additions/modifications
	COCOPELIA_HETERO_REUSE = 9,
	COCOPELIA_HETERO_BIDIRECTIONAL = 10
};
const char* printModel(ModelType mode);

enum ProblemType{
	BLAS1 = 0,
	BLAS2 = 1,
	BLAS3 = 2
};
const char* printProblem(ProblemType problem);

#ifndef COCONTROL_H
#define COCONTROL_H
typedef struct CoControl{
	long int T = 0; /// The tiling size used for 1D/2D Data split to tiles.
	int active_unit_num = -1; /// The number of units that will be used in the involving operation.
	int* active_unit_id_list;	/// The list of ids of said units.
	double* active_unit_score; /// The 'score' of each said units relative to the total task completion.
	double pred_t; /// The predicted seconds the whole operation will require using the above parameters.

	long int subkernel_num; /// The number of subkernels.
	int* Subkernels_per_dev; /// The number of subkernels derived from a unit's score that that unit unit will fire.
	int** Subkernel_dev_id_list; /// The sk_id ids of said sub-kernels, IF they are predefined and not dynamic.

	long long cache_limit = 0; /// The 'cache' size allocation limit for all devices in bytes, IF any.

}* CoControl_p;
#endif

typedef struct flagParams{
	char TransA;
	char TransB;
	int incx;
	int incy;
	//TODO: Add all flags used in BLAS, only applicable initialized/used for each routine.
}* flagParams_p;

typedef struct CoCo_model{
	Vdata_p V;
	CoModel_p link[LOC_NUM], revlink[LOC_NUM];
	const char* func;
	ProblemType problem;
	//double Ker_pot;
	long int D1, D2, D3;
	flagParams_p flags;
	void* GPUexec_model_ptr;
	int dev_id;

}* CoCoModel_p;

double CoCoAutotuneParameters(CoControl_p autotune_controller, const char* routine_name, void* initial_problem_wrap,
  CoCoModel_p* glob_model, short reuse_model_flag);

double PARALiaMultidevOptimizeTile(CoControl_p autotune_controller, CoCoModel_p* dev_model_list);

double PARALiaMultidevOptimizeSplit(CoControl_p autotune_controller, CoCoModel_p* dev_model_list);

/// A naive prediction of the full-overlap (~unreachable) performance of a modeled routine
double CoCopeLiaPredictFullOverlap(CoCoModel_p model);

/// A naive prediction of the zero-overlap (~worst case) performance of a modeled routine
double CoCopeLiaPredictZeroOverlap(CoCoModel_p model);

///  Mode-Generalized prediction wrapper
double CoCoPeLiaModelPredict(CoCoModel_p model, long int T, ModelType mode);

/// Mode-Generalized prediction wrapper for heterogeneous
double CoCoPeLiaModelPredictHetero(CoCo_model* model, int used_devs,
	int* used_dev_ids, double* used_dev_relative_scores, long int T, ModelType mode);

double WerkhovenModelPredictWrapper(CoCo_model* model, long int T, short t_exec_method);
double CoCopeLiaPredictBaseline(CoCoModel_p model, long int T);
double CoCopeLiaPredictDataLoc(CoCoModel_p model, long int T);
double CoCopeLiaPredictBidirectional(CoCoModel_p model, long int T);
double CoCopeLiaPredictReuse(CoCoModel_p model, long int T);
double CoCopeLiaPipelineEmulate(CoCoModel_p model, long int T);

double PARALiaPredictReuseKernelOverBLAS3(CoCoModel_p model, long int T);

long int CoCopeLiaMinT(CoCoModel_p model);
long int CoCopeLiaMaxT(CoCoModel_p model);

CoCoModel_p CoCoPeLiaTileModelInit(short dev_id, const char* func_name, void* func_data);
///  Predicts Best tile size for 3-way overlaped execution time for BLAS3 2-dim blocking.
double CoCoPeLiaModelOptimizeTile(CoControl_p autotune_controller, CoCoModel_p model, ModelType mode);

/// Each device gets 1/num_devices Subkernels without acounting for their size or location
void CoCoDistributeSubkernelsNaive(CoControl_p autotune_controller, int Subkernel_num);

/// A classic round-robin distribution without acounting for their size or location
void CoCoDistributeSubkernelsRoundRobin(CoControl_p autotune_controller, int Subkernel_num);

/// A round-robin distribution of chunk_size subkernels each time (if possible)
void CoCoDistributeSubkernelsRoundRobinChunk(CoControl_p autotune_controller,
	int Subkernel_num, int Chunk_size);

/// A round-robin distribution of chunk_size subkernels each time (if possible).
/// Reverse subkernel order per device after distribution.
void CoCoDistributeSubkernelsRoundRobinChunkReverse(CoControl_p autotune_controller,
	int Subkernel_num, int Chunk_size);

void CoCoDistributeSubkernels2DBlockCyclic(CoControl_p autotune_controller,
	int D1GridSz, int D2GridSz, int D3GridSz);

extern double link_cost_1D[LOC_NUM][LOC_NUM];
extern double link_cost_2D[LOC_NUM][LOC_NUM];
extern double link_used_1D[LOC_NUM][LOC_NUM];
extern double link_used_2D[LOC_NUM][LOC_NUM];

#ifdef ENABLE_TRANSFER_HOPS
#define MAX_ALLOWED_HOPS 2
#define HOP_PENALTY 0.5
extern short link_hop_num[LOC_NUM][LOC_NUM];
extern short link_hop_route[LOC_NUM][LOC_NUM][MAX_ALLOWED_HOPS];
extern double link_cost_hop_1D[LOC_NUM][LOC_NUM];
extern double link_cost_hop_2D[LOC_NUM][LOC_NUM];
#endif

#endif
