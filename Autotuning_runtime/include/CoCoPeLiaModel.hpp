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
	COCOPELIA_HETERO_REUSE = 8,
	COCOPELIA_HETERO_BIDIRECTIONAL = 9
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
	long int T = 0;
	short dev_num = -1;
	short dev_ids[LOC_NUM];
	int Subkernels_per_dev[LOC_NUM];
	int** Subkernel_dev_id_list;
	long long cache_limit = 0;
}* CoControl_p;
#endif

typedef struct tunableParams{
	long int T;
	double* rel_dev_score;
	double pred_t;
}* tunableParams_p;

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

tunableParams_p CoCoAutotuneParameters(const char* routine_name, void* initial_problem_wrap,
  CoControl_p* autotuned_vals_p, CoCoModel_p* glob_model, CoControl_p predef_vals, short reuse_model_flag);

short* CoCoPeLiaDeviceSelectBest(short used_devs, short avail_devs, short* avail_dev_ids,
	CoCoModel_p* avail_dev_model_list);

tunableParams_p CoCoPeLiaModelMultidevOptimizeTileAndSplit(short used_devs, short* used_dev_ids,
	CoCoModel_p* dev_model_list);

tunableParams_p CoCoPeLiaModelMultidevOptimizeSplit(short used_devs, short* used_dev_ids,
	CoCoModel_p* dev_model_list, long int T);

/// A naive prediction of the full-overlap (~unreachable) performance of a modeled routine
double CoCopeLiaPredictFullOverlap(CoCoModel_p model);

/// A naive prediction of the zero-overlap (~worst case) performance of a modeled routine
double CoCopeLiaPredictZeroOverlap(CoCoModel_p model);

///  Mode-Generalized prediction wrapper
double CoCoPeLiaModelPredict(CoCoModel_p model, long int T, ModelType mode);

/// Mode-Generalized prediction wrapper for heterogeneous
double CoCoPeLiaModelPredictHetero(CoCo_model* model, short used_devs,
	short* used_dev_ids, double* used_dev_relative_scores, long int T, ModelType mode);

double WerkhovenModelPredictWrapper(CoCo_model* model, long int T, short t_exec_method);
double CoCopeLiaPredictBaseline(CoCoModel_p model, long int T);
double CoCopeLiaPredictDataLoc(CoCoModel_p model, long int T);
double CoCopeLiaPredictBidirectional(CoCoModel_p model, long int T);
double CoCopeLiaPredictReuse(CoCoModel_p model, long int T);
double CoCopeLiaPipelineEmulate(CoCoModel_p model, long int T);

long int CoCopeLiaMinT(CoCoModel_p model);
long int CoCopeLiaMaxT(CoCoModel_p model);

CoCoModel_p CoCoPeLiaTileModelInit(short dev_id, const char* func_name, void* func_data);
///  Predicts Best tile size for 3-way overlaped execution time for BLAS3 2-dim blocking.
tunableParams_p CoCoPeLiaModelOptimizeTile(CoCoModel_p model, ModelType mode);

tunableParams_p tunableParamsInit();
const char* printTunableParams(tunableParams_p params);

/// Each device gets 1/num_devices Subkernels without acounting for their size or location
void CoCoDistributeSubkernelsNaive(CoControl_p autotune_vals, tunableParams_p best_pred_p,
	int Subkernel_num);

/// A classic round-robin distribution without acounting for their size or location
void CoCoDistributeSubkernelsRoundRobin(CoControl_p autotune_vals, tunableParams_p best_pred_p,
	int Subkernel_num);

/// A round-robin distribution of chunk_size subkernels each time (if possible)
void CoCoDistributeSubkernelsRoundRobinChunk(CoControl_p autotune_vals, tunableParams_p best_pred_p,
	int Subkernel_num, int Chunk_size);

/// A round-robin distribution of chunk_size subkernels each time (if possible).
/// Reverse subkernel order per device after distribution.
void CoCoDistributeSubkernelsRoundRobinChunkReverse(CoControl_p autotune_vals, tunableParams_p best_pred_p,
	int Subkernel_num, int Chunk_size);

void CoCoDistributeSubkernels2DBlockCyclic(CoControl_p autotune_vals,
	  tunableParams_p pred_p, int D1GridSz, int D2GridSz, int D3GridSz);
#endif
