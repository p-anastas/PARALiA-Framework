///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief Some CUDA function calls with added error-checking
///
#ifndef COCOPELIA_MODEL_H
#define COCOPELIA_MODEL_H

#include "CoCoPeLiaCoModel.hpp"

// TODO: To avoid mallocs, define a set vec size of 4 (No BLAS has that many data arguments anyway)
typedef struct V_struct{
	// Routine specific
	short numT;
	short dtype_sz;
	short in[4]; // Possibly modified from scalar_nz
	short out[4]; // Possibly modified from scalar_nz

	// Problem specific
	size_t *Dim1[4];
	size_t *Dim2[4];
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
	COCOPELIA_PIPELINE_EMULATE = 7
};

typedef struct tunableParams{
	size_t T;
	double pred_t;
	double cpuRatio;
}* tunableParams_p;

typedef struct flagParams{
	char TransA;
	char TransB;
	//TODO: Add all flags used in BLAS, only applicable initialized/used for each routine.
}* flagParams_p;

typedef struct CoCo_model{
	Vdata_p V;
	CoModel_p h2d;//ComModel_p h2d;
	CoModel_p d2h;//ComModel_p d2h;
	char* func;
	double Ker_pot;
	size_t D1, D2, D3;
	flagParams_p flags;
	void* GPUexec_model_ptr;
	// FIXME: Add cpu_exec prediction

}* CoCoModel_p;

///  Mode-Generalized prediction wrapper
double CoCoPeLiaModelPredict(CoCoModel_p model, size_t T, ModelType mode);

double WerkhovenModelPredictWrapper(CoCo_model* model, size_t T, short t_exec_method);
double CoCopeLiaPredictBaseline(CoCoModel_p model, size_t T);
double CoCopeLiaPredictDataLoc(CoCoModel_p model, size_t T);
double CoCopeLiaPredictBidirectional(CoCoModel_p model, size_t T);
double CoCopeLiaPredictReuse(CoCoModel_p model, size_t T);
double CoCopeLiaPipelineEmulate(CoCoModel_p model, size_t T);

///  Predicts Best tile size for 3-way overlaped execution time for BLAS3 2-dim blocking.
tunableParams_p CoCoPeLiaModelOptimizeTile(CoCoModel_p model, ModelType mode);

/// Choose the best way to approach h2d/d2h overlap
short CoCoModel_choose_transfer_mode3(CoCoModel_p model, size_t T);

///  Predicts Best tile size for 3-way overlaped execution time for BLAS1 1-dim blocking.
size_t CoCoModel_optimize1(CoCoModel_p model);

CoCoModel_p CoCoPeLiaModelInit(short dev_id, char* func, char flag1, char flag2, char flag3, size_t Dim1, size_t Dim2, size_t Dim3, short Loc1, short Loc2, short Loc3, short OutLoc1, short OutLoc2, short OutLoc3, size_t offset1, size_t offset2, size_t offset3);

CoCoModel_p CoCoModel_gemm_init(CoCoModel_p base_model, char TransA, char TransB, size_t M, size_t N, size_t K, short A_loc, short B_loc, short C_loc, short A_out_loc, short B_out_loc, short C_out_loc, size_t ldA, size_t ldB, size_t ldC, short dev_id, char* func);

CoCoModel_p CoCoModel_gemv_init(size_t M, size_t N, short A_loc, short x_loc, short y_loc, short dev_id, char* func);

CoCoModel_p CoCoModel_axpy_init(size_t N, short x_loc, short y_loc, short dev_id, char* func);

const char* printModel(ModelType mode);

tunableParams_p tunableParamsInit();

const char* printTunableParams(tunableParams_p params);

/// Each device gets 1/num_devices Subkernels without acounting for their size or location
void CoCoDistributeSubkernelsNaive(int* Subkernel_dev_id_list,
  int* Subkernels_per_dev, short num_devices, int Subkernel_num);

/// A classic round-robin distribution
void CoCoDistributeSubkernelsRoundRobin(int* Subkernel_dev_id_list,
  int* Subkernels_per_dev, short num_devices, int Subkernel_num);

#endif
