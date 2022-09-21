///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief Some CUDA function calls with added error-checking
///
#ifndef AUTOTUNING_RUNTIME_H
#define AUTOTUNING_RUNTIME_H

#include <cstdio>
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
	HETERO_REUSE = 9,
	HETERO_BIDIRECTIONAL = 10,
	FULL_OVERLAP = 11,
	NO_OVERLAP = 12
};
const char* printModel(ModelType mode);

enum ProblemType{
	BLAS1 = 0,
	BLAS2 = 1,
	BLAS3 = 2
};
const char* printProblem(ProblemType problem);

typedef struct flagParams{
	char TransA;
	char TransB;
	int incx;
	int incy;
	//TODO: Add all flags used in BLAS, only applicable initialized/used for each routine.
}* flagParams_p;

typedef class Modeler{
	public:
		Vdata_p V; /// The CoCoPeLia Modeling data struct.
		CoModel_p link[LOC_NUM], revlink[LOC_NUM];  /// All Transfer Links from and to the modeler's unit_id.
		const char* func; /// The BLAS function name string, used for any routine-specific calls/initalizations.
		ProblemType problem; /// The BLAS function level, used for translating GPUexec_model_ptr and get-ers.
		void* GPUexec_model_ptr;

		//double Ker_pot;
		long int D1, D2, D3;
		flagParams_p flags;
		int unit_id;

/********************** Initialization/Modification ***************************/
		Modeler(int dev_id, const char* func_name, void* func_data);
		~Modeler();
/******************************************************************************/
/**************************** Helper Fuctions *********************************/
		void print(); /// print the characteristics of a modeler to stderr
		long int getMinT();
		long int getMaxT();
		long int getSKNum(int T);
		long int getGPUexecLines();
		long int getGPUexecElem(int idx);
/******************************************************************************/
/************************ Prediction Functions ********************************/
		double predict(ModelType mode, long int T = -1, int used_devs = -1, int* used_dev_ids = NULL,
			double* used_dev_relative_scores = NULL);	///  Mode-Generalized prediction wrapper
/******************************************************************************/

}* MD_p;


typedef class ATC{
	public:
		long int T; /// The tiling size used for 1D/2D Data split to tiles.
		int active_unit_num; /// The number of units that will be used in the involving operation.
		int* active_unit_id_list;	/// The list of ids of said units.
		double* active_unit_score; /// The 'score' of each said units relative to the total task completion.
		double pred_t; /// The predicted seconds the whole operation will require using the above parameters.

		long int subkernel_num; /// The number of subkernels.
		int* Subkernels_per_unit_num; /// The number of subkernels derived from a unit's score that that unit unit will fire.
		int** Subkernels_per_unit_list; /// The sk_id ids of said sub-kernels, IF they are predefined and not dynamic.

		long long cache_limit; /// The 'cache' size allocation limit for all devices in bytes, IF any.
		MD_p* unit_modeler_list; /// The list of modelers for ALL available units (e.g. LOC_NUM)
/********************** Initialization/Modification ***************************/
	ATC();	/// Constructor
	~ATC(); /// Destructor
	void reset(); /// Resets controller to default parameters (untuned).
	void mimic_ATC(class ATC* other_ATC); /// Copy all characteristics of another autotune controller, using its modelers.
	void update_sk_num(long long int subkernel_num_in); /// Updates the autotuner for a given number of subkernels.
	/// Each device gets 1/num_devices Subkernels without acounting for their size or location
	/// A classic round-robin distribution without acounting for their size or location
	/// A round-robin distribution of chunk_size subkernels each time (if possible)
	/// Reverse subkernel order per device after chunk_size distribution.
	/// 2D block cyclic distribution
	void distribute_subkernels(int D1GridSz, int D2GridSz, int D3GridSz);
/******************************************************************************/
/****************************** Autotuning ************************************/
	double autotune_problem(const char* routine_name, void* initial_problem_wrap); 	/// Fire the autotuner for a given problem.
	void init_modelers(const char* routine_name, void* initial_problem_wrap);
	double optimize_tile(); ///  Predicts the best tile T for a multi-unit problem
	double optimize_tile_CoCoPeLia(int model_idx, ModelType mode); /// Predicts T using CoCoPeLia models for a single unit, defined at CoCoPeLiaModelWrap.cpp
	double optimize_split();
	void normalize_split();
	void update_link_weights(); /// Update link weights. Function defined at TransferLinks.cpp for wholeness.
/******************************************************************************/
/**************************** Helper Fuctions *********************************/
	void print(); /// Print the characteristics of the autotune controller to stderr
	const char* print_csv(); /// Print the basic characteristics of the autotune controller to a string in csv-friendly format (X,Y,...)
/******************************************************************************/

}* ATC_p;

extern double link_cost[LOC_NUM][LOC_NUM];
extern double link_used[LOC_NUM][LOC_NUM];


#ifdef ENABLE_TRANSFER_HOPS
#define MAX_ALLOWED_HOPS 2
#define HOP_PENALTY 0.5
extern short link_hop_num[LOC_NUM][LOC_NUM];
extern short link_hop_route[LOC_NUM][LOC_NUM][MAX_ALLOWED_HOPS];
extern double link_cost_hop[LOC_NUM][LOC_NUM];
#endif

#endif
