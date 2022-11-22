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
	FULL_OVERLAP = 9,
	NO_OVERLAP = 10,
	HETERO_REUSE = 11,
	HETERO_BIDIRECTIONAL = 12,
	PARALIA_HETERO_LINK_BASED = 13
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
		long int getFlops();
		long int getSKNum(int T);
		double getGPUexecWatts();
		long int getGPUexecLines();
		long int getGPUexecElem(int idx);
		void getDatalocs(int** dataloc_list_p, int* dataloc_num_p);
		void getWDatalocs(int** dataloc_list_p, int* dataloc_num_p);
/******************************************************************************/
/************************ Prediction Functions ********************************/
		/// Return an optimistic transfer time for a "request_ratio" of the per-unit available data
		/// e.g. for active_unit_num = 3 units and request_ratio = 1.7, 1/1.7 of request_size will be transfered
		/// from the 'closest' unit, 0.7/1.7 from the second closest and 0 from the 3rd.
		double predictBestFriends_t(double request_ratio, long long request_size, int active_unit_num, int* active_unit_id_list);
		/// Predicts the transfer time for request_size bytes if given by the average of all (other) available BWs
		double predictAvgBw_t(long long request_size, int active_unit_num, int* active_unit_id_list);

		double predict(ModelType mode, long int T = -1, int active_unit_num = -1, int* active_unit_id_list = NULL,
			double* active_unit_score = NULL);	///  Mode-Generalized prediction wrapper
/******************************************************************************/

}* MD_p;

#define MAX_ALLOWED_HOPS 1
#define MAX_HOP_ROUTES 1
#define HOP_PENALTY 0.15

typedef class LinkMap{
	public:
		// Empirically obtained values for links if used independently
		double link_lat[LOC_NUM][LOC_NUM] = {{0}};
		double link_bw[LOC_NUM][LOC_NUM] = {{0}};

		// Estimated bandwidth values for links if used silmuntaniously for a given problem
		double link_bw_shared[LOC_NUM][LOC_NUM] = {{0}};
		double link_bw_shared_hops[LOC_NUM][LOC_NUM] = {{0}};

		// Number of current link uses. TODO: For runtime optimization, not implemented
		long long link_uses[LOC_NUM][LOC_NUM] = {{0}};

		// The backend hop route used for each transfer.
		short link_hop_route[LOC_NUM][LOC_NUM][MAX_ALLOWED_HOPS][MAX_HOP_ROUTES] = {{{{0}}}};
		// The number of intermediate hops between unit memories each link utilizes.
		short link_hop_num[LOC_NUM][LOC_NUM] = {{0}};

		// The number of different available routes for each link. TODO: Not implemented
		short link_hop_route_num[LOC_NUM][LOC_NUM] = {{0}};

/********************** Initialization/Modification ***************************/
		LinkMap();
		~LinkMap();
/******************************************************************************/
/**************************** Helper Fuctions *********************************/
		void print_link_bw();
		void print_link_bw_shared();
		void print_link_bw_shared_hops();
/******************************************************************************/
/************************ Class main Functions ********************************/
		void update_link_weights(MD_p* list_of_models, int T);
		void update_link_shared_weights(MD_p* list_of_models,
			int* active_unit_id_list, int active_unit_num);
		void init_hop_routes(MD_p* list_of_models, int* active_unit_id_list, int unit_num);

/******************************************************************************/
/************************ Class ESPA Functions ********************************/

		void ESPA_init(MD_p* list_of_models, int* list_of_units,
			int* list_of_unit_percentages, int unit_num, int init_type);
		void ESPA_init_hop_routes(MD_p* list_of_models, int* list_of_units,
				int* list_of_unit_percentages, int unit_num, int init_type);
/******************************************************************************/
}* LinkMap_p;

typedef class ATC{
	public:
		long int T; /// The tiling size used for 1D/2D Data split to tiles.
		int active_unit_num; /// The number of units that will be used in the involving operation.
		int* active_unit_id_list;	/// The list of ids of said units.
		double* active_unit_score; /// The 'score' of each said units relative to the total task completion.
		double pred_t; /// The predicted seconds the whole operation will require using the above parameters.
		double pred_J; /// The predicted Joules the whole operation will require using the above parameters.
		double power_delay, energy_delay; /// The predicted power and energy delay products using the above parameters.
		long int subkernel_num; /// The number of subkernels.
		int* Subkernels_per_unit_num; /// The number of subkernels derived from a unit's score that that unit unit will fire.
		int** Subkernels_per_unit_list; /// The sk_id ids of said sub-kernels, IF they are predefined and not dynamic.

		long long cache_limit; /// The 'cache' size allocation limit for all devices in bytes, IF any.
		MD_p* unit_modeler_list; /// The list of modelers for ALL available units (e.g. LOC_NUM)
		LinkMap_p linkmap; /// The LinkMap representation of the system memory interconnection.
/********************** Initialization/Modification ***************************/
	ATC();	/// Constructor
	~ATC(); /// Destructor
	void reset(); /// Resets controller to default parameters (untuned).
	int diff_intialized_params_ATC(class ATC* other_ATC); /// Rerurns the number of parameters defined in other_ATC that defer from caller.
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
	double optimize_tile_CoCoPeLia(int model_idx, ModelType mode); /// Predicts T using CoCoPeLia models for a single unit, defined at Model_functions.cpp
	double optimize_split();
	void normalize_split();
/******************************************************************************/
/**************************** Helper Fuctions *********************************/
	void print(); /// Print the characteristics of the autotune controller to stderr
	const char* print_csv(); /// Print the basic characteristics of the autotune controller to a string in csv-friendly format (X,Y,...)
/******************************************************************************/

}* ATC_p;

extern double final_estimated_link_bw[LOC_NUM][LOC_NUM];
extern LinkMap_p final_estimated_linkmap;
#endif
