///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The 3-way concurency overlap prediction models for BLAS
///

#include <stdlib.h>
#include <math.h>

// for DBL_MAX
#include <float.h>

#include "CoCoPeLiaCoModel.hpp"
#include "CoCoPeLiaGPUexec.hpp"
#include "Autotuning_runtime.hpp"
#include "CoCoPeLiaModelLvl3.hpp"
#include "CoCoPeLiaModelLvl1.hpp"
#include "unihelpers.hpp"
#include "Werkhoven.hpp"

double PredictFullOverlap(MD_p model)
{
	switch(model->problem){
		case BLAS1:
			return PredictFullOverlapBLAS1(model);
		case BLAS2:
			error("PredictFullOverlap: BLAS 2 Not implemented\n");
			return 0;
		case BLAS3:
			return PredictFullOverlapBLAS3(model);
		default:
			error("PredictFullOverlap: Invalid Problem %s", printProblem(model->problem));
	}
	return 0;
}

double PredictZeroOverlap(MD_p model)
{
	switch(model->problem){
		case BLAS1:
			return PredictZeroOverlapBLAS1(model);
		case BLAS2:
			error("PredictZeroOverlap: BLAS 2 Not implemented\n");
			return 0;
		case BLAS3:
			return PredictZeroOverlapBLAS3(model);
		default:
			error("PredictZeroOverlap: Invalid Problem %s", printProblem(model->problem));
	}
	return 0;
}


double CoCopeLiaPredictBaseline(MD_p model, long int T)
{
	switch(model->problem){
		case BLAS1:
			//return CoCopeLiaPredictBaselineBLAS1(model, T);
		case BLAS2:
			error("CoCopeLiaPredictBaseline: BLAS 2 Not implemented\n");
			return 0;
		case BLAS3:
			return CoCopeLiaPredictBaselineBLAS3(model, T);
		default:
			error("CoCopeLiaPredictBaseline: Invalid Problem %s", printProblem(model->problem));
	}
	return 0;
}


double CoCopeLiaPredictDataLoc(MD_p model, long int T)
{
	switch(model->problem){
		case BLAS1:
			//return CoCopeLiaPredictDataLocBLAS1(model, T);
		case BLAS2:
			error("CoCopeLiaPredictDataLoc: BLAS 2 Not implemented\n");
			return 0;
		case BLAS3:
			return CoCopeLiaPredictDataLocBLAS3(model, T);
		default:
			error("CoCopeLiaPredictDataLoc: Invalid Problem %s", printProblem(model->problem));
	}
	return 0;
}


///  Predicts 3-way overlaped execution time for BLAS3 Square tilling blocking without data reuse.
double CoCopeLiaPredictBidirectional(MD_p model, long int T)
{
	switch(model->problem){
		case BLAS1:
			//return CoCopeLiaPredictBidirectionalBLAS1(model, T);
		case BLAS2:
			error("CoCopeLiaPredictBidirectional: BLAS 2 Not implemented\n");
			return 0;
		case BLAS3:
			return CoCopeLiaPredictBidirectionalBLAS3(model, T);
		default:
			error("CoCopeLiaPredictBidirectional: Invalid Problem %s", printProblem(model->problem));
	}
	return 0;
}


double CoCopeLiaPredictReuse(MD_p model, long int T)
{
	switch(model->problem){
		case BLAS1:
			//return CoCopeLiaPredictReuseBLAS1(model, T);
		case BLAS2:
			error("CoCopeLiaPredictReuse: BLAS 2 Not implemented\n");
			return 0;
		case BLAS3:
			return CoCopeLiaPredictReuseBLAS3(model, T);
		default:
			error("CoCopeLiaPredictReuse: Invalid Problem %s", printProblem(model->problem));
	}
	return 0;
}


/// TODO: currently d2h overlap is ignored
double CoCopeLiaPipelineEmulate(MD_p model, long int T)
{
	switch(model->problem){
		case BLAS1:
			//return CoCopeLiaPipelineEmulateBLAS1(model, T);
		case BLAS2:
			error("CoCopeLiaPipelineEmulate: BLAS 2 Not implemented\n");
			return 0;
		case BLAS3:
			return CoCopeLiaPipelineEmulateBLAS3(model, T);
		default:
			error("CoCopeLiaPipelineEmulate: Invalid Problem %s", printProblem(model->problem));
	}
	return 0;
}


double PredictReuseHetero(MD_p model, long int T, int used_devs = 1, int* used_dev_ids = NULL,
	double* used_dev_relative_scores = NULL)
{
	switch(model->problem){
		case BLAS1:
			//return CoCopeLiaPredictReuseHeteroBLAS1(model, T);
			error("PredictReuseHetero: BLAS 1 Not implemented\n");
		case BLAS2:
			error("PredictReuseHetero: BLAS 2 Not implemented\n");
			return 0;
		case BLAS3:
			return PredictReuseHeteroBLAS3(model, used_devs, used_dev_ids,
				used_dev_relative_scores, T);
		default:
			error("PredictReuseHetero: Invalid Problem %s", printProblem(model->problem));
	}
	return 0;
}

double PredictBidirectionalHetero(MD_p model, long int T, int used_devs = 1, int* used_dev_ids = NULL,
	double* used_dev_relative_scores = NULL)
{
	switch(model->problem){
		case BLAS1:
			return PredictBidirectionalHeteroBLAS1(model, used_devs, used_dev_ids,
				used_dev_relative_scores, T);
		case BLAS2:
			error("PredictBidirectionalHetero: BLAS 2 Not implemented\n");
		case BLAS3:
			error("PredictBidirectionalHetero: BLAS 3 Not implemented\n");
		default:
			error("PredictBidirectionalHetero: Invalid Problem %s", printProblem(model->problem));
	}
	return 0;
}


///  Itterates through benchmarked values for T and chooses the Tbest that minimizes total time.
double ATC::optimize_tile_CoCoPeLia(int model_idx, ModelType mode){
	short lvl = 3;
	double timer = csecond();
#ifdef DEBUG
	lprintf(lvl-1, "|-----> CoCoPeLiaModelOptimizeTile(mode=%d)\n", mode);
#endif
#ifdef TEST
	lprintf(lvl-1, "|-----> CoCoPeLiaModelOptimizeTile\n");
#endif

	MD_p model = unit_modeler_list[model_idx];

	long int min_T = 0, max_allowed_T = 0, ctr = 0;
	max_allowed_T = model->getMaxT();
	min_T = model->getMinT();
	if (min_T > max_allowed_T){
		T = max_allowed_T;
		// FIXME: Undefined performance for tiles < than the smaller microbenchmark
		pred_t = 0;
		timer = csecond() - timer;
		return timer;
	}
	double temp_t, min_t = model->predict(mode, model->getGPUexecElem(0));
	long int prev_trial_T = 0;
	if(min_t < 0) error("CoCoPeLiaModelOptimizeTile: First value in DM results in negative prediction");
	for (ctr = 1 ; ctr < model->getGPUexecLines(); ctr++){
		long int trial_T = model->getGPUexecElem(ctr);
		if (trial_T > max_allowed_T) break;
		if (trial_T == prev_trial_T) continue;
		temp_t = model->predict(mode, trial_T);

		//fprintf(stderr, "Checking T = %ld\n : t = %lf ms\n", trial_T, temp_t*1000);
		if (temp_t >= 0 && temp_t < min_t ){
			min_t = temp_t;
			min_T = trial_T;
		}
		prev_trial_T = trial_T;
	}
	T = min_T;
	pred_t = min_t;
	timer = csecond() - timer;
#ifdef TEST
	lprintf(lvl, "CoCoPeLiaModelOptimizeTile time:%lf ms\n", timer*1000);
#endif
#ifdef DEBUG
	lprintf(lvl, "T = %ld\n : t_min = %lf ms\n", min_T, min_t*1000);
	lprintf(lvl-1, "<-----|\n");
#endif
	return timer;
}

const char* printModel(ModelType mode){
	switch(mode) {
	case (WERKHOVEN):
		return "WERKHOVEN";
	case (WERKHOVEN_DATALOC):
		return "WERKHOVEN_DATALOC";
	case (WERKHOVEN_LOOKUP_EXEC_TILES):
		return "WERKHOVEN_LOOKUP_EXEC_TILES";
	case (COCOPELIA_BASELINE):
		return "COCOPELIA_BASELINE";
	case (COCOPELIA_DATALOC):
		return "COCOPELIA_DATALOC";
	case (COCOPELIA_BIDIRECTIONAL):
		return "COCOPELIA_BIDIRECTIONAL";
	case (COCOPELIA_REUSE):
		return "COCOPELIA_REUSE";
	case (COCOPELIA_PIPELINE_EMULATE):
		return "COCOPELIA_PIPELINE_EMULATE";
	case (HETERO_BIDIRECTIONAL):
		return "HETERO_BIDIRECTIONAL";
	case (HETERO_REUSE):
		return "HETERO_REUSE";
	case (FULL_OVERLAP):
		return "FULL_OVERLAP";
	case (NO_OVERLAP):
		return "NO_OVERLAP";
	default:
		error("printModel: Invalid mode\n");
	}
}

const char* printProblem(ProblemType problem){
	switch(problem) {
	case (BLAS1):
		return "BLAS1";
	case (BLAS2):
		return "BLAS2";
	case (BLAS3):
		return "BLAS3";
	default:
		error("printProblem: Invalid problem\n");
	}
}
