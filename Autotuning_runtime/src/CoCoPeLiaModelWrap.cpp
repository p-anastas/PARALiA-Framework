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
#include "CoCoPeLiaModel.hpp"
#include "CoCoPeLiaModelLvl3.hpp"
#include "CoCoPeLiaModelLvl1.hpp"
#include "unihelpers.hpp"
#include "Werkhoven.hpp"

double CoCopeLiaPredictFullOverlap(CoCoModel_p model)
{
	switch(model->problem){
		case BLAS1:
			return CoCopeLiaPredictFullOverlapBLAS1(model);
		case BLAS2:
			error("CoCopeLiaPredictFullOverlap: BLAS 2 Not implemented\n");
			return 0;
		case BLAS3:
			return CoCopeLiaPredictFullOverlapBLAS3(model);
		default:
			error("CoCopeLiaPredictFullOverlap: Invalid Problem %s", printProblem(model->problem));
	}
	return 0;
}

double CoCopeLiaPredictZeroOverlap(CoCoModel_p model)
{
	switch(model->problem){
		case BLAS1:
			return CoCopeLiaPredictZeroOverlapBLAS1(model);
		case BLAS2:
			error("CoCopeLiaPredictZeroOverlap: BLAS 2 Not implemented\n");
			return 0;
		case BLAS3:
			return CoCopeLiaPredictZeroOverlapBLAS3(model);
		default:
			error("CoCopeLiaPredictZeroOverlap: Invalid Problem %s", printProblem(model->problem));
	}
	return 0;
}


double CoCopeLiaPredictBaseline(CoCoModel_p model, long int T)
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


double CoCopeLiaPredictDataLoc(CoCoModel_p model, long int T)
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
double CoCopeLiaPredictBidirectional(CoCoModel_p model, long int T)
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


double CoCopeLiaPredictReuse(CoCoModel_p model, long int T)
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
double CoCopeLiaPipelineEmulate(CoCoModel_p model, long int T)
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


double CoCopeLiaPredictReuseHetero(CoCoModel_p model, short used_devs, int* used_dev_ids,
	double* used_dev_relative_scores, long int T)
{
	switch(model->problem){
		case BLAS1:
			//return CoCopeLiaPredictReuseHeteroBLAS1(model, T);
			error("CoCopeLiaPredictReuseHetero: BLAS 1 Not implemented\n");
		case BLAS2:
			error("CoCopeLiaPredictReuseHetero: BLAS 2 Not implemented\n");
			return 0;
		case BLAS3:
			return CoCopeLiaPredictReuseHeteroBLAS3(model, used_devs, used_dev_ids,
				used_dev_relative_scores, T);
		default:
			error("CoCopeLiaPredictReuseHetero: Invalid Problem %s", printProblem(model->problem));
	}
	return 0;
}

double CoCopeLiaPredictBidirectionalHetero(CoCoModel_p model, short used_devs, int* used_dev_ids,
	double* used_dev_relative_scores, long int T)
{
	switch(model->problem){
		case BLAS1:
			return CoCopeLiaPredictBidirectionalHeteroBLAS1(model, used_devs, used_dev_ids,
				used_dev_relative_scores, T);
		case BLAS2:
			error("CoCopeLiaPredictBidirectionalHetero: BLAS 2 Not implemented\n");
		case BLAS3:
			error("CoCopeLiaPredictBidirectionalHetero: BLAS 3 Not implemented\n");
		default:
			error("CoCopeLiaPredictBidirectionalHetero: Invalid Problem %s", printProblem(model->problem));
	}
	return 0;
}

long int CoCopeLiaMinT(CoCoModel_p model){
	switch(model->problem){
		case BLAS1:
			return CoCopeLiaMinAllowedTBLAS1(model);
		case BLAS2:
			error("CoCopeLiaMinT: BLAS 2 Not implemented\n");
			return 0;
		case BLAS3:
			return CoCopeLiaMinAllowedTBLAS3(model);
		default:
			error("CoCopeLiaMinT: Invalid Problem %s", printProblem(model->problem));
	}
	return 0;
}

long int CoCopeLiaMaxT(CoCoModel_p model){
	switch(model->problem){
		case BLAS1:
			return CoCopeLiaMaxAllowedTBLAS1(model);
		case BLAS2:
			error("CoCopeLiaMaxT: BLAS 2 Not implemented\n");
			return 0;
		case BLAS3:
			return CoCopeLiaMaxAllowedTBLAS3(model);
		default:
			error("CoCopeLiaMaxT: Invalid Problem %s", printProblem(model->problem));
	}
	return 0;
}

int CoCoPeLiaGPUexecGetLines(CoCoModel_p model){
	switch(model->problem){
		case BLAS1:
			return ((GPUexec1Model_p)model->GPUexec_model_ptr)->lines;
		case BLAS2:
			return ((GPUexec2Model_p)model->GPUexec_model_ptr)->lines;
		case BLAS3:
			return ((GPUexec3Model_p)model->GPUexec_model_ptr)->lines;
		default:
			error("CoCoPeLiaGPUexecGetLines: Invalid Problem %s", printProblem(model->problem));
	}
	return 0;
}

long int CoCoPeLiaGPUexecGetElem(CoCoModel_p model, int idx){
	switch(model->problem){
		case BLAS1:
			return ((GPUexec1Model_p)model->GPUexec_model_ptr)->T_lookup_buf[idx];
		case BLAS2:
			return ((GPUexec2Model_p)model->GPUexec_model_ptr)->T_lookup_buf[idx];
		case BLAS3:
			return ((GPUexec3Model_p)model->GPUexec_model_ptr)->T_lookup_buf[idx];
		default:
			error("CoCoPeLiaGPUexecGetLines: Invalid Problem %s", printProblem(model->problem));
	}
	return 0;
}

long int CoCopeLiaGetSKNum(CoCoModel_p model, int T){
	switch(model->problem){
		case BLAS1:
			error("CoCopeLiaGetSKNum: BLAS 1 Not implemented\n");
		case BLAS2:
			error("CoCopeLiaGetSKNum: BLAS 2 Not implemented\n");
			return 0;
		case BLAS3:
			return CoCopeLiaGetSKNumBLAS3(model, T);
		default:
			error("CoCopeLiaGetSKNum: Invalid Problem %s", printProblem(model->problem));
	}
	return 0;
}

void CoCoPeLiaNormalizeSplit(double* score_list, int list_len){
	for (int i = 0; i < list_len; i++)
	if(score_list[i] < NORMALIZE_NEAR_SPLIT_LIMIT){
		for (int j = 0; j < list_len; j++)
			if (i != j) score_list[j] = score_list[j]/(1 - score_list[i]);
		score_list[i] = 0;
	}
	else {
	      int flag_normalize[list_len] = {0}, normalize_num = 1;
	      double normalize_sum = score_list[i];
	      flag_normalize[i] = 1;
	      for (int j = i + 1; j < list_len; j++)
				if(abs(score_list[i] - score_list[j])/score_list[i]/list_len < NORMALIZE_NEAR_SPLIT_LIMIT){
		//printf("Normalizing score_list[%d] and score_list[%d] to %lf\n", i, j, (score_list[i] + score_list[j])/2);
					//score_list[j] = score_list[i] = (score_list[i] + score_list[j])/2;
		flag_normalize[j] = 1;
		normalize_sum+=score_list[j];
		normalize_num++;
	      }
	      for (int j = i ; j < list_len; j++) if(flag_normalize[j]) score_list[j] = normalize_sum/normalize_num;
	}
}

///  Initializes the model for gemm
CoCoModel_p CoCoPeLiaTileModelInit(short dev_id, const char* func, void* func_data){
	short lvl = 3;
#ifdef DEBUG
	lprintf(lvl-1, "|-----> CoCoPeLiaTileModelInit(dev_id=%d,func=%s)\n", dev_id, func);
#endif
	CoCoModel_p out_model = (CoCoModel_p) malloc(sizeof(struct CoCo_model));
	for(int idx = 0; idx < LOC_NUM; idx++){
		short dev_idx_id = (idx == LOC_NUM - 1)? -1 : idx;
		if(dev_idx_id!= dev_id){
			out_model->link[idx] = CoModel_init(dev_id, dev_idx_id);
			out_model->revlink[idx] = CoModel_init(dev_idx_id, dev_id);
		}
		else out_model->link[idx] = out_model->revlink[idx] = CoModel_init_local(dev_id);
	}
	out_model->V = (Vdata_p) malloc(sizeof(struct V_struct));
	out_model->flags = (flagParams_p) malloc(sizeof(struct flagParams));
	out_model->dev_id = dev_id;

	if ( !strcmp(func, "Daxpy") || !strcmp(func, "Saxpy")) out_model->problem = BLAS1;
	else if (0) out_model->problem = BLAS2;
	else if ( !strcmp(func, "Dgemm") || !strcmp(func, "Sgemm")) out_model->problem = BLAS3;
	else error("CoCoPeLiaModelInit: Problem type for '%s' func not integrated\n", func);

	switch(out_model->problem){
		case BLAS1:
			out_model->GPUexec_model_ptr = (void*) GPUexec1Model_init(dev_id, func);
			return CoCoModelFuncInitBLAS1(out_model, dev_id, func, func_data);
			break;
		case BLAS2:
			error("CoCoPeLiaTileModelInit: GPUexec2Model_init Not Implemented\n");
		 	out_model->GPUexec_model_ptr = (void*) GPUexec2Model_init(dev_id, func);
			break;
		case BLAS3:
			out_model->GPUexec_model_ptr = (void*) GPUexec3Model_init(dev_id, func);
			return CoCoModelFuncInitBLAS3(out_model, dev_id, func, func_data);
			break;
		default:
			error("CoCoPeLiaModelInit: Unreachable default reached\n");
			return NULL;
	}
}

double CoCoPeLiaModelPredictHetero(CoCo_model* model, int used_devs, int* used_dev_ids, double* used_dev_relative_scores, long int T, ModelType mode){
	switch(mode){
		case COCOPELIA_HETERO_BIDIRECTIONAL:
			return CoCopeLiaPredictBidirectionalHetero(model, used_devs, used_dev_ids, used_dev_relative_scores, T);
		case COCOPELIA_HETERO_REUSE:
			return CoCopeLiaPredictReuseHetero(model, used_devs, used_dev_ids, used_dev_relative_scores, T);

		default:
			error("CoCoPeLiaModelPredictHetero: Invalid mode %s\n", printModel(mode));
	}
}

double CoCoPeLiaModelPredict(CoCo_model* model, long int T, ModelType mode){
	switch(mode){
		case WERKHOVEN:
			return WerkhovenModelPredictWrapper(model, T, 0);
		case WERKHOVEN_DATALOC:
			return WerkhovenModelPredictWrapper(model, T, 1);
		case WERKHOVEN_LOOKUP_EXEC_TILES:
			return WerkhovenModelPredictWrapper(model, T, 2);
		case COCOPELIA_BASELINE:
			return CoCopeLiaPredictBaseline(model, T);
		case COCOPELIA_DATALOC:
			return CoCopeLiaPredictDataLoc(model, T);
		case COCOPELIA_BIDIRECTIONAL:
			return CoCopeLiaPredictBidirectional(model, T);
		case COCOPELIA_REUSE:
			return CoCopeLiaPredictReuse(model, T);
		case COCOPELIA_PIPELINE_EMULATE:
			return CoCopeLiaPipelineEmulate(model, T);
		default:
			error("CoCoPeLiaModelPredict: Invalid mode %s", printModel(mode));
	}
}

///  Itterates through benchmarked values for T and chooses the Tbest that minimizes total time.
double CoCoPeLiaModelOptimizeTile(ATC_p autotune_controller, CoCoModel_p model, ModelType mode){
	short lvl = 3;
	double timer = csecond();
#ifdef DEBUG
	lprintf(lvl-1, "|-----> CoCoPeLiaModelOptimizeTile(mode=%d)\n", mode);
#endif
#ifdef TEST
	lprintf(lvl-1, "|-----> CoCoPeLiaModelOptimizeTile\n");
#endif
	//TODO: Naive naive naive! Should replace with something better at some point.
	long int min_T = 0, max_allowed_T = 0, ctr = 0;
	max_allowed_T = CoCopeLiaMaxT(model);
	min_T = CoCopeLiaMinT(model);
	if (min_T > max_allowed_T){
		autotune_controller->T = max_allowed_T;
		// FIXME: Undefined performance for tiles < than the smaller microbenchmark
		autotune_controller->pred_t = 0;
		timer = csecond() - timer;
		return timer;
	}
	double temp_t, min_t = CoCoPeLiaModelPredict(model, CoCoPeLiaGPUexecGetElem(model, 0), mode);
	long int prev_trial_T = 0;
	if(min_t < 0) error("CoCoPeLiaModelOptimizeTile: First value in DM results in negative prediction");
	for (ctr = 1 ; ctr < CoCoPeLiaGPUexecGetLines(model); ctr++){
		long int trial_T = CoCoPeLiaGPUexecGetElem(model, ctr);
		if (trial_T > max_allowed_T) break;
		if (trial_T == prev_trial_T) continue;
		temp_t = CoCoPeLiaModelPredict(model, trial_T, mode);

		//fprintf(stderr, "Checking T = %ld\n : t = %lf ms\n", trial_T, temp_t*1000);
		if (temp_t >= 0 && temp_t < min_t ){
			min_t = temp_t;
			min_T = trial_T;
		}
		prev_trial_T = trial_T;
	}
	autotune_controller->T = min_T;
	autotune_controller->pred_t = min_t;
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
	case (COCOPELIA_HETERO_BIDIRECTIONAL):
		return "COCOPELIA_HETERO_BIDIRECTIONAL";
	case (COCOPELIA_HETERO_REUSE):
		return "COCOPELIA_HETERO_REUSE";
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

void PARALiaRemoveUselessDevices(ATC_p autotune_controller){
	for (int i = 0; i < autotune_controller->active_unit_num; i++)
	if(autotune_controller->active_unit_score[i] == 0.0 ) {
		for (int i_move = i; i_move < autotune_controller->active_unit_num - 1; i_move++){
			autotune_controller->active_unit_score[i_move] = autotune_controller->active_unit_score[i_move+1];
			autotune_controller->active_unit_id_list[i_move] = autotune_controller->active_unit_id_list[i_move+1];
		}
		i--;
		autotune_controller->active_unit_num--;
	}
}

void PARALIA_translate_unit_ids(int case_id, int* active_unit_num_p, int* active_unit_id_list){
	int mask;
	*active_unit_num_p = 0;
	for (int mask_offset = 0; mask_offset < LOC_NUM; mask_offset++){
		mask =  1 << mask_offset;
		if (case_id & mask){
			active_unit_id_list[*active_unit_num_p] = deidxize(mask_offset);
			(*active_unit_num_p)++;
			//lprintf(0, "PARALIA_translate_unit_ids(case_id = %d): mask = %d -> Adding unit %d to available\n",
			//	case_id, mask, deidxize(mask_offset));
		}
	}
}

double PARALiaAutotuneParameters(ATC_p autotune_controller, const char* routine_name, void* initial_problem_wrap,
  CoCoModel_p* glob_model, short reuse_model_flag, short autotune_controller_is_limited){
	short lvl = 3;
	double cpu_timer = csecond();
#ifdef PDEBUG
	lprintf(lvl, "CoCoAutotuneParameters(%p, %s, &%p, %p, %d)", autotune_controller, routine_name,
		initial_problem_wrap, glob_model, reuse_model_flag);
#endif

for (int dev_id_idx =0; dev_id_idx < LOC_NUM; dev_id_idx++){
	if(!reuse_model_flag){
		free(glob_model[dev_id_idx]);
		glob_model[dev_id_idx] = NULL;
		if(!autotune_controller_is_limited) autotune_controller->reset();
		else warning("PARALiaAutotuneParameters: Called with reuse_model_flag = 0 and \
			autotune_controller_is_limited = 1, controller will not be reset.");
	}
	else{
//#ifdef PDEBUG
		lprintf(lvl, "CoCoAutotuneParameters() reuse_model_flag = 1, Reusing Previous autotune_controller\n");
//#endif
		return  csecond() - cpu_timer;
	}
	if(glob_model[dev_id_idx] == NULL){
		glob_model[dev_id_idx] = CoCoPeLiaTileModelInit(deidxize(dev_id_idx), routine_name, initial_problem_wrap);
	}
}

	int autotune_eval_devices = 0;
	if (autotune_controller->active_unit_num > 0){
		if (autotune_controller->active_unit_id_list){
//#ifdef DEBUG
		lprintf(lvl, "Running on %d devices with dev_ids=[ ", autotune_controller->active_unit_num);
		for (int i =0; i < autotune_controller->active_unit_num; i++) fprintf(stderr, "%d ", autotune_controller->active_unit_id_list[i]);
		fprintf(stderr, "]\n");
//#endif
		}
		else{
			autotune_eval_devices = 1;
//#ifdef DEBUG
		lprintf(lvl, "Running on %d devices with tunable dev_ids\n", autotune_controller->active_unit_num);
//#endif
		}
	}
	else{
#ifdef ENABLE_CPU_WORKLOAD
		autotune_controller->active_unit_num = LOC_NUM;
#else
		autotune_controller->active_unit_num = LOC_NUM - 1;
#endif
		autotune_eval_devices = 1;
		for (int i =0; i < autotune_controller->active_unit_num; i++)
			autotune_controller->active_unit_id_list[i] = deidxize(i);
	}

	if (autotune_eval_devices){
		ATC_p temp_controller = new ATC();
		temp_controller->mimic_ATC(autotune_controller);
		int explored_cases = pow(2,LOC_NUM);
		autotune_controller->pred_t = DBL_MAX;
		int max_unit_num = autotune_controller->active_unit_num, initial_T = autotune_controller->T;
		double tile_selection_t = 0, split_selection_t = 0;
		for (int case_id = 1; case_id < explored_cases; case_id++){
				PARALIA_translate_unit_ids(case_id, &temp_controller->active_unit_num, temp_controller->active_unit_id_list);
				if(temp_controller->active_unit_num > max_unit_num) continue;
				if(initial_T <= 0) tile_selection_t += PARALiaMultidevOptimizeTile(temp_controller, glob_model);
				split_selection_t += PARALiaMultidevOptimizeSplit(temp_controller, glob_model);
#ifdef PDEBUG
				lprintf(lvl, "Recallibrating T = %ld prediction (2nd pass)\n", temp_controller->T);
#endif
				tile_selection_t += PARALiaMultidevOptimizeTile(temp_controller, glob_model);
#ifdef PDEBUG
				lprintf(lvl, "Recallibrating Split = [ ");
				for (int i =0; i < temp_controller->active_unit_num; i++) lprintf(0, "%.5lf ", temp_controller->active_unit_score[i]);
				lprintf(0, "] prediction (2nd pass)\n");
#endif
				split_selection_t += PARALiaMultidevOptimizeSplit(temp_controller, glob_model);

				if (temp_controller->pred_t < autotune_controller->pred_t) autotune_controller->mimic_ATC(temp_controller);
#ifdef PDEBUG
						lprintf(0, "==============================================\n");
						lprintf(0, "Autotune devices (iter %d): Tuning for active_unit_id_list = [ ", case_id);
						for (int i =0; i < temp_controller->active_unit_num; i++) lprintf(0, "%d ", temp_controller->active_unit_id_list[i]);
						lprintf(0, "] -> pred_t = %lf, best_pred_t = %lf\n", temp_controller->pred_t,  autotune_controller->pred_t);
						lprintf(0, "==============================================\n");
#endif
		}
	/// try all device combinations and their potential performance (possibly considerable preproc)
	/*int candidate_unit_id_list_of_lists[autotune_controller->active_unit_num][autotune_controller->active_unit_num], best_dev_num = 0;
	double candidate_pred_t[autotune_controller->active_unit_num],
		candidate_unit_id_score_of_lists[autotune_controller->active_unit_num][autotune_controller->active_unit_num];
	int candidate_T[autotune_controller->active_unit_num] = {0};
	for (int candidate_unit_num = 0; candidate_unit_num < autotune_controller->active_unit_num; candidate_unit_num++){
		else {
			used_devs_0idx = autotune_controller->active_unit_num-1;
			dev_ids_list_of_lists[used_devs_0idx] = autotune_controller->active_unit_id_list;
		}*/
	}
	else{
		double tile_selection_t = 0, split_selection_t = 0;
		if(autotune_controller->T <= 0) tile_selection_t = PARALiaMultidevOptimizeTile(autotune_controller, glob_model);
		split_selection_t = PARALiaMultidevOptimizeSplit(autotune_controller, glob_model);
#ifdef PDEBUG
		lprintf(lvl, "Recallibrating T = %ld prediction (2nd pass)\n", autotune_controller->T);
#endif
		tile_selection_t += PARALiaMultidevOptimizeTile(autotune_controller, glob_model);
#ifdef PDEBUG
		lprintf(lvl, "Recallibrating Split = [ ");
		for (int i =0; i < autotune_controller->active_unit_num; i++) fprintf(stderr, "%.5lf ", autotune_controller->active_unit_score[i]);
		lprintf(0, "] prediction (2nd pass)\n");
#endif
		split_selection_t += PARALiaMultidevOptimizeSplit(autotune_controller, glob_model);
	}

	PARALiaRemoveUselessDevices(autotune_controller);

	cpu_timer = csecond() - cpu_timer;

	if(!reuse_model_flag){
			lprintf(0, "====================================\n");
			lprintf(0, "CoCoAutotuneParameters: Autotuning complete-> t_autotune = %lf ms\n", cpu_timer*1000);
			lprintf(0, "autotune_controller: T=%ld,  active_unit_num=%d, Problem split = %s -> %s : t_pred = %lf ms\n",
				autotune_controller->T, autotune_controller->active_unit_num, printlist<int>(autotune_controller->active_unit_id_list, autotune_controller->active_unit_num),
				printlist<double>(autotune_controller->active_unit_score, autotune_controller->active_unit_num), autotune_controller->pred_t*1000);
			lprintf(0, "====================================\n");
	}

	return cpu_timer;
}

/*
short* CoCoPeLiaDeviceSelectBest(short used_devs, short avail_devs, short* avail_dev_ids,
	CoCoModel_p* avail_dev_model_list){
	short lvl = 3;
#ifdef PDEBUG
	lprintf(lvl, "====================================\n");
#endif
	if(used_devs > avail_devs)
		error("CoCoPeLiaDeviceSelectBest: used_devs(%d) > avail_devs(%d)\n", used_devs, avail_devs);
	short* used_dev_ids = (short*) malloc(sizeof(short)* used_devs), dev_ctr = 0;
	if(used_devs == avail_devs) for(int idx = 0; idx < used_devs; idx++) used_dev_ids[idx] = avail_dev_ids[idx];
	else{
		short checked[avail_devs];
    for(int idx = 0; idx < avail_devs; idx++) checked[idx] = 0;
		while(dev_ctr < used_devs){
			double best_score = 0;
			int best_idx = -1;
			for(int idx = 0; idx < avail_devs; idx++){
				double temp_score;
				if (!strcmp(DEV_COST_FUNC, "FULL-OVERLAP"))
					temp_score = CoCopeLiaPredictFullOverlap(avail_dev_model_list[idx]);
				else if (!strcmp(DEV_COST_FUNC, "ZERO-OVERLAP"))
					temp_score = CoCopeLiaPredictZeroOverlap(avail_dev_model_list[idx]);
				else error("CoCoPeLiaDeviceSelectBest: Unknown DEV_COST_FUNC = %s\n", DEV_COST_FUNC);
				if (temp_score != 0) temp_score = 1/temp_score;
				if(!checked[idx] && temp_score > best_score){
					best_score = temp_score;
					best_idx = idx;
				}
			}
			if(best_idx == -1) error("CoCoPeLiaDeviceSelectBest: best_idx not found in full itteration\n");
			checked[best_idx] = 1;
			used_dev_ids[dev_ctr] = deidxize(best_idx);
#ifdef PDEBUG
			lprintf(lvl, "Best_score(dev_ctr=%d, dev_id = %d) = %lf\n", dev_ctr, deidxize(best_idx), best_score);
#endif
			dev_ctr++;
		}
	}
#ifdef PDEBUG
	lprintf(lvl, "Best %d devices: [ ", used_devs);
	for (int i =0; i < used_devs; i++) fprintf(stderr, "%d ", used_dev_ids[i]);
	lprintf(0, "]\n");
#endif
	return used_dev_ids;
}*/

double PARALiaMultidevOptimizeTile(ATC_p autotune_controller, CoCoModel_p* dev_model_list){
	short lvl = 3;
	double timer = csecond();
#ifdef PDEBUG
lprintf(lvl, "PARALiaMultidevOptimizeTile( autotune_controller{ T=%ld, active_unit_num=%d, Problem split = %s -> %s : t_pred = %lf ms}, dev_model_list =%p)\n",
	autotune_controller->T, autotune_controller->active_unit_num, printlist<int>(autotune_controller->active_unit_id_list, autotune_controller->active_unit_num),
	printlist<double>(autotune_controller->active_unit_score, autotune_controller->active_unit_num), autotune_controller->pred_t*1000, dev_model_list);
#endif
	int best_idx = -1;
	double temp_score = 0;

	if (autotune_controller->active_unit_num <= 0)
	error("PARALiaMultidevOptimizeTile: Called with active_unit_num = %d\n", autotune_controller->active_unit_num);
	// All models are created for the same initial problem, therefore min_T, max_T and dimension related values are the same.
	short first_model_idx = idxize(autotune_controller->active_unit_id_list[0]);
	CoCoModel_p model = dev_model_list[first_model_idx];

	long int min_T = 0, max_allowed_T = 0, ctr = 0;
	max_allowed_T = CoCopeLiaMaxT(model);
	min_T = CoCopeLiaMinT(model);
#ifdef PDEBUG
	lprintf(lvl, "min_T = %ld, max_allowed_T = %ld\n",
		min_T, max_allowed_T);
#endif
	if (min_T >= max_allowed_T){
		autotune_controller->T = max_allowed_T;
//#ifdef PDEBUG
		lprintf(lvl, "min_T = %ld > max_allowed_T = %ld: returning T = %ld\n",
			min_T, max_allowed_T, max_allowed_T);
//#endif
		timer = csecond() - timer;
		return timer;
	}

/// TODO: Maybe define the lowest suggested Tile sizes?

/// Not very sciency, but also not a priority currently
#define MAX_DESIRED_SK_DEV 128
#define MIN_DESIRED_SK_DEV 32
	long int max_sk = MAX_DESIRED_SK_DEV*autotune_controller->active_unit_num,
					 min_sk = MIN_DESIRED_SK_DEV*autotune_controller->active_unit_num;
#ifdef PDEBUG
	lprintf(lvl, "Desired max_sk = %ld, min_sk = %ld\n",
		max_sk, min_sk);
#endif
	long int sk_num_of_max_T = CoCopeLiaGetSKNum(model, max_allowed_T), sk_num_of_min_T = CoCopeLiaGetSKNum(model, min_T);
#ifdef PDEBUG
	lprintf(lvl, "sk_num_of_max_T = %ld, sk_num_of_min_T = %ld\n",
		sk_num_of_max_T, sk_num_of_min_T);
#endif
	if (sk_num_of_max_T > min_sk) min_sk = sk_num_of_max_T;
	if (sk_num_of_min_T < max_sk) max_sk = sk_num_of_min_T;
	if (min_sk >= max_sk) min_sk = max_sk;
#ifdef PDEBUG
	lprintf(lvl, "min_sk = %ld, max_sk = %ld\n",
		min_sk, max_sk);
#endif

	//if (min_sk == max_sk && ()) min_sk = max_sk;
/// Assume having no remainders in SK-spliting is the most important.
/// TODO: For future could also take into account specific "good" tiles per device.

	int D1_dummy = model->D1, D2_Dummy = (model->D2 == -1)? D1_dummy: model->D2, D3_Dummy = (model->D3 == -1)? D1_dummy: model->D3;
	int candidate_T = gcd(D1_dummy, D2_Dummy, D3_Dummy);
	if(candidate_T == 1) candidate_T = std::min(D1_dummy, std::min(D2_Dummy, D3_Dummy));
	while(CoCopeLiaGetSKNum(model, candidate_T) < min_sk) candidate_T/=2;
	while(CoCopeLiaGetSKNum(model, candidate_T) > max_sk) candidate_T++;
	if (CoCopeLiaGetSKNum(model, candidate_T) < min_sk){
//#ifdef PDEBUG
		lprintf(lvl, "Default GCD method for obtaining T failed, resorting in secondary method - performance might degrade\n");
//#endif
			while(CoCopeLiaGetSKNum(model, candidate_T) < min_sk) candidate_T/=2;
	}

	autotune_controller->T = candidate_T;
#ifdef PDEBUG
	lprintf(lvl, "====================================\n");
	lprintf(lvl, "Predict T=%d : No t_pred provided\n", autotune_controller->T);
#endif
	timer = csecond() - timer;
#ifdef TEST
	lprintf(lvl, "Tile selection time:%lf ms\n", timer*1000);
	lprintf(lvl-1, "<-----|\n");
#endif
	return timer;
}

double PARALiaMultidevOptimizeSplit(ATC_p autotune_controller, CoCoModel_p* dev_model_list){
	short lvl = 3;
	double timer = csecond();
#ifdef PDEBUG
	lprintf(lvl, "PARALiaMultidevOptimizeSplit( autotune_controller{ T=%ld, active_unit_num=%d, Problem split = %s -> %s : t_pred = %lf ms}, dev_model_list =%p)\n",
		autotune_controller->T, autotune_controller->active_unit_num, printlist<int>(autotune_controller->active_unit_id_list, autotune_controller->active_unit_num),
		printlist<double>(autotune_controller->active_unit_score, autotune_controller->active_unit_num), autotune_controller->pred_t*1000, dev_model_list);
#endif

	if (autotune_controller->active_unit_num <= 0)
	error("PARALiaMultidevOptimizeSplit: Called with active_unit_num = %d\n", autotune_controller->active_unit_num);
	// All models are created for the same initial problem, therefore min_T, max_T and dimension related values are the same.
	short first_model_idx = idxize(autotune_controller->active_unit_id_list[0]);
	CoCoModel_p model = dev_model_list[first_model_idx];

	long int max_allowed_T = 0, ctr = 0;
	max_allowed_T = CoCopeLiaMaxT(model);
#ifdef PDEBUG
		lprintf(lvl, "max_allowed_T = %ld\n", max_allowed_T);
#endif
	if (autotune_controller->T > max_allowed_T)
		error("PARALiaMultidevOptimizeSplit: Give T = %ld > max_allowed_T = %ld\n", autotune_controller->T, max_allowed_T);

	double temp_t, min_overlap_t = 10000000, temp_score = 0;

	for(int idx = 0; idx < autotune_controller->active_unit_num; idx++){
		int cur_dev_idx = idxize(autotune_controller->active_unit_id_list[idx]);
		autotune_controller->active_unit_score[idx] = CoCopeLiaPredictFullOverlap(dev_model_list[cur_dev_idx]);
		if (autotune_controller->active_unit_score[idx] != 0) autotune_controller->active_unit_score[idx] = 1/autotune_controller->active_unit_score[idx];
		else warning("PARALiaMultidevOptimizeSplit: active_unit_score[%d] == 0\n", idx);
		temp_score+= autotune_controller->active_unit_score[idx];
	}
	for(int idx = 0; idx < autotune_controller->active_unit_num; idx++){
		autotune_controller->active_unit_score[idx] /= temp_score;
#ifdef PDEBUG
		lprintf(lvl, "Calculating Relative score for unit_id = %d (idx = %d ): autotune_controller->active_unit_score = %e\n",
				autotune_controller->active_unit_id_list[idx], idx, autotune_controller->active_unit_score[idx]);
#endif
	}
	double temp_overlap_t = 0;
	for(int idx = 0; idx < autotune_controller->active_unit_num; idx++){
		int cur_dev_id = autotune_controller->active_unit_id_list[idx], cur_dev_idx = idxize(cur_dev_id);
		model = dev_model_list[cur_dev_idx];
		ModelType used_model;
		switch(model->problem){
			case BLAS1:
				used_model = COCOPELIA_HETERO_BIDIRECTIONAL;
				break;
			case BLAS2:
				used_model = COCOPELIA_HETERO_BIDIRECTIONAL;
				break;
			case BLAS3:
				used_model = COCOPELIA_HETERO_REUSE;
				break;
			default:
				error("PARALiaMultidevOptimizeTileAndSplit:\
				model->problem switch default reached\n");
		}
		temp_t = CoCoPeLiaModelPredictHetero(model, autotune_controller->active_unit_num, autotune_controller->active_unit_id_list, autotune_controller->active_unit_score, autotune_controller->T, used_model);
		if(temp_t > 0) temp_overlap_t = fmax(temp_overlap_t, temp_t);
		else error("CoCoPeLiaModelPredictHetero(%p(dev_id = %d, (idx = %d )), T = %ld): negative prediction temp_t = %lf\n",
			model, cur_dev_id, cur_dev_idx, autotune_controller->T, temp_t);
#ifdef PDEBUG
		lprintf(lvl, "CoCoPeLiaModelPredictHetero(%p) for dev_id = %d (idx = %d ) with T = %ld: temp_overlap_t = %lf, temp_t = %lf\n",
			model, cur_dev_id, cur_dev_idx, autotune_controller->T, temp_overlap_t, temp_t);
#endif
	}
	CoCoPeLiaNormalizeSplit(autotune_controller->active_unit_score, autotune_controller->active_unit_num);
	autotune_controller->pred_t = temp_overlap_t;

#ifdef PDEBUG
	lprintf(lvl, "====================================\n");
	lprintf(lvl, "Best %d percentages : [ ", autotune_controller->active_unit_num);
	for (int i =0; i < autotune_controller->active_unit_num; i++) lprintf(0, "%.5lf ", autotune_controller->active_unit_score[i]);
	lprintf(0, "]\n");
	lprintf(lvl, "====================================\n");
#endif
	timer = csecond() - timer;
#ifdef TEST
	lprintf(lvl, "Optimization time:%lf ms\n", timer*1000);
	lprintf(lvl-1, "<-----|\n");
#endif
	return timer;
}

/* FIXME: DEPRECATED - kept for future model-based T selection update if needed
double PARALiaMultidevOptimizeTile_modelBased(ATC_p autotune_controller, short used_devs, short* used_dev_ids,
	int* dev_idx_ignore, CoCoModel_p* dev_model_list){
	short lvl = 3;
#ifdef DEBUG
	lprintf(lvl-1, "|-----> PARALiaMultidevOptimizeTile_modelBased(used_devs=%d, used_dev_ids= [ ", used_devs);
	for (int i =0; i < used_devs; i++) lprintf(0, "%d ", used_dev_ids[i]);
	lprintf(0, "]\n");
#endif
#ifdef TEST
	double timer = csecond();
#endif
	short first_model_idx = (used_dev_ids[0] == -1) ? LOC_NUM - 1 : used_dev_ids[0];
	CoCoModel_p model = dev_model_list[first_model_idx];
	autotune_controller->active_unit_score = NULL;
	int best_idx = -1;
	double temp_score = 0;

	long int min_T = 0, max_allowed_T = 0, ctr = 0;
	max_allowed_T = CoCopeLiaMaxT(model);
	min_T = CoCopeLiaMinT(model);
#ifdef PDEBUG
		lprintf(lvl, "min_T = %ld, max_allowed_T = %ld\n",
			min_T, max_allowed_T);
#endif
	if (min_T > max_allowed_T){
		outparams->T = max_allowed_T;
		// FIXME: Undefined expected performance for tiles < than the smaller microbenchmark
		outparams->pred_t = 0;
#ifdef PDEBUG
		lprintf(lvl, "min_T = %ld > max_allowed_T = %ld: returning T = %ld",
			min_T, max_allowed_T, max_allowed_T);
#endif
		return outparams;
	}
	double temp_t, min_overlap_t = 10000000;
	long int prev_trial_T = 0;

	int lines = CoCoPeLiaGPUexecGetLines(model);
	for (ctr = 0 ; ctr < lines ; ctr++){
		long int trial_T = CoCoPeLiaGPUexecGetElem(model, ctr);
		if (trial_T > max_allowed_T) break;
		if (trial_T ==  prev_trial_T) continue;

		double temp_overlap_t = 0;
		for(int idx = 0; idx < used_devs; idx++){
			if(dev_idx_ignore[idx]) continue;
			short cur_dev_id = used_dev_ids[idx], cur_dev_idx = (cur_dev_id == -1)? LOC_NUM - 1 : cur_dev_id;
			model = dev_model_list[cur_dev_idx];
			ModelType used_model;
			switch(model->problem){
				case BLAS1:
					used_model = COCOPELIA_BIDIRECTIONAL;
					break;
				case BLAS2:
					used_model = COCOPELIA_BIDIRECTIONAL;
					break;
				case BLAS3:
					used_model = COCOPELIA_REUSE;
					break;
				default:
					error("PARALiaMultidevOptimizeTile_modelBased:\
					model->problem switch default reached\n");
			}
				temp_t = CoCoPeLiaModelPredict(model, trial_T, used_model);
				double imb_time_multiplier = 1.0, reduce_time_multiplier = 1.0;
#ifdef TILE_IMBALANCE_PENALTY
					if (model->D1 != -1 && (model->D1%trial_T)) imb_time_multiplier+=TILE_IMBALANCE_PENALTY;
					if (model->D2 != -1 && (model->D2%trial_T)) imb_time_multiplier+=TILE_IMBALANCE_PENALTY;
					if (model->D3 != -1 && (model->D3%trial_T)) imb_time_multiplier+=TILE_IMBALANCE_PENALTY;
#endif
#ifdef REDUCE_PENALTY
					if ((model->D1/trial_T + ((model->D1%trial_T)? 1 : 0))*(model->D2/trial_T + ((model->D2%trial_T)? 1 : 0))
						*(model->D3/trial_T + ((model->D3%trial_T)? 1 : 0)) % used_devs) reduce_time_multiplier+=REDUCE_PENALTY;
#endif
			temp_t *= imb_time_multiplier *reduce_time_multiplier;
			if(temp_t > 0) temp_overlap_t = fmax(temp_overlap_t, temp_t);
			else error("CoCoPeLiaModelPredict(%p(dev_id = %d, (idx = %d )), trial_T = %ld): negative prediction temp_t = %lf\n",
				model, cur_dev_id, cur_dev_idx, trial_T, temp_t);
#ifdef PDEBUG
			lprintf(lvl, "CoCoPeLiaModelPredict(%p) for dev_id = %d (idx = %d ) with trial_T = %ld: temp_overlap_t = %lf, temp_t = %lf\n",
				model, cur_dev_id, cur_dev_idx, trial_T, temp_overlap_t, temp_t);
#endif
		}
		if (temp_overlap_t < min_overlap_t){
			min_overlap_t = temp_overlap_t;
			min_T = trial_T;
		}
		prev_trial_T = trial_T;
	}
	outparams->T = min_T;
	outparams->pred_t = min_overlap_t;
#ifdef PDEBUG
	lprintf(lvl, "====================================\n");
	lprintf(lvl, "Predict T=%ld : t_pred = %lf\n", outparams->T, outparams->pred_t);
#endif
#ifdef TEST
	timer = csecond() - timer;
	lprintf(lvl, "Tile selection time:%lf ms\n", timer*1000);
	lprintf(lvl-1, "<-----|\n");
#endif
#ifdef DEBUG
	lprintf(lvl, "outparams->T = %ld\n : outparams->pred_t = %lf ms\n", outparams->T, outparams->pred_t);
	lprintf(lvl-1, "<-----|\n");
#endif
	return outparams;
}*/
