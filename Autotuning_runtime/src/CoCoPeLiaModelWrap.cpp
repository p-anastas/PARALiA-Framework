///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The 3-way concurency overlap prediction models for BLAS
///

#include <stdlib.h>
#include <math.h>

#include "CoCoPeLiaCoModel.hpp"
#include "CoCoPeLiaGPUexec.hpp"
#include "CoCoPeLiaModel.hpp"
#include "CoCoPeLiaModelLvl3.hpp"
#include "CoCoPeLiaModelLvl1.hpp"
#include "unihelpers.hpp"
#include "Werkhoven.hpp"

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

void CoCoPeLiaRemoveUselessDevices(CoControl_p* autotuned_vals_p, tunableParams_p params){
	for (int i = 0; i < (*autotuned_vals_p)->dev_num; i++)
	if(params->rel_dev_score[i] == 0.0 ) {
		for (int i_move = i; i_move < (*autotuned_vals_p)->dev_num - 1; i_move++){
			params->rel_dev_score[i_move] = params->rel_dev_score[i_move+1];
			(*autotuned_vals_p)->dev_ids[i_move] = (*autotuned_vals_p)->dev_ids[i_move+1];
		}
		i--;
		(*autotuned_vals_p)->dev_num--;
	}
}

tunableParams_p CoCoAutotuneParameters(const char* routine_name, void* initial_problem_wrap,
  CoControl_p* autotuned_vals_p, CoCoModel_p* glob_model, CoControl_p predef_vals, short reuse_model_flag){
	short lvl = 3;
#ifdef DEBUG
	lprintf(lvl, "CoCoAutotuneParameters(%s, &%p, %p, %p, %p, %d)", routine_name, initial_problem_wrap,
	  *autotuned_vals_p, glob_model, predef_vals, reuse_model_flag);
#endif
#ifdef TEST
	double cpu_timer = csecond();
#endif
	if(!reuse_model_flag && *autotuned_vals_p){
#ifdef DEBUG
	lprintf(lvl, "Freeing autotuned_vals because reuse_model_flag = %d\n", reuse_model_flag);
#endif
		free((*autotuned_vals_p));
		*autotuned_vals_p = NULL;
#ifdef DDEBUG
	lprintf(lvl, "Allocating new autotuned_vals\n");
#endif
	}

	if(*autotuned_vals_p == NULL){
		*autotuned_vals_p = (CoControl_p) malloc(sizeof(struct CoControl));
	}
	CoControl_p autotuned_vals = *autotuned_vals_p;
	int autotune_eval_devices = 0;
	if (predef_vals && predef_vals->dev_num > 0){
		autotuned_vals->dev_num = predef_vals->dev_num;
		for (int i =0; i < autotuned_vals->dev_num; i++) autotuned_vals->dev_ids[i] = predef_vals->dev_ids[i];
#ifdef DEBUG
		lprintf(lvl, "Running on %d devices with dev_ids=[ ", autotuned_vals->dev_num);
		for (int i =0; i < autotuned_vals->dev_num; i++) fprintf(stderr, "%d ", autotuned_vals->dev_ids[i]);
		fprintf(stderr, "]\n");
#endif
	}
	else{
#ifdef ENABLE_CPU_WORKLOAD
		autotuned_vals->dev_num = LOC_NUM;
#else
		autotuned_vals->dev_num = DEV_NUM;
#endif
		autotune_eval_devices = 1;
		for (int i =0; i < autotuned_vals->dev_num; i++)
			autotuned_vals->dev_ids[i] = deidxize(i);
	}

	for (int i =0; i < autotuned_vals->dev_num; i++){
		short dev_id_idx = idxize(autotuned_vals->dev_ids[i]);
		if(!reuse_model_flag){
			free(glob_model[dev_id_idx]);
			glob_model[dev_id_idx] = NULL;
		}
		if(glob_model[dev_id_idx] == NULL){
			glob_model[dev_id_idx] = CoCoPeLiaTileModelInit(autotuned_vals->dev_ids[i], routine_name, initial_problem_wrap);
		}
	}

	short *dev_ids[autotuned_vals->dev_num], best_dev_num = 0;
	tunableParams_p pred_p[autotuned_vals->dev_num], best_pred_p = NULL;
  for (int used_devs = 0; used_devs < autotuned_vals->dev_num; used_devs++){
    dev_ids[autotuned_vals->dev_num] = NULL;
    pred_p[autotuned_vals->dev_num] = NULL;
  }
	if(predef_vals && predef_vals->T > 0 && !autotune_eval_devices){
		autotuned_vals->T = predef_vals->T;
		best_pred_p = CoCoPeLiaModelMultidevOptimizeSplit(autotuned_vals->dev_num,
			autotuned_vals->dev_ids, glob_model, autotuned_vals->T);
#ifdef PDEBUG
		lprintf(lvl, "====================================\n");
		lprintf(lvl, "Using predefined T=%ld and dev_num=%d, autotuned split = %s -> %s : t_pred = %lf\n",
			autotuned_vals->T, autotuned_vals->dev_num, printlist<short>(autotuned_vals->dev_ids, autotuned_vals->dev_num),
			printlist<double>(best_pred_p->rel_dev_score, autotuned_vals->dev_num), best_pred_p->pred_t*1000);
		lprintf(lvl, "====================================\n");
#endif
	}
	else if(predef_vals && predef_vals->T > 0 && autotune_eval_devices){
		autotuned_vals->T = predef_vals->T;
		for (int used_devs = 0; used_devs < autotuned_vals->dev_num; used_devs++){
			dev_ids[used_devs] = CoCoPeLiaDeviceSelectBest(used_devs + 1, autotuned_vals->dev_num,
				autotuned_vals->dev_ids, glob_model);
			pred_p[used_devs] = CoCoPeLiaModelMultidevOptimizeSplit(used_devs + 1,
				dev_ids[used_devs], glob_model, autotuned_vals->T);

			if (best_pred_p == NULL){
				best_pred_p = pred_p[used_devs];
				best_dev_num = used_devs + 1;
			}
			else if(best_pred_p->pred_t >= pred_p[used_devs]->pred_t){
				best_pred_p = pred_p[used_devs];
				best_dev_num = used_devs + 1;
			}
		}
		autotuned_vals->T = best_pred_p->T;
		autotuned_vals->dev_num = best_dev_num;
		for (int idx = 0; idx < autotuned_vals->dev_num; idx++)
			autotuned_vals->dev_ids[idx] = dev_ids[autotuned_vals->dev_num - 1][idx];
#ifdef PDEBUG
		lprintf(lvl, "====================================\n");
		lprintf(lvl, "Using predefined T=%ld, autotuned dev_num=%d and split = %s -> %s : t_pred = %lf\n",
			autotuned_vals->T, autotuned_vals->dev_num, printlist<short>(autotuned_vals->dev_ids, autotuned_vals->dev_num),
			printlist<double>(best_pred_p->rel_dev_score, autotuned_vals->dev_num), best_pred_p->pred_t*1000);
		lprintf(lvl, "====================================\n");
#endif
	}
	else if (predef_vals && predef_vals->T <= 0 && !autotune_eval_devices){
		best_pred_p = CoCoPeLiaModelMultidevOptimizeTileAndSplit(autotuned_vals->dev_num,
			autotuned_vals->dev_ids, glob_model);
		autotuned_vals->T = best_pred_p->T;
#ifdef PDEBUG
		lprintf(lvl, "====================================\n");
		lprintf(lvl, "Using predefined dev_num = %d, autotuned T = %ld and split = %s -> %s : t_pred = %lf\n",
			autotuned_vals->dev_num, autotuned_vals->T, printlist<short>(autotuned_vals->dev_ids, autotuned_vals->dev_num),
			printlist<double>(best_pred_p->rel_dev_score, autotuned_vals->dev_num), best_pred_p->pred_t*1000);
		lprintf(lvl, "====================================\n");
#endif
	}
	else if ((predef_vals && predef_vals->T <= 0 && autotune_eval_devices) || !predef_vals){
		for (int used_devs = 0; used_devs < autotuned_vals->dev_num; used_devs++){
			dev_ids[used_devs] = CoCoPeLiaDeviceSelectBest(used_devs + 1, autotuned_vals->dev_num,
				autotuned_vals->dev_ids, glob_model);
			pred_p[used_devs] = CoCoPeLiaModelMultidevOptimizeTileAndSplit(used_devs + 1,
				dev_ids[used_devs], glob_model);

			if (best_pred_p == NULL){
				best_pred_p = pred_p[used_devs];
				best_dev_num = used_devs + 1;
			}
			else if(best_pred_p->pred_t >= pred_p[used_devs]->pred_t){
				best_pred_p = pred_p[used_devs];
				best_dev_num = used_devs + 1;
			}
		}
		autotuned_vals->T = best_pred_p->T;
		autotuned_vals->dev_num = best_dev_num;
		for (int idx = 0; idx < autotuned_vals->dev_num; idx++)
			autotuned_vals->dev_ids[idx] = dev_ids[autotuned_vals->dev_num - 1][idx];
#ifdef PDEBUG
		lprintf(lvl, "====================================\n");
		lprintf(lvl, "Using autotuned dev_num = %d, T = %ld and split = %s -> %s : t_pred = %lf\n",
			autotuned_vals->dev_num, autotuned_vals->T, printlist<short>(autotuned_vals->dev_ids, autotuned_vals->dev_num),
			printlist<double>(best_pred_p->rel_dev_score, autotuned_vals->dev_num), best_pred_p->pred_t*1000);
		lprintf(lvl, "====================================\n");
#endif
	}
	else error("CoCoAutotuneParameters: Unknown predefined parameter combination\
	(predef_vals = %p, predef_vals->T = %ld , autotune_eval_devices = %d)\n", (predef_vals) ? (predef_vals): NULL, (predef_vals) ? predef_vals->T : -42, autotune_eval_devices);
	if (predef_vals && predef_vals->cache_limit > 0)
		autotuned_vals->cache_limit = predef_vals->cache_limit;
	else autotuned_vals->cache_limit = 0;

#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Device/T selection -> t_configure = %lf ms\n", cpu_timer*1000);
	cpu_timer = csecond();
#endif

	CoCoPeLiaRemoveUselessDevices(autotuned_vals_p, best_pred_p);

if(!reuse_model_flag){
		lprintf(0, "====================================\n");
		lprintf(0, "CoCoAutotuneParameters: T=%ld,  dev_num=%d, split = %s -> %s : t_pred = %lf\n",
			autotuned_vals->T, autotuned_vals->dev_num, printlist<short>(autotuned_vals->dev_ids, autotuned_vals->dev_num),
			printlist<double>(best_pred_p->rel_dev_score, autotuned_vals->dev_num), best_pred_p->pred_t*1000);
		lprintf(0, "====================================\n");
}

	return best_pred_p;
}

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
}

tunableParams_p CoCoPeLiaModelMultidevOptimizeTileAndSplit(short used_devs, short* used_dev_ids,
	CoCoModel_p* dev_model_list){
	short lvl = 3;
#ifdef DEBUG
	lprintf(lvl-1, "|-----> CoCoPeLiaModelMultidevOptimizeTileAndSplit(used_devs=%d, used_dev_ids= [ ", used_devs);
	for (int i =0; i < used_devs; i++) lprintf(0, "%d ", used_dev_ids[i]);
	lprintf(0, "]\n");
#endif
#ifdef TEST
	double timer = csecond();
#endif
	short first_model_idx = (used_dev_ids[0] == -1) ? LOC_NUM - 1 : used_dev_ids[0];
	CoCoModel_p model = dev_model_list[first_model_idx];
	tunableParams_p outparams = tunableParamsInit();
	//TODO: Naive naive naive! Should replace with something better at some point.
	long int min_T = 0, max_allowed_T = 0, ctr = 0;
	max_allowed_T = fmin(fmin(model->D1, model->D2),model->D3);
	min_T = ((GPUexec3Model_p)model->GPUexec_model_ptr)->T_lookup_buf[0];
#ifdef PDEBUG
		lprintf(lvl, "min_T = %ld, max_allowed_T = %ld\n",
			min_T, max_allowed_T);
#endif
	if (min_T > max_allowed_T){
		outparams->T = max_allowed_T;
		// FIXME: Undefined performance for tiles < than the smaller microbenchmark
		outparams->pred_t = 0;
#ifdef PDEBUG
		lprintf(lvl, "min_T = %ld > max_allowed_T = %ld: returning T = %ld",
			min_T, max_allowed_T, max_allowed_T);
#endif
		return outparams;
	}
	double temp_t, min_overlap_t = 10000000, temp_score = 0;
	long int prev_trial_T = 0;

	outparams->rel_dev_score = (double*) malloc(sizeof(double)*used_devs);
	int best_idx = -1;
	for(int idx = 0; idx < used_devs; idx++){
		outparams->rel_dev_score[idx] = CoCopeLiaPredictFullOverlap(dev_model_list[idx]);
		if (outparams->rel_dev_score[idx] != 0) outparams->rel_dev_score[idx] = 1/outparams->rel_dev_score[idx];
		else warning("CoCoPeLiaModelMultidevOptimizeTileAndSplit: rel_dev_score[%d] == 0\n", idx);
		temp_score+= outparams->rel_dev_score[idx];
	}
	for(int idx = 0; idx < used_devs; idx++){
		outparams->rel_dev_score[idx] /= temp_score;
#ifdef PDEBUG
			lprintf(lvl, "Calculating Relative score for dev_id = %d (idx = %d ): outparams->rel_dev_score = %e\n",
				used_dev_ids[idx], idx, outparams->rel_dev_score[idx]);
#endif
	}

	for (ctr = 0 ; ctr < ((GPUexec3Model_p)model->GPUexec_model_ptr)->lines ; ctr++){
		long int trial_T = ((GPUexec3Model_p)model->GPUexec_model_ptr)->T_lookup_buf[ctr];
		if (trial_T > max_allowed_T) break;
		if (trial_T ==  prev_trial_T) continue;
/*
		temp_score = temp_t = 0;
		double rel_dev_score[used_devs];
		for(int idx = 0; idx < used_devs; idx++){
			short cur_dev_id = used_dev_ids[idx], cur_dev_idx = (cur_dev_id == -1)? LOC_NUM - 1 : cur_dev_id;
			model = dev_model_list[cur_dev_idx];
			temp_t = CoCoPeLiaModelPredict(model, trial_T, COCOPELIA_REUSE);
			if(temp_t > 0){
				rel_dev_score[idx] = 1/temp_t;
				temp_score += rel_dev_score[idx];
			}
			else error("CoCoPeLiaModelPredict(%p(dev_id = %d, (idx = %d )), trial_T = %d): negative prediction temp_t = %lf\n",
				model, cur_dev_id, cur_dev_idx, trial_T, temp_t);
#ifdef PDEBUG
			lprintf(lvl, "CoCoPeLiaModelMultidevOptimizeTileAndSplit(%p) for dev_id = %d (idx = %d ) with trial_T = %d: dev_score = %e, temp_t = %e\n",
				model, cur_dev_id, cur_dev_idx, trial_T, rel_dev_score[idx], temp_t);
#endif
		}
		for(int idx = 0; idx < used_devs; idx++) rel_dev_score[idx] /= temp_score;
*/
		double temp_overlap_t = 0;
		for(int idx = 0; idx < used_devs; idx++){
			short cur_dev_id = used_dev_ids[idx], cur_dev_idx = (cur_dev_id == -1)? LOC_NUM - 1 : cur_dev_id;
			model = dev_model_list[cur_dev_idx];
			temp_t = CoCoPeLiaModelPredictHetero(model, used_devs, used_dev_ids, outparams->rel_dev_score, trial_T, COCOPELIA_HETERO_REUSE);
			if(temp_t > 0) temp_overlap_t = fmax(temp_overlap_t, temp_t);
			else error("CoCoPeLiaModelPredictHetero(%p(dev_id = %d, (idx = %d )), trial_T = %ld): negative prediction temp_t = %lf\n",
				model, cur_dev_id, cur_dev_idx, trial_T, temp_t);
#ifdef PDEBUG
			lprintf(lvl, "CoCoPeLiaModelPredictHetero(%p) for dev_id = %d (idx = %d ) with trial_T = %ld: temp_overlap_t = %lf, temp_t = %lf\n",
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
	CoCoPeLiaNormalizeSplit(outparams->rel_dev_score, used_devs);
#ifdef PDEBUG
	lprintf(lvl, "====================================\n");
	lprintf(lvl, "Best %d percentages : [ ", used_devs);
	for (int i =0; i < used_devs; i++) fprintf(stderr, "%.3lf ", outparams->rel_dev_score[i]);
	lprintf(0, "]\n");
	lprintf(lvl, "Predict T=%ld : t_pred = %lf\n", outparams->T, outparams->pred_t);
#endif
#ifdef TEST
	timer = csecond() - timer;
	lprintf(lvl, "Optimization time:%lf ms\n", timer*1000);
	lprintf(lvl-1, "<-----|\n");
#endif
#ifdef DEBUG
	lprintf(lvl, "outparams->T = %ld\n : outparams->pred_t = %lf ms\n", outparams->T, outparams->pred_t);
	lprintf(lvl-1, "<-----|\n");
#endif
	return outparams;
}

tunableParams_p CoCoPeLiaModelMultidevOptimizeSplit(short used_devs, short* used_dev_ids,
	CoCoModel_p* dev_model_list, long int T){
	short lvl = 3;
#ifdef DEBUG
	lprintf(lvl-1, "|-----> CoCoPeLiaModelMultidevOptimizeSplit(used_devs=%d, used_dev_ids= [ ", used_devs);
	for (int i =0; i < used_devs; i++) lprintf(0, "%d ", used_dev_ids[i]);
	lprintf(0, "]\n");
#endif
#ifdef TEST
	double timer = csecond();
#endif
	short first_model_idx = (used_dev_ids[0] == -1) ? LOC_NUM - 1 : used_dev_ids[0];
	CoCoModel_p model = dev_model_list[first_model_idx];
	tunableParams_p outparams = tunableParamsInit();
	//TODO: Naive naive naive! Should replace with something better at some point.
	long int max_allowed_T = 0, ctr = 0;
	max_allowed_T = fmin(fmin(model->D1, model->D2),model->D3);
#ifdef PDEBUG
		lprintf(lvl, "max_allowed_T = %ld\n", max_allowed_T);
#endif
	if (T > max_allowed_T)
		error("CoCoPeLiaModelMultidevOptimizeSplit: Give T = %ld > max_allowed_T = %ld\n", T, max_allowed_T);

	double temp_t, min_overlap_t = 10000000, temp_score = 0;

	outparams->rel_dev_score = (double*) malloc(sizeof(double)*used_devs);
	for(int idx = 0; idx < used_devs; idx++){
		int dev_idx = idxize(used_dev_ids[idx]);
		outparams->rel_dev_score[idx] = CoCopeLiaPredictFullOverlap(dev_model_list[dev_idx]);
		if (outparams->rel_dev_score[idx] != 0) outparams->rel_dev_score[idx] = 1/outparams->rel_dev_score[idx];
		else warning("CoCoPeLiaModelMultidevOptimizeSplit: rel_dev_score[%d] == 0\n", idx);
		temp_score+= outparams->rel_dev_score[idx];
	}
	for(int idx = 0; idx < used_devs; idx++){
		outparams->rel_dev_score[idx] /= temp_score;
#ifdef PDEBUG
		lprintf(lvl, "Calculating Relative score for dev_id = %d (idx = %d ): outparams->rel_dev_score = %e\n",
				used_dev_ids[idx], idx, outparams->rel_dev_score[idx]);
#endif
	}
	double temp_overlap_t = 0;
	for(int idx = 0; idx < used_devs; idx++){
		short cur_dev_id = used_dev_ids[idx], cur_dev_idx = idxize(cur_dev_id);
		model = dev_model_list[cur_dev_idx];
		temp_t = CoCoPeLiaModelPredictHetero(model, used_devs, used_dev_ids, outparams->rel_dev_score, T, COCOPELIA_HETERO_REUSE);
		if(temp_t > 0) temp_overlap_t = fmax(temp_overlap_t, temp_t);
		else error("CoCoPeLiaModelPredictHetero(%p(dev_id = %d, (idx = %d )), T = %ld): negative prediction temp_t = %lf\n",
			model, cur_dev_id, cur_dev_idx, T, temp_t);
#ifdef PDEBUG
		lprintf(lvl, "CoCoPeLiaModelPredictHetero(%p) for dev_id = %d (idx = %d ) with T = %ld: temp_overlap_t = %lf, temp_t = %lf\n",
			model, cur_dev_id, cur_dev_idx, T, temp_overlap_t, temp_t);
#endif
	}
	CoCoPeLiaNormalizeSplit(outparams->rel_dev_score, used_devs);
	outparams->T = T;
	outparams->pred_t = temp_overlap_t;

#ifdef PDEBUG
	lprintf(lvl, "====================================\n");
	lprintf(lvl, "Best %d percentages : [ ", used_devs);
	for (int i =0; i < used_devs; i++) fprintf(stderr, "%.5lf ", outparams->rel_dev_score[i]);
	lprintf(0, "]\n");
	lprintf(lvl, "Predict T=%ld : t_pred = %lf\n", outparams->T, outparams->pred_t);
#endif
#ifdef TEST
	timer = csecond() - timer;
	lprintf(lvl, "Optimization time:%lf ms\n", timer*1000);
	lprintf(lvl-1, "<-----|\n");
#endif
#ifdef DEBUG
	lprintf(lvl, "outparams->T = %ld\n : outparams->pred_t = %lf ms\n", outparams->T, outparams->pred_t);
	lprintf(lvl-1, "<-----|\n");
#endif
	return outparams;
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
	if ( !strcmp(func, "Dgemm") || !strcmp(func, "Sgemm")) out_model->problem = BLAS3;
	else error("CoCoPeLiaModelInit: Problem type for '%s' func not integrated\n", func);

	switch(out_model->problem){
		case BLAS1:
			out_model->GPUexec_model_ptr = (void*) GPUexec1Model_init(dev_id, func);
			return CoCoModelFuncInitBLAS1(out_model, dev_id, func, func_data);
			break;
		case BLAS2:
			error("Not Implemented\n");
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

double CoCopeLiaPredictReuseHetero(CoCo_model* model, short used_devs, short* used_dev_ids,
	double* used_dev_relative_scores, long int T){
	short lvl = 4;
	long int prob_dims = 0, reset_D1 = model->D1, reset_D2 = model->D2, reset_D3 = model->D3;
	double imb_time_multiplier = 1.0, reduce_time_multiplier = 1.0;
#define ENABLE_HETERO_RELATIVE_DIMS
#ifdef ENABLE_HETERO_RELATIVE_DIMS
	if (reset_D1 != -1){
#ifdef TILE_IMBALANCE_PENALTY
		if (reset_D1%T) imb_time_multiplier+=TILE_IMBALANCE_PENALTY;
#endif
		prob_dims++;
	}
	if (reset_D2 != -1){
#ifdef TILE_IMBALANCE_PENALTY
		if (reset_D2%T) imb_time_multiplier+=TILE_IMBALANCE_PENALTY;
#endif
		prob_dims++;
	}
	if (reset_D3 != -1){
#ifdef TILE_IMBALANCE_PENALTY
		if (reset_D3%T) imb_time_multiplier+=TILE_IMBALANCE_PENALTY;
#endif
#ifdef REDUCE_PENALTY
		if ((reset_D1/T + (reset_D1%T)? 1 : 0)*(reset_D2/T + (reset_D2%T)? 1 : 0)*(reset_D3/T + (reset_D3%T)? 1 : 0)%used_devs) reduce_time_multiplier+=REDUCE_PENALTY;
#endif
		prob_dims++;
	}
	short iloc = -1;
	for (int idx = 0; idx < used_devs; idx++)
		if (used_dev_ids[idx] == model->dev_id){ iloc = idx; break; }
	if (iloc == -1) error("CoCopeLiaPredictReuseHetero:  model->dev_id = %d not found in used_dev_ids[%d]\n",
		model->dev_id, used_devs);
	double problem_percentage = used_dev_relative_scores[iloc];
#ifdef PDEBUG
	lprintf(lvl, "CoCopeLiaPredictReuseHetero(dev_id=%d) prob_dims = %ld, problem_percentage = %lf\n",
		model->dev_id, prob_dims, problem_percentage);
#endif
	if (!strcmp(REL_PERF_MODE, "ROOT-PROBLEM")){
		if (reset_D1 != -1) model->D1 = (long int) reset_D1* 1.0* pow(problem_percentage, 1.0/prob_dims);
		if (reset_D2 != -1) model->D2 = (long int) reset_D2* 1.0* pow(problem_percentage, 1.0/prob_dims);
		if (reset_D3 != -1) model->D3 = (long int) reset_D3* 1.0* pow(problem_percentage, 1.0/prob_dims);
	}
#ifdef PDEBUG
	lprintf(lvl, "CoCopeLiaPredictReuseHetero(dev_id=%d) Modified Dims D1 = %ld, D2 = %ld, D3 = %ld, imb_time_multiplier = %lf, reduce_time_multiplier = %lf\n",
		model->dev_id, model->D1, model->D2, model->D3, imb_time_multiplier, reduce_time_multiplier);
#endif
#endif
	double result = imb_time_multiplier* reduce_time_multiplier* CoCopeLiaPredictReuse(model, T);
	if (!strcmp(REL_PERF_MODE, "PERCENTILE")) result*=problem_percentage;
	else if (!strcmp(REL_PERF_MODE, "ROOT-PROBLEM")){
		model->D1 = reset_D1;
		model->D2 = reset_D2;
		model->D3 = reset_D3;
	}
	else error("CoCopeLiaPredictReuseHetero: Unknown REL_PERF_MODE = %s\n", REL_PERF_MODE);
	return result;
}

double CoCoPeLiaModelPredictHetero(CoCo_model* model, short used_devs, short* used_dev_ids, double* used_dev_relative_scores, long int T, ModelType mode){
	switch(mode){
		case COCOPELIA_HETERO_REUSE:
			return CoCopeLiaPredictReuseHetero(model, used_devs, used_dev_ids, used_dev_relative_scores, T);
		default:
			error("CoCoPeLiaModelPredictHetero: Invalid mode %s", printModel(mode));
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

double CoCopeLiaPredictFullOverlap(CoCoModel_p model)
{
	switch(model->problem){
		case BLAS1:
			//return CoCopeLiaPredictFullOverlapBLAS1(model);
		case BLAS2:
			error("Not implemented\n");
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
			//return CoCopeLiaPredictZeroOverlapBLAS1(model);
		case BLAS2:
			error("Not implemented\n");
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
			error("Not implemented\n");
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
			error("Not implemented\n");
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
			error("Not implemented\n");
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
			error("Not implemented\n");
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
			error("Not implemented\n");
			return 0;
		case BLAS3:
			return CoCopeLiaPipelineEmulateBLAS3(model, T);
		default:
			error("CoCopeLiaPipelineEmulate: Invalid Problem %s", printProblem(model->problem));
	}
	return 0;
}

///  Itterates through benchmarked values for T and chooses the Tbest that minimizes total time.
tunableParams_p CoCoPeLiaModelOptimizeTile(CoCoModel_p model, ModelType mode){
	short lvl = 3;
#ifdef DEBUG
	lprintf(lvl-1, "|-----> CoCoPeLiaModelOptimizeTile(mode=%d)\n", mode);
#endif
#ifdef TEST
	lprintf(lvl-1, "|-----> CoCoPeLiaModelOptimizeTile\n");
	double timer = csecond();
#endif
	tunableParams_p outparams = tunableParamsInit();
	//TODO: Naive naive naive! Should replace with something better at some point.
	long int min_T = 0, max_allowed_T = 0, ctr = 0;
	max_allowed_T = fmin(fmin(model->D1, model->D2),model->D3);
	min_T = ((GPUexec3Model_p)model->GPUexec_model_ptr)->T_lookup_buf[0];
	if (min_T > max_allowed_T){
		outparams->T = max_allowed_T;
		// FIXME: Undefined performance for tiles < than the smaller microbenchmark
		outparams->pred_t = 0;
		return outparams;
	}
	double temp_t, min_t = CoCoPeLiaModelPredict(model, ((GPUexec3Model_p)model->GPUexec_model_ptr)->T_lookup_buf[0], mode);
	long int prev_trial_T = 0;
	if(min_t < 0) error("CoCoPeLiaModelOptimizeTile: First value in DM results in negative prediction");
	for (ctr = 1 ; ctr < ((GPUexec3Model_p)model->GPUexec_model_ptr)->lines ; ctr++){
		long int trial_T = ((GPUexec3Model_p)model->GPUexec_model_ptr)->T_lookup_buf[ctr];
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
	outparams->T = min_T;
	outparams->pred_t = min_t;
#ifdef TEST
	timer = csecond() - timer;
	lprintf(lvl, "Optimization time:%lf ms\n", timer*1000);
	lprintf(lvl-1, "<-----|\n");
#endif
#ifdef DEBUG
	lprintf(lvl, "T = %ld\n : t_min = %lf ms\n", min_T, min_t*1000);
	lprintf(lvl-1, "<-----|\n");
#endif
	return outparams;
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

tunableParams_p tunableParamsInit(){
	tunableParams_p outparams = (tunableParams_p) malloc(sizeof(struct tunableParams));
	outparams->T = 0;
	outparams->pred_t = 0;
	outparams->rel_dev_score = NULL;
	return outparams;
}

const char* printTunableParams(tunableParams_p params){
	char* buf = (char*) malloc(256*sizeof(char));
	sprintf(buf, "{%ld|%e}", params->T, params->pred_t);
	return buf;
}
