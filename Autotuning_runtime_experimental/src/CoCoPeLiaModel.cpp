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
#include "unihelpers.hpp"
#include "Werkhoven.hpp"

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
		short checked[avail_devs] = {0};
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

void CoCoPeLiaNormalizeSplit(double* score_list, int list_len){
	for (int i = 0; i < list_len; i++)
		for (int j = i + 1; j < list_len; j++)
			if(abs(score_list[i] - score_list[j])/score_list[i] < 0.05)
				score_list[j] = score_list[i] = (score_list[i] + score_list[j])/2;
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
	size_t min_T = 0, max_allowed_T = 0, ctr = 0;
	max_allowed_T = fmin(fmin(model->D1, model->D2),model->D3);
	min_T = ((GPUexec3Model_p)model->GPUexec_model_ptr)->T_lookup_buf[0];
#ifdef PDEBUG
		lprintf(lvl, "min_T = %d, max_allowed_T = %d\n",
			min_T, max_allowed_T);
#endif
	if (min_T > max_allowed_T){
		outparams->T = max_allowed_T;
		// FIXME: Undefined performance for tiles < than the smaller microbenchmark
		outparams->pred_t = 0;
#ifdef PDEBUG
		lprintf(lvl, "min_T = %d > max_allowed_T = %d: returning T = %d",
			min_T, max_allowed_T, max_allowed_T);
#endif
		return outparams;
	}
	double temp_t, min_overlap_t = 10000000, temp_score = 0;
	size_t prev_trial_T = 0;

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
		size_t trial_T = ((GPUexec3Model_p)model->GPUexec_model_ptr)->T_lookup_buf[ctr];
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
			else error("CoCoPeLiaModelPredictHetero(%p(dev_id = %d, (idx = %d )), trial_T = %d): negative prediction temp_t = %lf\n",
				model, cur_dev_id, cur_dev_idx, trial_T, temp_t);
#ifdef PDEBUG
			lprintf(lvl, "CoCoPeLiaModelPredictHetero(%p) for dev_id = %d (idx = %d ) with trial_T = %d: temp_overlap_t = %lf, temp_t = %lf\n",
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
	lprintf(lvl, "Predict T=%zu : t_pred = %lf\n", outparams->T, outparams->pred_t);
#endif
#ifdef TEST
	timer = csecond() - timer;
	lprintf(lvl, "Optimization time:%lf ms\n", timer*1000);
	lprintf(lvl-1, "<-----|\n");
#endif
#ifdef DEBUG
	lprintf(lvl, "outparams->T = %zu\n : outparams->pred_t = %lf ms\n", outparams->T, outparams->pred_t);
	lprintf(lvl-1, "<-----|\n");
#endif
	return outparams;
}

tunableParams_p CoCoPeLiaModelMultidevOptimizeSplit(short used_devs, short* used_dev_ids,
	CoCoModel_p* dev_model_list, int T){
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
	size_t max_allowed_T = 0, ctr = 0;
	max_allowed_T = fmin(fmin(model->D1, model->D2),model->D3);
#ifdef PDEBUG
		lprintf(lvl, "max_allowed_T = %d\n", max_allowed_T);
#endif
	if (T > max_allowed_T)
		error("CoCoPeLiaModelMultidevOptimizeSplit: Give T = %d > max_allowed_T = %d\n", T, max_allowed_T);

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
		else error("CoCoPeLiaModelPredictHetero(%p(dev_id = %d, (idx = %d )), T = %d): negative prediction temp_t = %lf\n",
			model, cur_dev_id, cur_dev_idx, T, temp_t);
#ifdef PDEBUG
		lprintf(lvl, "CoCoPeLiaModelPredictHetero(%p) for dev_id = %d (idx = %d ) with T = %d: temp_overlap_t = %lf, temp_t = %lf\n",
			model, cur_dev_id, cur_dev_idx, T, temp_overlap_t, temp_t);
#endif
	}
	CoCoPeLiaNormalizeSplit(outparams->rel_dev_score, used_devs);
	outparams->T = T;
	outparams->pred_t = temp_overlap_t;

#ifdef PDEBUG
	lprintf(lvl, "====================================\n");
	lprintf(lvl, "Best %d percentages : [ ", used_devs);
	for (int i =0; i < used_devs; i++) fprintf(stderr, "%.3lf ", outparams->rel_dev_score[i]);
	lprintf(0, "]\n");
	lprintf(lvl, "Predict T=%zu : t_pred = %lf\n", outparams->T, outparams->pred_t);
#endif
#ifdef TEST
	timer = csecond() - timer;
	lprintf(lvl, "Optimization time:%lf ms\n", timer*1000);
	lprintf(lvl-1, "<-----|\n");
#endif
#ifdef DEBUG
	lprintf(lvl, "outparams->T = %zu\n : outparams->pred_t = %lf ms\n", outparams->T, outparams->pred_t);
	lprintf(lvl-1, "<-----|\n");
#endif
	return outparams;
}

///  Initializes the model for gemm
CoCoModel_p CoCoPeLiaTileModelInit(short dev_id, char* func, void* func_data){
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
	out_model->GPUexec_model_ptr = (void*) GPUexec3Model_init(dev_id, func);
	out_model->V = (Vdata_p) malloc(sizeof(struct V_struct));
	out_model->flags = (flagParams_p) malloc(sizeof(struct flagParams));
	out_model->dev_id = dev_id;

	if ( !strcmp(func, "Dgemm") || !strcmp(func, "Sgemm")) return CoCoModel_gemm_init(out_model, dev_id, func, (gemm_backend_in_p) func_data);
	else error("CoCoPeLiaModelInit: Model for '%s' func not integrated", func);
}

double CoCopeLiaPredictReuseHetero(CoCo_model* model, short used_devs, short* used_dev_ids,
	double* used_dev_relative_scores, size_t T){
	short lvl = 4;
	size_t prob_dims = 0, reset_D1 = model->D1, reset_D2 = model->D2, reset_D3 = model->D3;
#define ENABLE_HETERO_RELATIVE_DIMS
#ifdef ENABLE_HETERO_RELATIVE_DIMS
	if (reset_D1 != -1) prob_dims++;
	if (reset_D2 != -1) prob_dims++;
	if (reset_D3 != -1) prob_dims++;
	short iloc = -1;
	for (int idx = 0; idx < used_devs; idx++)
		if (used_dev_ids[idx] == model->dev_id){ iloc = idx; break; }
	if (iloc == -1) error("CoCopeLiaPredictReuseHetero:  model->dev_id = %d not found in used_dev_ids[%d]\n",
		model->dev_id, used_devs);
	double problem_percentage = used_dev_relative_scores[iloc];
#ifdef PDEBUG
	lprintf(lvl, "CoCopeLiaPredictReuseHetero(dev_id=%d) prob_dims = %zu, problem_percentage = %lf\n",
		model->dev_id, prob_dims, problem_percentage);
#endif
	if (reset_D1 != -1) model->D1 = (size_t) reset_D1* 1.0* std::pow(problem_percentage, 1.0/prob_dims);
	if (reset_D2 != -1) model->D2 = (size_t) reset_D2* 1.0* std::pow(problem_percentage, 1.0/prob_dims);
	if (reset_D3 != -1) model->D3 = (size_t) reset_D3* 1.0* std::pow(problem_percentage, 1.0/prob_dims);
#ifdef PDEBUG
	lprintf(lvl, "CoCopeLiaPredictReuseHetero(dev_id=%d) Modified Dims D1 = %zu, D2 = %zu, D3 = %zu\n",
		model->dev_id, model->D1, model->D2, model->D3);
#endif
#endif
	double result = CoCopeLiaPredictReuse(model, T);
	model->D1 = reset_D1;
	model->D2 = reset_D2;
	model->D3 = reset_D3;
	return result;
}

double CoCoPeLiaModelPredictHetero(CoCo_model* model, short used_devs, short* used_dev_ids, double* used_dev_relative_scores, size_t T, ModelType mode){
	switch(mode){
		case COCOPELIA_HETERO_REUSE:
			return CoCopeLiaPredictReuseHetero(model, used_devs, used_dev_ids, used_dev_relative_scores, T);
		default:
			error("CoCoPeLiaModelPredictHetero: Invalid mode %s", printModel(mode));
	}
}

double CoCoPeLiaModelPredict(CoCo_model* model, size_t T, ModelType mode){
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
	short lvl = 4;
	double t_recv_full = 0, t_send_full = 0, t_exec_full = 0, t_total = 0;
	size_t maxT = GPUexec3MaxT((GPUexec3Model_p)model->GPUexec_model_ptr);
	size_t Tbig = GPUexec3NearestT((GPUexec3Model_p)model->GPUexec_model_ptr,
		fmin(maxT, fmin(fmin(model->D1,model->D2), model->D3)));
	//fprintf(stderr, "Tbig = %zu\n", Tbig);
	t_exec_full = (model->D1*1.0/Tbig * model->D2*1.0/Tbig * model->D3*1.0/Tbig)*
		GPUexec3Model_predict((GPUexec3Model_p)model->GPUexec_model_ptr, Tbig, model->flags->TransA, model->flags->TransB);
	if ( t_exec_full < 0){
		warning("CoCopeLiaPredictFullOverlap: GPUexec3Model_predict submodel returned negative value, abort prediction");
		return -1.0;
	}
	long long recv_sz = 0, send_sz = 0;
	for (int i = 0; i < model->V->numT; i++){
		recv_sz += model->V->in[i]*(*model->V->Dim1[i])*(*model->V->Dim2[i])*model->V->dtype_sz;
		send_sz += model->V->out[i]*(*model->V->Dim1[i])*(*model->V->Dim2[i])*model->V->dtype_sz;
		double t_recv_tmp = model->V->in[i]*t_com_predict(model->revlink[idxize(model->V->loc[i])],
			(*model->V->Dim1[i])*(*model->V->Dim2[i])*model->V->dtype_sz);
		double t_send_tmp =  model->V->out[i]*t_com_predict(model->revlink[idxize(model->V->out_loc[i])],
			(*model->V->Dim1[i])*(*model->V->Dim2[i])*model->V->dtype_sz);
		if(t_recv_tmp < 0 || t_send_tmp < 0 ){
				warning("CoCopeLiaPredictFullOverlap: t_com_predict submodel idx = %d\
					returned negative value, abort prediction", idxize(model->V->loc[i]));
				return -1.0;
		}
		t_recv_full+= t_recv_tmp;
		t_send_full+= t_send_tmp;
	}

	t_total = fmax(t_exec_full, fmax(t_recv_full, t_send_full));
#ifdef DPDEBUG
	fprintf(stderr, "CoCopelia FullOverlap :\n"
	"\tt_recv_full: %lf ms ( %lf Gb/s)\n"
	"\tt_exec_full: %lf ms (%lf GFlops/s)\n"
	"\tt_send_full: %lf ms ( %lf Gb/s)\n"
	"\tt_total: %lf ms (%lf GFlops/s)\n\n",
	t_recv_full*1000, Gval_per_s(recv_sz,t_recv_full),
	t_exec_full*1000, Gval_per_s(dgemm_flops(model->D1,model->D2,model->D3), t_exec_full),
	t_send_full*1000, Gval_per_s(send_sz,t_send_full),
	t_total*1000, Gval_per_s(dgemm_flops(model->D1,model->D2,model->D3), t_total));
#endif

	return t_total;
}

double CoCopeLiaPredictZeroOverlap(CoCoModel_p model)
{
	short lvl = 4;
	double t_recv_full = 0, t_send_full = 0, t_exec_full = 0, t_total = 0;
	size_t maxT = GPUexec3MaxT((GPUexec3Model_p)model->GPUexec_model_ptr);
	size_t Tbig = GPUexec3NearestT((GPUexec3Model_p)model->GPUexec_model_ptr,
		fmin(maxT, fmin(fmin(model->D1,model->D2), model->D3)));
	//fprintf(stderr, "Tbig = %zu\n", Tbig);
	t_exec_full = (model->D1*1.0/Tbig * model->D2*1.0/Tbig * model->D3*1.0/Tbig)*
		GPUexec3Model_predict((GPUexec3Model_p)model->GPUexec_model_ptr, Tbig, model->flags->TransA, model->flags->TransB);
	if ( t_exec_full < 0){
		warning("CoCopeLiaPredictZeroOverlap: GPUexec3Model_predict submodel returned negative value, abort prediction");
		return -1.0;
	}
	long long recv_sz = 0, send_sz = 0;
	for (int i = 0; i < model->V->numT; i++){
		recv_sz += model->V->in[i]*(*model->V->Dim1[i])*(*model->V->Dim2[i])*model->V->dtype_sz;
		send_sz += model->V->out[i]*(*model->V->Dim1[i])*(*model->V->Dim2[i])*model->V->dtype_sz;
		double t_recv_tmp = model->V->in[i]*t_com_predict(model->revlink[idxize(model->V->loc[i])],
			(*model->V->Dim1[i])*(*model->V->Dim2[i])*model->V->dtype_sz);
		double t_send_tmp =  model->V->out[i]*t_com_predict(model->revlink[idxize(model->V->out_loc[i])],
			(*model->V->Dim1[i])*(*model->V->Dim2[i])*model->V->dtype_sz);
		if(t_recv_tmp < 0 || t_send_tmp < 0 ){
				warning("CoCopeLiaPredictZeroOverlap: t_com_predict submodel idx = %d\
					returned negative value, abort prediction", idxize(model->V->loc[i]));
				return -1.0;
		}
		t_recv_full+= t_recv_tmp;
		t_send_full+= t_send_tmp;
	}

	t_total = t_exec_full + t_recv_full + t_send_full;

#ifdef DPDEBUG
	fprintf(stderr, "CoCopelia ZeroOverlap :\n"
	"\tt_recv_full: %lf ms ( %lf Gb/s)\n"
	"\tt_exec_full: %lf ms (%lf GFlops/s)\n"
	"\tt_send_full: %lf ms ( %lf Gb/s)\n"
	"\tt_total: %lf ms (%lf GFlops/s)\n\n",
	t_recv_full*1000, Gval_per_s(recv_sz,t_recv_full),
	t_exec_full*1000, Gval_per_s(dgemm_flops(model->D1,model->D2,model->D3), t_exec_full),
	t_send_full*1000, Gval_per_s(send_sz,t_send_full),
	t_total*1000, Gval_per_s(dgemm_flops(model->D1,model->D2,model->D3), t_total));
#endif

	return t_total;
}

double CoCopeLiaPredictBaseline(CoCoModel_p model, size_t T)
{
	short lvl = 4;
	double t_recv_T3[LOC_NUM] = {0}, t_send_T3[LOC_NUM] = {0}, t_exec_T3 = 0, t_total = 0;
	t_exec_T3 = GPUexec3Model_predict((GPUexec3Model_p) model->GPUexec_model_ptr, T, model->flags->TransA, model->flags->TransB);
	if ( t_exec_T3 < 0){
		warning("CoCopeLiaPredictReuse: GPUexec3Model_predict submodel returned negative value, abort prediction");
		return -1.0;
	}

	for(int idx = 0; idx < LOC_NUM; idx++){
		t_recv_T3[idx] = t_com_predict(model->revlink[idx], T*T*model->V->dtype_sz);
		t_send_T3[idx] = t_com_predict(model->link[idx], T*T*model->V->dtype_sz);
		if(t_recv_T3[idx] < 0 || t_send_T3[idx] < 0 ){
			warning("CoCopeLiaPredictReuse: t_com_predict submodel idx = %d returned negative value, abort prediction", idx);
			return -1.0;
		}
	}

	short mv_dev_id = -1, dev_id_initlocs[LOC_NUM] = {};
	for (int i = 0; i < model->V->numT; i++) dev_id_initlocs[idxize(model->V->loc[i])] ++;
	for (int idx = 0; idx < LOC_NUM; idx++)
		if (dev_id_initlocs[idxize(mv_dev_id)] <  dev_id_initlocs[idx]) mv_dev_id = deidxize(idx);
		else if (dev_id_initlocs[idxize(mv_dev_id)] ==  dev_id_initlocs[idx]
			&& t_recv_T3[idxize(mv_dev_id)] < t_recv_T3[idx]) mv_dev_id = deidxize(idx);

#ifdef DPDEBUG
	lprintf(lvl, "Selecting  mv_dev_id =%d\n", mv_dev_id);
#endif
	double mv_t_recv_T3 = t_recv_T3[idxize(mv_dev_id)], mv_t_send_T3 = t_send_T3[idxize(mv_dev_id)];

	double t_over_T3;
	size_t numTin = 0, numTout = 0;

	double ker_over =  (1.0*model->D1/T)*(1.0*model->D2/T)*(1.0*model->D3/T) - 1;
	for (int i = 0; i < model->V->numT; i++){
		if (*model->V->Dim1[i] < 1 || *model->V->Dim2[i] < 1) error("CoCopeLiaPredictBaseline: Invalid data struct dims");
		numTin += model->V->in[i];
		numTout += model->V->out[i];
	}
	t_over_T3 = fmax(numTin*mv_t_recv_T3, mv_t_send_T3*numTout);
	t_total = fmax(t_exec_T3, t_over_T3)* ker_over +
	+ t_exec_T3 + numTin * mv_t_recv_T3 + numTout * mv_t_send_T3;

#ifdef DPDEBUG
	fprintf(stderr, "CoCopelia (T=%zu) predicted :\n"
	"\t -> numTin = %d -> numTout = %d\n"
	"\tmv_t_recv_T3: %lf ms ( %lf Gb/s)\n"
	"\tt_execT3: %lf ms (%lf GFlops/s)\n"
	"\tmv_t_send_T3: %lf ms ( %lf Gb/s)\n"
	"\tt_total: %lf ms (%lf GFlops/s)\n\n",
	T, numTin, numTout,
	mv_t_recv_T3*1000, Gval_per_s(T*T*model->V->dtype_sz,mv_t_recv_T3),
	t_exec_T3*1000, Gval_per_s(dgemm_flops(T,T,T), t_exec_T3),
	mv_t_send_T3*1000, Gval_per_s(T*T*model->V->dtype_sz,mv_t_send_T3),
	t_total*1000, Gval_per_s(dgemm_flops(model->D1,model->D2,model->D3), t_total));
#endif

	return t_total;
}

double CoCopeLiaPredictDataLoc(CoCoModel_p model, size_t T)
{
	short lvl = 4;
	double t_recv_T3[LOC_NUM] = {0}, t_send_T3[LOC_NUM] = {0}, t_exec_T3 = 0, t_total = 0;
	t_exec_T3 = GPUexec3Model_predict((GPUexec3Model_p) model->GPUexec_model_ptr, T, model->flags->TransA, model->flags->TransB);
	if ( t_exec_T3 < 0){
		warning("CoCopeLiaPredictReuse: GPUexec3Model_predict submodel returned negative value, abort prediction");
		return -1.0;
	}

	for(int idx = 0; idx < LOC_NUM; idx++){
		t_recv_T3[idx] = t_com_predict(model->revlink[idx], T*T*model->V->dtype_sz);
		t_send_T3[idx] = t_com_predict(model->link[idx], T*T*model->V->dtype_sz);
		if(t_recv_T3[idx] < 0 || t_send_T3[idx] < 0 ){
			warning("CoCopeLiaPredictReuse: t_com_predict submodel idx = %d returned negative value, abort prediction", idx);
			return -1.0;
		}
	}

	short mv_dev_id = -1, dev_id_initlocs[LOC_NUM] = {};
	for (int i = 0; i < model->V->numT; i++) dev_id_initlocs[idxize(model->V->loc[i])] ++;
	for (int idx = 0; idx < LOC_NUM; idx++)
		if (dev_id_initlocs[idxize(mv_dev_id)] <  dev_id_initlocs[idx]) mv_dev_id = deidxize(idx);
		else if (dev_id_initlocs[idxize(mv_dev_id)] ==  dev_id_initlocs[idx]
			&& t_recv_T3[idxize(mv_dev_id)] < t_recv_T3[idx]) mv_dev_id = deidxize(idx);

#ifdef DPDEBUG
	lprintf(lvl, "Selecting  mv_dev_id =%d\n", mv_dev_id);
#endif
	double mv_t_recv_T3 = t_recv_T3[idxize(mv_dev_id)], mv_t_send_T3 = t_send_T3[idxize(mv_dev_id)];

	double t_over_T3;
	size_t numTin = 0, numTout = 0;

	double ker_over =  (1.0*model->D1/T)*(1.0*model->D2/T)*(1.0*model->D3/T) - 1;
	for (int i = 0; i < model->V->numT; i++){
		if (*model->V->Dim1[i] < 1 || *model->V->Dim2[i] < 1) error("CoCopeLiaPredictDataLoc: Invalid data struct dims");
		numTin += model->V->in[i] * remote(model->V->loc[i], model->dev_id);
		numTout += model->V->out[i] * remote(model->V->loc[i], model->dev_id);
	}
	t_over_T3 = fmax(numTin*mv_t_recv_T3, mv_t_send_T3*numTout);
	t_total = fmax(t_exec_T3, t_over_T3)* ker_over +
	+ t_exec_T3 + numTin * mv_t_recv_T3 + numTout * mv_t_send_T3;

#ifdef DPDEBUG
	fprintf(stderr, "CoCopelia (T=%zu) predicted :\n"
	"\t -> numTin = %d -> numTout = %d\n"
	"\tmv_t_recv_T3: %lf ms ( %lf Gb/s)\n"
	"\tt_execT3: %lf ms (%lf GFlops/s)\n"
	"\tmv_t_send_T3: %lf ms ( %lf Gb/s)\n"
	"\tt_total: %lf ms (%lf GFlops/s)\n\n",
	T, numTin, numTout,
	mv_t_recv_T3*1000, Gval_per_s(T*T*model->V->dtype_sz,mv_t_recv_T3),
	t_exec_T3*1000, Gval_per_s(dgemm_flops(T,T,T), t_exec_T3),
	mv_t_send_T3*1000, Gval_per_s(T*T*model->V->dtype_sz,mv_t_send_T3),
	t_total*1000, Gval_per_s(dgemm_flops(model->D1,model->D2,model->D3), t_total));
#endif

	return t_total;
}

///  Predicts 3-way overlaped execution time for BLAS3 Square tilling blocking without data reuse.
double CoCopeLiaPredictBidirectional(CoCoModel_p model, size_t T)
{
	short lvl = 4;
	double t_recv_T3[LOC_NUM] = {0}, t_send_T3[LOC_NUM] = {0}, t_exec_T3 = 0, t_total = 0;
	t_exec_T3 = GPUexec3Model_predict((GPUexec3Model_p) model->GPUexec_model_ptr, T, model->flags->TransA, model->flags->TransB);
	if ( t_exec_T3 < 0){
		warning("CoCopeLiaPredictReuse: GPUexec3Model_predict submodel returned negative value, abort prediction");
		return -1.0;
	}

	for(int idx = 0; idx < LOC_NUM; idx++){
		t_recv_T3[idx] = t_com_predict(model->revlink[idx], T*T*model->V->dtype_sz);
		t_send_T3[idx] = t_com_predict(model->link[idx], T*T*model->V->dtype_sz);
		if(t_recv_T3[idx] < 0 || t_send_T3[idx] < 0 ){
			warning("CoCopeLiaPredictReuse: t_com_predict submodel idx = %d returned negative value, abort prediction", idx);
			return -1.0;
		}
	}

	short mv_dev_id = -1, dev_id_initlocs[LOC_NUM] = {};
	for (int i = 0; i < model->V->numT; i++) dev_id_initlocs[idxize(model->V->loc[i])] ++;
	for (int idx = 0; idx < LOC_NUM; idx++)
		if (dev_id_initlocs[idxize(mv_dev_id)] <  dev_id_initlocs[idx]) mv_dev_id = deidxize(idx);
		else if (dev_id_initlocs[idxize(mv_dev_id)] ==  dev_id_initlocs[idx]
			&& t_recv_T3[idxize(mv_dev_id)] < t_recv_T3[idx]) mv_dev_id = deidxize(idx);

#ifdef DPDEBUG
	lprintf(lvl, "Selecting  mv_dev_id =%d\n", mv_dev_id);
#endif
	double mv_t_recv_T3 = t_recv_T3[idxize(mv_dev_id)], mv_t_send_T3 = t_send_T3[idxize(mv_dev_id)];

	double t_over_T3;
	size_t numTin = 0, numTout = 0;

	double ker_over =  (1.0*model->D1/T)*(1.0*model->D2/T)*(1.0*model->D3/T) - 1;
	for (int i = 0; i < model->V->numT; i++){
		if (*model->V->Dim1[i] < 1 || *model->V->Dim2[i] < 1) error("CoCopeLiaPredictBidirectional: Invalid data struct dims");
		numTin += model->V->in[i] * remote(model->V->loc[i], model->dev_id);
		numTout += model->V->out[i] * remote(model->V->loc[i], model->dev_id);
	}
	// Use bidirectional magic here if needed
	t_over_T3 = t_com_bid_predict(model->revlink[idxize(mv_dev_id)], model->link[idxize(mv_dev_id)],
		T*T*model->V->dtype_sz*numTin,  T*T*model->V->dtype_sz*numTout);
	t_total = fmax(t_exec_T3, t_over_T3)* ker_over +
	+ t_exec_T3 + numTin * mv_t_recv_T3 + numTout * mv_t_send_T3;

#ifdef DPDEBUG
	fprintf(stderr, "CoCopelia (T=%zu) predicted :\n"
	"\t -> numTin = %d -> numTout = %d\n"
	"\tmv_t_recv_T3: %lf ms ( %lf Gb/s)\n"
	"\tt_execT3: %lf ms (%lf GFlops/s)\n"
	"\mv_t_send_T3t: %lf ms ( %lf Gb/s)\n"
	"\tt_over_T3: %lf ms\n"
	"\tt_total: %lf ms (%lf GFlops/s)\n\n",
	T, numTin, numTout,
	mv_t_recv_T3*1000, Gval_per_s(T*T*model->V->dtype_sz,mv_t_recv_T3),
	t_exec_T3*1000, Gval_per_s(dgemm_flops(T,T,T), t_exec_T3),
	mv_t_send_T3*1000, Gval_per_s(T*T*model->V->dtype_sz,mv_t_send_T3),
	t_over_T3*1000,
	t_total*1000, Gval_per_s(dgemm_flops(model->D1,model->D2,model->D3), t_total));
#endif

	return t_total;

}

double CoCopeLiaPredictReuse(CoCoModel_p model, size_t T)
{
	short lvl = 4;
	double t_recv_T3[LOC_NUM] = {0}, t_send_T3[LOC_NUM] = {0}, t_exec_T3 = 0, t_total = 0;
	t_exec_T3 = GPUexec3Model_predict((GPUexec3Model_p) model->GPUexec_model_ptr, T, model->flags->TransA, model->flags->TransB);
	if ( t_exec_T3 < 0){
		warning("CoCopeLiaPredictReuse: GPUexec3Model_predict submodel returned negative value, abort prediction");
		return -1.0;
	}

	for(int idx = 0; idx < LOC_NUM; idx++){
		t_recv_T3[idx] = t_com_predict(model->revlink[idx], T*T*model->V->dtype_sz);
		t_send_T3[idx] = t_com_predict(model->link[idx], T*T*model->V->dtype_sz);
		if(t_recv_T3[idx] < 0 || t_send_T3[idx] < 0 ){
			warning("CoCopeLiaPredictReuse: t_com_predict submodel idx = %d returned negative value, abort prediction", idx);
			return -1.0;
		}
	}

	short mv_dev_id = -1, dev_id_initlocs[LOC_NUM] = {};
	for (int i = 0; i < model->V->numT; i++) dev_id_initlocs[idxize(model->V->loc[i])] ++;
	for (int idx = 0; idx < LOC_NUM; idx++)
		if (dev_id_initlocs[idxize(mv_dev_id)] <  dev_id_initlocs[idx]) mv_dev_id = deidxize(idx);
		else if (dev_id_initlocs[idxize(mv_dev_id)] ==  dev_id_initlocs[idx]
			&& t_recv_T3[idxize(mv_dev_id)] < t_recv_T3[idx]) mv_dev_id = deidxize(idx);

#ifdef DPDEBUG
	lprintf(lvl, "Selecting  mv_dev_id =%d\n", mv_dev_id);
#endif
	double mv_t_recv_T3 = t_recv_T3[idxize(mv_dev_id)], mv_t_send_T3 = t_send_T3[idxize(mv_dev_id)];

	size_t numTin = 0, numTout = 0;

	double zero_over = 0, one_over = 0, two_over = 0;
	zero_over =  (1.0*model->D1/T)*(1.0*model->D2/T)*(1.0*model->D3/T) - 1;
	for (int i = 0; i < model->V->numT; i++){
		if (*model->V->Dim1[i] < 1 || *model->V->Dim2[i] < 1) error("CoCopeLiaPredictReuse: Invalid data struct dims");
		numTin += model->V->in[i] * remote(model->V->loc[i], model->dev_id);
		numTout += model->V->out[i] * remote(model->V->out_loc[i], model->dev_id);
		one_over+= model->V->in[i] * remote(model->V->loc[i], model->dev_id)*((1.0*(*model->V->Dim1[i]))/T)*((1.0*(*model->V->Dim2[i]))/T); // - 1 The -1 only if two_over is ommited

		if (mv_t_recv_T3 > t_exec_T3) {
		// two_over kernels calculated
			for (int j = i + 1; j < model->V->numT; j++)
				if (model->V->in[i] * remote(model->V->loc[i], model->dev_id) && model->V->in[j] * remote(model->V->loc[j], model->dev_id)){
					if ( model->V->Dim1[i] == model->V->Dim1[j] || model->V->Dim1[i] == model->V->Dim2[j]) two_over += ((1.0*(*model->V->Dim1[i]))/T) - 1;
					else if ( model->V->Dim2[i] == model->V->Dim1[j] || model->V->Dim2[i] == model->V->Dim2[j]) two_over += ((1.0*(*model->V->Dim2[i]))/T) - 1;
					else error("CoCopeLiaPredictReuse: something is wrong with my brilliant pointer comparisson idea");
			}
		}
	}
	// Performance Cheat
	if ( 2* mv_t_recv_T3 > t_exec_T3 && t_exec_T3 > mv_t_recv_T3)  two_over += (1.0*model->D3/T);
	one_over -= (2*two_over + numTin);
	zero_over -= (one_over + two_over);
	t_total = t_exec_T3*(1 + zero_over) +
	fmax(t_exec_T3, mv_t_recv_T3)* one_over +
	fmax(t_exec_T3, mv_t_recv_T3*2)* two_over +
	+ numTin * mv_t_recv_T3 + numTout * mv_t_send_T3;

#ifdef DPDEBUG
	lprintf(lvl, "CoCopelia (T=%d) predicted :\n"
	"\tmv_t_recv_T3: %lf ms ( %lf Gb/s)\n"
	"\t -> two_over = %lf -> one_over = %lf -> zero_over = %lf\n"
	"\tt_execT3: %lf ms (%lf GFlops/s)\n"
	"\tmv_t_send_T3: %lf ms ( %lf Gb/s)\n"
	"\tt_total: %lf ms (%lf GFlops/s)\n\n",
	T, mv_t_recv_T3*1000, Gval_per_s(T*T*model->V->dtype_sz,mv_t_recv_T3),
	two_over, one_over, zero_over,
	t_exec_T3*1000, Gval_per_s(dgemm_flops(T,T,T), t_exec_T3),
	mv_t_send_T3*1000, Gval_per_s(T*T*model->V->dtype_sz,mv_t_send_T3),
	t_total*1000, Gval_per_s(dgemm_flops(model->D1,model->D2,model->D3), t_total));
#endif

	return t_total;
}

/// TODO: currently d2h overlap is ignored
double CoCopeLiaPipelineEmulate(CoCoModel_p model, size_t T){
	CoModel_p h2d_model = model->revlink[LOC_NUM-1], d2h_model = model->link[LOC_NUM-1];
	double t_h2d_T3 = 0, t_d2h_T3 = 0, t_exec_T3 = 0, t_total = 0;
	t_exec_T3 = GPUexec3Model_predict((GPUexec3Model_p) model->GPUexec_model_ptr, T, model->flags->TransA, model->flags->TransB);
	t_h2d_T3 = t_com_predict(h2d_model, T*T*model->V->dtype_sz); //CoTile_predict(h2d_model, T, model->V->dtype_sz);
	t_d2h_T3 = t_com_predict(d2h_model, T*T*model->V->dtype_sz);//CoTile_predict(d2h_model, T, model->V->dtype_sz);

	if ( t_exec_T3 < 0 || t_h2d_T3 < 0 || t_d2h_T3 < 0 ){
		if(t_exec_T3 < 0) warning("CoCopeLiaPipelineEmulate: GPUexec3Model_predict submodel returned negative value, abort prediction");
		if(t_h2d_T3 < 0) warning("CoCopeLiaPipelineEmulate: t_com_predict submodel returned negative value, abort prediction");
		if(t_d2h_T3 < 0) warning("CoCopeLiaPipelineEmulate: t_com_predict submodel returned negative value, abort prediction");
		return -1.0;
	}

	size_t D1_last = model->D1%T , D2_last = model->D2%T, D3_last= model->D3%T;
	size_t D1_parts = model->D1/T , D2_parts = model->D2/T, D3_parts = model->D3/T;
	if (D1_last > T/2) D1_parts++;
	else D1_last+=T;
	if (D2_last > T/2) D2_parts++;
	else D2_last+=T;
	if (D3_last > T/2) D3_parts++;
	else D3_last+=T;
	//printf("D1_parts=%zu,D2_parts=%zu,D3_parts=%zu\n", D1_parts, D2_parts, D3_parts);
	//printf("D1_last=%zu,D2_last=%zu,D3_last=%zu\n", D1_last, D2_last, D3_last);

	if (!D1_parts || !D2_parts || !D3_parts) error("CoCoModel_pipeline_simulate3: Some dim cut is considered 0");
	size_t numTin = 0, numTout = 0;
	short *idx_matrix[model->V->numT];

	for (int i = 0; i < model->V->numT; i++){
		if (*model->V->Dim1[i] < 1 || *model->V->Dim2[i] < 1) error("CoCoModel_pipeline_simulate3: Invalid data struct dims");
		size_t Dim1_num = *model->V->Dim1[i]/T;
		if(*model->V->Dim1[i]%T > T/2) Dim1_num++;
		size_t Dim2_num = *model->V->Dim2[i]/T;
		if(*model->V->Dim2[i]%T > T/2) Dim2_num++;
		idx_matrix[i] = (short*) calloc (Dim1_num*Dim2_num, sizeof(short));
		/// First element 0 because its not accounted in pipeline
		if (model->V->loc[i]) for (int j = 0; j < Dim1_num*Dim2_num; j++) idx_matrix[i][j] = model->V->loc[i];
		//printf("Dim1_num=%zu,Dim2_num=%zu\n", Dim1_num, Dim2_num);
		//matrix_visualize(idx_matrix[i], Dim1_num, Dim2_num);
		numTin += model->V->in[i] * remote(model->V->loc[i], model->dev_id);
		numTout += model->V->out[i] * model->V->out_loc[i];
	}

	/// TODO: GEMM specific
	double t_pipe_exec = 0 , t_pipe_h2d = 0, t_pipe_d2h = 0;
	double t_exec_T3_adj = 0, t_h2d_T3_A = 0, t_h2d_T3_B = 0, t_h2d_T3_C = 0;
	for (int mi = 0; mi < D1_parts; mi++)
		for (int ni = 0 ; ni < D2_parts; ni++)
			for (int ki = 0; ki < D3_parts; ki++){
				//printf("t_exec_T3_adj=%lf,t_h2d_T3_A=%lf,t_h2d_T3_B=%lf,t_h2d_T3_C=%lf\n", t_exec_T3_adj, t_h2d_T3_A,t_h2d_T3_B,t_h2d_T3_C);
				if (mi + ni + ki != 0) t_pipe_exec= fmax(t_pipe_h2d, t_pipe_exec) + t_exec_T3_adj;
				t_exec_T3_adj = t_exec_T3;
				t_h2d_T3_A = idx_matrix[0][mi*D3_parts+ki]*t_h2d_T3;
				t_h2d_T3_B = idx_matrix[1][ki*D2_parts+ni]*t_h2d_T3;
				t_h2d_T3_C = idx_matrix[2][mi*D2_parts+ni]*t_h2d_T3;
				idx_matrix[0][mi*D3_parts+ki] = idx_matrix[1][ki*D2_parts+ni] = idx_matrix[2][mi*D2_parts+ni] = 0;
				if (mi == D1_parts -1){
					t_exec_T3_adj*=1.0*D1_last/T;
					t_h2d_T3_A*=1.0*D1_last/T;
					t_h2d_T3_C*=1.0*D1_last/T;
				}
				if (ni == D2_parts -1){
					t_exec_T3_adj*=1.0*D2_last/T;
					t_h2d_T3_B*=1.0*D2_last/T;
					t_h2d_T3_C*=1.0*D2_last/T;
				}
				if (ki == D3_parts -1){
					t_exec_T3_adj*=1.0*D3_last/T;
					t_h2d_T3_A*=1.0*D3_last/T;
					t_h2d_T3_B*=1.0*D3_last/T;
				}
				t_pipe_h2d+= t_h2d_T3_A + t_h2d_T3_B + t_h2d_T3_C;
				//printf("t_pipe_exec=%lf,t_pipe_h2d=%lf\n", t_pipe_exec,t_pipe_h2d);
			}

	t_total = fmax(t_pipe_exec,t_pipe_h2d) + t_exec_T3_adj + (1.0*D1_last/T)*(1.0*D2_last/T)*t_d2h_T3*model->V->out_loc[2];

	/*for (int i = 0; i < model->V->numT; i++){
		if (*model->V->Dim1[i] < 1 || *model->V->Dim2[i] < 1) error("CoCoModel_predict3: Invalid data struct dims");
		size_t Dim1_num = *model->V->Dim1[i]/T;
		if(*model->V->Dim1[i]%T > T/2) Dim1_num++;
		size_t Dim2_num = *model->V->Dim2[i]/T;
		if(*model->V->Dim2[i]%T > T/2) Dim2_num++;
		matrix_visualize(idx_matrix[i], Dim1_num, Dim2_num);
	}*/

	/*
	fprintf(stderr, "CoCopelia Simulator(T=%zu) predicted :\n"
	"\tt_h2d_T3: %lf ms ( %lf Gb/s)\n"
	"\tt_execT3: %lf ms (%lf GFlops/s)\n"
	"\tt_d2h_T3: %lf ms ( %lf Gb/s)\n"
	"\tt_total: %lf ms (%lf GFlops/s)\n\n",
	T, t_h2d_T3*1000, Gval_per_s(T*T*model->V->dtype_sz,t_h2d_T3),
	t_exec_T3*1000, Gval_per_s(dgemm_flops(T,T,T), t_exec_T3),
	t_d2h_T3*1000, Gval_per_s(T*T*model->V->dtype_sz,t_d2h_T3),
	t_total*1000, Gval_per_s(dgemm_flops(model->D1,model->D2,model->D3), t_total));
	*/
	return t_total;
}

///  Initializes the model for gemm
CoCoModel_p CoCoModel_gemm_init(CoCoModel_p out_model, short dev_id, char* func, gemm_backend_in_p func_data){
	char TransA = func_data->TransA, TransB = func_data->TransB;
	size_t M = func_data->M, N = func_data->N, K = func_data->K;
	short A_loc, A_out_loc = A_loc = CoCoGetPtrLoc(*func_data->A),
				B_loc, B_out_loc = B_loc = CoCoGetPtrLoc(*func_data->B),
				C_loc, C_out_loc = C_loc = CoCoGetPtrLoc(*func_data->C);
	size_t ldA = func_data->ldA, ldB = func_data->ldB, ldC = func_data->ldC;
	short lvl = 3;
#ifdef DEBUG
	lprintf(lvl-1, "|-----> CoCoModel_gemm_init(model, %c, %c, %zu, %zu, %zu, %d, %d, %d, %d, %d, %d, %zu, %zu, %zu, %d, %s)\n", TransA, TransB, M, N, K, A_loc, B_loc, C_loc, A_out_loc, B_out_loc, C_out_loc, ldA, ldB, ldC, dev_id, func);
#endif
	out_model->func = func;
	// Gemm Routine info
	out_model->V->numT = 3;

	if (!strcmp(func, "Dgemm")) out_model->V->dtype_sz = sizeof(double);
	else if (!strcmp(func, "Sgemm")) out_model->V->dtype_sz = sizeof(float);

	out_model->V->in[0] = 1;
	out_model->V->in[1] = 1;
	out_model->V->in[2] = 1;

	out_model->V->out[0] = 0;
	out_model->V->out[1] = 0;
	out_model->V->out[2] = 1;

	// Gemm Problem Specific values for Routine info functions
	out_model->flags->TransA = TransA;
	out_model->flags->TransB = TransB;

	out_model->D1 = M;
	out_model->D2 = N;
	out_model->D3 = K;

	out_model->V->Dim1[0] = &out_model->D1;
	out_model->V->Dim1[1] = &out_model->D3;
	out_model->V->Dim1[2] = &out_model->D1;

	out_model->V->Dim2[0] = &out_model->D3;
	out_model->V->Dim2[1] = &out_model->D2;
	out_model->V->Dim2[2] = &out_model->D2;

	out_model->V->loc[0] = A_loc;
	out_model->V->loc[1] = B_loc;
	out_model->V->loc[2] = C_loc;

	out_model->V->out_loc[0] = A_out_loc;
	out_model->V->out_loc[1] = B_out_loc;
	out_model->V->out_loc[2] = C_out_loc;

#ifdef DEBUG
	lprintf(lvl, "CoCoModel_gemm initalized for %s->\nInitial problem dims: D1 = %zu, D2 = %zu, D3 = %zu\n"
	"Data tiles : A(%zu,%zu), B(%zu,%zu), C(%zu,%zu) in loc (%d,%d,%d)\n", \
	func, out_model->D1, out_model->D2, out_model->D3, out_model->D1, out_model->D3, out_model->D3, out_model->D2, out_model->D1, out_model->D2, out_model->V->out_loc[0], out_model->V->out_loc[1], out_model->V->out_loc[2]);
	lprintf(lvl-1, "<-----|\n");
#endif
	return out_model;
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
	size_t min_T = 0, max_allowed_T = 0, ctr = 0;
	max_allowed_T = fmin(fmin(model->D1, model->D2),model->D3);
	min_T = ((GPUexec3Model_p)model->GPUexec_model_ptr)->T_lookup_buf[0];
	if (min_T > max_allowed_T){
		outparams->T = max_allowed_T;
		// FIXME: Undefined performance for tiles < than the smaller microbenchmark
		outparams->pred_t = 0;
		return outparams;
	}
	double temp_t, min_t = CoCoPeLiaModelPredict(model, ((GPUexec3Model_p)model->GPUexec_model_ptr)->T_lookup_buf[0], mode);
	size_t prev_trial_T = 0;
	if(min_t < 0) error("CoCoPeLiaModelOptimizeTile: First value in DM results in negative prediction");
	for (ctr = 1 ; ctr < ((GPUexec3Model_p)model->GPUexec_model_ptr)->lines ; ctr++){
		size_t trial_T = ((GPUexec3Model_p)model->GPUexec_model_ptr)->T_lookup_buf[ctr];
		if (trial_T > max_allowed_T) break;
		if (trial_T == prev_trial_T) continue;
		temp_t = CoCoPeLiaModelPredict(model, trial_T, mode);

		//fprintf(stderr, "Checking T = %zu\n : t = %lf ms\n", trial_T, temp_t*1000);
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
	lprintf(lvl, "T = %zu\n : t_min = %lf ms\n", min_T, min_t*1000);
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
		error("printModel: Invalid mode");
	}
}

tunableParams_p tunableParamsInit(){
	tunableParams_p outparams = (tunableParams_p) malloc(sizeof(struct tunableParams));
	outparams->T = 0;
	outparams->pred_t = 0;
	return outparams;
}

const char* printTunableParams(tunableParams_p params){
	char* buf = (char*) malloc(256*sizeof(char));
	sprintf(buf, "{%zu|%e}", params->T, params->pred_t);
	return buf;
}
