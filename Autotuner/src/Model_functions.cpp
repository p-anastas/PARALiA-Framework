///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The 3-way concurency overlap prediction models for BLAS.
///

#include <stdlib.h>
#include <math.h>

// for DBL_MAX
#include <float.h>

#include "CoModel.hpp"
#include "GPUexec_lookup.hpp"
#include "Autotuner.hpp"
#include "ModelLvl3.hpp"
#include "ModelLvl1.hpp"
#include "linkmap.hpp"
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


double PredictReuseHetero(MD_p model, long int T, int active_unit_num = 1, int* active_unit_id_list = NULL,
	double* active_unit_score = NULL)
{
	switch(model->problem){
		case BLAS1:
			//return CoCopeLiaPredictReuseHeteroBLAS1(model, T);
			error("PredictReuseHetero: BLAS 1 Not implemented\n");
		case BLAS2:
			error("PredictReuseHetero: BLAS 2 Not implemented\n");
			return 0;
		case BLAS3:
			return PredictReuseHeteroBLAS3(model, T, active_unit_num, active_unit_id_list,
				active_unit_score);
		default:
			error("PredictReuseHetero: Invalid Problem %s", printProblem(model->problem));
	}
	return 0;
}

double PredictBidirectionalHetero(MD_p model, long int T, int active_unit_num = 1, int* active_unit_id_list = NULL,
	double* active_unit_score = NULL)
{
	switch(model->problem){
		case BLAS1:
			return PredictBidirectionalHeteroBLAS1(model, T, active_unit_num, active_unit_id_list,
				active_unit_score);
		case BLAS2:
			error("PredictBidirectionalHetero: BLAS 2 Not implemented\n");
		case BLAS3:
			error("PredictBidirectionalHetero: BLAS 3 Not implemented\n");
		default:
			error("PredictBidirectionalHetero: Invalid Problem %s", printProblem(model->problem));
	}
	return 0;
}

double share_mult(int dest, int src, int* active_unit_id_list, int active_unit_num){
	double mult = 1.0;
	if(links_share_bandwidth[idxize(dest)][idxize(src)][0] != -42 && 
		final_link_active[links_share_bandwidth[idxize(dest)][idxize(src)][0]]
					[links_share_bandwidth[idxize(dest)][idxize(src)][1]]) mult = 2.0;
	return mult;
}

double* PARALiaPredictLinkHeteroBLAS3_v2(MD_p model, long int T, int active_unit_num, int* active_unit_id_list,
	double* active_unit_score){
		double* result = (double*) calloc(4,sizeof(double));
		int used_unit_idx = -1;
		for(int unit_idx = 0; unit_idx < active_unit_num; unit_idx++) if(model->unit_id == active_unit_id_list[unit_idx]) used_unit_idx = unit_idx;
		if (used_unit_idx == - 1) error("PARALiaPredictLinkHeteroBLAS3: Model %p with unit_id = %d not present in given active_unit_id_list = %s\n",
			model, model->unit_id, printlist<int>(active_unit_id_list, active_unit_num));

		double t_recv_full = 0, t_send_full = 0, t_exec_full = 0, t_total = 0;
		long int maxT = GPUexec3MaxT((GPUexec3Model_p)model->GPUexec_model_ptr);
		long int Tbig = GPUexec3NearestT((GPUexec3Model_p)model->GPUexec_model_ptr,
			fmin(maxT, fmin(fmin(model->D1,model->D2), model->D3)));
		//fprintf(stderr, "Tbig = %ld\n", Tbig);
		t_exec_full = (model->D1*1.0/Tbig * model->D2*1.0/Tbig * model->D3*1.0/Tbig)*
			GPUexec3Model_predict((GPUexec3Model_p)model->GPUexec_model_ptr, Tbig, model->flags->TransA, model->flags->TransB);
		t_exec_full*= active_unit_score[used_unit_idx];
		if ( t_exec_full < 0)
			error("PARALiaPredictLinkHeteroBLAS3: GPUexec3Model_predict submodel returned negative value, abort prediction");

		long long recv_sz = 0, recv_sz_RONLY = 0, send_sz = 0;
		int recv_num_RONLY = 0;
		for (int i = 0; i < model->V->numT; i++){
			long long tmp_recv_sz = (long long) model->V->in[i]*(*model->V->Dim1[i])*(*model->V->Dim2[i])*
				model->V->dtype_sz*active_unit_score[used_unit_idx];
			long long tmp_send_sz =  (long long) model->V->out[i]*(*model->V->Dim1[i])*(*model->V->Dim2[i])*
				model->V->dtype_sz*active_unit_score[used_unit_idx];
			double t_recv_tmp = share_mult(model->unit_id, model->V->loc[i], active_unit_id_list, active_unit_num)*
				t_com_predict_shared(model->link[idxize(model->V->loc[i])], tmp_recv_sz);
			double t_send_tmp = share_mult(model->V->out_loc[i], model->unit_id, active_unit_id_list, active_unit_num)*
				t_com_predict_shared(model->link[idxize(model->V->out_loc[i])], tmp_send_sz);
			if(t_recv_tmp < 0 || t_send_tmp < 0 )	
				error("PARALiaPredictLinkHeteroBLAS3: t_recv_tmp(%d<-%d) = %lf, t_send_tmp(%d<-%d) = %lf,\
					abort prediction", model->unit_id, model->V->loc[i], t_recv_tmp, 
					model->V->out_loc[i], model->unit_id, t_send_tmp );

			recv_sz += tmp_recv_sz;
			send_sz += tmp_send_sz;
			if(!model->V->out[i] && model->V->loc[i] != model->unit_id) { recv_sz_RONLY+= recv_sz; recv_num_RONLY++; }
			t_recv_full+= t_recv_tmp;
			t_send_full+= t_send_tmp;
		}

		/// TODO: Extra transfers created from internal dims due to multi-unit spliting.
		/// Algorithm may vary for other BLAS3, but not at that bridge yet.
		/// The assumtion for extra transfers is made based on the 2D cyclic distribution,
		/// but the estimation is also useful for other distributions as a best case scenario (worse distributions -> more extra transfers).
		int D1_parts = sqrt(active_unit_num);
		int D2_parts = D1_parts;
		if (D1_parts ==0) { D2_parts = active_unit_num; D1_parts = 1; }
		else { /* find the most square decomposition of autotune_controller->active_unit_num in D1_parts x D2_parts */
			int g;
			for (g = D1_parts+1; g>0; --g) if (active_unit_num % g == 0) break;
			if (g==0) { D1_parts = active_unit_num; D2_parts = 1; }
			else { D1_parts = g; D2_parts = active_unit_num/g; }
		}
		double extra_transfer_ratio = (recv_num_RONLY)? (1.0*((D1_parts-1) + (D2_parts -1)))/recv_num_RONLY: 0;

#ifdef DPDEBUG
		fprintf(stderr, "PARALiaPredictLinkHeteroBLAS3(unit_num = %d) : D1_parts = %d, D2_parts = %d, extra_transfer_ratio = %lf\n",
		active_unit_num, D1_parts, D2_parts, extra_transfer_ratio);
#endif
		long long recv_sz_extra = extra_transfer_ratio * recv_sz_RONLY;
		//double t_recv_extra_optimistic_old = model->predictBestFriends_t(extra_transfer_ratio, recv_sz_extra, active_unit_num, active_unit_id_list);
	 	//double t_recv_extra_pesimistic = model->predictAvgBw_t(recv_sz_extra, active_unit_num, active_unit_id_list);
		double t_recv_extra_optimistic = model->predictSumBw_t(recv_sz_extra, active_unit_num, active_unit_id_list);

		double t_recv_extra = t_recv_extra_optimistic; // (t_recv_extra_optimistic + t_recv_extra_pesimistic)/2;

		t_total = fmax(t_exec_full, fmax(t_recv_full, fmax(t_recv_extra, t_send_full)));
				
		result[0] = t_exec_full; 
		result[1] = t_recv_full; 
		result[2] = t_recv_extra; 
		result[3] = t_send_full; 

#ifdef PDEBUG
		fprintf(stderr, "PARALia  PredictLinkHetero (Unit = %d, Unit_ratio = %.2lf%%):\n"
		"\tt_recv_full: %lf ms ( %lf Gb/s)\n"
		"\tt_recv_extra: %lf ms ( %lf Gb/s) -> (%.2lf%% bytes of full)\n"
		"\tt_exec_full: %lf ms (%lf GFlops/s)\n"
		"\tt_send_full: %lf ms ( %lf Gb/s)\n"
		"\tt_max: %lf ms (%lf GFlops/s)\n\n",
		model->unit_id, 100*active_unit_score[used_unit_idx], t_recv_full*1000, Gval_per_s(recv_sz,t_recv_full),
		t_recv_extra*1000, Gval_per_s(recv_sz_extra,t_recv_extra), 100*extra_transfer_ratio,
		t_exec_full*1000, Gval_per_s(model->getFlops()*active_unit_score[used_unit_idx], t_exec_full),
		t_send_full*1000, Gval_per_s(send_sz,t_send_full),
		t_total*1000, Gval_per_s(model->getFlops()*active_unit_score[used_unit_idx], t_total));
#endif

		return result;
	}

double* PARALiaPredictLinkHetero_v2(MD_p model, long int T, int active_unit_num, int* active_unit_id_list,
	double* active_unit_score)
{
	switch(model->problem){
		case BLAS1:
			error("PARALiaPredictLinkHetero_v2: BLAS 3 Not implemented\n");
		case BLAS2:
			error("PARALiaPredictLinkHetero_v2: BLAS 2 Not implemented\n");
		case BLAS3:
			return PARALiaPredictLinkHeteroBLAS3_v2(model, T, active_unit_num, active_unit_id_list,
				active_unit_score);
		default:
			error("PARALiaPredictLinkHetero_v2: Invalid Problem %s", printProblem(model->problem));
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
