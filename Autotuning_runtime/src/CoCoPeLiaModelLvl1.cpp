///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The 3-way concurency overlap prediction models for BLAS1
///

#include <stdlib.h>
#include <math.h>

#include "CoCoPeLiaCoModel.hpp"
#include "CoCoPeLiaGPUexec.hpp"
#include "CoCoPeLiaModel.hpp"
#include "CoCoPeLiaModelLvl1.hpp"
#include "unihelpers.hpp"
#include "Werkhoven.hpp"


///  Initializes the model for gemm
CoCoModel_p CoCoModel_axpy_init(CoCoModel_p out_model, int dev_id, const char* func, axpy_backend_in_p func_data){
	long int N = func_data->N;
	short x_loc, x_out_loc = x_loc = CoCoGetPtrLoc(*func_data->x),
				y_loc, y_out_loc = y_loc = CoCoGetPtrLoc(*func_data->y);
	long int incx = func_data->incx, incy = func_data->incy;
	short lvl = 3;
#ifdef DEBUG
	lprintf(lvl-1, "|-----> CoCoModel_axpy_init(model,%ld, %d, %d, %d, %d, %ld, %ld, %d, %s)\n",
		N, x_loc, y_loc, x_out_loc, y_out_loc, incx, incy, dev_id, func);
#endif
	out_model->func = func;
	// Axpy Routine info
	out_model->V->numT = 2;

	if (!strcmp(func, "Daxpy")) out_model->V->dtype_sz = sizeof(double);
	else if (!strcmp(func, "Saxpy")) out_model->V->dtype_sz = sizeof(float);

	out_model->V->in[0] = 1;
	out_model->V->in[1] = 1;

	out_model->V->out[0] = 0;
	out_model->V->out[1] = 1;

	out_model->D1 = N;
	out_model->D2 = 1;
	out_model->D3 = 1;

	out_model->V->Dim1[0] = &out_model->D1;
	out_model->V->Dim1[1] = &out_model->D1;

	out_model->V->Dim2[0] = &out_model->D2;
	out_model->V->Dim2[1] = &out_model->D2;

	out_model->V->loc[0] = x_loc;
	out_model->V->loc[1] = y_loc;

	out_model->V->out_loc[0] = x_out_loc;
	out_model->V->out_loc[1] = y_out_loc;

#ifdef DEBUG
	lprintf(lvl, "CoCoModel_axpy initalized for %s->\nInitial problem dims: D1 = %ld, D2 = %ld, D3 = %ld\n"
	"Data tiles : x(%ld), y(%ld), in loc (%d,%d)\n", \
	func, out_model->D1, out_model->D2, out_model->D3, out_model->D1, out_model->D1, out_model->V->out_loc[0], out_model->V->out_loc[1]);
	lprintf(lvl-1, "<-----|\n");
#endif
	return out_model;
}

long int CoCopeLiaMinAllowedTBLAS1(CoCoModel_p model){
		return GPUexec1MinT((GPUexec1Model_p)model->GPUexec_model_ptr);
}

long int CoCopeLiaMaxAllowedTBLAS1(CoCoModel_p model){
		return model->D1;
}

///  Initializes the model for gemm
CoCoModel_p CoCoModelFuncInitBLAS1(CoCoModel_p out_model, int dev_id, const char* func, void* func_data){
	if ( !strcmp(func, "Daxpy") || !strcmp(func, "Saxpy"))
		return CoCoModel_axpy_init(out_model, dev_id, func, (axpy_backend_in_p) func_data);
	else error("CoCoModelFuncInitBLAS1: func %s not implemented\n", func);
}

double CoCopeLiaPredictFullOverlapBLAS1(CoCoModel_p model)
{
	short lvl = 4;
	double t_recv_full = 0, t_send_full = 0, t_exec_full = 0, t_total = 0;
	long int maxT = GPUexec1MaxT((GPUexec1Model_p)model->GPUexec_model_ptr);
	long int minT = CoCopeLiaMinAllowedTBLAS1(model);
	long int Tbig = GPUexec1NearestT((GPUexec1Model_p)model->GPUexec_model_ptr, fmin(maxT, minT));
	//fprintf(stderr, "Tbig = %ld\n", Tbig);
	t_exec_full = (model->D1*1.0/Tbig)* GPUexec1Model_predict((GPUexec1Model_p)model->GPUexec_model_ptr, Tbig);
	if ( t_exec_full < 0){
		warning("CoCopeLiaPredictFullOverlapBLAS1: GPUexec31odel_predict submodel returned negative value, abort prediction\n");
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
	fprintf(stderr, "CoCopelia FullOverlap BLAS1 (dev = %d):\n"
	"\tt_recv_full: %lf ms ( %lf Gb/s)\n"
	"\tt_exec_full: %lf ms (%lf GFlops/s)\n"
	"\tt_send_full: %lf ms ( %lf Gb/s)\n"
	"\tt_total: %lf ms (%lf GFlops/s)\n\n",
	model->dev_id,
	t_recv_full*1000, Gval_per_s(recv_sz,t_recv_full),
	t_exec_full*1000, Gval_per_s(axpy_flops(model->D1), t_exec_full),
	t_send_full*1000, Gval_per_s(send_sz,t_send_full),
	t_total*1000, Gval_per_s(axpy_flops(model->D1), t_total));
#endif

	return t_total;
}

double CoCopeLiaPredictZeroOverlapBLAS1(CoCoModel_p model)
{
	short lvl = 4;
	double t_recv_full = 0, t_send_full = 0, t_exec_full = 0, t_total = 0;
	long int maxT = GPUexec1MaxT((GPUexec1Model_p)model->GPUexec_model_ptr);
	long int minT = CoCopeLiaMinAllowedTBLAS1(model);
	long int Tbig = GPUexec1NearestT((GPUexec1Model_p)model->GPUexec_model_ptr, fmin(maxT, minT));
	//fprintf(stderr, "Tbig = %ld\n", Tbig);
	t_exec_full = (model->D1*1.0/Tbig)* GPUexec1Model_predict((GPUexec1Model_p)model->GPUexec_model_ptr, Tbig);
	if ( t_exec_full < 0){
		warning("CoCopeLiaPredictZeroOverlapBLAS1: GPUexec31odel_predict submodel returned negative value, abort prediction\n");
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
				warning("CoCopeLiaPredictZeroOverlapBLAS1: t_com_predict submodel idx = %d\
					returned negative value, abort prediction", idxize(model->V->loc[i]));
				return -1.0;
		}
		t_recv_full+= t_recv_tmp;
		t_send_full+= t_send_tmp;
	}

	t_total = t_exec_full + t_recv_full + t_send_full;
#ifdef DPDEBUG
	fprintf(stderr, "CoCopelia ZeroOverlap BLAS1 (dev = %d):\n"
	"\tt_recv_full: %lf ms ( %lf Gb/s)\n"
	"\tt_exec_full: %lf ms (%lf GFlops/s)\n"
	"\tt_send_full: %lf ms ( %lf Gb/s)\n"
	"\tt_total: %lf ms (%lf GFlops/s)\n\n",
	model->dev_id,
	t_recv_full*1000, Gval_per_s(recv_sz,t_recv_full),
	t_exec_full*1000, Gval_per_s(axpy_flops(model->D1), t_exec_full),
	t_send_full*1000, Gval_per_s(send_sz,t_send_full),
	t_total*1000, Gval_per_s(axpy_flops(model->D1), t_total));
#endif

	return t_total;
}

///  Predicts 3-way overlaped execution time for BLAS3 Square tilling blocking without data reuse.
double CoCopeLiaPredictBidirectionalBLAS1(CoCoModel_p model, long int T)
{
	short lvl = 4;
	double t_recv_T1[LOC_NUM] = {0}, t_send_T1[LOC_NUM] = {0}, t_exec_T1 = 0, t_total = 0;
	t_exec_T1 = GPUexec1Model_predict((GPUexec1Model_p) model->GPUexec_model_ptr, T);
	if ( t_exec_T1 < 0){
		warning("CoCopeLiaPredictBidirectionalBLAS1: GPUexec1Model_predict submodel returned negative value, abort prediction");
		return -1.0;
	}

	for(int idx = 0; idx < LOC_NUM; idx++){
		t_recv_T1[idx] = t_com_predict(model->revlink[idx],T*model->V->dtype_sz);
		t_send_T1[idx] = t_com_predict(model->link[idx], T*model->V->dtype_sz);
		if(t_recv_T1[idx] < 0 || t_send_T1[idx] < 0 ){
			warning("CoCopeLiaPredictBidirectionalBLAS1: t_com_predict submodel idx = %d returned negative value, abort prediction", idx);
			return -1.0;
		}
	}

	int mv_dev_id = -1, dev_id_initlocs[LOC_NUM] = {};
	for (int i = 0; i < model->V->numT; i++) dev_id_initlocs[idxize(model->V->loc[i])] ++;
	for (int idx = 0; idx < LOC_NUM; idx++)
		if (dev_id_initlocs[idxize(mv_dev_id)] <  dev_id_initlocs[idx]) mv_dev_id = deidxize(idx);
		else if (dev_id_initlocs[idxize(mv_dev_id)] ==  dev_id_initlocs[idx]
			&& t_recv_T1[idxize(mv_dev_id)] < t_recv_T1[idx]) mv_dev_id = deidxize(idx);

#ifdef DPDEBUG
	lprintf(lvl, "Selecting  mv_dev_id =%d\n", mv_dev_id);
#endif
	double mv_t_recv_T1 = t_recv_T1[idxize(mv_dev_id)], mv_t_send_T1 = t_send_T1[idxize(mv_dev_id)];

	double t_over_T1;
	int numTin = 0, numTout = 0;

	double ker_over =  (1.0*model->D1/T) - 1;
	for (int i = 0; i < model->V->numT; i++){
		if (*model->V->Dim1[i] < 1 || *model->V->Dim2[i] < 1) error("CoCopeLiaPredictBidirectional: Invalid data struct dims");
		numTin += model->V->in[i] * remote(model->V->loc[i], model->dev_id);
		numTout += model->V->out[i] * remote(model->V->loc[i], model->dev_id);
	}
	// Use bidirectional magic here if needed
	t_over_T1 = t_com_bid_predict(model->revlink[idxize(mv_dev_id)], model->link[idxize(mv_dev_id)],
		T*model->V->dtype_sz*numTin,  T*model->V->dtype_sz*numTout);
	t_total = fmax(t_exec_T1, t_over_T1)* ker_over +
	+ fmax(t_exec_T1, numTout * mv_t_send_T1) + numTin * mv_t_recv_T1 + numTout * mv_t_send_T1;

#ifdef DPDEBUG
	fprintf(stderr, "CoCopelia Bidirectional(T=%ld)  BLAS1 (dev = %d) predicted :\n"
	"\t -> numTin = %d -> numTout = %d\n"
	"\tmv_t_recv_T1: %lf ms ( %lf Gb/s)\n"
	"\tt_execT1: %lf ms (%lf GFlops/s)\n"
	"\tmv_t_send_T1: %lf ms ( %lf Gb/s)\n"
	"\tt_over_T1: %lf ms\n"
	"\tt_total: %lf ms (%lf GFlops/s)\n\n",
	T, model->dev_id, numTin, numTout,
	mv_t_recv_T1*1000, Gval_per_s(T*model->V->dtype_sz,mv_t_recv_T1),
	t_exec_T1*1000, Gval_per_s(axpy_flops(T), t_exec_T1),
	mv_t_send_T1*1000, Gval_per_s(T*model->V->dtype_sz,mv_t_send_T1),
	t_over_T1*1000,
	t_total*1000, Gval_per_s(axpy_flops(model->D1), t_total));
#endif

	return t_total;

}

double CoCopeLiaPredictBidirectionalHeteroBLAS1(CoCo_model* model, int used_devs, int* used_dev_ids,
	double* used_dev_relative_scores, long int T){
	short lvl = 4;
	long int prob_dims = 0, reset_D1 = model->D1;
	double imb_time_multiplier = 1.0, reduce_time_multiplier = 1.0;
#define ENABLE_HETERO_RELATIVE_DIMS
#ifdef ENABLE_HETERO_RELATIVE_DIMS
	if (reset_D1 != 1){
#ifdef TILE_IMBALANCE_PENALTY
		if (reset_D1%T) imb_time_multiplier+=TILE_IMBALANCE_PENALTY;
#endif
		prob_dims++;
	}
	short iloc = -1;
	for (int idx = 0; idx < used_devs; idx++)
		if (used_dev_ids[idx] == model->dev_id){ iloc = idx; break; }
	if (iloc == -1) error("CoCopeLiaPredictBidirectionalHeteroBLAS1:  model->dev_id = %d not found in used_dev_ids[%d]\n",
		model->dev_id, used_devs);
	double problem_percentage = used_dev_relative_scores[iloc];
#ifdef PDEBUG
	lprintf(lvl, "CoCopeLiaPredictBidirectionalHeteroBLAS1(dev_id=%d) prob_dims = %ld, problem_percentage = %lf\n",
		model->dev_id, prob_dims, problem_percentage);
#endif
	if (!strcmp(REL_PERF_MODE, "ROOT-PROBLEM")){
		if (reset_D1 != 1) model->D1 = (long int) reset_D1* 1.0* pow(problem_percentage, 1.0/prob_dims);
	}
#ifdef PDEBUG
	lprintf(lvl, "CoCopeLiaPredictBidirectionalHeteroBLAS1(dev_id=%d) Modified Dims D1 = %ld, imb_time_multiplier = %lf\n",
		model->dev_id, model->D1, imb_time_multiplier);
#endif
#endif
	double result = imb_time_multiplier* reduce_time_multiplier* CoCopeLiaPredictBidirectionalBLAS1(model, T);
	if (!strcmp(REL_PERF_MODE, "PERCENTILE")){
		#ifdef DPDEBUG
			lprintf(lvl, "CoCopeLiaPredictBidirectionalHeteroBLAS1(dev_id=%d) REL_PERF_MODE = PERCENTILE:\
			Modifying result with problem_percentage = %lf, old_res = %lf ms, new_res = %lf ms\n",
				model->dev_id, problem_percentage, result*1000, result*problem_percentage*1000);
		#endif
		result*=problem_percentage;
	}
	else if (!strcmp(REL_PERF_MODE, "ROOT-PROBLEM")){
		model->D1 = reset_D1;
	}
	else error("CoCopeLiaPredictBidirectionalHeteroBLAS1: Unknown REL_PERF_MODE = %s\n", REL_PERF_MODE);
	return result;
}
