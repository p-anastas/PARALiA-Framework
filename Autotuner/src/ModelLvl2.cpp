///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The 3-way concurency overlap prediction models for BLAS2
///

#include <stdlib.h>
#include <math.h>

#include "CoModel.hpp"
#include "GPUexec_lookup.hpp"
#include "Autotuner.hpp"
#include "ModelLvl1.hpp"
#include "unihelpers.hpp"
#include "Werkhoven.hpp"


///  Initializes the model for gemm
void CoCoModel_gemv_init(MD_p out_model, int dev_id, const char* func, void* func_data_wrapped){
  char TransA;
  long int M , N;
  short A_loc, A_out_loc, x_loc, x_out_loc, y_loc, y_out_loc;
  long int ldA, incx, incy;

  if (!strcmp(func, "Dgemv")) {
    gemv_backend_in<double>* func_data = (gemv_backend_in<double>*) func_data_wrapped;
    out_model->V->dtype_sz = sizeof(double);
    TransA = func_data->TransA;
  	M = func_data->M;
    N = func_data->N;
  	A_out_loc = A_loc = CoCoGetPtrLoc(*func_data->A);
  	x_out_loc = x_loc = CoCoGetPtrLoc(*func_data->x);
    y_out_loc = y_loc = CoCoGetPtrLoc(*func_data->y);
  	ldA = func_data->ldA;
    incx = func_data->incx;
    incy = func_data->incy;
  }
  else if (!strcmp(func, "Sgemv")){
    gemv_backend_in<float>* func_data = (gemv_backend_in<float>*) func_data_wrapped;
    out_model->V->dtype_sz = sizeof(float);
    TransA = func_data->TransA;
  	M = func_data->M;
    N = func_data->N;
  	A_out_loc = A_loc = CoCoGetPtrLoc(*func_data->A);
  	x_out_loc = x_loc = CoCoGetPtrLoc(*func_data->x);
    y_out_loc = y_loc = CoCoGetPtrLoc(*func_data->y);
  	ldA = func_data->ldA;
    incx = func_data->incx;
    incy = func_data->incy;
  }
	short lvl = 3;
#ifdef DEBUG
	lprintf(lvl-1, "|-----> CoCoModel_gemv_init(model,%c, %ld, %ld, %d, %d, %d, %d, %d, %d, %ld, %ld, %ld, %d, %s)\n",
		TransA, M, N, A_loc, x_loc, y_loc, A_out_loc, x_out_loc, y_out_loc, incx, incy, lda, dev_id, func);
#endif
	out_model->func = func;
	// Gemv Routine info
	out_model->V->numT = 3;

	if (!strcmp(func, "Dgemv")) out_model->V->dtype_sz = sizeof(double);

	out_model->V->in[0] = 1;
	out_model->V->in[1] = 1;
	out_model->V->in[2] = 1;

	out_model->V->out[0] = 0;
	out_model->V->out[1] = 0;
	out_model->V->out[2] = 1;

	out_model->flags->TransA = TransA;

	out_model->D1 = M;
	out_model->D2 = N;
	out_model->D3 = 1;

	out_model->V->Dim1[0] = &out_model->D1;
	out_model->V->Dim1[1] = &out_model->D1;
	out_model->V->Dim1[2] = &out_model->D2;

	out_model->V->Dim2[0] = &out_model->D2;
	out_model->V->Dim2[1] = &out_model->D3;
	out_model->V->Dim2[2] = &out_model->D3;

	out_model->V->loc[0] = A_loc;
	out_model->V->loc[1] = x_loc;
	out_model->V->loc[2] = y_loc;

	out_model->V->out_loc[0] = A_out_loc;
	out_model->V->out_loc[1] = x_out_loc;
	out_model->V->out_loc[2] = y_out_loc;

#ifdef DEBUG
	lprintf(lvl, "CoCoModel_gemv initalized for %s->\nInitial problem dims: D1 = %ld, D2 = %ld, D3 = %ld\n"
	"Data tiles : x(%ld), y(%ld), in loc (%d,%d)\n", \
	func, out_model->D1, out_model->D2, out_model->D3, out_model->D1, out_model->D1, out_model->V->out_loc[0], out_model->V->out_loc[1]);
	lprintf(lvl-1, "<-----|\n");
#endif
}


long int MinAllowedTBLAS2(MD_p model){
		long int result = GPUexec2MinT((GPUexec2Model_p)model->GPUexec_model_ptr);
		return std::max(result, (long int) 1024);
}

long int MaxAllowedTBLAS2(MD_p model){
		return fmin(model->D1, model->D2);
}

long int GetSKNumBLAS2(MD_p model, int T){
	return (model->D1/T + ((model->D1%T)? 1:0))
		*(model->D2/T + ((model->D2%T)? 1:0));
}

///  Initializes the model for gemm
void ModelFuncInitBLAS2(MD_p out_model, int dev_id, const char* func, void* func_data){
	if ( !strcmp(func, "Dgemv"))
		return CoCoModel_gemv_init(out_model, dev_id, func, func_data);
	else error("ModelFuncInitBLAS2: func %s not implemented\n", func);
}

double PredictFullOverlapBLAS2(MD_p model)
{
	short lvl = 4;
	double t_recv_full = 0, t_send_full = 0, t_exec_full = 0, t_total = 0;
	long int maxT = GPUexec2MaxT((GPUexec2Model_p)model->GPUexec_model_ptr);
	long int Tbig = GPUexec2NearestT((GPUexec2Model_p)model->GPUexec_model_ptr, fmin(maxT, fmin(model->D1,model->D2)));
	//fprintf(stderr, "Tbig = %ld\n", Tbig);
	t_exec_full = (model->D1*1.0/Tbig * model->D2*1.0/Tbig)* GPUexec2Model_predict((GPUexec2Model_p)model->GPUexec_model_ptr, Tbig, model->flags->TransA);
	if ( t_exec_full < 0){
		warning("CoCopeLiaPredictFullOverlapBLAS2: GPUexec31odel_predict submodel returned negative value, abort prediction\n");
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
	fprintf(stderr, "CoCopelia FullOverlap BLAS2 (dev = %d):\n"
	"\tt_recv_full: %lf ms ( %lf Gb/s)\n"
	"\tt_exec_full: %lf ms (%lf GFlops/s)\n"
	"\tt_send_full: %lf ms ( %lf Gb/s)\n"
	"\tt_total: %lf ms (%lf GFlops/s)\n\n",
	model->unit_id,
	t_recv_full*1000, Gval_per_s(recv_sz,t_recv_full),
	t_exec_full*1000, Gval_per_s(gemv_flops(model->D1, model->D2), t_exec_full),
	t_send_full*1000, Gval_per_s(send_sz,t_send_full),
	t_total*1000, Gval_per_s(gemv_flops(model->D1, model->D2), t_total));
#endif

	return t_total;
}

double PredictZeroOverlapBLAS2(MD_p model)
{
	short lvl = 4;
	double t_recv_full = 0, t_send_full = 0, t_exec_full = 0, t_total = 0;
	long int maxT = GPUexec2MaxT((GPUexec2Model_p)model->GPUexec_model_ptr);
	long int Tbig = GPUexec2NearestT((GPUexec2Model_p)model->GPUexec_model_ptr, fmin(maxT, fmin(model->D1,model->D2)));
	//fprintf(stderr, "Tbig = %ld\n", Tbig);
	t_exec_full = (model->D1*1.0/Tbig * model->D2*1.0/Tbig)* GPUexec2Model_predict((GPUexec2Model_p)model->GPUexec_model_ptr, Tbig, model->flags->TransA);
	if ( t_exec_full < 0){
		warning("CoCopeLiaPredictZeroOverlapBLAS2: GPUexec31odel_predict submodel returned negative value, abort prediction\n");
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
				warning("CoCopeLiaPredictZeroOverlapBLAS2: t_com_predict submodel idx = %d\
					returned negative value, abort prediction", idxize(model->V->loc[i]));
				return -1.0;
		}
		t_recv_full+= t_recv_tmp;
		t_send_full+= t_send_tmp;
	}

	t_total = t_exec_full + t_recv_full + t_send_full;
#ifdef DPDEBUG
	fprintf(stderr, "CoCopelia ZeroOverlap BLAS2 (dev = %d):\n"
	"\tt_recv_full: %lf ms ( %lf Gb/s)\n"
	"\tt_exec_full: %lf ms (%lf GFlops/s)\n"
	"\tt_send_full: %lf ms ( %lf Gb/s)\n"
	"\tt_total: %lf ms (%lf GFlops/s)\n\n",
	model->unit_id,
	t_recv_full*1000, Gval_per_s(recv_sz,t_recv_full),
	t_exec_full*1000, Gval_per_s(gemv_flops(model->D1, model->D2), t_exec_full),
	t_send_full*1000, Gval_per_s(send_sz,t_send_full),
	t_total*1000, Gval_per_s(gemv_flops(model->D1, model->D2), t_total));
#endif

	return t_total;
}

///  Predicts 3-way overlaped execution time for BLAS3 Square tilling blocking without data reuse.
double CoCopeLiaPredictBidirectionalBLAS2(MD_p model, long int T)
{
	short lvl = 4;
	double t_recv_T2[LOC_NUM] = {0}, t_send_T2[LOC_NUM] = {0}, t_exec_T2 = 0, t_total = 0;
	t_exec_T2 = GPUexec2Model_predict((GPUexec2Model_p) model->GPUexec_model_ptr, T, model->flags->TransA);
	if ( t_exec_T2 < 0){
		warning("CoCopeLiaPredictBidirectionalBLAS2: GPUexec2Model_predict submodel returned negative value, abort prediction");
		return -1.0;
	}

	for(int idx = 0; idx < LOC_NUM; idx++){
		t_recv_T2[idx] = t_com_predict(model->revlink[idx],T*model->V->dtype_sz);
		t_send_T2[idx] = t_com_predict(model->link[idx], T*model->V->dtype_sz);
		if(t_recv_T2[idx] < 0 || t_send_T2[idx] < 0 ){
			warning("CoCopeLiaPredictBidirectionalBLAS2: t_com_predict submodel idx = %d returned negative value, abort prediction", idx);
			return -1.0;
		}
	}

	int mv_dev_id = -1, dev_id_initlocs[LOC_NUM] = {};
	for (int i = 0; i < model->V->numT; i++) dev_id_initlocs[idxize(model->V->loc[i])] ++;
	for (int idx = 0; idx < LOC_NUM; idx++)
		if (dev_id_initlocs[idxize(mv_dev_id)] <  dev_id_initlocs[idx]) mv_dev_id = deidxize(idx);
		else if (dev_id_initlocs[idxize(mv_dev_id)] ==  dev_id_initlocs[idx]
			&& t_recv_T2[idxize(mv_dev_id)] < t_recv_T2[idx]) mv_dev_id = deidxize(idx);

#ifdef DPDEBUG
	lprintf(lvl, "Selecting  mv_dev_id =%d\n", mv_dev_id);
#endif
	double mv_t_recv_T2 = t_recv_T2[idxize(mv_dev_id)], mv_t_send_T2 = t_send_T2[idxize(mv_dev_id)];

	double t_over_T2;
	int numTin = 0, numTout = 0;

	double ker_over =  (1.0*model->D1/T) - 1;
	for (int i = 0; i < model->V->numT; i++){
		if (*model->V->Dim1[i] < 1 || *model->V->Dim2[i] < 1) error("CoCopeLiaPredictBidirectional: Invalid data struct dims");
		numTin += model->V->in[i] * remote(model->V->loc[i], model->unit_id);
		numTout += model->V->out[i] * remote(model->V->loc[i], model->unit_id);
	}
	// Use bidirectional magic here if needed
	t_over_T2 = t_com_bid_predict(model->revlink[idxize(mv_dev_id)], model->link[idxize(mv_dev_id)],
		T*model->V->dtype_sz*numTin,  T*model->V->dtype_sz*numTout);
	t_total = fmax(t_exec_T2, t_over_T2)* ker_over +
	+ fmax(t_exec_T2, numTout * mv_t_send_T2) + numTin * mv_t_recv_T2 + numTout * mv_t_send_T2;

#ifdef DPDEBUG
	fprintf(stderr, "CoCopelia Bidirectional(T=%ld)  BLAS2 (dev = %d) predicted :\n"
	"\t -> numTin = %d -> numTout = %d\n"
	"\tmv_t_recv_T2: %lf ms ( %lf Gb/s)\n"
	"\tt_execT2: %lf ms (%lf GFlops/s)\n"
	"\tmv_t_send_T2: %lf ms ( %lf Gb/s)\n"
	"\tt_over_T2: %lf ms\n"
	"\tt_total: %lf ms (%lf GFlops/s)\n\n",
	T, model->unit_id, numTin, numTout,
	mv_t_recv_T2*1000, Gval_per_s(T*model->V->dtype_sz,mv_t_recv_T2),
	t_exec_T2*1000, Gval_per_s(gemv_flops(T, T), t_exec_T2),
	mv_t_send_T2*1000, Gval_per_s(T*model->V->dtype_sz,mv_t_send_T2),
	t_over_T2*1000,
	t_total*1000, Gval_per_s(gemv_flops(model->D1, model->D2), t_total));
#endif

	return t_total;

}
/////// Here i am //////////
double PredictBidirectionalHeteroBLAS2(MD_p model, long int T, int used_devs, int* used_dev_ids,
	double* used_dev_relative_scores){
	short lvl = 4;
	long int prob_dims = 0, reset_D1 = model->D1, reset_D2 = model->D2;
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
		if (used_dev_ids[idx] == model->unit_id){ iloc = idx; break; }
	if (iloc == -1) error("CoCopeLiaPredictBidirectionalHeteroBLAS2:  model->unit_id = %d not found in used_dev_ids[%d]\n",
		model->unit_id, used_devs);
	double problem_percentage = used_dev_relative_scores[iloc];
#ifdef PDEBUG
	lprintf(lvl, "CoCopeLiaPredictBidirectionalHeteroBLAS2(dev_id=%d) prob_dims = %ld, problem_percentage = %lf\n",
		model->unit_id, prob_dims, problem_percentage);
#endif
	if (!strcmp(REL_PERF_MODE, "ROOT-PROBLEM")){
		if (reset_D1 != 1) model->D1 = (long int) reset_D1* 1.0* pow(problem_percentage, 1.0/prob_dims);
	}
#ifdef PDEBUG
	lprintf(lvl, "CoCopeLiaPredictBidirectionalHeteroBLAS2(dev_id=%d) Modified Dims D1 = %ld, imb_time_multiplier = %lf\n",
		model->unit_id, model->D1, imb_time_multiplier);
#endif
#endif
	double result = imb_time_multiplier* reduce_time_multiplier* CoCopeLiaPredictBidirectionalBLAS2(model, T);
	if (!strcmp(REL_PERF_MODE, "PERCENTILE")){
		#ifdef DPDEBUG
			lprintf(lvl, "CoCopeLiaPredictBidirectionalHeteroBLAS2(dev_id=%d) REL_PERF_MODE = PERCENTILE:\
			Modifying result with problem_percentage = %lf, old_res = %lf ms, new_res = %lf ms\n",
				model->unit_id, problem_percentage, result*1000, result*problem_percentage*1000);
		#endif
		result*=problem_percentage;
	}
	else if (!strcmp(REL_PERF_MODE, "ROOT-PROBLEM")){
		model->D1 = reset_D1;
	}
	else error("CoCopeLiaPredictBidirectionalHeteroBLAS2: Unknown REL_PERF_MODE = %s\n", REL_PERF_MODE);
	return result;
}
