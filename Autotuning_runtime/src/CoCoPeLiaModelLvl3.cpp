///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The 3-way concurency overlap prediction models for BLAS3
///

#include <stdlib.h>
#include <math.h>

#include "CoCoPeLiaCoModel.hpp"
#include "CoCoPeLiaGPUexec.hpp"
#include "CoCoPeLiaModel.hpp"
#include "CoCoPeLiaModelLvl3.hpp"
#include "unihelpers.hpp"
#include "Werkhoven.hpp"


double CoCopeLiaPredictFullOverlapBLAS3(CoCoModel_p model)
{
	short lvl = 4;
	double t_recv_full = 0, t_send_full = 0, t_exec_full = 0, t_total = 0;
	long int maxT = GPUexec3MaxT((GPUexec3Model_p)model->GPUexec_model_ptr);
	long int Tbig = GPUexec3NearestT((GPUexec3Model_p)model->GPUexec_model_ptr,
		fmin(maxT, fmin(fmin(model->D1,model->D2), model->D3)));
	//fprintf(stderr, "Tbig = %ld\n", Tbig);
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
	t_exec_full*1000, Gval_per_s(gemm_flops(model->D1,model->D2,model->D3), t_exec_full),
	t_send_full*1000, Gval_per_s(send_sz,t_send_full),
	t_total*1000, Gval_per_s(gemm_flops(model->D1,model->D2,model->D3), t_total));
#endif

	return t_total;
}

double CoCopeLiaPredictZeroOverlapBLAS3(CoCoModel_p model)
{
	short lvl = 4;
	double t_recv_full = 0, t_send_full = 0, t_exec_full = 0, t_total = 0;
	long int maxT = GPUexec3MaxT((GPUexec3Model_p)model->GPUexec_model_ptr);
	long int Tbig = GPUexec3NearestT((GPUexec3Model_p)model->GPUexec_model_ptr,
		fmin(maxT, fmin(fmin(model->D1,model->D2), model->D3)));
	//fprintf(stderr, "Tbig = %ld\n", Tbig);
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
	t_exec_full*1000, Gval_per_s(gemm_flops(model->D1,model->D2,model->D3), t_exec_full),
	t_send_full*1000, Gval_per_s(send_sz,t_send_full),
	t_total*1000, Gval_per_s(gemm_flops(model->D1,model->D2,model->D3), t_total));
#endif

	return t_total;
}

double CoCopeLiaPredictBaselineBLAS3(CoCoModel_p model, long int T)
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
	int numTin = 0, numTout = 0;

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
	fprintf(stderr, "CoCopelia (T=%ld) predicted :\n"
	"\t -> numTin = %d -> numTout = %d\n"
	"\tmv_t_recv_T3: %lf ms ( %lf Gb/s)\n"
	"\tt_execT3: %lf ms (%lf GFlops/s)\n"
	"\tmv_t_send_T3: %lf ms ( %lf Gb/s)\n"
	"\tt_total: %lf ms (%lf GFlops/s)\n\n",
	T, numTin, numTout,
	mv_t_recv_T3*1000, Gval_per_s(T*T*model->V->dtype_sz,mv_t_recv_T3),
	t_exec_T3*1000, Gval_per_s(gemm_flops(T,T,T), t_exec_T3),
	mv_t_send_T3*1000, Gval_per_s(T*T*model->V->dtype_sz,mv_t_send_T3),
	t_total*1000, Gval_per_s(gemm_flops(model->D1,model->D2,model->D3), t_total));
#endif

	return t_total;
}

double CoCopeLiaPredictDataLocBLAS3(CoCoModel_p model, long int T)
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
	int numTin = 0, numTout = 0;

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
	fprintf(stderr, "CoCopelia (T=%ld) predicted :\n"
	"\t -> numTin = %d -> numTout = %d\n"
	"\tmv_t_recv_T3: %lf ms ( %lf Gb/s)\n"
	"\tt_execT3: %lf ms (%lf GFlops/s)\n"
	"\tmv_t_send_T3: %lf ms ( %lf Gb/s)\n"
	"\tt_total: %lf ms (%lf GFlops/s)\n\n",
	T, numTin, numTout,
	mv_t_recv_T3*1000, Gval_per_s(T*T*model->V->dtype_sz,mv_t_recv_T3),
	t_exec_T3*1000, Gval_per_s(gemm_flops(T,T,T), t_exec_T3),
	mv_t_send_T3*1000, Gval_per_s(T*T*model->V->dtype_sz,mv_t_send_T3),
	t_total*1000, Gval_per_s(gemm_flops(model->D1,model->D2,model->D3), t_total));
#endif

	return t_total;
}

///  Predicts 3-way overlaped execution time for BLAS3 Square tilling blocking without data reuse.
double CoCopeLiaPredictBidirectionalBLAS3(CoCoModel_p model, long int T)
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
	int numTin = 0, numTout = 0;

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
	fprintf(stderr, "CoCopelia (T=%ld) predicted :\n"
	"\t -> numTin = %d -> numTout = %d\n"
	"\tmv_t_recv_T3: %lf ms ( %lf Gb/s)\n"
	"\tt_execT3: %lf ms (%lf GFlops/s)\n"
	"\tmv_t_send_T3t: %lf ms ( %lf Gb/s)\n"
	"\tt_over_T3: %lf ms\n"
	"\tt_total: %lf ms (%lf GFlops/s)\n\n",
	T, numTin, numTout,
	mv_t_recv_T3*1000, Gval_per_s(T*T*model->V->dtype_sz,mv_t_recv_T3),
	t_exec_T3*1000, Gval_per_s(gemm_flops(T,T,T), t_exec_T3),
	mv_t_send_T3*1000, Gval_per_s(T*T*model->V->dtype_sz,mv_t_send_T3),
	t_over_T3*1000,
	t_total*1000, Gval_per_s(gemm_flops(model->D1,model->D2,model->D3), t_total));
#endif

	return t_total;

}

double CoCopeLiaPredictReuseBLAS3(CoCoModel_p model, long int T)
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

#ifdef PDEBUG
	lprintf(lvl, "Selecting  mv_dev_id =%d\n", mv_dev_id);
#endif
	double mv_t_recv_T3 = t_recv_T3[idxize(mv_dev_id)], mv_t_send_T3 = t_send_T3[idxize(mv_dev_id)];

	int numTin = 0, numTout = 0;

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
	lprintf(lvl, "CoCopelia (T=%ld) predicted :\n"
	"\tmv_t_recv_T3: %lf ms ( %lf Gb/s)\n"
	"\t -> two_over = %lf -> one_over = %lf -> zero_over = %lf\n"
	"\tt_execT3: %lf ms (%lf GFlops/s)\n"
	"\tmv_t_send_T3: %lf ms ( %lf Gb/s)\n"
	"\tt_total: %lf ms (%lf GFlops/s)\n\n",
	T, mv_t_recv_T3*1000, Gval_per_s(T*T*model->V->dtype_sz,mv_t_recv_T3),
	two_over, one_over, zero_over,
	t_exec_T3*1000, Gval_per_s(gemm_flops(T,T,T), t_exec_T3),
	mv_t_send_T3*1000, Gval_per_s(T*T*model->V->dtype_sz,mv_t_send_T3),
	t_total*1000, Gval_per_s(gemm_flops(model->D1,model->D2,model->D3), t_total));
#endif

	return t_total;
}

/// TODO: currently d2h overlap is ignored
double CoCopeLiaPipelineEmulateBLAS3(CoCoModel_p model, long int T){
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

	long int D1_last = model->D1%T , D2_last = model->D2%T, D3_last= model->D3%T;
	long int D1_parts = model->D1/T , D2_parts = model->D2/T, D3_parts = model->D3/T;
	if (D1_last > T/2) D1_parts++;
	else D1_last+=T;
	if (D2_last > T/2) D2_parts++;
	else D2_last+=T;
	if (D3_last > T/2) D3_parts++;
	else D3_last+=T;
	//printf("D1_parts=%ld,D2_parts=%ld,D3_parts=%ld\n", D1_parts, D2_parts, D3_parts);
	//printf("D1_last=%ld,D2_last=%ld,D3_last=%ld\n", D1_last, D2_last, D3_last);

	if (!D1_parts || !D2_parts || !D3_parts) error("CoCoModel_pipeline_simulate3: Some dim cut is considered 0");
	int numTin = 0, numTout = 0;
	short *idx_matrix[model->V->numT];

	for (int i = 0; i < model->V->numT; i++){
		if (*model->V->Dim1[i] < 1 || *model->V->Dim2[i] < 1) error("CoCoModel_pipeline_simulate3: Invalid data struct dims");
		long int Dim1_num = *model->V->Dim1[i]/T;
		if(*model->V->Dim1[i]%T > T/2) Dim1_num++;
		long int Dim2_num = *model->V->Dim2[i]/T;
		if(*model->V->Dim2[i]%T > T/2) Dim2_num++;
		idx_matrix[i] = (short*) calloc (Dim1_num*Dim2_num, sizeof(short));
		/// First element 0 because its not accounted in pipeline
		if (model->V->loc[i]) for (int j = 0; j < Dim1_num*Dim2_num; j++) idx_matrix[i][j] = model->V->loc[i];
		//printf("Dim1_num=%ld,Dim2_num=%ld\n", Dim1_num, Dim2_num);
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
		long int Dim1_num = *model->V->Dim1[i]/T;
		if(*model->V->Dim1[i]%T > T/2) Dim1_num++;
		long int Dim2_num = *model->V->Dim2[i]/T;
		if(*model->V->Dim2[i]%T > T/2) Dim2_num++;
		matrix_visualize(idx_matrix[i], Dim1_num, Dim2_num);
	}*/

	/*
	fprintf(stderr, "CoCopelia Simulator(T=%ld) predicted :\n"
	"\tt_h2d_T3: %lf ms ( %lf Gb/s)\n"
	"\tt_execT3: %lf ms (%lf GFlops/s)\n"
	"\tt_d2h_T3: %lf ms ( %lf Gb/s)\n"
	"\tt_total: %lf ms (%lf GFlops/s)\n\n",
	T, t_h2d_T3*1000, Gval_per_s(T*T*model->V->dtype_sz,t_h2d_T3),
	t_exec_T3*1000, Gval_per_s(gemm_flops(T,T,T), t_exec_T3),
	t_d2h_T3*1000, Gval_per_s(T*T*model->V->dtype_sz,t_d2h_T3),
	t_total*1000, Gval_per_s(gemm_flops(model->D1,model->D2,model->D3), t_total));
	*/

	return t_total;
}

double CoCopeLiaPredictReuseHeteroBLAS3(CoCo_model* model, int used_devs, int* used_dev_ids,
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
	if (iloc == -1) error("CoCopeLiaPredictReuseHeteroBLAS3:  model->dev_id = %d not found in used_dev_ids[%d]\n",
		model->dev_id, used_devs);
	double problem_percentage = used_dev_relative_scores[iloc];
#ifdef PDEBUG
	lprintf(lvl, "CoCopeLiaPredictReuseHeteroBLAS3(dev_id=%d) prob_dims = %ld, problem_percentage = %lf\n",
		model->dev_id, prob_dims, problem_percentage);
#endif
	if (!strcmp(REL_PERF_MODE, "ROOT-PROBLEM")){
		if (reset_D1 != -1) model->D1 = (long int) reset_D1* 1.0* pow(problem_percentage, 1.0/prob_dims);
		if (reset_D2 != -1) model->D2 = (long int) reset_D2* 1.0* pow(problem_percentage, 1.0/prob_dims);
		if (reset_D3 != -1) model->D3 = (long int) reset_D3* 1.0* pow(problem_percentage, 1.0/prob_dims);
	}
#ifdef PDEBUG
	lprintf(lvl, "CoCopeLiaPredictReuseHeteroBLAS3(dev_id=%d) Modified Dims D1 = %ld, D2 = %ld, D3 = %ld, imb_time_multiplier = %lf, reduce_time_multiplier = %lf\n",
		model->dev_id, model->D1, model->D2, model->D3, imb_time_multiplier, reduce_time_multiplier);
#endif
#endif
	double result = imb_time_multiplier* reduce_time_multiplier* CoCopeLiaPredictReuseBLAS3(model, T);
	if (!strcmp(REL_PERF_MODE, "PERCENTILE")) result*=problem_percentage;
	else if (!strcmp(REL_PERF_MODE, "ROOT-PROBLEM")){
		model->D1 = reset_D1;
		model->D2 = reset_D2;
		model->D3 = reset_D3;
	}
	else error("CoCopeLiaPredictReuseHeteroBLAS3: Unknown REL_PERF_MODE = %s\n", REL_PERF_MODE);
	return result;
}

///  Initializes the model for gemm
CoCoModel_p CoCoModel_gemm_init(CoCoModel_p out_model, int dev_id, const char* func, gemm_backend_in_p func_data){
	char TransA = func_data->TransA, TransB = func_data->TransB;
	long int M = func_data->M, N = func_data->N, K = func_data->K;
	short A_loc, A_out_loc = A_loc = CoCoGetPtrLoc(*func_data->A),
				B_loc, B_out_loc = B_loc = CoCoGetPtrLoc(*func_data->B),
				C_loc, C_out_loc = C_loc = CoCoGetPtrLoc(*func_data->C);
	long int ldA = func_data->ldA, ldB = func_data->ldB, ldC = func_data->ldC;
	short lvl = 3;
#ifdef DEBUG
	lprintf(lvl-1, "|-----> CoCoModel_gemm_init(model, %c, %c, %ld, %ld, %ld, %d, %d, %d, %d, %d, %d, %ld, %ld, %ld, %d, %s)\n", TransA, TransB, M, N, K, A_loc, B_loc, C_loc, A_out_loc, B_out_loc, C_out_loc, ldA, ldB, ldC, dev_id, func);
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
	lprintf(lvl, "CoCoModel_gemm initalized for %s->\nInitial problem dims: D1 = %ld, D2 = %ld, D3 = %ld\n"
	"Data tiles : A(%ld,%ld), B(%ld,%ld), C(%ld,%ld) in loc (%d,%d,%d)\n", \
	func, out_model->D1, out_model->D2, out_model->D3, out_model->D1, out_model->D3, out_model->D3, out_model->D2, out_model->D1, out_model->D2, out_model->V->out_loc[0], out_model->V->out_loc[1], out_model->V->out_loc[2]);
	lprintf(lvl-1, "<-----|\n");
#endif
	return out_model;
}

long int CoCopeLiaMinAllowedTBLAS3(CoCoModel_p model){
		return GPUexec3MinT((GPUexec3Model_p)model->GPUexec_model_ptr);
}

long int CoCopeLiaMaxAllowedTBLAS3(CoCoModel_p model){
		return fmin(fmin(model->D1, model->D2),model->D3);
}

long int CoCopeLiaGetSKNumBLAS3(CoCoModel_p model, int T){
		return (model->D1/T + ((model->D1%T)? 1:0))
			*(model->D2/T + ((model->D2%T)? 1:0))
			*(model->D3/T + ((model->D3%T)? 1:0));
}

///  Initializes the model for gemm
CoCoModel_p CoCoModelFuncInitBLAS3(CoCoModel_p out_model, int dev_id, const char* func, void* func_data){
	if ( !strcmp(func, "Dgemm") || !strcmp(func, "Sgemm"))
		return CoCoModel_gemm_init(out_model, dev_id, func, (gemm_backend_in_p) func_data);
	else error("CoCoModelFuncInitBLAS1: func %s not implemented\n", func);
}
