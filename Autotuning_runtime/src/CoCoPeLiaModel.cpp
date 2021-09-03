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

///  Initializes the model for gemm
CoCoModel_p CoCoPeLiaModelInit(short dev_id, char* func, char flag1, char flag2, char flag3, size_t Dim1, size_t Dim2, size_t Dim3, short Loc1, short Loc2, short Loc3, short OutLoc1, short OutLoc2, short OutLoc3, size_t offset1, size_t offset2, size_t offset3){
	CoCoModel_p out_model = (CoCoModel_p) malloc(sizeof(struct CoCo_model));
	out_model->h2d = CoModel_init(dev_id, -1);
	out_model->d2h = CoModel_init(-1, dev_id);
	out_model->GPUexec_model_ptr = (void*) GPUexec3Model_init(dev_id, func);
	out_model->V = (Vdata_p) malloc(sizeof(struct V_struct));
	out_model->flags = (flagParams_p) malloc(sizeof(struct flagParams));

	if ( !strcmp(func, "Dgemm") || !strcmp(func, "Sgemm")) return CoCoModel_gemm_init(out_model, flag2, flag3, Dim1, Dim2, Dim3, Loc1, Loc2, Loc3, OutLoc1, OutLoc2, OutLoc3, offset1, offset2, offset3, dev_id, func);
	else error("CoCoPeLiaModelInit: Model for '%s' func not integrated", func); 
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

double CoCopeLiaPredictBaseline(CoCoModel_p model, size_t T)
{
	double t_h2d_T3 = 0, t_d2h_T3 = 0, t_exec_T3 = 0, t_total = 0, t_over_T3 = 0;
	t_exec_T3 = GPUexec3Model_predict((GPUexec3Model_p) model->GPUexec_model_ptr, T, model->flags->TransA, model->flags->TransB);
	t_h2d_T3 = t_com_predict(model->h2d, T*T*model->V->dtype_sz); //CoTile_predict(model->h2d, T, model->V->dtype_sz);	
	t_d2h_T3 = t_com_predict(model->d2h, T*T*model->V->dtype_sz);//CoTile_predict(model->d2h, T, model->V->dtype_sz);
	if ( t_exec_T3 < 0 || t_h2d_T3 < 0 || t_d2h_T3 < 0 ){
		if(t_exec_T3 < 0) warning("CoCopeLiaPredictBaseline: GPUexec3Model_predict submodel returned negative value, abort prediction");
		if(t_h2d_T3 < 0) warning("CoCopeLiaPredictBaseline: t_com_predict submodel returned negative value, abort prediction");
		if(t_d2h_T3 < 0) warning("CoCopeLiaPredictBaseline: t_com_predict submodel returned negative value, abort prediction");
		return -1.0;
	}

	double t_in_T, t_out_T;
	size_t numTin = 0, numTout = 0;
	
	double ker_over =  (1.0*model->D1/T)*(1.0*model->D2/T)*(1.0*model->D3/T) - 1;
	for (int i = 0; i < model->V->numT; i++){
		if (*model->V->Dim1[i] < 1 || *model->V->Dim2[i] < 1) error("CoCopeLiaPredictBaseline: Invalid data struct dims");
		numTin += model->V->in[i]; 
		numTout += model->V->out[i];
	}
	t_over_T3 = fmax(numTin*t_h2d_T3, t_d2h_T3*numTout);
	t_total = fmax(t_exec_T3, t_over_T3)* ker_over +
	+ t_exec_T3 + numTin * t_h2d_T3 + numTout * t_d2h_T3;

	/*
	fprintf(stderr, "CoCopelia (T=%zu) predicted :\n"
	"\t -> numTin = %d -> numTout = %d\n"
	"\tt_h2d_T3: %lf ms ( %lf Gb/s)\n"
	"\tt_execT3: %lf ms (%lf GFlops/s)\n"
	"\tt_d2h_T3: %lf ms ( %lf Gb/s)\n"
	"\tt_total: %lf ms (%lf GFlops/s)\n\n", 
	T, numTin, numTout,
	t_h2d_T3*1000, Gval_per_s(T*T*model->V->dtype_sz,t_h2d_T3),  
	t_exec_T3*1000, Gval_per_s(dgemm_flops(T,T,T), t_exec_T3), 
	t_d2h_T3*1000, Gval_per_s(T*T*model->V->dtype_sz,t_d2h_T3),
	t_total*1000, Gval_per_s(dgemm_flops(model->D1,model->D2,model->D3), t_total));
	*/

	return t_total;
}

double CoCopeLiaPredictDataLoc(CoCoModel_p model, size_t T)
{
	double t_h2d_T3 = 0, t_d2h_T3 = 0, t_exec_T3 = 0, t_total = 0, t_over_T3 = 0;
	t_exec_T3 = GPUexec3Model_predict((GPUexec3Model_p) model->GPUexec_model_ptr, T, model->flags->TransA, model->flags->TransB);
	t_h2d_T3 = t_com_predict(model->h2d, T*T*model->V->dtype_sz); //CoTile_predict(model->h2d, T, model->V->dtype_sz);	
	t_d2h_T3 = t_com_predict(model->d2h, T*T*model->V->dtype_sz);//CoTile_predict(model->d2h, T, model->V->dtype_sz);

	if ( t_exec_T3 < 0 || t_h2d_T3 < 0 || t_d2h_T3 < 0 ){
		if(t_exec_T3 < 0) warning("CoCopeLiaPredictDataLoc: GPUexec3Model_predict submodel returned negative value, abort prediction");
		if(t_h2d_T3 < 0) warning("CoCopeLiaPredictDataLoc: t_com_predict submodel returned negative value, abort prediction");
		if(t_d2h_T3 < 0) warning("CoCopeLiaPredictDataLoc: t_com_predict submodel returned negative value, abort prediction");
		return -1.0;
	}

	double t_in_T, t_out_T;
	size_t numTin = 0, numTout = 0;
	
	double ker_over =  (1.0*model->D1/T)*(1.0*model->D2/T)*(1.0*model->D3/T) - 1;
	for (int i = 0; i < model->V->numT; i++){
		if (*model->V->Dim1[i] < 1 || *model->V->Dim2[i] < 1) error("CoCopeLiaPredictDataLoc: Invalid data struct dims");
		numTin += model->V->in[i] * model->V->loc[i]; 
		numTout += model->V->out[i] * model->V->loc[i];
	}
	t_over_T3 = fmax(numTin*t_h2d_T3, t_d2h_T3*numTout);
	t_total = fmax(t_exec_T3, t_over_T3)* ker_over +
	+ t_exec_T3 + numTin * t_h2d_T3 + numTout * t_d2h_T3;

	/*
	fprintf(stderr, "CoCopelia (T=%zu) predicted :\n"
	"\t -> numTin = %d -> numTout = %d\n"
	"\tt_h2d_T3: %lf ms ( %lf Gb/s)\n"
	"\tt_execT3: %lf ms (%lf GFlops/s)\n"
	"\tt_d2h_T3: %lf ms ( %lf Gb/s)\n"
	"\tt_total: %lf ms (%lf GFlops/s)\n\n", 
	T, numTin, numTout,
	t_h2d_T3*1000, Gval_per_s(T*T*model->V->dtype_sz,t_h2d_T3),  
	t_exec_T3*1000, Gval_per_s(dgemm_flops(T,T,T), t_exec_T3), 
	t_d2h_T3*1000, Gval_per_s(T*T*model->V->dtype_sz,t_d2h_T3),
	t_total*1000, Gval_per_s(dgemm_flops(model->D1,model->D2,model->D3), t_total));
	*/

	return t_total;
}

///  Predicts 3-way overlaped execution time for BLAS3 Square tilling blocking without data reuse.
double CoCopeLiaPredictBidirectional(CoCoModel_p model, size_t T)
{
	double t_h2d_T3 = 0, t_d2h_T3 = 0, t_exec_T3 = 0, t_total = 0, t_over_T3 = 0;
	t_exec_T3 = GPUexec3Model_predict((GPUexec3Model_p) model->GPUexec_model_ptr, T, model->flags->TransA, model->flags->TransB);
	t_h2d_T3 = t_com_predict(model->h2d, T*T*model->V->dtype_sz); //CoTile_predict(model->h2d, T, model->V->dtype_sz);	
	t_d2h_T3 = t_com_predict(model->d2h, T*T*model->V->dtype_sz);//CoTile_predict(model->d2h, T, model->V->dtype_sz);

	if ( t_exec_T3 < 0 || t_h2d_T3 < 0 || t_d2h_T3 < 0 ){
		if(t_exec_T3 < 0) warning("CoCopeLiaPredictBidirectional: GPUexec3Model_predict submodel returned negative value, abort prediction");
		if(t_h2d_T3 < 0) warning("CoCopeLiaPredictBidirectional: t_com_predict submodel returned negative value, abort prediction");
		if(t_d2h_T3 < 0) warning("CoCopeLiaPredictBidirectional: t_com_predict submodel returned negative value, abort prediction");
		return -1.0;
	}

	double t_in_T, t_out_T;
	size_t numTin = 0, numTout = 0;
	
	double ker_over =  (1.0*model->D1/T)*(1.0*model->D2/T)*(1.0*model->D3/T) - 1;
	for (int i = 0; i < model->V->numT; i++){
		if (*model->V->Dim1[i] < 1 || *model->V->Dim2[i] < 1) error("CoCopeLiaPredictBidirectional: Invalid data struct dims");
		numTin += model->V->in[i] * model->V->loc[i]; 
		numTout += model->V->out[i] * model->V->loc[i];
	}
	// Use bidirectional magic here if needed
	t_over_T3 = t_com_bid_predict(model->h2d, model->d2h, T*T*model->V->dtype_sz*numTin,  T*T*model->V->dtype_sz*numTout);
	t_total = fmax(t_exec_T3, t_over_T3)* ker_over +
	+ t_exec_T3 + numTin * t_h2d_T3 + numTout * t_d2h_T3;

	/*
	fprintf(stderr, "CoCopelia (T=%zu) predicted :\n"
	"\t -> numTin = %d -> numTout = %d\n"
	"\tt_h2d_T3: %lf ms ( %lf Gb/s)\n"
	"\tt_execT3: %lf ms (%lf GFlops/s)\n"
	"\tt_d2h_T3: %lf ms ( %lf Gb/s)\n"
	"\tt_over_T3: %lf ms\n"
	"\tt_total: %lf ms (%lf GFlops/s)\n\n", 
	T, numTin, numTout,
	t_h2d_T3*1000, Gval_per_s(T*T*model->V->dtype_sz,t_h2d_T3),  
	t_exec_T3*1000, Gval_per_s(dgemm_flops(T,T,T), t_exec_T3), 
	t_d2h_T3*1000, Gval_per_s(T*T*model->V->dtype_sz,t_d2h_T3),
	t_over_T3*1000,
	t_total*1000, Gval_per_s(dgemm_flops(model->D1,model->D2,model->D3), t_total));
	*/

	return t_total;

}

double CoCopeLiaPredictReuse(CoCoModel_p model, size_t T)
{
	//fprintf(stderr, "\nCoCopeLiaPredictReuse ->\nProblem dims: D1 = %zu, D2 = %zu, D3 = %zu\nVdata(%d)->\n", model->D1, model->D2, model->D3, model->V->numT);

	double t_h2d_T3 = 0, t_d2h_T3 = 0, t_exec_T3 = 0, t_total = 0;
	t_exec_T3 = GPUexec3Model_predict((GPUexec3Model_p) model->GPUexec_model_ptr, T, model->flags->TransA, model->flags->TransB);
	t_h2d_T3 = t_com_predict(model->h2d, T*T*model->V->dtype_sz); //CoTile_predict(model->h2d, T, model->V->dtype_sz);	
	t_d2h_T3 = t_com_predict(model->d2h, T*T*model->V->dtype_sz);//CoTile_predict(model->d2h, T, model->V->dtype_sz);

	if ( t_exec_T3 < 0 || t_h2d_T3 < 0 || t_d2h_T3 < 0 ){
		if(t_exec_T3 < 0) warning("CoCopeLiaPredictReuse: GPUexec3Model_predict submodel returned negative value, abort prediction");
		if(t_h2d_T3 < 0) warning("CoCopeLiaPredictReuse: t_com_predict submodel returned negative value, abort prediction");
		if(t_d2h_T3 < 0) warning("CoCopeLiaPredictReuse: t_com_predict submodel returned negative value, abort prediction");
		return -1.0;
	}

	double t_in_T, t_out_T;
	size_t numTin = 0, numTout = 0;
	
	double zero_over = 0, one_over = 0, two_over = 0; 
	zero_over =  (1.0*model->D1/T)*(1.0*model->D2/T)*(1.0*model->D3/T) - 1;
	for (int i = 0; i < model->V->numT; i++){
		if (*model->V->Dim1[i] < 1 || *model->V->Dim2[i] < 1) error("CoCopeLiaPredictReuse: Invalid data struct dims");
		numTin += model->V->in[i] * model->V->loc[i]; 
		numTout += model->V->out[i] * model->V->out_loc[i]; 
		one_over+= model->V->in[i] * model->V->loc[i]*((1.0*(*model->V->Dim1[i]))/T)*((1.0*(*model->V->Dim2[i]))/T); // - 1 The -1 only if two_over is ommited
				
		if (t_h2d_T3 > t_exec_T3) {
		// two_over kernels calculated
			for (int j = i + 1; j < model->V->numT; j++)
				if (model->V->in[i] * model->V->loc[i] && model->V->in[j] * model->V->loc[j]){
					if ( model->V->Dim1[i] == model->V->Dim1[j] || model->V->Dim1[i] == model->V->Dim2[j]) two_over += ((1.0*(*model->V->Dim1[i]))/T) - 1; 
					else if ( model->V->Dim2[i] == model->V->Dim1[j] || model->V->Dim2[i] == model->V->Dim2[j]) two_over += ((1.0*(*model->V->Dim2[i]))/T) - 1; 
					else error("CoCopeLiaPredictReuse: something is wrong with my brilliant pointer comparisson idea");
			}
		} 
	}	
	// Performance Cheat
	if ( 2* t_h2d_T3 > t_exec_T3 && t_exec_T3 > t_h2d_T3)  two_over += (1.0*model->D3/T); 
	one_over -= (2*two_over + numTin); 
	zero_over -= (one_over + two_over); 
	t_total = t_exec_T3*(1 + zero_over) + 
	fmax(t_exec_T3, t_h2d_T3)* one_over +
	fmax(t_exec_T3, t_h2d_T3*2)* two_over +
	+ numTin * t_h2d_T3 + numTout * t_d2h_T3;

	/*
	fprintf(stderr, "CoCopelia (T=%d) predicted :\n"
	"\tt_h2d_T3: %lf ms ( %lf Gb/s)\n"
	"\t -> two_over = %lf -> one_over = %lf -> zero_over = %lf\n"
	"\tt_execT3: %lf ms (%lf GFlops/s)\n"
	"\tt_d2h_T3: %lf ms ( %lf Gb/s)\n"
	"\tt_total: %lf ms (%lf GFlops/s)\n\n", 
	T, t_h2d_T3*1000, Gval_per_s(T*T*model->V->dtype_sz,t_h2d_T3),  
	two_over, one_over, zero_over,
	t_exec_T3*1000, Gval_per_s(dgemm_flops(T,T,T), t_exec_T3), 
	t_d2h_T3*1000, Gval_per_s(T*T*model->V->dtype_sz,t_d2h_T3),
	t_total*1000, Gval_per_s(dgemm_flops(model->D1,model->D2,model->D3), t_total));
	*/
		
	return t_total; 
}

/// TODO: currently d2h overlap is ignored
double CoCopeLiaPipelineEmulate(CoCoModel_p model, size_t T){
	double t_h2d_T3 = 0, t_d2h_T3 = 0, t_exec_T3 = 0, t_total = 0;
	t_exec_T3 = GPUexec3Model_predict((GPUexec3Model_p) model->GPUexec_model_ptr, T, model->flags->TransA, model->flags->TransB);
	t_h2d_T3 = t_com_predict(model->h2d, T*T*model->V->dtype_sz); //CoTile_predict(model->h2d, T, model->V->dtype_sz);	
	t_d2h_T3 = t_com_predict(model->d2h, T*T*model->V->dtype_sz);//CoTile_predict(model->d2h, T, model->V->dtype_sz);
	
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
		numTin += model->V->in[i] * model->V->loc[i]; 
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
CoCoModel_p CoCoModel_gemm_init(CoCoModel_p out_model, char TransA, char TransB, size_t M, size_t N, size_t K, short A_loc, short B_loc, short C_loc, short A_out_loc, short B_out_loc, short C_out_loc, size_t ldA, size_t ldB, size_t ldC, short dev_id, char* func){
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
	if(min_t < 0) error("CoCoPeLiaModelOptimizeTile: First value in DM results in negative prediction");
	for (ctr = 1 ; ctr < ((GPUexec3Model_p)model->GPUexec_model_ptr)->lines ; ctr++){
		size_t trial_T = ((GPUexec3Model_p)model->GPUexec_model_ptr)->T_lookup_buf[ctr];
		if (trial_T > max_allowed_T) break;
		temp_t = CoCoPeLiaModelPredict(model, trial_T, mode);
		
		//fprintf(stderr, "Checking T = %zu\n : t = %lf ms\n", trial_T, temp_t*1000);	
		if (temp_t >= 0 && temp_t < min_t ){
			min_t = temp_t; 
			min_T = trial_T;
		}
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
	outparams->pred_t = outparams->cpuRatio = 0;
	return outparams;
}

const char* printTunableParams(tunableParams_p params){
	char* buf = (char*) malloc(256*sizeof(char));

	sprintf(buf, "{%zu|%.2lf|%e}", params->T, params->cpuRatio, params->pred_t);
	return buf;
}
	

