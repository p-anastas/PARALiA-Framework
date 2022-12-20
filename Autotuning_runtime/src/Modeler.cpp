///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The 3-way concurency overlap prediction models for BLAS
///

#include <stdlib.h>
#include <math.h>

#include "CoCoPeLiaCoModel.hpp"
#include "CoCoPeLiaGPUexec.hpp"
#include "Autotuning_runtime.hpp"
#include "Model_functions.hpp"
#include "CoCoPeLiaModelLvl3.hpp"
#include "CoCoPeLiaModelLvl1.hpp"
#include "unihelpers.hpp"
#include "Werkhoven.hpp"

/********************** Initialization/Modification ***************************/
///  Initializes the this for gemm
Modeler::Modeler(int dev_id, const char* func, void* func_data){
	short lvl = 3;
#ifdef DEBUG
	lprintf(lvl-1, "|-----> Modeler::Modeler(dev_id=%d,func=%s)\n", dev_id, func);
#endif
	for(int idx = 0; idx < LOC_NUM; idx++){
		short dev_idx_id = deidxize(idx);
		if(dev_idx_id!= dev_id){
			link[idx] = CoModel_init(dev_id, dev_idx_id);
			revlink[idx] = CoModel_init(dev_idx_id, dev_id);
		}
		else link[idx] = revlink[idx] = CoModel_init_local(dev_id);
	}
	V = (Vdata_p) malloc(sizeof(struct V_struct));
	flags = (flagParams_p) malloc(sizeof(struct flagParams));
	unit_id = dev_id;

	if ( !strcmp(func, "Daxpy") || !strcmp(func, "Saxpy")) problem = BLAS1;
	else if (0) problem = BLAS2;
	else if ( !strcmp(func, "Dgemm") || !strcmp(func, "Sgemm")) problem = BLAS3;
	else error("Modeler::Modeler: Problem type for '%s' func not integrated\n", func);

	switch(problem){
		case BLAS1:
			GPUexec_model_ptr = (void*) GPUexec1Model_init(dev_id, func);
			CoCoModelFuncInitBLAS1(this, dev_id, func, func_data);
			break;
		case BLAS2:
			error("Modeler::Modeler: GPUexec2Model_init Not Implemented\n");
		 	GPUexec_model_ptr = (void*) GPUexec2Model_init(dev_id, func);
			break;
		case BLAS3:
			GPUexec_model_ptr = (void*) GPUexec3Model_init(dev_id, func);
			CoCoModelFuncInitBLAS3(this, dev_id, func, func_data);
			break;
		default:
			error("Modeler::Modeler: Unreachable default reached\n");
	}
}

Modeler::~Modeler(){
	free(flags);
	free(V);
	/// FIXME: Should also free CoCoPeLia sub-models (e.g. CoModels, GPUexec models.)
}
/******************************************************************************/
/**************************** Helper Fuctions *********************************/

void Modeler::print(){
	fprintf(stderr, "->V = %p\n->link = [ ", V );
	for (int i = 0 ; i < LOC_NUM; i++) fprintf(stderr, "%p ", link[i]);
	fprintf(stderr, "]\n->revlink = [ ");
	for (int i = 0 ; i < LOC_NUM; i++) fprintf(stderr, "%p ", revlink[i]);
	fprintf(stderr, "] \n->func = %s\
	\n->problem = %s\n->D2, D2, D3  = %ld, %ld, %ld\n->flags = Not printed\n->GPUexec_model_ptr = %p\n->unit_id = %d\n\n",
	func, printProblem(problem),
	D1, D2, D3, GPUexec_model_ptr, unit_id);
	return;
}

long int Modeler::getMinT(){
	switch(problem){
		case BLAS1:
			return CoCopeLiaMinAllowedTBLAS1(this);
		case BLAS2:
			error("CoCopeLiaMinT: BLAS 2 Not implemented\n");
			return 0;
		case BLAS3:
			return CoCopeLiaMinAllowedTBLAS3(this);
		default:
			error("CoCopeLiaMinT: Invalid Problem %s", printProblem(problem));
	}
	return 0;
}

long int Modeler::getMaxT(){
	switch(problem){
		case BLAS1:
			return CoCopeLiaMaxAllowedTBLAS1(this);
		case BLAS2:
			error("CoCopeLiaMaxT: BLAS 2 Not implemented\n");
			return 0;
		case BLAS3:
			return CoCopeLiaMaxAllowedTBLAS3(this);
		default:
			error("CoCopeLiaMaxT: Invalid Problem %s", printProblem(problem));
	}
	return 0;
}

long int Modeler::getFlops(){
	if (strcmp(func,"gemm")) return gemm_flops(D1,D2,D3);
	else if (strcmp(func,"axpy")) return axpy_flops(D1);
	else error("Modeler::getFlops() not implemented for %s\n", func);
	return -1;
}

long int Modeler::getSKNum(int T){
	switch(problem){
		case BLAS1:
			return CoCopeLiaGetSKNumBLAS1(this, T);
		case BLAS2:
			error("CoCopeLiaGetSKNum: BLAS 2 Not implemented\n");
			return 0;
		case BLAS3:
			return CoCopeLiaGetSKNumBLAS3(this, T);
		default:
			error("CoCopeLiaGetSKNum: Invalid Problem %s", printProblem(problem));
	}
	return 0;
}


// Currently return the Watts for the last(larger) measurement
double Modeler::getGPUexecFull(){
	switch(problem){
		case BLAS1:
			{
				long int maxT = GPUexec1MaxT((GPUexec1Model_p)GPUexec_model_ptr);
				long int Tbig = GPUexec1NearestT((GPUexec1Model_p)GPUexec_model_ptr,
					fmin(maxT, D1));
				//fprintf(stderr, "Tbig = %ld\n", Tbig);
				return (D1*1.0/Tbig)* // TODO: Something (minor) might be missing here with flags->incx, flags->incy, should keep in mind.
					GPUexec1Model_predict((GPUexec1Model_p)GPUexec_model_ptr, Tbig);
			}
		case BLAS2:
			error("getGPUexecFull: BLAS 2 Not implemented\n");
		case BLAS3:
			{
				long int maxT = GPUexec3MaxT((GPUexec3Model_p)GPUexec_model_ptr);
				long int Tbig = GPUexec3NearestT((GPUexec3Model_p)GPUexec_model_ptr,
					fmin(maxT, fmin(fmin(D1,D2), D3)));
					//fprintf(stderr, "Tbig = %ld\n", Tbig);
				return (D1*1.0/Tbig * D2*1.0/Tbig * D3*1.0/Tbig)*
					GPUexec3Model_predict((GPUexec3Model_p)GPUexec_model_ptr, Tbig, flags->TransA, flags->TransB);
			}
		default:
			error("CoCoPeLiaGPUexecGetLines: Invalid Problem %s", printProblem(problem));
	}
	return 0;
}

// Currently return the Watts for the last(larger) measurement
double Modeler::getGPUexecWatts(){
	switch(problem){
		case BLAS1:
			return ((GPUexec1Model_p)GPUexec_model_ptr)->av_W_buf[((GPUexec1Model_p)GPUexec_model_ptr)->lines-1];
		case BLAS2:
			return ((GPUexec2Model_p)GPUexec_model_ptr)->av_W_buf[((GPUexec2Model_p)GPUexec_model_ptr)->lines-1];
		case BLAS3:
			return ((GPUexec3Model_p)GPUexec_model_ptr)->av_W_buf[((GPUexec3Model_p)GPUexec_model_ptr)->lines-1];
		default:
			error("CoCoPeLiaGPUexecGetLines: Invalid Problem %s", printProblem(problem));
	}
	return 0;
}

long int Modeler::getGPUexecLines(){
	switch(problem){
		case BLAS1:
			return ((GPUexec1Model_p)GPUexec_model_ptr)->lines;
		case BLAS2:
			return ((GPUexec2Model_p)GPUexec_model_ptr)->lines;
		case BLAS3:
			return ((GPUexec3Model_p)GPUexec_model_ptr)->lines;
		default:
			error("CoCoPeLiaGPUexecGetLines: Invalid Problem %s", printProblem(problem));
	}
	return 0;
}

long int Modeler::getGPUexecElem(int idx){
	switch(problem){
		case BLAS1:
			return ((GPUexec1Model_p)GPUexec_model_ptr)->T_lookup_buf[idx];
		case BLAS2:
			return ((GPUexec2Model_p)GPUexec_model_ptr)->T_lookup_buf[idx];
		case BLAS3:
			return ((GPUexec3Model_p)GPUexec_model_ptr)->T_lookup_buf[idx];
		default:
			error("CoCoPeLiaGPUexecGetLines: Invalid Problem %s", printProblem(problem));
	}
	return 0;
}

void Modeler::getDatalocs(int** dataloc_list_p, int* dataloc_num_p){
	for (int data_chunk = 0; data_chunk < V->numT; data_chunk++)
	 if (!is_in_list(V->loc[data_chunk], (*dataloc_list_p), (*dataloc_num_p)))
	 	(*dataloc_list_p)[(*dataloc_num_p)++] = V->loc[data_chunk];
}

void Modeler::getWDatalocs(int** dataloc_list_p, int* dataloc_num_p){
	for (int data_chunk = 0; data_chunk < V->numT; data_chunk++)
	 if (!is_in_list(V->loc[data_chunk], (*dataloc_list_p), (*dataloc_num_p)) && V->out[data_chunk])
	 	(*dataloc_list_p)[(*dataloc_num_p)++] = V->loc[data_chunk];
}

/******************************************************************************/
/************************ Prediction Functions ********************************/

double Modeler::predictBestFriends_t(double request_ratio, long long request_size, int active_unit_num, int* active_unit_id_list)
{
		double total_t = 0, remaining_request_ratio = request_ratio;
		int remaining_unit_id_list[active_unit_num], remaining_unit_num = active_unit_num;
		for (int i = 0; i < active_unit_num; i++) remaining_unit_id_list[i] = active_unit_id_list[i];
		while(remaining_request_ratio > 0.0){
			double curr_ratio = (remaining_request_ratio >= 1.0)? 1.0 : remaining_request_ratio;
			int i, i_out = 0, closest_remaining_unit_idx = idxize(remaining_unit_id_list[0]);
			for (i = 0; i < remaining_unit_num; i++){
				if (remaining_unit_id_list[i] == unit_id) continue;
				if(final_estimated_link_bw[idxize(unit_id)][idxize(remaining_unit_id_list[i])] >
					 final_estimated_link_bw[idxize(unit_id)][closest_remaining_unit_idx]){
					 	closest_remaining_unit_idx = idxize(remaining_unit_id_list[i]);
						i_out = i;
					}
			}
#ifdef DPDEBUG
			lprintf(0, "Closest %d friend Unit with bw[%d][%d] = %lf\n", active_unit_num - remaining_unit_num,
				idxize(unit_id), closest_remaining_unit_idx, final_estimated_link_bw[idxize(unit_id)][closest_remaining_unit_idx]);
#endif
			total_t+= t_com_predict_shared(link[closest_remaining_unit_idx], (long long)((1.0*curr_ratio/request_ratio)*request_size));
			remaining_unit_num--;
			remaining_unit_id_list[i_out] = remaining_unit_id_list[remaining_unit_num];
			remaining_request_ratio-=curr_ratio;
		}

		return total_t;
}

double Modeler::predictAvgBw_t(long long request_size, int active_unit_num, int* active_unit_id_list)
{
		double total_t = 0;
		for (int i = 0; i < active_unit_num; i++){
			if (active_unit_id_list[i] == unit_id) continue;
			total_t += t_com_predict_shared(link[idxize(active_unit_id_list[i])],
					request_size/(active_unit_num - 1));
		}
		return total_t;
}

double Modeler::predict(ModelType mode, long int T, int used_devs, int* used_dev_ids, double* used_dev_relative_scores){
	switch(mode){
		case WERKHOVEN:
			return WerkhovenModelPredictWrapper(this, T, 0);
		case WERKHOVEN_DATALOC:
			return WerkhovenModelPredictWrapper(this, T, 1);
		case WERKHOVEN_LOOKUP_EXEC_TILES:
			return WerkhovenModelPredictWrapper(this, T, 2);
		case COCOPELIA_BASELINE:
			return CoCopeLiaPredictBaseline(this, T);
		case COCOPELIA_DATALOC:
			return CoCopeLiaPredictDataLoc(this, T);
		case COCOPELIA_BIDIRECTIONAL:
			return CoCopeLiaPredictBidirectional(this, T);
		case COCOPELIA_REUSE:
			return CoCopeLiaPredictReuse(this, T);
		case COCOPELIA_PIPELINE_EMULATE:
			return CoCopeLiaPipelineEmulate(this, T);
		case FULL_OVERLAP:
			return PredictFullOverlap(this);
		case NO_OVERLAP:
			return PredictZeroOverlap(this);
		case HETERO_REUSE:
			if (T == -1 || used_devs == -1 || !used_dev_ids || !used_dev_relative_scores)
				error("Called Modeler::predict(mode=HETERO_REUSE) with undefined arguments\n");
			return PredictReuseHetero(this, T, used_devs, used_dev_ids, used_dev_relative_scores);
		case HETERO_BIDIRECTIONAL:
			if (T == -1 || used_devs == -1 || !used_dev_ids || !used_dev_relative_scores)
				error("Called Modeler::predict(mode=HETERO_BIDIRECTIONAL) with undefined arguments\n");
			return PredictBidirectionalHetero(this, T, used_devs, used_dev_ids, used_dev_relative_scores);
		case PARALIA_HETERO_LINK_BASED:
			if (T == -1 || used_devs == -1 || !used_dev_ids || !used_dev_relative_scores)
				error("Called Modeler::predict(mode=HETERO_REUSE) with undefined arguments\n");
			return PARALiaPredictLinkHetero(this, T, used_devs, used_dev_ids, used_dev_relative_scores);
		default:
			error("CoCoPeLiaModelPredict: Invalid mode %s", printModel(mode));
			return 0;
	}
}
