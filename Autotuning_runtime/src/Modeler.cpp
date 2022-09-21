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
#include "CoCoPeLiaModelWrap.hpp"
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

long int Modeler::getSKNum(int T){
	switch(problem){
		case BLAS1:
			error("CoCopeLiaGetSKNum: BLAS 1 Not implemented\n");
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

/******************************************************************************/
/************************ Prediction Functions ********************************/

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
		case HETERO_REUSE:
			if (T == -1 || used_devs == -1 || !used_dev_ids || !used_dev_relative_scores)
				error("Called Modeler::predict(mode=HETERO_REUSE) with undefined arguments\n");
			return PredictReuseHetero(this, T, used_devs, used_dev_ids, used_dev_relative_scores);
		case HETERO_BIDIRECTIONAL:
			if (T == -1 || used_devs == -1 || !used_dev_ids || !used_dev_relative_scores)
				error("Called Modeler::predict(mode=HETERO_BIDIRECTIONAL) with undefined arguments\n");
			return PredictBidirectionalHetero(this, T, used_devs, used_dev_ids, used_dev_relative_scores);
		case FULL_OVERLAP:
			return PredictFullOverlap(this);
		case NO_OVERLAP:
			return PredictZeroOverlap(this);
		default:
			error("CoCoPeLiaModelPredict: Invalid mode %s", printModel(mode));
			return 0;
	}
}
