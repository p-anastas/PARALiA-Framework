///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief Some CUDA function calls with added error-checking
///
#ifndef COCOPELIA_MODEL_LVL1_H
#define COCOPELIA_MODEL_LVL1_H

#include "CoCoPeLiaCoModel.hpp"
#include "backend_wrappers.hpp"

CoCoModel_p CoCoModelFuncInitBLAS1(CoCoModel_p out_model, short dev_id, const char* func, void* func_data);

long int CoCopeLiaMinAllowedTBLAS1(CoCoModel_p model);
long int CoCopeLiaMaxAllowedTBLAS1(CoCoModel_p model);

/// A naive prediction of the full-overlap (~unreachable) performance of a modeled routine
double CoCopeLiaPredictFullOverlapBLAS1(CoCoModel_p model);
/// A naive prediction of the zero-overlap (~worst case) performance of a modeled routine
double CoCopeLiaPredictZeroOverlapBLAS1(CoCoModel_p model);

double WerkhovenModelPredictWrapperBLAS1(CoCo_model* model, long int T, short t_exec_method);
double CoCopeLiaPredictBaselineBLAS1(CoCoModel_p model, long int T);
double CoCopeLiaPredictDataLocBLAS1(CoCoModel_p model, long int T);
double CoCopeLiaPredictBidirectionalBLAS1(CoCoModel_p model, long int T);
double CoCopeLiaPredictReuseBLAS1(CoCoModel_p model, long int T);
double CoCopeLiaPipelineEmulateBLAS1(CoCoModel_p model, long int T);
double CoCopeLiaPredictBidirectionalHeteroBLAS1(CoCoModel_p model, short used_devs, short* used_dev_ids,
	double* used_dev_relative_scores, long int T);
  
#endif
