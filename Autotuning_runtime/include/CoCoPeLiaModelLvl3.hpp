///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief Some CUDA function calls with added error-checking
///
#ifndef COCOPELIA_MODEL_LVL3_H
#define COCOPELIA_MODEL_LVL3_H

#include "CoCoPeLiaCoModel.hpp"
#include "backend_wrappers.hpp"

CoCoModel_p CoCoModelFuncInitBLAS3(CoCoModel_p out_model, short dev_id, const char* func, void* func_data);

/// A naive prediction of the full-overlap (~unreachable) performance of a modeled routine
double CoCopeLiaPredictFullOverlapBLAS3(CoCoModel_p model);
/// A naive prediction of the zero-overlap (~worst case) performance of a modeled routine
double CoCopeLiaPredictZeroOverlapBLAS3(CoCoModel_p model);

double WerkhovenModelPredictWrapperBLAS3(CoCo_model* model, long int T, short t_exec_method);
double CoCopeLiaPredictBaselineBLAS3(CoCoModel_p model, long int T);
double CoCopeLiaPredictDataLocBLAS3(CoCoModel_p model, long int T);
double CoCopeLiaPredictBidirectionalBLAS3(CoCoModel_p model, long int T);
double CoCopeLiaPredictReuseBLAS3(CoCoModel_p model, long int T);
double CoCopeLiaPipelineEmulateBLAS3(CoCoModel_p model, long int T);

#endif
