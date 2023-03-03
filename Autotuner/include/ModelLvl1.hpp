///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief Some CUDA function calls with added error-checking
///
#ifndef MODEL_LVL1_H
#define MODEL_LVL1_H

#include "CoModel.hpp"
#include "backend_wrappers.hpp"

void ModelFuncInitBLAS1(MD_p out_model, int dev_id, const char* func, void* func_data);

long int MinAllowedTBLAS1(MD_p model);
long int MaxAllowedTBLAS1(MD_p model);
long int GetSKNumBLAS1(MD_p model, int T);

/// A naive prediction of the full-overlap (~unreachable) performance of a modeled routine
double PredictFullOverlapBLAS1(MD_p model);
/// A naive prediction of the zero-overlap (~worst case) performance of a modeled routine
double PredictZeroOverlapBLAS1(MD_p model);

double WerkhovenModelPredictWrapperBLAS1(MD_p model, long int T, short t_exec_method);
double CoCopeLiaPredictBaselineBLAS1(MD_p model, long int T);
double CoCopeLiaPredictDataLocBLAS1(MD_p model, long int T);
double CoCopeLiaPredictBidirectionalBLAS1(MD_p model, long int T);
double CoCopeLiaPredictReuseBLAS1(MD_p model, long int T);
double CoCopeLiaPipelineEmulateBLAS1(MD_p model, long int T);

double PredictBidirectionalHeteroBLAS1(MD_p model, long int T, int used_devs, int* used_dev_ids,
	double* used_dev_relative_scores);

#endif
