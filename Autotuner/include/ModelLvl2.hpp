///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
/// \author Theodoridis Aristomenis (atheodor@cslab.ece.ntua.gr)
///
/// \brief Some CUDA function calls with added error-checking
///
#ifndef MODEL_LVL2_H
#define MODEL_LVL2_H

#include "CoModel.hpp"
#include "backend_wrappers.hpp"

void ModelFuncInitBLAS2(MD_p out_model, int dev_id, const char* func, void* func_data);

long int MinAllowedTBLAS2(MD_p model);
long int MaxAllowedTBLAS2(MD_p model);
long int GetSKNumBLAS2(MD_p model, int T);

/// A naive prediction of the full-overlap (~unreachable) performance of a modeled routine
double PredictFullOverlapBLAS2(MD_p model);
/// A naive prediction of the zero-overlap (~worst case) performance of a modeled routine
double PredictZeroOverlapBLAS2(MD_p model);

double WerkhovenModelPredictWrapperBLAS2(MD_p model, long int T, short t_exec_method);
double CoCopeLiaPredictBaselineBLAS2(MD_p model, long int T);
double CoCopeLiaPredictDataLocBLAS2(MD_p model, long int T);
double CoCopeLiaPredictBidirectionalBLAS2(MD_p model, long int T);
double CoCopeLiaPredictReuseBLAS2(MD_p model, long int T);
double CoCopeLiaPipelineEmulateBLAS2(MD_p model, long int T);

double PredictReuseHeteroBLAS2(MD_p model, long int T, int used_devs, int* used_dev_ids,
	double* used_dev_relative_scores);

#endif
