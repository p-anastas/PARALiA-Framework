///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief Some CUDA function calls with added error-checking
///
#ifndef MODEL_LVL3_H
#define MODEL_LVL3_H

#include "CoModel.hpp"
#include "backend_wrappers.hpp"

void ModelFuncInitBLAS3(MD_p out_model, int dev_id, const char* func, void* func_data);

long int MinAllowedTBLAS3(MD_p model);
long int MaxAllowedTBLAS3(MD_p model);
long int GetSKNumBLAS3(MD_p model, int T);

/// A naive prediction of the full-overlap (~unreachable) performance of a modeled routine
double PredictFullOverlapBLAS3(MD_p model);
/// A naive prediction of the zero-overlap (~worst case) performance of a modeled routine
double PredictZeroOverlapBLAS3(MD_p model);

double WerkhovenModelPredictWrapperBLAS3(MD_p model, long int T, short t_exec_method);
double CoCopeLiaPredictBaselineBLAS3(MD_p model, long int T);
double CoCopeLiaPredictDataLocBLAS3(MD_p model, long int T);
double CoCopeLiaPredictBidirectionalBLAS3(MD_p model, long int T);
double CoCopeLiaPredictReuseBLAS3(MD_p model, long int T);
double CoCopeLiaPipelineEmulateBLAS3(MD_p model, long int T);

double PredictReuseHeteroBLAS3(MD_p model, long int T, int used_devs, int* used_dev_ids,
	double* used_dev_relative_scores);

double* PredictHeteroFullOverlapBLAS3_v2(MD_p model, long int T, int active_unit_num, int* active_unit_id_list,
	double* active_unit_score);
#endif
