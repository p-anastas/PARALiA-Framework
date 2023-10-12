///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief Some CUDA function calls with added error-checking
///
#ifndef COCOPELIA_MODEL_WRAP_H
#define COCOPELIA_MODEL_WRAP_H

#include "Autotuner.hpp"

/// A naive prediction of the full-overlap (~unreachable) performance of a modeled routine
double PredictFullOverlap(MD_p model);
/// A naive prediction of the zero-overlap (~worst case) performance of a modeled routine
double PredictZeroOverlap(MD_p model);

double WerkhovenModelPredictWrapper(MD_p model, long int T, short t_exec_method);
double CoCopeLiaPredictBaseline(MD_p model, long int T);
double CoCopeLiaPredictDataLoc(MD_p model, long int T);
double CoCopeLiaPredictBidirectional(MD_p model, long int T);
double CoCopeLiaPredictReuse(MD_p model, long int T);
double CoCopeLiaPipelineEmulate(MD_p model, long int T);
double CoCopeLiaPipelineEmulate(MD_p model, long int T);

double PARALiaPerfPenaltyModifier(MD_p model, long int T, int active_unit_num);
double PARALiaPredictReuseKernelOverBLAS3(MD_p model, long int T);
double PredictReuseHetero(MD_p model, long int T, int used_devs,
  int* used_dev_ids, double* used_dev_relative_scores);
double PredictBidirectionalHetero(MD_p model, long int T,
  int used_devs, int* used_dev_ids, double* used_dev_relative_scores);
double* PARALiaPredictLinkHetero_v2(MD_p model, long int T, int used_devs, int* used_dev_ids,
	double* used_dev_relative_scores);
/******************************************************************************/

#endif
