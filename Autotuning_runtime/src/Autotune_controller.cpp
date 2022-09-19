///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The DGEMM CoCopeLia implementation.
///

//#include <cblas.h>

#include "CoCoPeLiaModel.hpp"
//#include "unihelpers.hpp"


ATC::ATC(){
	active_unit_id_list = (int*) malloc(LOC_NUM*sizeof(int));
	active_unit_score = (double*) malloc(LOC_NUM*sizeof(double));
	Subkernels_per_unit_num = (int*) malloc(LOC_NUM*sizeof(int));
	Subkernels_per_unit_list = (int**) malloc(LOC_NUM*sizeof(int*));
	for (int d = 0; d < LOC_NUM; d++) Subkernels_per_unit_list[d] = NULL;
	T = active_unit_num = subkernel_num = -1;
	pred_t = -1.0;
	cache_limit = 0;
}

ATC::~ATC(){
	free(active_unit_id_list);
	free(active_unit_score);
	free(Subkernels_per_unit_num);
	for (int d = 0; d < LOC_NUM; d++) free(Subkernels_per_unit_list[d]);
	free(Subkernels_per_unit_list);
}

char* ATC::print(){
	char* outstring = (char*) malloc(256*sizeof(char));
	int dev_ids_token = 0;
	int ctr = 0, itter = 0;
	if (active_unit_num > 0) for (int i = 0; i < active_unit_num; i++) dev_ids_token+=pow(10,idxize(active_unit_id_list[i]));
	sprintf(outstring, "%ld,%d,%d,%lld",  T, active_unit_num, dev_ids_token, cache_limit);
	return outstring;
}

void ATC::update_sk_num(long long int subkernel_num_in){
	subkernel_num = subkernel_num_in;
	for (int d = 0; d < LOC_NUM; d++) Subkernels_per_unit_list[d] = (int*) malloc(subkernel_num*sizeof(int));
}
