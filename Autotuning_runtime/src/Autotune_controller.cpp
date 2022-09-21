///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The DGEMM CoCopeLia implementation.
///

//#include <cblas.h>

#include "Autotuning_runtime.hpp"
//#include "unihelpers.hpp"


ATC::ATC(){
	active_unit_id_list = (int*) malloc(LOC_NUM*sizeof(int));
	active_unit_score = (double*) malloc(LOC_NUM*sizeof(double));
	Subkernels_per_unit_num = (int*) malloc(LOC_NUM*sizeof(int));
	Subkernels_per_unit_list = (int**) malloc(LOC_NUM*sizeof(int*));
	for (int d = 0; d < LOC_NUM; d++){
		Subkernels_per_unit_list[d] = NULL;
		Subkernels_per_unit_num[d] = 0;
	}
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

void ATC::print(){
	//int dev_ids_token = 0;
	//int ctr = 0, itter = 0;
	//if (active_unit_num > 0) for (int i = 0; i < active_unit_num; i++) dev_ids_token+=pow(10,idxize(active_unit_id_list[i]));
	fprintf(stderr, "Autotune controller:\n->T = %ld\n->active_unit_num = %d\n->active_unit_id_list = %s\n->active_unit_score = %s\
	\n->pred_t = %lf\n->subkernel_num = %d\n->Subkernels_per_unit_num = %s\n", T, active_unit_num,
	printlist<int>(active_unit_id_list, active_unit_num),
	printlist<double>(active_unit_score, active_unit_num),
	pred_t, subkernel_num,
	printlist<int>(Subkernels_per_unit_num, active_unit_num));
 	if(subkernel_num != -1)
	for (int d = 0; d < active_unit_num; d++) fprintf(stderr, "Subkernels_per_unit_list[%d] = %s\n", d,
		printlist<int>(Subkernels_per_unit_list[d], subkernel_num));
	fprintf(stderr, "\n");	
	return;
}

const char* ATC::print_csv(){
	char* outstring = (char*) malloc(256*sizeof(char));
	int dev_ids_token = 0;
	int ctr = 0, itter = 0;
	if (active_unit_num > 0) for (int i = 0; i < active_unit_num; i++) dev_ids_token+=pow(10,idxize(active_unit_id_list[i]));
	sprintf(outstring, "%ld,%d,%d,%lld",  T, active_unit_num, dev_ids_token, cache_limit);
	return outstring;
}

void ATC::update_sk_num(long long int subkernel_num_in){
	int prev_sk_num = subkernel_num;
	subkernel_num = subkernel_num_in;
	if (prev_sk_num == -1)  for (int d = 0; d < LOC_NUM; d++){
		Subkernels_per_unit_list[d] = (int*) malloc(subkernel_num*sizeof(int));
		for (int sk = 0; sk < subkernel_num; sk++) Subkernels_per_unit_list[d][sk] = -1;
	}
	else if (prev_sk_num < subkernel_num) for (int d = 0; d < LOC_NUM; d++){
		free(Subkernels_per_unit_list[d]);
		Subkernels_per_unit_list[d] = (int*) malloc(subkernel_num*sizeof(int));
		for (int sk = 0; sk < subkernel_num; sk++) Subkernels_per_unit_list[d][sk] = -1;
	}
}

void ATC::mimic_ATC(ATC_p other_ATC){
	T = other_ATC->T;
	active_unit_num = other_ATC->active_unit_num;
	for (int d = 0; d < other_ATC->active_unit_num; d++) active_unit_id_list[d] = other_ATC->active_unit_id_list[d];
	for (int d = 0; d < other_ATC->active_unit_num; d++) active_unit_score[d] = other_ATC->active_unit_score[d];
	pred_t = other_ATC->pred_t;
	cache_limit = other_ATC->cache_limit;

	if(subkernel_num != -1){
		for (int d = 0; d < LOC_NUM; d++){
			free(Subkernels_per_unit_list[d]);
			Subkernels_per_unit_num[d] = 0;
			Subkernels_per_unit_list[d] = NULL;
		}
		subkernel_num = -1;
	}

	if (other_ATC->subkernel_num != -1){
		for (int d = 0; d < other_ATC->active_unit_num; d++){
			Subkernels_per_unit_num[d] = other_ATC->Subkernels_per_unit_num[d];
			Subkernels_per_unit_list[d] = (int*) malloc(other_ATC->subkernel_num*sizeof(int));
			for (int sk = 0; sk < other_ATC->subkernel_num; sk++)
				Subkernels_per_unit_list[d][sk] = other_ATC->Subkernels_per_unit_list[d][sk];
		}
		subkernel_num = other_ATC->subkernel_num;
	}
}

void ATC::reset(){
	for (int d = 0; d < LOC_NUM; d++){
		free(Subkernels_per_unit_list[d]);
		Subkernels_per_unit_list[d] = NULL;
		Subkernels_per_unit_num[d] = 0;
	}
	T = active_unit_num = subkernel_num = -1;
	pred_t = -1.0;
	cache_limit = 0;
}
