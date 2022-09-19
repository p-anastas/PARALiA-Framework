///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The DGEMM CoCopeLia implementation.
///

//#include <cblas.h>

//#include "Asset.hpp"
//#include "unihelpers.hpp"

//#include "backend_wrappers.hpp"


char* CoControlPrint(CoControl_p input){
	char* outstring = (char*) malloc(256*sizeof(char));
	int dev_ids_token = 0;
	int ctr = 0, itter = 0;
	if (input == NULL) sprintf(outstring,"-1,-1,-1,-1");
	else{
		if (input->active_unit_num > 0) for (int i = 0; i < input->active_unit_num; i++) dev_ids_token+=pow(10,idxize(input->active_unit_id_list[i]));
		sprintf(outstring, "%ld,%d,%d,%lld",  input->T, input->active_unit_num, dev_ids_token, input->cache_limit);
	}
	return outstring;
}

char* CoCoImplementationPrint(){
	char* string_out = (char*) malloc (256*sizeof(char));
#ifdef ENABLE_MUTEX_LOCKING
#ifdef MULTIDEVICE_REDUCTION_ENABLE
	sprintf(string_out, "ML-MR-BL%d", MAX_BUFFERING_L);
#else
	sprintf(string_out, "ML");
#endif
#elif ENABLE_PARALLEL_BACKEND
	sprintf(string_out, "PB-L%d", MAX_BACKEND_L);
#elif MULTIDEVICE_REDUCTION_ENABLE
	sprintf(string_out, "MR-BL%d", MAX_BUFFERING_L);
#elif UNIHELPER_LOCKFREE_ENABLE
	sprintf(string_out, "UL");
#elif BUFFER_REUSE_ENABLE
	sprintf(string_out, "BR");
#elif BACKEND_RES_REUSE_ENABLE
	sprintf(string_out, "BRR");
#elif ASYNC_ENABLE
	sprintf(string_out, "ASYNC");
#else
	sprintf(string_out, "SYNC");
#endif
	return string_out;
}

char* CoCoDistributionPrint(){
	char* string_out = (char*) malloc (256*sizeof(char));
#ifdef RUNTIME_SCHEDULER_VERSION
#ifdef DISTRIBUTION
	sprintf(string_out, "RT-%s", DISTRIBUTION);
#else
#error
#endif
#else
#ifdef DISTRIBUTION
	sprintf(string_out, "ST-%s", DISTRIBUTION);
#else
#error
#endif
#endif
	return string_out;
}
