///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief Some testing helper functions for file output.
///

#include <cstdlib>
#include <cmath>
#include "unihelpers.hpp"
#include "CoCoPeLia.hpp"

char* CoControlPrint(CoControl_p input){
	char* outstring = (char*) malloc(256*sizeof(char));
	int dev_ids_token = 0;
	int ctr = 0, itter = 0;
	if (input == NULL) sprintf(outstring,"-1,-1,-1,-1");
	else{
		if (input->dev_num > 0)for (int i = 0; i < input->dev_num; i++)dev_ids_token+=pow(10,idxize(input->dev_ids[i]));
		sprintf(outstring, "%d,%d,%d,%lld",  input->T, input->dev_num, dev_ids_token, input->cache_limit);
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
#ifdef DISTRIBUTION
	sprintf(string_out, "%s", DISTRIBUTION);
#else
#error
#endif
	return string_out;
}

void ParseInputLvl1(const int argc, const char *argv[], CoControl_p* predef_control_values, double* alpha,
	size_t* D1, size_t* inc1, size_t* inc2, short* loc1, short* loc2, short* outloc1, short* outloc2){
	if(argc != 13) error("Incorrect input arguments. Usage: ./correct_run\n\tdev_num(auto if <0)\
	\n\tdev_ids(form example: 0101 for devices 0,2 - ignore if dev_num < 0)\n\tT(auto if <=0)\
	\n\tcpu_ratio(auto if <0, set to 1 if dev_num=0)\n\t alpha D1 ic1 inc2 loc1 loc2 outloc1 outloc2\n");

	int dev_num = atoi(argv[1]);
	size_t dev_ids_token = atoi(argv[2]);
	int T = atoi(argv[3]);
	long long cache_limit = atof(argv[4]);

	CoControl_p temp_p;

	//Tunning Parameters
	if (dev_num < 0 && cache_limit <= 0 && T <=0 ) temp_p = *predef_control_values = NULL;
	else{
		fprintf(stderr, "Using predefined control parameters from input\n");
		*predef_control_values = (CoControl_p) malloc (sizeof(struct CoControl));
		temp_p = *predef_control_values;
		temp_p->cache_limit = cache_limit;
		temp_p->T = T;
		if (dev_num < 0){
			temp_p->dev_num = -1;
		}
		else{
			temp_p->dev_num = dev_num;
			int ctr = 0, itter = 0;
			do {
				if (dev_ids_token%10 == 1) temp_p->dev_ids[ctr++] = itter;
				itter++;
				dev_ids_token/=10;
			} while ( dev_ids_token > 0);
			if (ctr != dev_num) error("ParseInputLvl1: Read different device Ids in total (%d) than dev_num implied (%d)\n", ctr, dev_num);
		}
	}

	//Problem Parameters
	*alpha = atof(argv[5]);
	*D1 = atoi(argv[6]);
	*inc1 = atoi(argv[7]);
	*inc2 = atoi(argv[8]);
	*loc1 = atoi(argv[9]);
	*loc2 = atoi(argv[10]);
	*outloc1 = atoi(argv[11]);
	*outloc2 = atoi(argv[12]);

	char* control_str = CoControlPrint(temp_p);
	fprintf(stderr, "ParseInputLvl1: Parsed configuration:\n\tControl: %s\n\talpha: %lf\
	\n\tD1: %zu, inc1: %zu, inc2: %zu\n\tloc1: %d, loc2: %d, outloc1: %d, outloc2: %d\n",
	control_str, *alpha, *D1, *inc1, *inc2, *loc1, *loc2, *outloc1, *outloc2);
	free(control_str);
	return;
}

void ParseInputLvl3(const int argc, const char *argv[], CoControl_p* predef_control_values,
		char* TransA, char* TransB, double* alpha, double* beta, size_t* D1, size_t* D2, size_t* D3,
		short* loc1, short* loc2, short* loc3, short* outloc){
	if(argc != 16) error("Incorrect input arguments. Usage: ./correct_run\
	\n\tdev_num(auto if <0)\n\tdev_ids(form example: 0101 for devices 0,2 - ignore if dev_num < 0)\n\tT(auto if <=0)\
	\n\tcpu_ratio(auto if <0, set to 1 if dev_num=0)\n\tTransA TransB alpha beta D1 D2 D3 loc1 loc2 loc3 outloc \n");

	int dev_num = atoi(argv[1]);
	size_t dev_ids_token = atoi(argv[2]);
	int T = atoi(argv[3]);
	long long cache_limit = atof(argv[4]);

	CoControl_p temp_p;

	//Tunning Parameters
	if (dev_num < 0 && cache_limit <= 0 && T <=0 ) temp_p = *predef_control_values = NULL;
	else{
		fprintf(stderr, "Using predefined control parameters from input\n");
		*predef_control_values = (CoControl_p) malloc (sizeof(struct CoControl));
		temp_p = *predef_control_values;
		temp_p->cache_limit = cache_limit;
		temp_p->T = T;
		if (dev_num < 0){
			temp_p->dev_num = -1;
		}
		else{
			temp_p->dev_num = dev_num;
			int ctr = 0, itter = 0;
			do {
				if (dev_ids_token%10 == 1) temp_p->dev_ids[ctr++] = deidxize(itter);
				itter++;
				dev_ids_token/=10;
			} while ( dev_ids_token > 0);
			if (ctr != dev_num) error("ParseInputLvl3: Read different device Ids in total (%d) than dev_num implied (%d)\n", ctr, dev_num);
		}
	}

	//Problem Parameters
	*TransA = argv[5][0];
	*TransB = argv[6][0];
	*alpha = atof(argv[7]);
	*beta = atof(argv[8]);
	*D1 = atoi(argv[9]);
	*D2 = atoi(argv[10]);
	*D3 = atoi(argv[11]);
	*loc1 = atoi(argv[12]);
	*loc2 = atoi(argv[13]);
	*loc3 = atoi(argv[14]);
	*outloc = atoi(argv[15]);

	char* control_str = CoControlPrint(temp_p);
	fprintf(stderr, "ParseInputLvl3: Parsed configuration:\n\tControl: %s\n\tTransA: %c, TransB: %c\n\talpha: %lf, beta: %lf\n\tD1: %zu, D2: %zu, D3: %zu\n\tloc1: %d, loc2: %d, loc3: %d, outloc: %d\n",
	control_str, *TransA, *TransB, *alpha, *beta, *D1, *D2, *D3, *loc1, *loc2, *loc3, *outloc);
	free(control_str);
	return;
}

void CheckLogLvl1(char* filename, CoControl_p predef_control_values,
	double alpha, size_t D1, size_t inc1, size_t inc2, short loc1, short loc2, short outloc1, short outloc2){
	FILE* fp = fopen(filename,"r");
	if (!fp) {
		fp = fopen(filename,"w+");
		if (!fp) error("CheckLogLvl1: LogFile %s failed to open\n", filename);
		else warning("CheckLogLvl1: Generating Logfile %s...\n", filename);
	}
	char buffer[256], search_string[256];
	char* control_str = CoControlPrint(predef_control_values);
	sprintf(search_string, "%s, %.5lf,%zu,%zu,%zu,%d,%d,%d,%d", control_str, alpha, D1, inc1, inc2, loc1, loc2, outloc1, outloc2);
	while (fgets(buffer, sizeof(buffer), fp) != NULL){
		if(strstr(buffer, search_string) != NULL){
   			fprintf(stderr,"CheckLogLvl3: entry %s, %.5lf,%zu,%zu,%zu,%d,%d,%d,%d found. Quiting...\n",
					control_str, alpha, D1, inc1, inc2, loc1, loc2, outloc1, outloc2);
			fclose(fp);
			exit(1);
		}
	}
	free(control_str);
    	fclose(fp);
	return;
}

void CheckLogLvl3(char* filename, CoControl_p predef_control_values, char TransA, char TransB, double alpha, double beta, size_t D1, size_t D2, size_t D3, short loc1, short loc2, short loc3, short outloc){
	FILE* fp = fopen(filename,"r");
	if (!fp) {
		fp = fopen(filename,"w+");
		if (!fp) error("CheckLogLvl3: LogFile %s failed to open\n", filename);
		else warning("CheckLogLvl3: Generating Logfile %s...\n", filename);
	}
	char buffer[256], search_string[256];
	char* control_str = CoControlPrint(predef_control_values);
	sprintf(search_string, "%s, %c,%c,%.5lf,%.5lf,%zu,%zu,%zu,%d,%d,%d,%d", control_str, TransA, TransB, alpha, beta, D1, D2, D3, loc1, loc2, loc3, outloc);
	while (fgets(buffer, sizeof(buffer), fp) != NULL){
		if(strstr(buffer, search_string) != NULL){
   			fprintf(stderr,"CheckLogLvl3: entry %s, %c,%c,%.5lf,%.5lf,%zu,%zu,%zu,%d,%d,%d,%d found. Quiting...\n", control_str, TransA, TransB, alpha, beta, D1, D2, D3, loc1, loc2, loc3, outloc);
			fclose(fp);
			exit(1);
		}
	}
	free(control_str);
    	fclose(fp);
	return;
}

void StoreLogLvl1(char* filename, CoControl_p predef_control_values, double alpha, size_t D1,
	size_t inc1, size_t inc2, short loc1, short loc2, short outloc1, short outloc2, double timer){
	FILE* fp = fopen(filename,"a");
	if (!fp) error("report_results: LogFile failed to open");
	char* control_str = CoControlPrint(predef_control_values);
   	fprintf(fp,"%s, %.5lf,%zu,%zu,%zu,%d,%d,%d,%d", control_str, alpha, D1, inc1, inc2, loc1, loc2, outloc1, outloc2);
	free(control_str);
        fclose(fp);
	return;
}

void StoreLogLvl3(char* filename, CoControl_p predef_control_values, char TransA, char TransB, double alpha, double beta, size_t D1, size_t D2, size_t D3, short loc1, short loc2, short loc3, short outloc, double timer){
	FILE* fp = fopen(filename,"a");
	if (!fp) error("report_results: LogFile failed to open");
	char* control_str = CoControlPrint(predef_control_values);
   	fprintf(fp,"%s, %c,%c,%.5lf,%.5lf,%zu,%zu,%zu,%d,%d,%d,%d, %e\n",  control_str, TransA, TransB, alpha, beta, D1, D2, D3, loc1, loc2, loc3, outloc, timer);
	free(control_str);
        fclose(fp);
	return;
}
