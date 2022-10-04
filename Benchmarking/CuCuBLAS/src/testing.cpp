///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief Some testing helper functions for file output.
///

#include <cstdlib>
#include <cmath>
#include "unihelpers.hpp"
#include "CoCoPeLia.hpp"

char* CoCoImplementationPrint(){
	char* string_out = (char*) malloc (1024*sizeof(char));
	char* string_helper = (char*) malloc (1024*sizeof(char));
#ifndef ASYNC_ENABLE
	strcat(string_out, "_SYNC");
#endif
#ifndef UNIHELPER_LOCKFREE_ENABLE
	sprintf(string_out, "_UN-LC");
#endif
#ifndef BUFFER_REUSE_ENABLE
	sprintf(string_out, "_NO-BUF-RE");
#endif
#ifndef BACKEND_RES_REUSE_ENABLE
	sprintf(string_out, "_NO-UN-RE");
#endif
#ifdef ENABLE_PARALLEL_BACKEND
	sprintf(string_helper, "_UN-PB-L%d", MAX_BACKEND_L);
	strcat(string_out, string_helper);
#else
	strcat(string_out, "_UN-NO-PB");
#endif
#ifdef ENABLE_TRANSFER_HOPS
#ifdef ENABLE_TRANSFER_W_HOPS
	strcat(string_out, "_ALL-");
#else
	strcat(string_out, "_RONLY-");
#endif
	sprintf(string_helper, "HOPS-%d-%.2lf", MAX_ALLOWED_HOPS, HOP_PENALTY);
	strcat(string_out, string_helper);
#endif
//#ifdef DDEBUG
	printf("%s\n", string_out);
//#endif
	free(string_helper);
	return string_out;

}

char* CoCoDistributionPrint(){
	char* string_out = (char*) malloc (1024*sizeof(char));
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
void ParseInputLvl1(const int argc, const char *argv[], ATC_p* predef_control_values, double* alpha,
	long int* D1, long int* inc1, long int* inc2, short* loc1, short* loc2, short* outloc1, short* outloc2){
	if(argc != 13) error("Incorrect input arguments. Usage: ./correct_run\n\tactive_unit_num(auto if <0)\
	\n\tdev_ids(form example: 0101 for devices 0,2 - ignore if active_unit_num < 0)\n\tT(auto if <=0)\
	\n\tcache_max_size(auto if <0)\n\t alpha D1 inc1 inc2 loc1 loc2 outloc1 outloc2\n");

	int active_unit_num = atoi(argv[1]);
	long int dev_ids_token = atoi(argv[2]);
	int T = atoi(argv[3]);
	long long cache_limit = atof(argv[4]);

	ATC_p temp_p;

	//Tunning Parameters
	if (active_unit_num < 0 && cache_limit <= 0 && T <=0 ) temp_p = *predef_control_values = NULL;
	else{
		fprintf(stderr, "Using predefined control parameters from input\n");
		*predef_control_values = new ATC();
		temp_p = *predef_control_values;
		temp_p->cache_limit = cache_limit;
		temp_p->T = T;
		if (active_unit_num < 0){
			temp_p->active_unit_num = -1;
		}
		else{
			temp_p->active_unit_num = active_unit_num;
			int ctr = 0, itter = 0;
			do {
				if (dev_ids_token%10 == 1) temp_p->active_unit_id_list[ctr++] = itter;
				itter++;
				dev_ids_token/=10;
			} while ( dev_ids_token > 0);
			if (ctr != active_unit_num) error("ParseInputLvl1: Read different device Ids in total (%d) than active_unit_num implied (%d)\n", ctr, active_unit_num);
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

	fprintf(stderr, "ParseInputLvl1: ");
	if (*predef_control_values) (*predef_control_values)->print();
	fprintf(stderr, "Routine values:\talpha: %lf\n\tD1: %zu, inc1: %zu, inc2: %zu\
	\n\tloc1: %d, loc2: %d, outloc1: %d, outloc2: %d\n",
	*alpha, *D1, *inc1, *inc2, *loc1, *loc2, *outloc1, *outloc2);

	return;
}

void ParseInputLvl3(const int argc, const char *argv[], ATC_p* predef_control_values,
		char* TransA, char* TransB, double* alpha, double* beta, long int* D1, long int* D2, long int* D3,
		short* loc1, short* loc2, short* loc3, short* outloc){
	if(argc != 16) error("Incorrect input arguments. Usage: ./correct_run\
	\n\tactive_unit_num(auto if <0)\n\tdev_ids(form example: 0101 for devices 0,2 - ignore if active_unit_num < 0)\n\tT(auto if <=0)\
	\n\tcache_max_size(auto if <0)\n\tTransA TransB alpha beta D1 D2 D3 loc1 loc2 loc3 outloc \n");

	int active_unit_num = atoi(argv[1]);
	long int dev_ids_token = atoi(argv[2]);
	int T = atoi(argv[3]);
	long long cache_limit = atof(argv[4]);

	ATC_p temp_p;

	//Tunning Parameters
	if (active_unit_num < 0 && cache_limit <= 0 && T <=0 ) temp_p = *predef_control_values = NULL;
	else{
		fprintf(stderr, "Using predefined control parameters from input\n");
		*predef_control_values = new ATC();
		temp_p = *predef_control_values;
		temp_p->cache_limit = cache_limit;
		temp_p->T = T;
		if (active_unit_num < 0){
			temp_p->active_unit_num = -1;
		}
		else{
			temp_p->active_unit_num = active_unit_num;
			int ctr = 0, itter = 0;
			do {
				if (dev_ids_token%10 == 1) temp_p->active_unit_id_list[ctr++] = deidxize(itter);
				itter++;
				dev_ids_token/=10;
			} while ( dev_ids_token > 0);
			if (ctr != active_unit_num) error("ParseInputLvl3: Read different device Ids in total (%d) than active_unit_num implied (%d)\n", ctr, active_unit_num);
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

	fprintf(stderr, "ParseInputLvl3: ");
	if (*predef_control_values) (*predef_control_values)->print();
	fprintf(stderr, "Routine values:\n\tTransA: %c, TransB: %c\n\talpha: %lf, beta: %lf\n\tD1: %zu, D2: %zu, D3: %zu\n\tloc1: %d, loc2: %d, loc3: %d, outloc: %d\n",
	*TransA, *TransB, *alpha, *beta, *D1, *D2, *D3, *loc1, *loc2, *loc3, *outloc);

	return;
}

void CheckLogLvl1(char* filename, ATC_p predef_control_values,
	double alpha, long int D1, long int inc1, long int inc2, short loc1, short loc2, short outloc1, short outloc2){
	FILE* fp = fopen(filename,"r");
	if (!fp) {
		fp = fopen(filename,"w+");
		if (!fp) error("CheckLogLvl1: LogFile %s failed to open\n", filename);
		else warning("CheckLogLvl1: Generating Logfile %s...\n", filename);
	}
	char buffer[1024], search_string[1024];
	const char* control_str = (predef_control_values) ? predef_control_values->print_csv() : "-1,-1,-1,-1";
	sprintf(search_string, "%s, %.5lf,%zu,%zu,%zu,%d,%d,%d,%d", control_str, alpha, D1, inc1, inc2, loc1, loc2, outloc1, outloc2);
	while (fgets(buffer, sizeof(buffer), fp) != NULL){
		if(strstr(buffer, search_string) != NULL){
   			fprintf(stderr,"CheckLogLvl3: entry %s, %.5lf,%zu,%zu,%zu,%d,%d,%d,%d found. Quiting...\n",
					control_str, alpha, D1, inc1, inc2, loc1, loc2, outloc1, outloc2);
			fclose(fp);
			exit(1);
		}
	}

    	fclose(fp);
	return;
}

void CheckLogLvl3(char* filename, ATC_p predef_control_values, char TransA, char TransB, double alpha, double beta, long int D1, long int D2, long int D3, short loc1, short loc2, short loc3, short outloc){
	FILE* fp = fopen(filename,"r");
	if (!fp) {
		fp = fopen(filename,"w+");
		if (!fp) error("CheckLogLvl3: LogFile %s failed to open\n", filename);
		else warning("CheckLogLvl3: Generating Logfile %s...\n", filename);
	}
	char buffer[1024], search_string[1024];
	const char* control_str = (predef_control_values) ? predef_control_values->print_csv() : "-1,-1,-1,-1";
	sprintf(search_string, "%s, %c,%c,%.5lf,%.5lf,%zu,%zu,%zu,%d,%d,%d,%d", control_str, TransA, TransB, alpha, beta, D1, D2, D3, loc1, loc2, loc3, outloc);
	while (fgets(buffer, sizeof(buffer), fp) != NULL){
		if(strstr(buffer, search_string) != NULL){
   			fprintf(stderr,"CheckLogLvl3: entry %s, %c,%c,%.5lf,%.5lf,%zu,%zu,%zu,%d,%d,%d,%d found. Quiting...\n", control_str, TransA, TransB, alpha, beta, D1, D2, D3, loc1, loc2, loc3, outloc);
			fclose(fp);
			exit(1);
		}
	}

    	fclose(fp);
	return;
}

void StoreLogLvl1(char* filename, ATC_p predef_control_values, double alpha, long int D1,
	long int inc1, long int inc2, short loc1, short loc2, short outloc1, short outloc2, double timer){
	FILE* fp = fopen(filename,"a");
	if (!fp) error("report_results: LogFile failed to open");
	const char* control_str = (predef_control_values) ? predef_control_values->print_csv() : "-1,-1,-1,-1";
   	fprintf(fp,"%s, %.5lf,%zu,%zu,%zu,%d,%d,%d,%d", control_str, alpha, D1, inc1, inc2, loc1, loc2, outloc1, outloc2);

        fclose(fp);
	return;
}

void StoreLogLvl3(char* filename, ATC_p predef_control_values, char TransA, char TransB, double alpha, double beta, long int D1, long int D2, long int D3, short loc1, short loc2, short loc3, short outloc, double timer){
	FILE* fp = fopen(filename,"a");
	if (!fp) error("report_results: LogFile failed to open");
	const char* control_str = (predef_control_values) ? predef_control_values->print_csv() : "-1,-1,-1,-1";
   	fprintf(fp,"%s, %c,%c,%.5lf,%.5lf,%zu,%zu,%zu,%d,%d,%d,%d, %e\n",  control_str, TransA, TransB, alpha, beta, D1, D2, D3, loc1, loc2, loc3, outloc, timer);

        fclose(fp);
	return;
}
