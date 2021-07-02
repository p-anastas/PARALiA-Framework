///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief Some testing helper functions for file output.
///

#include <cstdlib>
#include "unihelpers.hpp"

void check_benchmark(char *filename){
	FILE* fp = fopen(filename,"r");
	if (!fp) { 
		fp = fopen(filename,"w+");
		if (!fp) error("report_results: LogFile failed to open");
		else warning("Generating Logfile...");
		fclose(fp);
	}
	else {
		fprintf(stderr,"Benchmark found: %s\n", filename);
		fclose(fp);	
		exit(1); 
	}
	return;		  	
}

void ParseInputLvl3(const int argc, const char *argv[], short* dev_id, char* TransA, char* TransB, double* alpha, double* beta, size_t* D1, size_t* D2, size_t* D3, short* loc1, short* loc2, short* loc3, short* outloc, int* T, double* cpu_ratio){
	if(argc != 15) error("Incorrect input arguments. Usage: ./correct_run dev_id TransA TransB alpha beta D1 D2 D3 loc1 loc2 loc3 outloc T cpu_ratio\n");

	// Control Parameters
	*dev_id = atoi(argv[1]);

	//Problem Parameters
	*TransA = argv[2][0];
	*TransB = argv[3][0];
	*alpha = atof(argv[4]);
	*beta = atof(argv[5]);	
	*D1 = atoi(argv[6]);
	*D2 = atoi(argv[7]);
	*D3 = atoi(argv[8]);
	*loc1 = atoi(argv[9]);
	*loc2 = atoi(argv[10]);
	*loc3 = atoi(argv[11]);
	*outloc = atoi(argv[12]);

	//Tunning Parameters
	*T = atoi(argv[13]);
	*cpu_ratio = atoi(argv[14]);

	fprintf(stderr, "ParseInputLvl3: Parsed configuration:\n\tdev_id: %d\n\tTransA: %c, TransB: %c\n\talpha: %lf, beta: %lf\n\tD1: %zu, D2: %zu, D3: %zu\n\tloc1: %d, loc2: %d, loc3: %d, outloc: %d\n\tT: %d\n\tcpu_ratio: %.3lf\n",
	*dev_id, *TransA, *TransB, *alpha, *beta, *D1, *D2, *D3, *loc1, *loc2, *loc3, *outloc, *T, *cpu_ratio);
	return; 
}


void CheckLogLvl3(char* filename, short dev_id, char TransA, char TransB, double alpha, double beta, size_t D1, size_t D2, size_t D3, short loc1, short loc2, short loc3, short outloc, int T, double cpu_ratio){
	FILE* fp = fopen(filename,"r");
	if (!fp) { 
		fp = fopen(filename,"w+");
		if (!fp) error("CheckLogLvl3: LogFile failed to open");
		else warning("CheckLogLvl3: Generating Logfile...");
	}	
	char buffer[256], search_string[256];
	sprintf(search_string, "%d, %c,%c,%.5lf,%.5lf,%zu,%zu,%zu,%d,%d,%d,%d, %d,%.3lf", dev_id, TransA, TransB, alpha, beta, D1, D2, D3, loc1, loc2, loc3, outloc, T, cpu_ratio);
	while (fgets(buffer, sizeof(buffer), fp) != NULL){
		if(strstr(buffer, search_string) != NULL){
   			fprintf(stderr,"CheckLogLvl3: entry %d, %c,%c,%.5lf,%.5lf,%zu,%zu,%zu,%d,%d,%d,%d, %d,%.3lf found. Quiting...\n", dev_id, TransA, TransB, alpha, beta, D1, D2, D3, loc1, loc2, loc3, outloc, T, cpu_ratio);
			fclose(fp);	
			exit(1); 
		}
	}			
    	fclose(fp);
	return;
}

void StoreLogLvl3(char* filename, short dev_id, char TransA, char TransB, double alpha, double beta, size_t D1, size_t D2, size_t D3, short loc1, short loc2, short loc3, short outloc, int T, double cpu_ratio, double av_time, double min_time, double max_time){
	FILE* fp = fopen(filename,"a");
	if (!fp) error("report_results: LogFile failed to open");
   	fprintf(fp,"%d, %c,%c,%.5lf,%.5lf,%zu,%zu,%zu,%d,%d,%d,%d, %d,%.3lf, %e,%e,%e\n",  dev_id, TransA, TransB, alpha, beta, D1, D2, D3, loc1, loc2, loc3, outloc, T, cpu_ratio, av_time, min_time, max_time);
        fclose(fp); 
	return;
}
 
