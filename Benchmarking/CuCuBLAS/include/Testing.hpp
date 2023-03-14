///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The external wrapper for CoCoPelia + wrapped cuBLASXt
///
#ifndef TESTING_H
#define TESTING_H

// The number of devices (e.g. CUDA GPUs) used by other GPU libraries that don't use the CPU
#define DEV_NUM (LOC_NUM -1)

/// Return a string with the active Cmake implemetation flag used
char* CoCoImplementationPrint();

/// Return a string with the active Cmake Subkernel Distribution flag
char* CoCoDistributionPrint();

void ParseInputLvl3(const int argc, const char *argv[], ATC_p* predef_control_values, char* TransA, char* TransB,
	double* alpha, double* beta, long int* D1, long int* D2, long int* D3, short* loc1, short* loc2, short* loc3, short* outloc);
void CheckLogLvl3(char* filename, ATC_p predef_control_values, char TransA, char TransB,
	double alpha, double beta, long int D1, long int D2, long int D3, short loc1, short loc2, short loc3, short outloc);
void StoreLogLvl3(char* filename, ATC_p predef_control_values, char TransA, char TransB, double alpha, double beta,
	long int D1, long int D2, long int D3, short loc1, short loc2, short loc3, short outloc, double timer, double pred_t, double pred_J);

void ParseInputLvl2(const int argc, const char *argv[], ATC_p* predef_control_values, char* TransA,
	double* alpha, double* beta, long int* D1, long int* D2, long int* inc1, long int* inc2, short* loc1, short* loc2, short* loc3, short* outloc);
void CheckLogLvl2(char* filename, ATC_p predef_control_values, char TransA, double alpha, double beta,
	long int D1, long int D2, long int inc1, long int inc2, short loc1, short loc2, short loc3, short outloc);
void StoreLogLvl2(char* filename, ATC_p predef_control_values, char TransA, double alpha, double beta,
	long int D1, long int D2, long int inc1, long int inc2, short loc1, short loc2, short loc3, short outloc, double timer, double pred_t, double pred_J);

void ParseInputLvl1(const int argc, const char *argv[], ATC_p* predef_control_values, double* alpha,
	long int* D1, long int* inc1, long int* inc2, short* loc1, short* loc2, short* outloc1, short* outloc2);
void CheckLogLvl1(char* filename, ATC_p predef_control_values,
	double alpha, long int D1, long int inc1, long int inc2, short loc1, short loc2, short outloc1, short outloc2);
void StoreLogLvl1(char* filename, ATC_p predef_control_values,
	double alpha, long int D1, long int inc1, long int inc2, short loc1, short loc2, short outloc1, short outloc2, double timer);

#endif
