///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The external wrapper for CoCoPelia + wrapped cuBLASXt
///
#ifndef TESTING_H
#define TESTING_H

void ParseInputLvl3(const int argc, const char *argv[], CoControl_p* predef_control_values, char* TransA, char* TransB,
	double* alpha, double* beta, size_t* D1, size_t* D2, size_t* D3, short* loc1, short* loc2, short* loc3, short* outloc);
void CheckLogLvl3(char* filename, CoControl_p predef_control_values, char TransA, char TransB,
	double alpha, double beta, size_t D1, size_t D2, size_t D3, short loc1, short loc2, short loc3, short outloc);
void StoreLogLvl3(char* filename, CoControl_p predef_control_values, char TransA, char TransB,
	double alpha, double beta, size_t D1, size_t D2, size_t D3, short loc1, short loc2, short loc3, short outloc, double timer);

void ParseInputLvl1(const int argc, const char *argv[], CoControl_p* predef_control_values, double* alpha,
	size_t* D1, size_t* inc1, size_t* inc2, short* loc1, short* loc2, short* outloc1, short* outloc2);
void CheckLogLvl1(char* filename, CoControl_p predef_control_values,
	double alpha, size_t D1, size_t inc1, size_t inc2, short loc1, short loc2, short outloc1, short outloc2);
void StoreLogLvl1(char* filename, CoControl_p predef_control_values,
	double alpha, size_t D1, size_t inc1, size_t inc2, short loc1, short loc2, short outloc1, short outloc2, double timer);

#endif
