///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The external wrapper for CoCoPelia + wrapped cuBLASXt
///
#ifndef COCOPELIA_H
#define COCOPELIA_H

typedef struct CoControl{
	int T = 0;
	int* dev_ids = NULL;
	int dev_num = -1;
	float cpu_ratio = 0.0;
}* CoControl_p;

char* CoControlPrint(CoControl_p input);
/// The CoCopeLia Dgemm implementation. A prediction model is used to select a tile from the micro-benchmarked tile candidates with CoCopeLia_optimize3.
CoControl_p CoCopeLiaDgemm(char TransA,  char TransB, size_t M, size_t N, size_t K, double alpha, double* A, size_t ldA, double* B, size_t ldB, double beta, double* C, size_t ldC);

/// A modification of CoCopeLiaDgemm but with a given T (mainly for performance/debug purposes)
CoControl_p CoCopeLiaDgemmControled(char TransA,  char TransB, size_t M, size_t N, size_t K, double alpha, double* A, size_t ldA, double* B, size_t ldB, double beta, double* C, size_t ldC, CoControl_p predef_control_values);

void ParseInputLvl3(const int argc, const char *argv[], CoControl_p* predef_control_values, char* TransA, char* TransB, double* alpha, double* beta, size_t* D1, size_t* D2, size_t* D3, short* loc1, short* loc2, short* loc3, short* outloc);
void CheckLogLvl3(char* filename, CoControl_p predef_control_values, char TransA, char TransB, double alpha, double beta, size_t D1, size_t D2, size_t D3, short loc1, short loc2, short loc3, short outloc);
void StoreLogLvl3(char* filename, CoControl_p predef_control_values, char TransA, char TransB, double alpha, double beta, size_t D1, size_t D2, size_t D3, short loc1, short loc2, short loc3, short outloc, double timer);

///Clears all GPU mem allocation done by CoCopeLiaDgemm at target device
void CoCopeLiaDgemm_flush_gpu_mem_buf(short dev_id);
#endif
