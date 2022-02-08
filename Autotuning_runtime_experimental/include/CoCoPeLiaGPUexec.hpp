#ifndef COCOPELIAGPUEXEC_H
#define COCOPELIAGPUEXEC_H

/*
typedef struct  BLAS1_data{
}* GPUexec1Model_p;

typedef struct  BLAS2_data{
}* GPUexec2Model_p;

/// Load parameters from file and return  BLAS 1 execution time model
GPUexec1Model_p GPUexec1Model_init(short dev,  char* func);

double GPUexec1Model_predict(GPUexec1Model_p model, size_t D1);

/// Load parameters from file and return  BLAS 2 execution time model
GPUexec2Model_p GPUexec2Model_init(short dev,  char* func);

double GPUexec2Model_predict(GPUexec2Model_p model, size_t D1,  size_t D2);
*/

typedef struct  BLAS3_data{
	short dev_id; 
	char* func;
	
	// TODO: add more complicated selection taking error margins into account.
	short mode;
	size_t* T_lookup_buf;
	double* av_time_buf;
	size_t lines;
	char* TransA_buf;
	char* TransB_buf;
	// TODO: These can be used for more robust results or for worst/best case performance prediction
}* GPUexec3Model_p;


/// Helper functions for getting important database limits and info
size_t GPUexec3MinT(GPUexec3Model_p model);
size_t GPUexec3MaxT(GPUexec3Model_p model);
size_t GPUexec3NearestT(GPUexec3Model_p model, size_t Tin);

/// Load parameters from file and return  BLAS 3 execution time model
GPUexec3Model_p GPUexec3Model_init(short dev,  char* func);

double GPUexec3Model_predict(GPUexec3Model_p model, size_t T, char TransA, char TransB);

#endif
