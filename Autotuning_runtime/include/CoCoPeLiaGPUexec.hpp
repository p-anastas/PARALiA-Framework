#ifndef COCOPELIAGPUEXEC_H
#define COCOPELIAGPUEXEC_H

/* ============================BLAS1 Models================================= */

typedef struct  BLAS1_data{
	short dev_id;
	const char* func;

	// TODO: add more complicated selection taking error margins into account.
	short mode;
	long int* T_lookup_buf;
	double* av_time_buf;
	long int lines;
	// TODO: These can be used for more robust results or for worst/best case performance prediction
}* GPUexec1Model_p;

GPUexec1Model_p GPUexec1Model_init(short dev, const char* func);

double GPUexec1Model_predict(GPUexec1Model_p model, long int T);

/* ============================BLAS2 Models================================= */

typedef struct  BLAS2_data{
}* GPUexec2Model_p;

/// Load parameters from file and return  BLAS 2 execution time model
GPUexec2Model_p GPUexec2Model_init(short dev, const char* func);

double GPUexec2Model_predict(GPUexec2Model_p model, long int T);

/* ============================BLAS3 Models================================= */

typedef struct  BLAS3_data{
	short dev_id;
	const char* func;

	// TODO: add more complicated selection taking error margins into account.
	short mode;
	long int* T_lookup_buf;
	double* av_time_buf;
	long int lines;
	char* TransA_buf;
	char* TransB_buf;
	// TODO: These can be used for more robust results or for worst/best case performance prediction
}* GPUexec3Model_p;


/// Helper functions for getting important database limits and info
long int GPUexec3MinT(GPUexec3Model_p model);
long int GPUexec3MaxT(GPUexec3Model_p model);
long int GPUexec3NearestT(GPUexec3Model_p model, long int Tin);

/// Load parameters from file and return  BLAS 3 execution time model
GPUexec3Model_p GPUexec3Model_init(short dev, const char* func);

double GPUexec3Model_predict(GPUexec3Model_p model, long int T, char TransA, char TransB);

#endif
