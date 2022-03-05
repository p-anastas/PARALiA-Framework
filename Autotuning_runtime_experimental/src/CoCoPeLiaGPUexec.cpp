///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The execution lookup functions for CoCopeLia.
///

#include <stdlib.h>
#include <math.h>

#include "unihelpers.hpp"
#include "CoCoPeLiaGPUexec.hpp"

size_t GPUexec3MinT(GPUexec3Model_p model){
	size_t result = model->T_lookup_buf[0];
	for (size_t ctr = 0; ctr < model->lines; ctr++)
		if (model->T_lookup_buf[ctr] < result) result = model->T_lookup_buf[ctr];
	return result;
}
size_t GPUexec3MaxT(GPUexec3Model_p model){
	size_t result = model->T_lookup_buf[0];
	for (size_t ctr = 0; ctr < model->lines; ctr++)
		if (model->T_lookup_buf[ctr] > result) result = model->T_lookup_buf[ctr];
	return result;
}

size_t GPUexec3NearestT(GPUexec3Model_p model, size_t Tin){
	//TODO: Assumes sorted, must fix at some point
	if (Tin < model->T_lookup_buf[0]) return  model->T_lookup_buf[0];
	size_t ctr = 0, result =  model->T_lookup_buf[ctr++], prev = 0;
	while(result <= Tin){
		if (result == Tin) return result;
		prev = result;
		result = model->T_lookup_buf[ctr++];
	}
	return prev;
}

GPUexec3Model_p GPUexec3Model_init(short dev_id, const char* func){
	short lvl = 3;
#ifdef DEBUG
	lprintf(lvl-1, "|-----> GPUexec3Model_init(func=%s)\n", func);
#endif
#ifdef TEST
	lprintf(lvl-1, "|-----> GPUexec3Model_init\n");
	double timer = csecond();
#endif
	short dsize = 1, matrixNum;
	if (strcmp(func, "Dgemm") && strcmp(func, "Sgemm") ) error("GPUexec3Model_init: Invalid/Not implemented func");
	else {
		matrixNum = 3;
		if (!strcmp(func, "Dgemm")) dsize = sizeof(double);
		else if (!strcmp(func, "Sgemm")) dsize = sizeof(float);
	}
	GPUexec3Model_p out_model = (GPUexec3Model_p) malloc(sizeof(struct  BLAS3_data));
	char filename[256];
	sprintf(filename, "%s/Processed/%s_lookup-table_dev-%d.log", DEPLOYDB, func, dev_id);
	FILE* fp = fopen(filename,"r");
	if (!fp) {
		fprintf(stderr, "GPUexec3Model_init: Logfile = %s\n", filename);
		error("GPUexec3Model_init: t_exec3 LogFile not generated");
	}
	size_t bench_lines = count_lines(fp);
	out_model->lines = bench_lines;
	out_model->T_lookup_buf = (size_t*) malloc( bench_lines* sizeof(size_t));
	out_model->av_time_buf = (double*) malloc( bench_lines* sizeof(double));
	out_model->TransA_buf = (char*) malloc( bench_lines* sizeof(char));
	out_model->TransB_buf = (char*) malloc( bench_lines* sizeof(char));
#ifdef PDEBUG
	lprintf(lvl, "Reading %zu lines from %s\n", bench_lines, filename);
#endif
	int items;
	size_t trashdata, trashdata2, conv_itter;
	double error_margin, Dtrashdata;
	for (int i = 0; i < bench_lines; i++){
		items = fscanf(fp, "%c,%c,%zu,%zu,%zu, %lf,%lf,%zu,%lf\n",
			&out_model->TransA_buf[i], &out_model->TransB_buf[i], &out_model->T_lookup_buf[i],
			&trashdata, &trashdata2, &out_model->av_time_buf[i], &error_margin, &conv_itter, &Dtrashdata);
		if (items != 9) error("GPUexec3Model_init: Problem in reading model");
#ifdef DPDEBUG
		lprintf(lvl, "Scanned entry %d: T = %zu, TransA = %c, TransB = %c -> t_av = %lf ms\n",
		i, out_model->T_lookup_buf[i], out_model->TransA_buf[i], out_model->TransB_buf[i], out_model->av_time_buf[i]*1000);
#endif
    	}
	fclose(fp);
	out_model->dev_id = dev_id;
	out_model->func = func;
#ifdef TEST
	timer = csecond() - timer;
	lprintf(lvl, "Initialization time:%lf ms\n", timer*1000);
	lprintf(lvl-1, "<-----|\n");
#endif
#ifdef DEBUG
	lprintf(lvl-1, "<-----|\n");
#endif
	return out_model;
}

double GPUexec3Model_predict(GPUexec3Model_p model, size_t T, char TransA, char TransB){
	double result = -1;
	//TODO: Ultra naive, not (currently) important for performance but come on mate...
	for (int i = 0; i < model->lines; i++)
		if(model->T_lookup_buf[i] == T && model->TransA_buf[i] == TransA && model->TransB_buf[i] == TransB){
#ifdef DPDEBUG
			lprintf(4, "GPUexec3Model_predict: Found T = %d, TransA = %c, TransB = %c in loc %d -> t_av = %lf ms\n",
				model->T_lookup_buf[i], model->TransA_buf[i], model->TransB_buf[i], i, model->av_time_buf[i]*1000);
#endif
			return model->av_time_buf[i];
	}
#ifdef PDEBUG
	if (result == -1) warning("GPUexec3Model_predict: Performing Linear regression for prediction of exec_t of T_in=%d based on T=%d\n",
		T, GPUexec3NearestT(model, T));
#endif
	int nearest_T = GPUexec3NearestT(model, T);
	long double reg_t = 0;
	for (int i = 0; i < model->lines; i++)
		if(model->T_lookup_buf[i] == nearest_T && model->TransA_buf[i] == TransA && model->TransB_buf[i] == TransB){
			reg_t = model->av_time_buf[i];
			break;
		}
	if(reg_t == 0) error("Failed to find nearest_T = %d for regression\n", nearest_T);
	// Cheat for gemm, not generic (?)
	result = (double) (reg_t*pow(T,3)/(pow(nearest_T,3)));
	return result;
}
