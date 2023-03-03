///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The execution lookup functions for CoCopeLia.
///

#include <stdlib.h>
#include <math.h>

#include "unihelpers.hpp"
#include "GPUexec_lookup.hpp"

/* ============================BLAS3 Models================================= */

long int GPUexec3MinT(GPUexec3Model_p model){
	long int result = model->T_lookup_buf[0];
	for (long int ctr = 0; ctr < model->lines; ctr++)
		if (model->T_lookup_buf[ctr] < result) result = model->T_lookup_buf[ctr];
	return result;
}

long int GPUexec3MaxT(GPUexec3Model_p model){
	long int result = model->T_lookup_buf[0];
	for (long int ctr = 0; ctr < model->lines; ctr++)
		if (model->T_lookup_buf[ctr] > result) result = model->T_lookup_buf[ctr];
	return result;
}

long int GPUexec3NearestT(GPUexec3Model_p model, long int Tin){
	//TODO: Assumes sorted, must fix at some point
	if (Tin < model->T_lookup_buf[0]) return  model->T_lookup_buf[0];
	long int ctr = 0, result =  model->T_lookup_buf[ctr++], prev = 0;
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
	short dsize = 1;//, matrixNum;
	if (strcmp(func, "Dgemm") && strcmp(func, "Sgemm") ) error("GPUexec3Model_init: Invalid/Not implemented func");
	else {
		//matrixNum = 3;
		if (!strcmp(func, "Dgemm")) dsize = sizeof(double);
		else if (!strcmp(func, "Sgemm")) dsize = sizeof(float);
	}
	GPUexec3Model_p out_model = (GPUexec3Model_p) malloc(sizeof(struct  BLAS3_data));
	char filename[1024];
	sprintf(filename, "%s/Processed/%s_lookup-table_dev-%d.log", DEPLOYDB, func, dev_id);
	FILE* fp = fopen(filename,"r");
	if (!fp) {
		fprintf(stderr, "GPUexec3Model_init: Logfile = %s\n", filename);
		error("GPUexec3Model_init: t_exec3 LogFile not generated");
	}
	long int bench_lines = count_lines(fp);
	out_model->lines = bench_lines;
	out_model->T_lookup_buf = (long int*) malloc( bench_lines* sizeof(long int));
	out_model->av_time_buf = (double*) malloc( bench_lines* sizeof(double));
	out_model->av_W_buf = (double*) malloc( bench_lines* sizeof(double));
	out_model->TransA_buf = (char*) malloc( bench_lines* sizeof(char));
	out_model->TransB_buf = (char*) malloc( bench_lines* sizeof(char));
#ifdef PDEBUG
	lprintf(lvl, "Reading %ld lines from %s\n", bench_lines, filename);
#endif
	int items;
	long int trashdata, trashdata2, conv_itter;
	double error_margin, Dtrashdata, Joules;
	for (int i = 0; i < bench_lines; i++){
		items = fscanf(fp, "%c,%c,%ld,%ld,%ld,%lf,%lf,%lf,%lf,%ld,%lf\n",
			&out_model->TransA_buf[i], &out_model->TransB_buf[i], &out_model->T_lookup_buf[i],
			&trashdata, &trashdata2, &out_model->av_time_buf[i], &out_model->av_W_buf[i], &Joules, &error_margin, &conv_itter, &Dtrashdata);
		if (items != 11) error("GPUexec3Model_init: Problem in reading model");
#ifdef DPDEBUG
		lprintf(lvl, "Scanned entry %d: T = %ld, TransA = %c, TransB = %c -> t_av = %lf ms, W_av = %lf W, J_total = %lf J\n",
		i, out_model->T_lookup_buf[i], out_model->TransA_buf[i], out_model->TransB_buf[i],
		out_model->av_time_buf[i]*1000, out_model->av_W_buf[i], Joules);
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

double GPUexec3Model_predict(GPUexec3Model_p model, long int T, char TransA, char TransB){
	double result = -1;
	//TODO: Ultra naive, not (currently) important for performance but come on mate...
	for (int i = 0; i < model->lines; i++)
		if(model->T_lookup_buf[i] == T && model->TransA_buf[i] == TransA && model->TransB_buf[i] == TransB){
#ifdef DPDEBUG
			lprintf(4, "GPUexec3Model_predict: Found T = %ld, TransA = %c, TransB = %c in loc %d -> t_av = %lf ms\n",
				model->T_lookup_buf[i], model->TransA_buf[i], model->TransB_buf[i], i, model->av_time_buf[i]*1000);
#endif
			return model->av_time_buf[i];
	}
#ifdef PDEBUG
	if (result == -1) warning("GPUexec3Model_predict: Performing Linear regression for prediction of exec_t of T_in=%ld based on T=%ld\n",
		T, GPUexec3NearestT(model, T));
#endif
	long int nearest_T = GPUexec3NearestT(model, T);
	long double reg_t = 0;
	for (int i = 0; i < model->lines; i++)
		if(model->T_lookup_buf[i] == nearest_T && model->TransA_buf[i] == TransA && model->TransB_buf[i] == TransB){
			reg_t = model->av_time_buf[i];
			break;
		}
	if(reg_t == 0) error("Failed to find nearest_T = %ld for regression\n", nearest_T);
	// Cheat for gemm, not generic (?)
	result = (double) (reg_t*pow(T,3)/(pow(nearest_T,3)));
	return result;
}

/* ============================BLAS2 Models================================= */
GPUexec2Model_p GPUexec2Model_init(short dev_id, const char* func){
	error("GPUexec2Model_init: Not implemented\n");
	return NULL;
}
/* ============================BLAS1 Models================================= */

long int GPUexec1MinT(GPUexec1Model_p model){
	long int result = model->T_lookup_buf[0];
	for (long int ctr = 0; ctr < model->lines; ctr++)
		if (model->T_lookup_buf[ctr] < result) result = model->T_lookup_buf[ctr];
	return result;
}
long int GPUexec1MaxT(GPUexec1Model_p model){
	long int result = model->T_lookup_buf[0];
	for (long int ctr = 0; ctr < model->lines; ctr++)
		if (model->T_lookup_buf[ctr] > result) result = model->T_lookup_buf[ctr];
	return result;
}

long int GPUexec1NearestT(GPUexec1Model_p model, long int Tin){
	//TODO: Assumes sorted, must fix at some point
	if (Tin < model->T_lookup_buf[0]) return  model->T_lookup_buf[0];
	long int ctr = 0, result =  model->T_lookup_buf[ctr++], prev = 0;
	while(result <= Tin){
		if (result == Tin) return result;
		prev = result;
		result = model->T_lookup_buf[ctr++];
	}
	return prev;
}

GPUexec1Model_p GPUexec1Model_init(short dev_id, const char* func){
	short lvl = 3;
#ifdef DEBUG
	lprintf(lvl-1, "|-----> GPUexec1Model_init(func=%s)\n", func);
#endif
#ifdef TEST
	lprintf(lvl-1, "|-----> GPUexec1Model_init\n");
	double timer = csecond();
#endif
	short dsize = 1, matrixNum;
	if (strcmp(func, "Daxpy") && strcmp(func, "Saxpy") ) error("GPUexec1Model_init: Invalid/Not implemented func");
	else {
		matrixNum = 3;
		if (!strcmp(func, "Daxpy")) dsize = sizeof(double);
		else if (!strcmp(func, "Saxpy")) dsize = sizeof(float);
	}
	GPUexec1Model_p out_model = (GPUexec1Model_p) malloc(sizeof(struct  BLAS1_data));
	char filename[256];
	sprintf(filename, "%s/Processed/%s_lookup-table_dev-%d.log", DEPLOYDB, func, dev_id);
	FILE* fp = fopen(filename,"r");
	if (!fp) {
		fprintf(stderr, "GPUexec1Model_init: Logfile = %s\n", filename);
		error("GPUexec1Model_init: t_exec1 LogFile not generated");
	}
	long int bench_lines = count_lines(fp);
	out_model->lines = bench_lines;
	out_model->T_lookup_buf = (long int*) malloc( bench_lines* sizeof(long int));
	out_model->av_time_buf = (double*) malloc( bench_lines* sizeof(double));
	out_model->av_W_buf = (double*) malloc( bench_lines* sizeof(double));
#ifdef PDEBUG
	lprintf(lvl, "Reading %ld lines from %s\n", bench_lines, filename);
#endif
	int items;
	long int conv_itter;
	double error_margin, Dtrashdata, Joules;
	for (int i = 0; i < bench_lines; i++){
		items = fscanf(fp, "%ld,%lf,%lf,%lf,%lf,%ld,%lf\n",
			&out_model->T_lookup_buf[i], &out_model->av_time_buf[i], &out_model->av_W_buf[i], &Joules, &error_margin, &conv_itter, &Dtrashdata);
		if (items != 7) error("GPUexec1Model_init: Problem in reading model");
#ifdef DPDEBUG
		lprintf(lvl, "Scanned entry %d: T = %ld -> t_av = %lf ms, W_av = %lf W, J_total = %lf J\n",
		i, out_model->T_lookup_buf[i], out_model->av_time_buf[i]*1000, out_model->av_W_buf[i], Joules);
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

double GPUexec1Model_predict(GPUexec1Model_p model, long int T){
	double result = -1;
	//TODO: Ultra naive, not (currently) important for performance but come on mate...
	for (int i = 0; i < model->lines; i++)
		if(model->T_lookup_buf[i] == T){
#ifdef DPDEBUG
			lprintf(4, "GPUexec1Model_predict: Found T = %ld in loc %d -> t_av = %lf ms\n",
				model->T_lookup_buf[i], i, model->av_time_buf[i]*1000);
#endif
			return model->av_time_buf[i];
	}
#ifdef PDEBUG
	if (result == -1) warning("GPUexec1Model_predict: Performing Linear regression for prediction of exec_t of T_in=%ld based on T=%ld\n",
		T, GPUexec1NearestT(model, T));
#endif
	long int nearest_T = GPUexec1NearestT(model, T);
	long double reg_t = 0;
	for (int i = 0; i < model->lines; i++)
		if(model->T_lookup_buf[i] == nearest_T){
			reg_t = model->av_time_buf[i];
			break;
		}
	if(reg_t == 0) error("Failed to find nearest_T = %ld for regression\n", nearest_T);
	// Cheat for gemm, not generic (?)
	result = (double) (reg_t*pow(T,3)/(pow(nearest_T,3)));
	return result;
}
