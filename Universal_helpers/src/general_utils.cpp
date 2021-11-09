///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief Some convinient C/C++ utilities for CoCopeLia.
///

#include "unihelpers.hpp"

#include <float.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <omp.h>
#include <math.h>

double csecond(void) {
  struct timespec tms;

  if (clock_gettime(CLOCK_REALTIME, &tms)) {
    return (0.0);
  }
  /// seconds, multiplied with 1 million
  int64_t micros = tms.tv_sec * 1000000;
  /// Add full microseconds
  micros += tms.tv_nsec / 1000;
  /// round up if necessary
  if (tms.tv_nsec % 1000 >= 500) {
    ++micros;
  }
  return ((double)micros / 1000000.0);
}

void tab_print(int lvl){
	for (int rep=0;rep<lvl;rep++) fprintf(stderr, "\t");
}

void _printf(const char *fmt, va_list ap)
{
    if (fmt) vfprintf(stderr, fmt, ap);
    //putc('\n', stderr);
}

void warning(const char *fmt, ...) {
	fprintf(stderr, "WARNING -> ");
	va_list ap;
	va_start(ap, fmt);
	_printf(fmt, ap);
	va_end(ap);
}

void error(const char *fmt, ...) {
	fprintf(stderr, "ERROR ->");
	va_list ap;
	va_start(ap, fmt);
	_printf(fmt, ap);
	va_end(ap);
	exit(1);
}

void massert(bool condi, const char *fmt, ...) {
	if (!condi) {
		va_list ap;
		va_start(ap, fmt);
		_printf(fmt, ap);
		va_end(ap);
		exit(1);
  	}
}

void lprintf(short lvl, const char *fmt, ...){
	tab_print(lvl);
	va_list ap;
	va_start(ap, fmt);
	_printf(fmt, ap);
	va_end(ap);
}

const char *print_mem(mem_layout mem) {
  if (mem == ROW_MAJOR)
    return "Row major";
  else if (mem == COL_MAJOR)
    return "Col major";
  else
    return "ERROR";
}

double dabs(double x){
	if (x < 0) return -x;
	else return x;
}

inline float Serror(float a, float b) {
  if (a == 0) return (float) dabs((float)(a - b)); 
  return dabs(a - b)/a;
}

inline double Derror(double a, double b) {
  if (a == 0) return dabs(a - b); 
  return dabs(a - b)/a;
}

size_t Dvec_diff(double* a, double* b, long long size, double eps) {
	size_t failed = 0;
	//#pragma omp parallel for
	for (long long i = 0; i < size; i++)
		if (Derror(a[i], b[i]) > eps){
			//#pragma omp atomic 
			failed++;
		}
	return failed;
}

size_t Svec_diff(float* a, float* b, long long size, float eps) {
  	size_t failed = 0;
	//#pragma omp parallel for
  	for (long long i = 0; i < size; i++)
		if (Serror(a[i], b[i]) > eps){
			//#pragma omp atomic 
			failed++;
		}
  	return failed;
}

short Stest_equality(float* C_comp, float* C, long long size) {
  size_t acc = 4, failed;
  float eps = 1e-4;
  failed = Svec_diff(C_comp, C, size, eps);
  while (eps > FLT_MIN && !failed && acc < 30) {
    eps *= 0.1;
    acc++;
    failed = Svec_diff(C_comp, C, size, eps);
  }
  if (4==acc) {
	fprintf(stderr, "Test failed %zu times\n", failed);
	int ctr = 0, itt = 0; 
	while (ctr < 10 & itt < size){
		if (Serror(C_comp[itt], C[itt]) > eps){
			fprintf(stderr, "Baseline vs Tested: %.10f vs %.10f\n", C_comp[itt], C[itt]);
			ctr++;
		}
		itt++;
	}
  } else
    fprintf(stderr, "Test passed(Accuracy= %zu digits, %zu/%lld breaking for %zu)\n\n",
            acc, failed, size, acc + 1);
  return (short) acc; 
}

short Dtest_equality(double* C_comp, double* C, long long size) {
  size_t acc = 8, failed;
  double eps = 1e-8;
  failed = Dvec_diff(C_comp, C, size, eps);
  while (eps > DBL_MIN && !failed && acc < 30) {
    eps *= 0.1;
    acc++;
    failed = Dvec_diff(C_comp, C, size, eps);
  }
  if (8==acc) {
	fprintf(stderr, "Test failed %zu times\n", failed);
	int ctr = 0, itt = 0; 
	while (ctr < 10 & itt < size){
		if (Derror(C_comp[itt], C[itt]) > eps){
			fprintf(stderr, "Baseline vs Tested: %.15lf vs %.15lf\n", C_comp[itt], C[itt]);
			ctr++;
		}
		itt++;
	}
  } else
    fprintf(stderr, "Test passed(Accuracy= %zu digits, %zu/%lld breaking for %zu)\n\n",
            acc, failed, size, acc + 1);
  return (short) acc; 
}

size_t count_lines(FILE* fp){
	if (!fp) error("count_lines: fp = 0 ");
	int ctr = 0; 
	char chr = getc(fp);
	while (chr != EOF){
		//Count whenever new line is encountered
		if (chr == '\n') ctr++;
		//take next character from file.
		chr = getc(fp);
	}
	fseek(fp, 0, SEEK_SET);;
	return ctr;
}

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

double Gval_per_s(long long value, double time){
  return value / (time * 1e9);
}

long long dgemm_flops(size_t M, size_t N, size_t K){
	return (long long) M * K * (2 * N + 1);
}

long long dgemm_memory(size_t M, size_t N, size_t K, size_t A_loc, size_t B_loc, size_t C_loc){
	return (M * K * A_loc + K * N * B_loc + M * N * C_loc)*sizeof(double); 
}

long long daxpy_flops(size_t N){
	return (long long) 2* N;
}


/*
long long dgemv_flops(size_t M, size_t N){
	return (long long) M * (2 * N + 1);
}

long long dgemv_bytes(size_t M, size_t N){
	return (M * N + N + M * 2)*sizeof(double) ; 
}


long long dgemm_bytes(size_t M, size_t N, size_t K){
	return (M * K + K * N + M * N * 2)*sizeof(double) ; 
}

long long sgemm_bytes(size_t M, size_t N, size_t K){
	return (M * K + K * N + M * N * 2)*sizeof(float) ; 
}
*/






