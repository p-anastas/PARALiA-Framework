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

int gcd (int a, int b, int c)
{
  for(int i = std::min(a, std::min(b, c)); i>1; i--) if(( a%i == 0 ) and ( b%i == 0 ) and ( c%i == 0 )) return i;
  return 1;
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
//#ifdef DEBUG
	fprintf(stderr, "WARNING -> ");
	va_list ap;
	va_start(ap, fmt);
	_printf(fmt, ap);
	va_end(ap);
//#endif
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

template<typename VALUETYPE>
const char *printlist(VALUETYPE *list, int length)
{
  char* outstring = (char*) malloc(abs(length)*10*sizeof(char));
  std::string printfCmd(" ");
  sprintf(outstring, "[");
  if (std::is_same<VALUETYPE, short>::value) printfCmd += "%hd";
	if (std::is_same<VALUETYPE, int>::value) printfCmd += "%d";
  else if (std::is_same<VALUETYPE, float>::value) printfCmd += "%3.3f";
  else if (std::is_same<VALUETYPE, double>::value) printfCmd += "%3.3lf";
  for (int i =0; i < length; i++) sprintf(outstring + strlen(outstring), printfCmd.c_str(), list[i]);
	sprintf(outstring + strlen(outstring), " ]");
  return outstring;
}

template const char *printlist<double>(double *list, int length);
template const char *printlist<float>(float *list, int length);
template const char *printlist<int>(int *list, int length);
template const char *printlist<short>(short *list, int length);

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

long int Dvec_diff(double* a, double* b, long long size, double eps) {
	long int failed = 0;
	//#pragma omp parallel for
	for (long long i = 0; i < size; i++)
		if (Derror(a[i], b[i]) > eps){
			//#pragma omp atomic
			failed++;
		}
	return failed;
}

long int Svec_diff(float* a, float* b, long long size, float eps) {
  	long int failed = 0;
	//#pragma omp parallel for
  	for (long long i = 0; i < size; i++)
		if (Serror(a[i], b[i]) > eps){
			//#pragma omp atomic
			failed++;
		}
  	return failed;
}

short Stest_equality(float* C_comp, float* C, long long size) {
  long int acc = 4, failed;
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
    return 0;
  } else
    fprintf(stderr, "Test passed(Accuracy= %zu digits, %zu/%lld breaking for %zu)\n\n",
            acc, failed, size, acc + 1);
  return (short) acc;
}

short Dtest_equality(double* C_comp, double* C, long long size) {
  long int acc = 8, failed;
  double eps = 1e-8;
  failed = Dvec_diff(C_comp, C, size, eps);
  while (eps > DBL_MIN && !failed && acc < 30) {
    eps *= 0.1;
    acc++;
    failed = Dvec_diff(C_comp, C, size, eps);
  }
  if (8==acc) {
  	fprintf(stderr, "Test failed %zu times\n", failed);
  	int ctr = 0;
    long long itt = 0;
  	while (ctr < 10 & itt < size){
  		if (Derror(C_comp[itt], C[itt]) > eps){
  			fprintf(stderr, "Baseline vs Tested(adr = %p, itt = %lld): %.15lf vs %.15lf\n", &C[itt], itt, C_comp[itt], C[itt]);
  			ctr++;
  		}
  		itt++;
  	}
  return 0;
  }
  else
    fprintf(stderr, "Test passed(Accuracy= %zu digits, %zu/%lld breaking for %zu)\n\n",
            acc, failed, size, acc + 1);
  return (short) acc;
}

long int count_lines(FILE* fp){
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

long long gemm_flops(long int M, long int N, long int K){
	return (long long) M * N * (2 * K + 1);
}

long long gemm_memory(long int M, long int N, long int K, long int A_loc, long int B_loc, long int C_loc, short dsize){
	return (M * K * A_loc + K * N * B_loc + M * N * C_loc)*dsize;
}

long long gemv_flops(long int M, long int N){
	return (long long) M * (2 * N + 1);
}

long long gemv_memory(long int M, long int N, long int A_loc, long int x_loc, long int y_loc, short dsize){
	return (M * N * A_loc + N * x_loc + M * y_loc)*dsize;
}

long long axpy_flops(long int  N){
	return (long long) 2* N;
}

long long axpy_memory(long int  N, long int x_loc, long int y_loc, short dsize){
	return (long long) N*(x_loc + y_loc)*dsize;
}

long long dot_flops(long int  N){
	return (long long) 2* N;
}

long long dot_memory(long int  N, long int x_loc, long int y_loc, short dsize){
	return (long long) N*(x_loc + y_loc)*dsize;
}

void translate_binary_to_unit_list(int case_id, int* active_unit_num_p, int* active_unit_id_list){
	int mask;
	*active_unit_num_p = 0;
	for (int mask_offset = 0; mask_offset < LOC_NUM; mask_offset++){
		mask =  1 << mask_offset;
		if (case_id & mask){
			active_unit_id_list[*active_unit_num_p] = deidxize(mask_offset);
			(*active_unit_num_p)++;
			//lprintf(0, "PARALIA_translate_unit_ids(case_id = %d): mask = %d -> Adding unit %d to available\n",
			//	case_id, mask, deidxize(mask_offset));
		}
	}
}
/*
long long dgemv_flops(long int M, long int N){
	return (long long) M * (2 * N + 1);
}

long long dgemv_bytes(long int M, long int N){
	return (M * N + N + M * 2)*sizeof(double) ;
}


long long dgemm_bytes(long int M, long int N, long int K){
	return (M * K + K * N + M * N * 2)*sizeof(double) ;
}

long long sgemm_bytes(long int M, long int N, long int K){
	return (M * K + K * N + M * N * 2)*sizeof(float) ;
}
*/
