///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The headers for Micro-benchmark limits, step, etc
///

#ifndef UNIHELPERS_H
#define UNIHELPERS_H

#include <stdio.h>
#include <cstring>
#include <stdarg.h>
#include <cblas.h>
#include <cuda.h>
#include "cublas_v2.h"

long long dgemm_flops(size_t M, size_t N, size_t K);
long long dgemm_memory(size_t M, size_t N, size_t K, size_t A_loc, size_t B_loc, size_t C_loc);

/// Memory layout struct for matrices
enum mem_layout { ROW_MAJOR = 0, COL_MAJOR };
const char *print_mem(mem_layout mem);

/// Print name of loc for transfers
const char *print_loc(short loc);

size_t CoCopeLiaGetMaxSqdimLvl3(short matrixNum, short dsize, size_t step);

size_t CoCopeLiaGetMaxVecdimLvl1(short vecNum, short dsize, size_t step);

short Dtest_equality(double* C_comp, double* C, long long size);
short Stest_equality(float* C_comp, float* C, long long size);
/// Check if there are CUDA errors on the stack
void cudaCheckErrors();


///TODO: Generalised (contiguous) data management functions

/// Malloc in loc with error-checking
void* CoCoMalloc(long long N_bytes, short loc);

/// Free in loc with error-checking
void CoCoFree(void * ptr, short loc);

/// Memcpy between two locations with errorchecking
void CoCoMemcpy(void* dest, void* src, long long N_bytes, short loc_dest, short loc_src);

/// Memcpy between two locations with errorchecking
void CoCoMemcpy2D(void* dest, void* src, long long N_bytes, short loc_dest, short loc_src);

/// Asunchronous Memcpy between two locations WITHOUT synchronous errorchecking. Use with caution.
void CoCoMemcpyAsync(void* dest, void* src, long long N_bytes, short loc_dest, short loc_src);

/// Asunchronous Memcpy between two locations WITHOUT synchronous errorchecking. Use with caution.
void CoCoMemcpy2DAsync(void* dest, size_t ldest, void* src, size_t lsrc, size_t rows, size_t cols, short elemSize, short loc_dest, short loc_src, cudaStream_t stream);

/// Initalize vector in loc with error-checking
template<typename VALUETYPE>
extern void CoCoVecInit(VALUETYPE *vec, long long length, int seed, short loc);
/// Helper for Parallel OpenMP vector initialization
template<typename VALUETYPE>
extern void CoCoParallelVecInitHost(VALUETYPE *vec, long long length, int seed);
inline double Drandom() { return (double)rand() / (double)RAND_MAX;}


double Gval_per_s(long long value, double time);

double csecond();

#if !defined(PRINTFLIKE)
#if defined(__GNUC__)
#define PRINTFLIKE(n,m) __attribute__((format(printf,n,m)))
#else
#define PRINTFLIKE(n,m) /* If only */
#endif /* __GNUC__ */
#endif /* PRINTFLIKE */

void lprintf(short lvl, const char *fmt, ...)PRINTFLIKE(2,3);
void massert(bool condi, const char *fmt, ...)PRINTFLIKE(2,3);
void error(const char *fmt, ...)PRINTFLIKE(1,2);
void warning(const char *fmt, ...)PRINTFLIKE(1,2);

/// Timers for benchmarks:
///CPU accurate timer:
double csecond();

///GPU CUDA timer for inter-stream timing, (ms accuracy)
typedef struct gpu_timer {
  cudaEvent_t start;
  cudaEvent_t stop;
  float ms;
} * gpu_timer_p;
gpu_timer_p gpu_timer_init();
void gpu_timer_start(gpu_timer_p timer, cudaStream_t stream);
void gpu_timer_stop(gpu_timer_p timer, cudaStream_t stream);
float gpu_timer_get(gpu_timer_p timer);

size_t count_lines(FILE* fp);

void check_benchmark(char *filename);

void TransposeTranslate(char TransChar, CBLAS_TRANSPOSE* cblasFlag, cublasOperation_t* cuBLASFlag, size_t* ldim, size_t dim1, size_t dim2);

cublasOperation_t OpCblasToCublas(CBLAS_TRANSPOSE src);
CBLAS_TRANSPOSE OpCublasToCblas(cublasOperation_t src);
cublasOperation_t OpCharToCublas(char src);
CBLAS_TRANSPOSE OpCharToCblas(char src);
char PrintCublasOp(cublasOperation_t src);

void ParseInputLvl3(const int argc, const char *argv[], short* dev_id, char* TransA, char* TransB, double* alpha, double* beta, size_t* D1, size_t* D2, size_t* D3, short* loc1, short* loc2, short* loc3, short* outloc, int* T, double* cpu_ratio);
void CheckLogLvl3(char* filename, short dev_id, char TransA, char TransB, double alpha, double beta, size_t D1, size_t D2, size_t D3, short loc1, short loc2, short loc3, short outloc, int T, double cpu_ratio);
void StoreLogLvl3(char* filename, short dev_id, char TransA, char TransB, double alpha, double beta, size_t D1, size_t D2, size_t D3, short loc1, short loc2, short loc3, short outloc, int T, double cpu_ratio, double av_time, double min_time, double max_time);

#endif
