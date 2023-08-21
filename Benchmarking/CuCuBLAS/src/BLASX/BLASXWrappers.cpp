///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief BLAS lvl 3 wrappers for benchmarks.
///
#include <cassert>
#include <cublasXt.h>
#include <cblas.h>

#include "backend_wrappers.hpp"
#include <BLASxModifiedcblas.h>
#include "PARALiA.hpp"
#include "unihelpers.hpp"

#ifdef TTEST /// C programmers hate him PETROFIX
extern int b_trans_ctr;
extern long long b_bytes[100000];
extern int b_locs[10000][2];
extern double b_timers[100000][3];
extern int b_timer_ctr[LOC_NUM][LOC_NUM];
extern double b_link_gbytes_s[LOC_NUM][LOC_NUM];

void b_reseTTEST(){
	for(int k = 0; k < b_trans_ctr; k++){
		b_bytes[k] = 0;
		for(int m = 0; m < 3; m++) b_timers[k][m] = 0;
	}
	b_trans_ctr = 0;
	for (int d1 = 0; d1 < LOC_NUM; d1++)
		for (int d2 = 0; d2 < LOC_NUM; d2++){
			b_timer_ctr[d1][d2] = 0; 
			b_link_gbytes_s[d1][d2] = 0; 
		}
}

void b_HopMemcpyPrint(){
	printf("\n Tranfers Full:\n");
	FILE* fp = fopen("temp_blasx_trans.log", "w+");
	for(int k = 0; k < b_trans_ctr; k++){
		int src = b_locs[k][0], dest = b_locs[k][1];
		b_timer_ctr[idxize(dest)][idxize(src)]++;
		double time = (b_timers[k][2] - b_timers[k][1]), pipe_time = (b_timers[k][2] - b_timers[k][0]);
		b_link_gbytes_s[idxize(dest)][idxize(src)]+=Gval_per_s(b_bytes[k], time);
		printf( "Normal 2D Trasfer %d->%d : total_t = %lf ms ( %.3lf Gb/s ), pipelined_t = %lf ms ( %.3lf Gb/s )\n", 
			src, dest, 1000*time, Gval_per_s(b_bytes[k], time), 1000*pipe_time, Gval_per_s(b_bytes[k], pipe_time));
		fprintf(fp, "%d,%d,[ %d %d ],%ld,%lf,%lf,%lf\n", src, dest, src, dest, b_bytes[k], b_timers[k][0], b_timers[k][1], b_timers[k][2]);
	}
		
	printf("\n Full Tranfer Map:\n   |");
	for (int d2 = 0; d2 < LOC_NUM; d2++)
		printf( "  %2d  |", deidxize(d2));
	printf( "\n   |");
	for (int d2 = 0; d2 < LOC_NUM; d2++)
		printf( "-------");
	printf( "\n");
	for (int d1 = 0; d1 < LOC_NUM; d1++){
		printf( "%2d | ", deidxize(d1));
		for (int d2 = 0; d2 < LOC_NUM; d2++){
			printf( "%4d | ", b_timer_ctr[d1][d2]);
		}
		printf( "\n");
	}

	printf("\n Full Tranfer Map Achieved Bandwidths (GB/s):\n   |");
	for (int d2 = 0; d2 < LOC_NUM; d2++)
		printf( "  %2d   |", deidxize(d2));
	printf( "\n   |");
	for (int d2 = 0; d2 < LOC_NUM; d2++)
		printf( "--------");
	printf( "\n");
	for (int d1 = 0; d1 < LOC_NUM; d1++){
		printf( "%2d | ", deidxize(d1));
		for (int d2 = 0; d2 < LOC_NUM; d2++)
			if (b_timer_ctr[d1][d2]) printf( "%.2lf | ", b_link_gbytes_s[d1][d2]/b_timer_ctr[d1][d2]);
			else printf( "  -   | ");
		printf( "\n");
	}
	fclose(fp);
	b_reseTTEST();
}

#endif 
void BLASxFlushGPUBuf(short dev_num, int dev_ids[] ){
	for(int i=0; i<dev_num;i++){ 
		CoCoPeLiaSelectDevice(deidxize(i));
		CoCoSyncCheckErr();
	}
	BLASx_LRU_free(dev_num, dev_ids);
}

double BLASxDgemmWrap(char TransA, char TransB, long int M, long int N, long int K, double alpha, double* A, long int ldA, double* B, long int ldB, double beta, double* C, long int ldC, long int T, double cpu_ratio, short dev_num, int dev_ids[] ){
	short lvl = 1;
	double total_t = csecond();
#ifdef DEBUG
	lprintf(lvl-1, "|-----> BLASxDgemmWrap(%c,%c,%zu,%zu,%zu,%lf,A(%d),%zu,B(%d),%zu,%lf,C(%d),%zu)\n",
		TransA, TransB, M, N, K, alpha, CoCoGetPtrLoc(A), ldA,
		CoCoGetPtrLoc(B), ldB, beta, CoCoGetPtrLoc(C), ldC);
#endif

#ifdef TEST
	lprintf(lvl-1, "|-----> BLASxDgemmWrap\n");
	double cpu_timer = csecond();
#endif
	CBLAS_TRANSPOSE cpu_op_A = OpCharToCblas(TransA), cpu_op_B = OpCharToCblas(TransB);
	BLASx_dgemm(CblasColMajor,cpu_op_A,cpu_op_B,M,N,K,alpha,A,ldA, B,ldB, beta,C, ldC, T, dev_num, dev_ids);
	CoCoSyncCheckErr();
#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "BLASx execution time -> t_kernel = %lf ms\n", cpu_timer*1000);
#endif
#ifdef TTEST
	b_HopMemcpyPrint();
#endif	
	total_t = csecond() - total_t;
	return total_t;

}

double BLASxExDgemmWrap(char TransA, char TransB, long int M, long int N, long int K, double alpha, double* A, long int ldA, double* B, long int ldB, double beta, double* C, long int ldC, long int T, double cpu_ratio, short dev_num, int dev_ids[] ){
	short lvl = 1;
	double total_t = csecond();
#ifdef DEBUG
	lprintf(lvl-1, "|-----> BLASxExDgemmWrap(%c,%c,%zu,%zu,%zu,%lf,A(%d),%zu,B(%d),%zu,%lf,C(%d),%zu)\n",
		TransA, TransB, M, N, K, alpha, CoCoGetPtrLoc(A), ldA,
		CoCoGetPtrLoc(B), ldB, beta, CoCoGetPtrLoc(C), ldC);
#endif

#ifdef TEST
	lprintf(lvl-1, "|-----> BLASxExDgemmWrap\n");
	double cpu_timer = csecond();
#endif
	CBLAS_TRANSPOSE cpu_op_A = OpCharToCblas(TransA), cpu_op_B = OpCharToCblas(TransB);
	BLASx_gpubuf_dgemm(CblasColMajor,cpu_op_A,cpu_op_B,M,N,K,alpha,A,ldA, B,ldB, beta,C, ldC, T, dev_num, dev_ids);
	CoCoSyncCheckErr();
#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "BLASx execution time -> t_kernel = %lf ms\n", cpu_timer*1000);
#endif
	total_t = csecond() - total_t;
#ifdef TTEST
	b_HopMemcpyPrint();
#endif	
	return total_t;

}
