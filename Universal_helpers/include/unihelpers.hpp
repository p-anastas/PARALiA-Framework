///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The headers for functions for general use throught CoCoPeLia
///

#ifndef UNIHELPERS_H
#define UNIHELPERS_H

#include<iostream>
#include <string>

#include <cstdlib>
#include <cstring>

#include <stdio.h>
#include <stdarg.h>

/*****************************************************/
/// Generalised "Command queue" and "Event" definition (e.g. CUDA streams and Events for CUDA backend)

typedef class Event* Event_p;

typedef class CommandQueue
{
	private:
	public:
		void* cqueue_backend_ptr;

		CommandQueue();
		void sync_barrier();
		void wait_for_event(Event_p Wevent);
		std::string name;
		void print() { std::cout << "Command Queue : " << name; }

}* CQueue_p;

class Event
{
	private:
	public:
		void* event_backend_ptr;

		Event();
		void sync_barrier();
		void record_to_queue(CQueue_p Rr);
		short is_complete();

};

/*****************************************************/
/// Generalised data management functions & helpers

// Sync all devices and search for enchountered errors.
void CoCoSyncCheckErr();

// Search for enchountered errors without synchronization.
void CoCoASyncCheckErr();

// Enable available links for target device with the provided list of (other) devices
void CoCoEnableLinks(short target_dev_i, short dev_ids[], short num_devices);

// Malloc in loc with error-checking
void* CoCoMalloc(long long N_bytes, short loc);

// Free in loc with error-checking
void CoCoFree(void * ptr, short loc);

// Memcpy between two locations with errorchecking
void CoCoMemcpy(void* dest, void* src, long long N_bytes, short loc_dest, short loc_src);

// Memcpy between two locations with errorchecking
void CoCoMemcpy2D(void* dest, size_t ldest, void* src, size_t lsrc, size_t rows, size_t cols, short elemSize, short loc_dest, short loc_src);

// Asunchronous Memcpy between two locations WITHOUT synchronous errorchecking. Use with caution.
void CoCoMemcpyAsync(void* dest, void* src, long long N_bytes, short loc_dest, short loc_src, CQueue_p transfer_medium);

// Asunchronous Memcpy between two locations WITHOUT synchronous errorchecking. Use with caution.
void CoCoMemcpy2DAsync(void* dest, size_t ldest, void* src, size_t lsrc, size_t rows, size_t cols, short elemSize, short loc_dest, short loc_src, CQueue_p transfer_medium);

// Initalize vector in loc with error-checking
template<typename VALUETYPE>
extern void CoCoVecInit(VALUETYPE *vec, long long length, int seed, short loc);
// Helper for Parallel OpenMP vector initialization
template<typename VALUETYPE>
extern void CoCoParallelVecInitHost(VALUETYPE *vec, long long length, int seed);

// Return the max dim size (which is a multiple of 'step') for 'Asset2DNum' square assets on 'loc'
size_t CoCoGetMaxDimSqAsset2D(short Asset2DNum, short dsize, size_t step, short loc);

// Return the max dim size (which is a multiple of 'step') for 'Asset1DNum' assets on 'loc'
size_t CoCoGetMaxDimAsset1D(short Asset1DNum, short dsize, size_t step, short loc);

short CoCoGetPtrLoc(const void * in_ptr);

/*****************************************************/
/// Timers for benchmarks
// CPU accurate timer
double csecond();

// Event timer for background Event timing (~usually ms accuracy)
typedef class Event_timer
{
	private:
		Event_p Event_start;
		Event_p Event_stop;
		double time_ms;
	public:

		Event_timer();
		void start_point(CQueue_p start_queue);
		void stop_point(CQueue_p stop_queue);
		double sync_get_time();
}* Event_timer_p;

/*****************************************************/
/// Print functions
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

/*****************************************************/
/// Enum(and internal symbol) functions
// Memory layout struct for matrices
enum mem_layout { ROW_MAJOR = 0, COL_MAJOR = 1 };
const char *print_mem(mem_layout mem);

// Print name of loc for transfers
//const char *print_loc(short loc); FIXME: Remove

/*****************************************************/
/// General benchmark functions

inline double Drandom() { return (double)rand() / (double)RAND_MAX;}
short Dtest_equality(double* C_comp, double* C, long long size);
short Stest_equality(float* C_comp, float* C, long long size);

double Gval_per_s(long long value, double time);
long long dgemm_flops(size_t M, size_t N, size_t K);
long long dgemm_memory(size_t M, size_t N, size_t K, size_t A_loc, size_t B_loc, size_t C_loc);

long long daxpy_flops(size_t N);

size_t count_lines(FILE* fp); // TODO: Where is this used?
void check_benchmark(char *filename);

/*****************************************************/

#endif
