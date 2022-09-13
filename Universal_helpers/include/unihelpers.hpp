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

#include <atomic>

/*****************************************************/
/// Generalised "Command queue" and "Event" definition (e.g. CUDA streams and Events for CUDA backend)

typedef class Event* Event_p;

typedef class CommandQueue
{
	private:
	public:
#ifdef ENABLE_PARALLEL_BACKEND
		void* cqueue_backend_ptr[MAX_BACKEND_L];
		void* cqueue_backend_data[MAX_BACKEND_L];
		int backend_ctr = 0;
#else
		void* cqueue_backend_ptr;
		void* cqueue_backend_data;
#endif
		int dev_id;

		//Constructor
		CommandQueue(int dev_id);

		//Destructor
		~CommandQueue();
		void sync_barrier();
		void add_host_func(void* func, void* data);
		void wait_for_event(Event_p Wevent);
#ifdef ENABLE_PARALLEL_BACKEND
		int request_parallel_backend();
		void set_parallel_backend(int backend_ctr);
#endif
		std::string name;
		void print() { std::cout << "Command Queue : " << name; }

}* CQueue_p;

enum event_status{
	UNRECORDED = 0, /// Event has not been recorded yet.
	RECORDED = 1, /// Recorded but not guaranteed to be complete.
	COMPLETE = 2,  /// Complete but not yet ran 'check' function (for cache updates etc)
	CHECKED = 3,  /// Complete and Checked/Updated caches etc.
	GHOST = 4  /// Does not exist in the time continuum.
};

/// Returns a string representation for event_status
const char* print_event_status(event_status in_status);

class Event
{
	private:
		event_status status;
	public:
		void* event_backend_ptr;
		int id, dev_id;

		/// Constructors
		Event(int dev_id);
		/// Destructors
		~Event();
		/// Functions
		void sync_barrier();
		void record_to_queue(CQueue_p Rr);
		event_status query_status();
		void checked();
		void reset();
		void soft_reset();

};

/*****************************************************/
/// Generalised data management functions & helpers

// Return current device used
int CoCoPeLiaGetDevice();

// Select device for current running pthread
void CoCoPeLiaSelectDevice(short dev_id);

// Return free memory and total memory for current device
void CoCoPeLiaDevGetMemInfo(long long* free_dev_mem, long long* max_dev_mem);

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
void CoCoMemcpy2DAsync(void* dest, size_t ldest, void* src, size_t lsrc, size_t rows, size_t cols,
	short elemSize, short loc_dest, short loc_src, CQueue_p transfer_medium);

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

// Struct for multi-hop optimized transfers
typedef struct link_road{
	int hop_num;
	int hop_uid_list[LOC_NUM];
	void* hop_buf_list[LOC_NUM];
	int hop_ldim_list[LOC_NUM];

	CQueue_p hop_cqueue_list[LOC_NUM-1];
	Event_p hop_event_list[LOC_NUM-1];
}* link_road_p;

// A memcpy implementation using multiple units as intermendiate hops for a better transfer bandwidth
void FasTCoCoMemcpy2DAsync(link_road_p roadMap, size_t rows, size_t cols, short elemSize);

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
		int dev_id;

		Event_timer(int dev_id);
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

template<typename VALUETYPE>
extern const char *printlist(VALUETYPE *list, int length);

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
long long gemm_flops(size_t M, size_t N, size_t K);
long long gemm_memory(size_t M, size_t N, size_t K, size_t A_loc, size_t B_loc, size_t C_loc, short dsize);

long long axpy_flops(long int  N);
long long axpy_memory(long int N, size_t x_loc, size_t y_loc, short dsize);

size_t count_lines(FILE* fp); // TODO: Where is this used?
void check_benchmark(char *filename);

/*****************************************************/
int gcd (int a, int b, int c);
inline short idxize(short num){ return (num == -1) ? LOC_NUM - 1: num;}
inline short deidxize(short idx){ return (idx == LOC_NUM - 1) ? -1 : idx;}
inline short remote(short loc, short other_loc){ return (loc == other_loc) ? 0 : 1;}

#endif
