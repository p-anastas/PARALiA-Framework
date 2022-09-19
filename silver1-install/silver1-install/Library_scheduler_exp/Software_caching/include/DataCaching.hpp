///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
/// \author Theodoridis Aristomenis (theodoridisaristomenis@gmail.com)
///
/// \brief The header containing the caching functions for data scheduling and management in heterogeneous multi-device systems.
///

#ifndef DATACACHNING_H
#define DATACACHNING_H

// Scheduling policy for block removal in cache.
 // "N": Naive
 // "F": FIFO
 // "M": MRU
 // "L": LRU

#if CACHE_SCHEDULING_POLICY=='N'
	#define NAIVE
#elif CACHE_SCHEDULING_POLICY=='F'
	#define FIFO
#elif CACHE_SCHEDULING_POLICY=='M'
	#define MRU
#elif CACHE_SCHEDULING_POLICY=='L'
	#define LRU
#endif

#include<iostream>
#include <string>
#include <mutex>
#include <atomic>

#include "unihelpers.hpp"

enum state{
	INVALID = 0, /// Cache Block is not valid.
	NATIVE = 1, /// Cache Block is native in memory, Should never be scheduled out or have its Adrs freed
	EXCLUSIVE = 2,  /// is being modified locally.
	SHARABLE = 3,  /// is available for sharing only but cannot be modified.
	AVAILABLE = 4, /// Is available with no operations performed or waiting on it.
};
const char* print_state(state in_state);

typedef class Cache* Cache_p;
typedef class CacheBlock* CBlock_p;

typedef struct Node_LL* Node_LL_p;
typedef class LinkedList* LinkedList_p;

/// A wrapper for additional info passed to DataCaching for a block,
/// in order to be able to write it back to native mem.
/// For 1D blocks dim1 = 1, dim2 = dim, ldim, ldim_wb = dim.
/// For 2D blocks currently ldim = dim1 always since they are cached as 1D blocks
typedef struct writeback_info{
		CBlock_p Native_block;
		int* WB_master_p;
		int dim1, dim2;
		int ldim, ldim_wb;
		int dtype_sz;
		CQueue_p wb_queue;
}* writeback_info_p;

// A class for each cache block.
typedef class CacheBlock{
	private:
	public:
		int id; // A unique per DevCache id for each block
		std::string Name; // Including it in all classes for potential debugging
		Cache_p Parent;		// Is this needed?
		void** Owner_p; // A pointer to the pointer that is used externally to associate with this block.
		writeback_info_p WritebackData_p;
		long long Size; // Included here but should be available at parent DevCache (?)

		// Current reads/writes + read/write requests waiting for access.
		std::atomic<int> PendingReaders, PendingWriters;  //Atomic for no lock
		// int PendingReaders, PendingWriters; if std::atomic becomes too annoying, must have block lock to change these
		void* Adrs;
		state State; // The (lazy) current state of the block. Must ALWAYS be <= the actual state
		Event_p Available;
		int Lock; // I like integers, but maybe consider using a more sophisticated/faster/friendly lock.
		// std::mutex Lock;
		//Constructor
		CacheBlock(int id, Cache_p Parent, long long Size);
		//Destructor
		~CacheBlock();

		// Functions
		void draw_block(bool lockfree=false);
		void allocate(bool lockfree=false);
		void add_reader(bool lockfree=false); // These might or might not be too much since DevCache will have to take part in calling them anyway.
		void add_writer(bool lockfree=false); // All add/remove should either use atomics or ensure the block is locked.
		void remove_reader(bool lockfree=false);
		void remove_writer(bool lockfree=false);
		void set_owner(void** owner_adrs, bool lockfree=false);
		void reset(bool lockfree=false, bool forceReset=false);  // Cleans a block to be given to someone else
		void init_writeback_info(CBlock_p WB_block, int* RW_master_p, int dim1, int dim2,
			int ldim, int ldim_wb, int dtype_sz, CQueue_p wb_queue, bool lockfree=false);
		void write_back(bool lockfree=false);
		state get_state();
		state set_state(state new_state, bool lockfree=false); // Return prev state
		int update_state(bool lockfree=false); // Force state check for Cblock, return 1 if state was changed, 0 if same old.
		void lock();
		void unlock();
		bool is_locked();

}* CBlock_p;

/// Device-wise software cache class declaration
typedef class Cache{
	private:
	public:
		int id; // A unique id per cache
		int dev_id; /// Pressumably this should be sufficient for current use cases instead of id, since we use only 1 cache/dev
		std::string Name; // Including it in all classes for potential debugging
		long long Size; // The sum of a cache's CBlock_sizes.
		int Lock; // I like integers, but maybe consider using a more sophisticated/faster/friendly lock.
		// std::mutex Lock;
		void* cont_buf_head; /// Used only if ENABLE_CACHE_CONTINUOUS_ALLOC

		int SerialCtr; // Number of blocks currently in cache.
		int BlockNum; // Number of Blocks the cache holds
		long long BlockSize; // Size allocated for each block - in reality it can hold less data
		CBlock_p* Blocks;

		Node_LL_p* Hash;
		LinkedList_p InvalidQueue; // Contains all invalid blocks.
		LinkedList_p Queue; // Contains a queue for blocks based on who came in first.

		//Constructor
		Cache(int dev_id, long long block_num, long long block_size);
		//Destructor
		~Cache();

		// Functions
		void draw_cache(bool print_blocks=true, bool print_queue=true, bool lockfree=false);
		void allocate(bool lockfree=false);
		void reset(bool lockfree=false, bool forceReset=false);
		CBlock_p assign_Cblock(state start_state=AVAILABLE, bool lockfree=false);

		void lock();
		void unlock();
		bool is_locked();

		double timer; // Keeps total time spend in cache operations-code

}* Cache_p;

typedef struct CBlock_wrap{
	CBlock_p CBlock;
	bool lockfree;
}* CBlock_wrap_p;

void* CBlock_RR_wrap(void* CBlock_wraped);

void* CBlock_RW_wrap(void* CBlock_wraped);

void* CBlock_INV_wrap(void* CBlock_wraped);

void* CBlock_RR_INV_wrap(void* CBlock_wraped);

void* CBlock_RW_INV_wrap(void* CBlock_wraped);

// Node for linked list.
typedef struct Node_LL{
	Node_LL_p next;
	Node_LL_p previous;
	int idx;
	bool valid;
}* Node_LL_p;

typedef class LinkedList{
private:
	Node_LL_p iter;
	Cache_p Parent;
public:
	std::string Name; // Including it in all classes for potential debugging
	Node_LL_p start;
	Node_LL_p end;
	int length;
	int lock_ll;
	// std::mutex lock_ll;

	// Constructor
	LinkedList(Cache_p cache, std::string name="LinkedList");
	// Destructor
	~LinkedList();

	// Functions
	void draw_queue(bool lockfree=false);
	void invalidate(Node_LL_p node, bool lockfree=false);
	void push_back(int idx, bool lockfree=false);
	Node_LL_p start_iterration(); // Queue has to be locked by user function
	Node_LL_p next_in_line();	// Queue has to be locked by user function
	Node_LL_p remove(Node_LL_p node, bool lockfree=false);
	void put_first(Node_LL_p node, bool lockfree=false);
	void put_last(Node_LL_p node, bool lockfree=false);
	void lock();
	void unlock();
	bool is_locked();
}* LinkedList_p;


int CacheSelectBlockToRemove_naive(Cache_p cache, bool lockfree=false);
int CacheSelectExclusiveBlockToRemove_naive(Cache_p cache, bool lockfree=false);
Node_LL* CacheSelectBlockToRemove_fifo_mru_lru(Cache_p cache, bool lockfree=false);
Node_LL* CacheSelectExclusiveBlockToRemove_fifo_mru_lru(Cache_p cache, bool lockfree=false);
// Node_LL* CacheSelectBlockToRemove_mru_lru(Cache_p cache, bool lockfree=false);

extern Cache_p Global_Cache[LOC_NUM];
#endif
