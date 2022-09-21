/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The header containing the "Asset" definition for data scheduling and management in heterogeneous multi-device systems.
///

#ifndef ASSET_H
#define ASSET_H

#include<iostream>
#include <string>
#include <mutex> // std::mutex

#include "Autotuning_runtime.hpp"
#include "DataCaching.hpp"
#include <atomic>

template <typename dtype>
class Tile1D
{
		// Variables
	public:
		int WriteBackLoc;
		int id, GridId;
		int dim;
		int R_flag, W_flag, W_total, RW_master;
		int RW_lock;
		std::atomic<int> RW_lock_holders;
//#ifdef ENABLE_PARALLEL_BACKEND
		int RW_Master_backend_ctr;
//#endif

		CBlock_p WriteBackBlock;
		CBlock_p StoreBlock[LOC_NUM];

		int inc[LOC_NUM];
		int RunTileMap[LOC_NUM];

		// Constructor
		Tile1D<dtype>(void* tile_addr, int T1tmp,
			int inc, int inGrid1, CBlock_p init_loc_block_p);

		//Destructor
		~Tile1D();

    // General Functions
		int dtypesize() { return sizeof(dtype); }
		int size() { return dtypesize()*dim; }
		short getWriteBackLoc();
		short getClosestReadLoc(short dev_id_in);
		double getMinLinkCost(short dev_id_in);
		double getMinLinkCostPenaltizeFetch(short dev_id_in);
		short isLocked();

};

template <typename dtype>
class Tile2D
{
		// Variables
	public:
		int WriteBackLoc;
		int id, GridId1, GridId2;
		int dim1, dim2;
		int R_flag, W_flag, W_total, RW_master;
		int RW_lock;
		std::atomic<int> RW_lock_holders;
//#ifdef ENABLE_PARALLEL_BACKEND
		int RW_Master_backend_ctr;
//#endif

		CBlock_p WriteBackBlock;
		CBlock_p StoreBlock[LOC_NUM];

		int ldim[LOC_NUM];
		int RunTileMap[LOC_NUM];

		// Constructor
		Tile2D<dtype>(void* tile_addr, int T1tmp, int T2tmp,
			int ldim, int inGrid1, int inGrid2, CBlock_p init_loc_block_p);

		//Destructor
		~Tile2D();

    // General Functions
		int dtypesize() { return sizeof(dtype); }
		int size() { return dtypesize()*dim1*dim2; }
		short getWriteBackLoc();
		short getClosestReadLoc(short dev_id_in);
		double getMinLinkCost(short dev_id_in);
		double getMinLinkCostPenaltizeFetch(short dev_id_in);
		short isLocked();

};

template <typename dtype>
class Asset2D
{
	// Variables
	private:
	public:
	int id;
	char transpose;
	int GridSz1, GridSz2;
	int loc;
	dtype *adrs;
	int ldim;
	int dim1, dim2;
	short pin_internally;
	Tile2D<dtype> **Tile_map;

	// Constructor, sets dim1, dim2, ldim, adrs and derives loc from get_loc(adr)
	Asset2D<dtype>(void* adrr, int in_dim1,
		int in_dim2, int in_ldim, char transpose);

	// General Functions
	void InitTileMap(int T1, int T2, Cache_p* init_loc_cache_p);
	void DestroyTileMap();
	Tile2D<dtype>* getTile(int iloc1, int iloc2);
	int dtypesize() { return sizeof(dtype); }
	int size() { return dtypesize()*dim1*dim2; }
	void DrawTileMap();

	// Backend Functions
	void prepareAsync(pthread_t* thread_id,
		pthread_attr_t attr);
	void resetProperties();
};

template <typename dtype>
class Asset1D
{
	// Variables
	private:
	public:
	int id;
	int GridSz;
	int loc;
	dtype *adrs;
	int dim;
	int inc;
	short pin_internally;
	Tile1D<dtype> **Tile_map;

	// Constructor, sets dim1, dim2, ldim, adrs and derives loc from get_loc(adr)
	Asset1D<dtype>(void* adrr, int in_dim, int in_inc);

	// General Functions
	void InitTileMap(int T, Cache_p* init_loc_cache_p);
	void DestroyTileMap();
	Tile1D<dtype>* getTile(int iloc);
	int dtypesize() { return sizeof(dtype); }
	int size() { return dtypesize()*dim; }
	void DrawTileMap();

	// Backend Functions
	void prepareAsync(pthread_t* thread_id,
		pthread_attr_t attr);
	void resetProperties();
};

#endif
