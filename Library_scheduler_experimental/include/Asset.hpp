/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The header containing the "Asset" definition for data scheduling and management in heterogeneous multi-device systems.
///

#ifndef ASSET_H
#define ASSET_H

#include<iostream>
#include <string>
#include <mutex> // std::mutex

#define LOC_NUM (DEV_NUM + 1)

#include "DataCaching.hpp"

// TODO: Not used yet (BLAS 3 only use Asset2D)
template <typename dtype>
class Asset1D
{
	private:
	public:
		int loc;
		dtype *adrs;
		int dim;
		std::string name;

		int dsize() { return sizeof(dtype); }

};

template <typename dtype>
class Tile2D
{
		// Variables
	private:
	public:
		int id, GridId1, GridId2;
		int dim1, dim2;
		int R_flag, W_flag, RW_master;
#ifdef ENABLE_MUTEX_LOCKING
		std::mutex RW_lock;
#else
		int RW_lock;
#endif

		Event* available[LOC_NUM];
		void *adrs[LOC_NUM];
		int ldim[LOC_NUM];
		int CacheLocId[LOC_NUM];
		//int PendingUsage[LOC_NUM];

		// Constructor
		Tile2D<dtype>(void* tile_addr, int T1tmp, int T2tmp, int ldim, int inGrid1, int inGrid2);

    // General Functions
		int dtypesize() { return sizeof(dtype); }
		int size() { return dtypesize()*dim1*dim2; }
		short getWriteBackLoc();
		short getClosestReadLoc(short dev_id_in);
		void Set_RW_lock(void* wrapped_value);

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
	Asset2D<dtype>(void* adrr, int in_dim1, int in_dim2, int in_ldim, char transpose);

	// General Functions
	void InitTileMap(int T1, int T2);
	void DestroyTileMap();
	Tile2D<dtype>* getTile(int iloc1, int iloc2);
	int dtypesize() { return sizeof(dtype); }
	int size() { return dtypesize()*dim1*dim2; }

	// Backend Functions
	void* prepareAsync(pthread_t* thread_id, pthread_attr_t attr);
	void resetProperties();
};

#endif
