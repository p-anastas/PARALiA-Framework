/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The header containing the "Asset" definition for data scheduling and management in heterogeneous multi-device systems.
///

#ifndef DECOM_H
#define DECOM_H

#include<iostream>
#include <string>
#include <mutex> // std::mutex

#include "Autotuner.hpp"
#include "linkmap.hpp"
#include <atomic>

#include "DataTile.hpp"

class Decom2D
{
	// Variables
	private:
	public:
	dtype_enum dtype;
	void *adrs;
	int id;
	char transpose;
	int GridSz1, GridSz2;
	int loc;
	//dtype *adrs;
	int ldim;
	int dim1, dim2;
	short pin_internally;
	Tile2D **Tile_map;

	// Constructor, sets dim1, dim2, ldim, adrs and derives loc from get_loc(adr)
	Decom2D(void* adrr, int in_dim1, int in_dim2, int in_ldim, char transpose, dtype_enum dtype_in);

	// General Functions
	void InitTileMap(int T1, int T2, Buffer_p* init_loc_cache_p);
	void WBTileMap();
	void SyncTileMap();
	void DestroyTileMap();
	Tile2D* getTile(int iloc1, int iloc2);
	int dtypesize() {
			if (dtype == DOUBLE) return sizeof(double);
			else if (dtype == FLOAT) return sizeof(float);
			else error("dtypesize: Unknown type");}
	int size() { return dtypesize()*dim1*dim2; }
	void DrawTileMap();

	// Backend Functions
	void prepareAsync(pthread_t* thread_id,
		pthread_attr_t attr);
	void resetProperties();
};


class Decom1D
{
	// Variables
	private:
	public:
	dtype_enum dtype;
	void *adrs;
	int id;
	int GridSz;
	int loc;
	//dtype *adrs;
	int dim;
	int inc;
	short pin_internally;
	Tile1D **Tile_map;

	// Constructor, sets dim1, dim2, ldim, adrs and derives loc from get_loc(adr)
	Decom1D(void* adrr, int in_dim, int in_inc, dtype_enum dtype_in);

	// General Functions
	void InitTileMap(int T, Buffer_p* init_loc_cache_p);
	void SyncTileMap();
	void DestroyTileMap();
	Tile1D* getTile(int iloc);
	int dtypesize() {
			if (dtype == DOUBLE) return sizeof(double);
			else if (dtype == FLOAT) return sizeof(float);
			else error("dtypesize: Unknown type");}
	int size() { return dtypesize()*dim; }
	void DrawTileMap();

	// Backend Functions
	void prepareAsync(pthread_t* thread_id,
		pthread_attr_t attr);
	void resetProperties();
};

#endif
