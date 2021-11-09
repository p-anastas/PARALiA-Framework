/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The header containing the "Asset" definition for data scheduling and management in heterogeneous multi-device systems. 
///

#ifndef ASSET_H
#define ASSET_H

#include<iostream>
#include <string>

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
		void print() { std::cout << "Asset1D : " << name; }

};

enum state{
	INVALID = 0,
	REQUESTED = 1,
	AVAILABLE = 2,
	WRITING = 3
};

template <typename dtype>
class Datum2D
{
		// Variables
	private:
	public:
		std::string name;
		int dim1, dim2;

		dtype *adrs[DEV_NUM];
		int ldim[DEV_NUM];
		state cachemap[DEV_NUM]; 

        // Functions		
		int dtypesize() { return sizeof(dtype); }
		int size() { return dtypesize()*dim1*dim2; }
		void print() { std::cout << "Asset2D : " << name; }

};

template <typename dtype>
class Asset2D
{
		// Variables
    private:
		int DatumNum1, DatumNum2;
    public:
		int loc;
		dtype *adrs;
		int ldim;
		int dim1, dim2;
		Datum2D<dtype> *Tile_map;
    	std::string name;
    	
		// Constructor, sets dim1, dim2, ldim, adrs and derives loc from get_loc(adr)
        Asset2D<dtype>(void* adrr, int in_dim1, int in_dim2, int in_ldim);
           
        // Functions
    	Datum2D<dtype> getTile(int iloc1, int iloc2);
    	int dtypesize() { return sizeof(dtype); }
    	int size() { return dtypesize()*dim1*dim2; }
    	void print() { std::cout << "Asset2D : " << name; }

};

#endif
