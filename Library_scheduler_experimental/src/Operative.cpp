///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The "Asset" related function implementations.
///

#include "Operative.hpp"
#include "unihelpers.hpp"

Operative::Operative(short TileNum_in){
	data_available = new Event();
	operation_complete = new Event();
	TileNum = TileNum_in;
	TileDimlist = (short*) malloc(TileNum*sizeof(short));
	TileDtypeList = (DTYPE_SUP*) malloc(TileNum*sizeof(DTYPE_SUP));
	TileList = (void**) malloc(TileNum*sizeof(void*));
}
