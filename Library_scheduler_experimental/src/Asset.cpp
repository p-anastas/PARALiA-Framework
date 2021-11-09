///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The "Asset" related function implementations. 
///

#include "Asset.hpp"
#include "unihelpers.hpp"

    	
template class Datum2D<double>;
template class Asset2D<double>;

template<typename dtype> Datum2D<dtype> Asset2D<dtype>::getTile(int iloc1, int iloc2){
    if(iloc1 >= DatumNum1) error("%s::getTile : iloc1 >= DatumNum1 (%d vs %d)\n", name.c_str(), iloc1, DatumNum1);
    else if(iloc2 >= DatumNum2) error("%s::getTile : iloc2 >= DatumNum2 (%d vs %d)\n", name.c_str(), iloc2, DatumNum2);
    return Tile_map[iloc1*DatumNum2 + iloc2];
}

template<typename dtype> Asset2D<dtype>::Asset2D( void* in_adr, int in_dim1, int in_dim2, int in_ldim){
    ldim = in_ldim;
    dim1 = in_dim1;
    dim2 = in_dim2;
    adrs = (dtype*) in_adr;
    loc = CoCoGetPtrLoc(in_adr);
}

