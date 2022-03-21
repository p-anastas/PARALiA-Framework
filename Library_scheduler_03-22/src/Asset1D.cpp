///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The "Asset" related function implementations.
///

#include "Asset.hpp"
#include "unihelpers.hpp"

template class Asset1D<double>;

template<typename dtype> Tile1D<dtype>* Asset1D<dtype>::getTile(int iloc){
  if(iloc >= GridSz) error("Asset1D::getTile : iloc >= GridSz (%d vs %d)\n", iloc, GridSz);
  return Tile_map[iloc];
}

template<typename dtype> Asset1D<dtype>::Asset1D(void* in_adr, int in_dim, int in_inc){
  dim = in_dim;
  adrs = (dtype*) in_adr;
  loc = CoCoGetPtrLoc(in_adr);
  inc = in_inc;
}

template<typename dtype> void Asset1D<dtype>::InitTileMap(int T, Cache_p init_loc_cache_p){
  short lvl = 2;

  #ifdef DEBUG
  	lprintf(lvl-1, "|-----> Asset1D<dtype>::InitTileMap(%d)\n", T);
  #endif

  GridSz = dim/T;
  int TLast = dim%T;
  // TODO: Padding instead of resize so all data fit in buffer without complex mechanism.
  // Can degrade performance for small div sizes.
  if (TLast > 0) GridSz++;
  else TLast=T;

  Tile_map = (Tile1D<dtype>**) malloc(sizeof(Tile2D<dtype>*)*GridSz);

  int current_ctr, Ttmp;
  void* tile_addr = NULL;
  for (int itt = 0; itt < GridSz; itt++){
    if ( itt == GridSz - 1) Ttmp = TLast;
    else  Ttmp = T;
    current_ctr = itt;
    tile_addr = adrs + itt*T*inc;
    Tile_map[current_ctr] = new Tile1D<dtype>(tile_addr, Ttmp, inc, itt, init_loc_cache_p->assign_Cblock());
  }
  #ifdef DEBUG
  	lprintf(lvl-1, "<-----|\n");
  #endif
}

template void Asset1D<double>::InitTileMap(int T, Cache_p init_loc_cache_p);

template<typename dtype> void Asset1D<dtype>::DestroyTileMap(){
  int current_ctr;
  for (int itt = 0 ; itt < GridSz; itt++){
    current_ctr = itt;
    delete Tile_map[current_ctr];
  }
  free(Tile_map);
}

template void Asset1D<double>::DestroyTileMap();
