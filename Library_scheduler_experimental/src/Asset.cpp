///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The "Asset" related function implementations.
///

#include "Asset.hpp"
#include "unihelpers.hpp"

template class Tile2D<double>;
template class Asset2D<double>;

template<typename dtype> Tile2D<dtype>* Asset2D<dtype>::getTile(int iloc1, int iloc2){
  if(iloc1 >= GridSz1) error("%s::getTile : iloc1 >= GridSz1 (%d vs %d)\n", name.c_str(), iloc1, GridSz1);
  else if(iloc2 >= GridSz2) error("%s::getTile : iloc2 >= GridSz2 (%d vs %d)\n", name.c_str(), iloc2, GridSz2);
  return Tile_map[iloc1*GridSz2 + iloc2];
}

template<typename dtype> Asset2D<dtype>::Asset2D( void* in_adr, int in_dim1, int in_dim2, int in_ldim){
  ldim = in_ldim;
  dim1 = in_dim1;
  dim2 = in_dim2;
  adrs = (dtype*) in_adr;
  loc = CoCoGetPtrLoc(in_adr);
}

template<typename dtype> void Asset2D<dtype>::InitTileMap(int T1, int T2){
  short lvl = 2;

  #ifdef DEBUG
  	lprintf(lvl-1, "|-----> Asset2D<dtype>::InitTileMap(%zu,%zu)\n", T1, T2);
  #endif

  GridSz1 = dim1/T1;
	GridSz2 = dim2/T2;
  int T1Last = dim1%T1, T2Last = dim2%T2;
	if (T1Last > T1/4) GridSz1++;
	else T1Last+=T1;
  if (T2Last > T2/4) GridSz2++;
  else T2Last+=T2;

  Tile_map = (Tile2D<dtype>**) malloc(sizeof(Tile2D<dtype>*)*GridSz1*GridSz2);

  int current_ctr, T1tmp, T2tmp;
  void* tile_addr;
  for (int itt1 = 0; itt1 < GridSz1; itt1++){
    if ( itt1 == GridSz1 - 1) T1tmp = T1Last;
    else  T1tmp = T1;
		for (int itt2 = 0 ; itt2 < GridSz2; itt2++){
      if ( itt2 == GridSz2 - 1) T2tmp = T2Last;
      else  T2tmp = T2;
      current_ctr = itt1*GridSz2 + itt2;
      tile_addr = adrs + itt1*T1tmp + itt2*T2tmp*ldim;
      Tile_map[current_ctr] = new Tile2D<dtype>(tile_addr, T1tmp, T2tmp, ldim);
     }
   }
   #ifdef DEBUG
   	lprintf(lvl-1, "<-----|\n");
   #endif
}

template void Asset2D<double>::InitTileMap(int T1, int T2);

template<typename dtype>  Tile2D<dtype>::Tile2D(void * in_addr, int in_dim1, int in_dim2, int in_ldim)
{
  short lvl = 3;

  #ifdef DEBUG
    lprintf(lvl-1, "|-----> Tile2D<dtype>::Tile2D(in_addr,%d,%d,%d)\n", in_dim1, in_dim2, in_ldim);
  #endif

  dim1 = in_dim1;
  dim2 = in_dim2;
  short init_loc = CoCoGetPtrLoc(in_addr);
  if (init_loc < 0) init_loc = LOC_NUM -1;
  for (int iloc = 0; iloc < LOC_NUM; iloc++){
    if (iloc = init_loc){
       adrs[iloc] = in_addr;
       ldim[iloc] = in_ldim;
       cachemap[iloc] = MASTER;
    }
    else{
      adrs[iloc] = NULL;
      ldim[iloc] = in_dim2;
      cachemap[iloc] = INVALID;
    }
  }
  #ifdef DEBUG
  	lprintf(lvl-1, "<-----|\n");
  #endif
}
