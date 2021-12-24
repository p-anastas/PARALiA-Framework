///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The "Asset" related function implementations.
///

#include "Asset.hpp"
#include "unihelpers.hpp"

int Tile_num = 0;

template class Tile2D<double>;
template class Asset2D<double>;

template<typename dtype> Tile2D<dtype>* Asset2D<dtype>::getTile(int iloc1, int iloc2){
  if(iloc1 >= GridSz1) error("Asset2D::getTile : iloc1 >= GridSz1 (%d vs %d)\n", iloc1, GridSz1);
  else if(iloc2 >= GridSz2) error("Asset2D::getTile : iloc2 >= GridSz2 (%d vs %d)\n", iloc2, GridSz2);
  return Tile_map[iloc1*GridSz2 + iloc2];
}

template<typename dtype> Asset2D<dtype>::Asset2D( void* in_adr, int in_dim1, int in_dim2, int in_ldim, char in_transpose){
  ldim = in_ldim;
  dim1 = in_dim1;
  dim2 = in_dim2;
  adrs = (dtype*) in_adr;
  loc = CoCoGetPtrLoc(in_adr);
  transpose = in_transpose;
}

template<typename dtype> void Asset2D<dtype>::InitTileMap(int T1, int T2){
  short lvl = 2;

  #ifdef DEBUG
  	lprintf(lvl-1, "|-----> Asset2D<dtype>::InitTileMap(%d,%d)\n", T1, T2);
  #endif

  GridSz1 = dim1/T1;
	GridSz2 = dim2/T2;
  int T1Last = dim1%T1, T2Last = dim2%T2;
  // TODO: Padding instead of resize so all data fit in buffer without complex mechanism.
  // Can degrade performance for small div sizes.
  if (T1Last > 0) GridSz1++;
  else T1Last=T1;
  if (T2Last > 0) GridSz2++;
  else T2Last=T2;
	//if (T1Last > T1/4) GridSz1++;
	//else T1Last+=T1;
  //if (T2Last > T2/4) GridSz2++;
  //else T2Last+=T2;

  Tile_map = (Tile2D<dtype>**) malloc(sizeof(Tile2D<dtype>*)*GridSz1*GridSz2);

  int current_ctr, T1tmp, T2tmp;
  void* tile_addr = NULL;
  for (int itt1 = 0; itt1 < GridSz1; itt1++){
    if ( itt1 == GridSz1 - 1) T1tmp = T1Last;
    else  T1tmp = T1;
		for (int itt2 = 0 ; itt2 < GridSz2; itt2++){
      if ( itt2 == GridSz2 - 1) T2tmp = T2Last;
      else  T2tmp = T2;
      current_ctr = itt1*GridSz2 + itt2;
      /// For column major format assumed with T1tmp = rows and T2tmp = cols
      if (transpose == 'N'){
         tile_addr = adrs + itt1*T1 + itt2*T2*ldim;
         Tile_map[current_ctr] = new Tile2D<dtype>(tile_addr, T1tmp, T2tmp, ldim, itt1, itt2);
       }
      else if (transpose == 'T'){
        tile_addr = adrs + itt1*T1*ldim + itt2*T2;
        Tile_map[current_ctr] = new Tile2D<dtype>(tile_addr, T2tmp, T1tmp, ldim, itt2, itt1);
      }
      else error("Asset2D<dtype>::InitTileMap: Unknown transpose type\n");

     }
   }
   #ifdef DEBUG
   	lprintf(lvl-1, "<-----|\n");
   #endif
}

template void Asset2D<double>::InitTileMap(int T1, int T2);

template<typename dtype> void Asset2D<dtype>::DestroyTileMap(){
  int current_ctr;
  for (int itt1 = 0; itt1 < GridSz1; itt1++)
    for (int itt2 = 0 ; itt2 < GridSz2; itt2++){
      current_ctr = itt1*GridSz2 + itt2;
      delete Tile_map[current_ctr];
      Tile_num--;
    }
  free(Tile_map);
}

template void Asset2D<double>::DestroyTileMap();

template<typename dtype>  Tile2D<dtype>::Tile2D(void * in_addr, int in_dim1, int in_dim2, int in_ldim, int inGrid1, int inGrid2)
{
  short lvl = 3;

  #ifdef DEBUG
    lprintf(lvl-1, "|-----> Tile2D(%d)::Tile2D(in_addr(%d),%d,%d,%d, %d, %d)\n",
      Tile_num, CoCoGetPtrLoc(in_addr), in_dim1, in_dim2, in_ldim, inGrid1, inGrid2);
  #endif

  dim1 = in_dim1;
  dim2 = in_dim2;
  GridId1 = inGrid1;
  GridId2 = inGrid2;
  id = Tile_num;
  Tile_num++;
  short prev_loc = CoCoPeLiaGetDevice();
  for (int iloc = 0; iloc < LOC_NUM -1; iloc++){
    	CoCoPeLiaSelectDevice(iloc);
      available[iloc] = new Event();
      if(!iloc) available[LOC_NUM -1] = new Event();
  }
  CoCoPeLiaSelectDevice(prev_loc);
  short init_loc = CoCoGetPtrLoc(in_addr);
  if (init_loc < 0) init_loc = LOC_NUM -1;
  for (int iloc = 0; iloc < LOC_NUM; iloc++){
    //PendingUsage[iloc] = 0;
    if (iloc == init_loc){
       adrs[iloc] = in_addr;
       ldim[iloc] = in_ldim;
       CacheLocId[iloc] = -1;
       //available[iloc]->record_to_queue(NULL); FIXME: this might result in cache bugs
    }
    else{
      adrs[iloc] = NULL;
      /// For column major format assumed = in_dim1, else in_dim2
      ldim[iloc] = in_dim1;
      CacheLocId[iloc] = -42;
    }
  }
  W_flag = R_flag = 0;
  #ifdef DEBUG
  	lprintf(lvl-1, "<-----|\n");
  #endif
}

template<typename dtype> short Tile2D<dtype>::getWriteBackLoc(){
  short pos = 0;
  state temp;
  for (pos =0; pos < LOC_NUM; pos++) if (CacheLocId[pos] == -1) break;
  if (pos >= LOC_NUM) error("Tile2D<dtype>::getWriteBackLoc: No initial location found for tile - bug.");
  else if (pos == LOC_NUM - 1) return -1;
  else return pos;
}

// TODO: to make this more sophisticated, we have to add prediction data in it.
// For now: closest = already in device, then other GPUs, then host (since devices in order etc, host after in CacheLocId)
template<typename dtype> short Tile2D<dtype>::getClosestReadLoc(short dev_id_in){
  short lvl = 5;
#ifdef DEBUG
  lprintf(lvl-1, "|-----> Tile2D(%d)::getClosestReadLoc(%d)\n", id, dev_id_in);
#endif
  short pos = 0;
  for (pos =0; pos < LOC_NUM; pos++){
    if (pos == dev_id_in) continue;
    if (CacheLocId[pos] == -1) break;
    else if (CacheLocId[pos] > -1){
      state temp = CoCacheUpdateBlockState(pos, CacheLocId[pos]);
      if (temp == AVAILABLE || temp == R) break;
#ifdef DDEBUG
  lprintf(lvl, "|-----> Tile2D(%d)::getClosestReadLoc(%d): Selecting cached tile in loc =%d \n", id, dev_id_in, pos);
#endif
    }
  }
  if (pos >= LOC_NUM) error("Tile2D(%d)::getClosestReadLoc(%d): No location found for tile - bug.", id, dev_id_in);
  else if (pos == LOC_NUM - 1) return -1;
  else return pos;
}
