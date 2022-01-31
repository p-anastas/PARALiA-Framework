///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The "Asset" related function implementations.
///

#include "Asset.hpp"
#include "unihelpers.hpp"

int Tile1D_num = 0;

template class Tile1D<double>;
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

template<typename dtype> void Asset1D<dtype>::InitTileMap(int T){
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
    Tile_map[current_ctr] = new Tile1D<dtype>(tile_addr, Ttmp, inc, itt);
  }
  #ifdef DEBUG
  	lprintf(lvl-1, "<-----|\n");
  #endif
}

template void Asset1D<double>::InitTileMap(int T);

template<typename dtype> void Asset1D<dtype>::DestroyTileMap(){
  int current_ctr;
  for (int itt = 0 ; itt < GridSz; itt++){
    current_ctr = itt;
    delete Tile_map[current_ctr];
    Tile1D_num--;
  }
  free(Tile_map);
}

template void Asset1D<double>::DestroyTileMap();

template<typename dtype>  Tile1D<dtype>::Tile1D(void * in_addr, int in_dim, int in_inc, int inGrid)
{
  short lvl = 3;

  #ifdef DEBUG
    lprintf(lvl-1, "|-----> Tile1D(%d)::Tile1D(in_addr(%d), %d, %d, %d)\n",
      Tile1D_num, CoCoGetPtrLoc(in_addr), in_dim, in_inc, inGrid);
  #endif

  dim = in_dim;
  GridId = inGrid;
  id = Tile1D_num;
  Tile1D_num++;
  short prev_loc = CoCoPeLiaGetDevice();
  for (int iloc = 0; iloc < LOC_NUM -1; iloc++){
    	CoCoPeLiaSelectDevice(iloc);
      available[iloc] = new Event();
      if(!iloc) available[LOC_NUM -1] = new Event(); // "Host" event is initialized in device 0
  }
  CoCoPeLiaSelectDevice(prev_loc);
  short init_loc = CoCoGetPtrLoc(in_addr);
  if (init_loc < 0) init_loc = LOC_NUM -1;
  for (int iloc = 0; iloc < LOC_NUM; iloc++){
    if (iloc == init_loc){
       adrs[iloc] = in_addr;
       inc[iloc] = in_inc;
       CacheLocId[iloc] = -1;
       //available[iloc]->record_to_queue(NULL);
    }
    else{
      adrs[iloc] = NULL;
      /// For column major format assumed = in_dim1, else in_dim2
      inc[iloc] = in_inc;
      CacheLocId[iloc] = -42;
    }
  }
  W_flag = R_flag = 0;
#ifdef ENABLE_MUTEX_LOCKING
	RW_lock.lock();
#else
  RW_lock = 1;
#endif
  RW_master = -42;
  #ifdef DEBUG
  	lprintf(lvl-1, "<-----|\n");
  #endif
}

template<typename dtype> short Tile1D<dtype>::getWriteBackLoc(){
  short pos = 0;
  state temp;
  for (pos =0; pos < LOC_NUM; pos++) if (CacheLocId[pos] == -1) break;
  if (pos >= LOC_NUM) error("Tile2D<dtype>::getWriteBackLoc: No initial location found for tile - bug.");
  else if (pos == LOC_NUM - 1) return -1;
  else return pos;
}

// TODO: to make this more sophisticated, we have to add prediction data in it.
// For now: closest = already in device, then other GPUs, then host (since devices in order etc, host after in CacheLocId)
template<typename dtype> short Tile1D<dtype>::getClosestReadLoc(short dev_id_in){
  short lvl = 5;
#ifdef DEBUG
  lprintf(lvl-1, "|-----> Tile1D(%d)::getClosestReadLoc(%d)\n", id, dev_id_in);
#endif
  short pos = 0;
  for (pos =0; pos < LOC_NUM; pos++){
    if (pos == dev_id_in) continue;
    if (CacheLocId[pos] == -1) break;
    else if (CacheLocId[pos] > -1){
      state temp = CoCacheUpdateBlockState(pos, CacheLocId[pos]);
      if (temp == AVAILABLE || temp == R) break;
#ifdef DDEBUG
  lprintf(lvl, "|-----> Tile1D(%d)::getClosestReadLoc(%d): Selecting cached tile in loc =%d \n", id, dev_id_in, pos);
#endif
    }
  }
  if (pos >= LOC_NUM) error("Tile1D(%d)::getClosestReadLoc(%d): No location found for tile - bug.", id, dev_id_in);
  else if (pos == LOC_NUM - 1) return -1;
  else return pos;
}
