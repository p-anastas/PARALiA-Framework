///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The "Asset" related function implementations.
///

#include "Asset.hpp"
#include "unihelpers.hpp"

int Tile2D_num = 0;

template class Tile2D<double>;

template<typename dtype>  Tile2D<dtype>::Tile2D(void * in_addr, int in_dim1, int in_dim2, int in_ldim, int inGrid1, int inGrid2)
{
  short lvl = 3;

  #ifdef DEBUG
    lprintf(lvl-1, "|-----> Tile2D(%d)::Tile2D(in_addr(%d),%d,%d,%d, %d, %d)\n",
      Tile2D_num, CoCoGetPtrLoc(in_addr), in_dim1, in_dim2, in_ldim, inGrid1, inGrid2);
  #endif

  dim1 = in_dim1;
  dim2 = in_dim2;
  GridId1 = inGrid1;
  GridId2 = inGrid2;
  id = Tile2D_num;
  Tile2D_num++;
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
    RunTileMap[iloc] = 0;
    if (iloc == init_loc){
       adrs[iloc] = in_addr;
       ldim[iloc] = in_ldim;
       CacheLocId[iloc] = -1;
       //available[iloc]->record_to_queue(NULL);
    }
    else{
      adrs[iloc] = NULL;
      /// For column major format assumed = in_dim1, else in_dim2
      ldim[iloc] = in_dim1;
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

template<typename dtype>  Tile2D<dtype>::~Tile2D()
{
  short lvl = 3;
#ifdef DEBUG
  lprintf(lvl-1, "|-----> Tile2D(%d)::~Tile2D()\n", Tile2D_num);
#endif
  Tile2D_num--;
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
