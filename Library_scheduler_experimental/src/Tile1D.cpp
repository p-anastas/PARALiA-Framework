///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The "Asset" related function implementations.
///

#include "Asset.hpp"
#include "unihelpers.hpp"

int Tile1D_num = 0;

template class Tile1D<double>;

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
  for (int iloc = 0; iloc < LOC_NUM; iloc++){
    	CoCoPeLiaSelectDevice(deidxize(iloc));
      available[iloc] = new Event(deidxize(iloc));
  }
  CoCoPeLiaSelectDevice(prev_loc);
  short init_loc = CoCoGetPtrLoc(in_addr);
  short init_loc_idx = idxize(init_loc);
  for (int iloc = 0; iloc < LOC_NUM; iloc++){
    RunTileMap[iloc] = 0;
    if (iloc == init_loc_idx){
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
  W_flag = R_flag = W_total = 0;
#ifdef ENABLE_MUTEX_LOCKING
	//RW_lock.lock();
#else
  RW_lock = 0;
#endif
  RW_master = init_loc;
  #ifdef DEBUG
  	lprintf(lvl-1, "<-----|\n");
  #endif
}

template<typename dtype>  Tile1D<dtype>::~Tile1D()
{
  short lvl = 3;
#ifdef DEBUG
  lprintf(lvl-1, "|-----> Tile1D(%d)::~Tile1D()\n", Tile1D_num);
#endif
  Tile1D_num--;
}

template<typename dtype> short Tile1D<dtype>::getWriteBackLoc(){
  short pos = 0;
  state temp;
  for (pos =0; pos < LOC_NUM; pos++) if (CacheLocId[pos] == -1) break;
  if (pos >= LOC_NUM) error("Tile2D<dtype>::getWriteBackLoc: No initial location found for tile - bug.");
  else if (pos == LOC_NUM - 1) return -1;
  else return pos;
}

template<typename dtype> short Tile1D<dtype>::isLocked(){
#ifdef ENABLE_MUTEX_LOCKING
		error("Not sure how to do this correctly\n");
    return 0;
#else
		if(RW_lock) return 1;
    else return 0;
#endif
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
      state temp = CacheGetBlockState(pos, CacheLocId[pos]);
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
