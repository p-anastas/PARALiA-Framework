///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The "Asset" related function implementations.
///

#include "Asset.hpp"
#include "unihelpers.hpp"

int Tile1D_num = 0;

template class Tile1D<double>;

template<typename dtype>  Tile1D<dtype>::Tile1D(void * in_addr, int in_dim,
  int in_inc, int inGrid, CBlock_p init_loc_block_p)
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
  short init_loc = CoCoGetPtrLoc(in_addr);
  short init_loc_idx = idxize(init_loc);
  WriteBackLoc = init_loc;
  for (int iloc = 0; iloc < LOC_NUM; iloc++){
    RunTileMap[iloc] = 0;
    if (iloc == init_loc_idx){
      StoreBlock[iloc] = init_loc_block_p;
      StoreBlock[iloc]->Adrs = in_addr;
      StoreBlock[iloc]->set_owner((void**)&StoreBlock[iloc]);
      inc[iloc] = in_inc;
      StoreBlock[iloc]->Available->record_to_queue(NULL);
    }
    else{
      StoreBlock[iloc] = NULL;
      inc[iloc] = in_inc;
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
  return WriteBackLoc;
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
    if (idxize(pos) == dev_id_in){
      if (StoreBlock[pos]!= NULL)
        error("Tile1D(%d)::getClosestReadLoc(%d): Should not be called, Tile already available in %d.\n",  id, dev_id_in, dev_id_in);
      continue;
    }
    if (StoreBlock[pos]!= NULL){
      state temp = StoreBlock[pos]->State;
      if (temp == AVAILABLE || temp == SHARABLE) break;
#ifdef DDEBUG
  lprintf(lvl, "|-----> Tile1D(%d)::getClosestReadLoc(%d): Selecting cached tile in loc =%d \n", id, dev_id_in, pos);
#endif
    }
  }
  if (pos >= LOC_NUM) error("Tile1D(%d)::getClosestReadLoc(%d): No location found for tile - bug.", id, dev_id_in);
  return idxize(pos);
}
