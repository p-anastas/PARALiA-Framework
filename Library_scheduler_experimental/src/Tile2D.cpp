///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The "Asset" related function implementations.
///

#include "Asset.hpp"
#include "unihelpers.hpp"

int Tile2D_num = 0;

double link_cost[LOC_NUM][LOC_NUM];

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
  for (pos =0; pos < LOC_NUM; pos++) if (CacheLocId[pos] == -1) break;
  if (pos >= LOC_NUM) error("Tile2D<dtype>::getWriteBackLoc: No initial location found for tile - bug.");
  else if (pos == LOC_NUM - 1) return -1;
  else return pos;
}

template<typename dtype> short Tile2D<dtype>::isLocked(){
#ifdef ENABLE_MUTEX_LOCKING
		error("Not sure how to do this correctly\n");
    return 0;
#else
		if(RW_lock) return 1;
    else return 0;
#endif
}

template<typename dtype> short Tile2D<dtype>::getClosestReadLoc(short dev_id_in){
  short lvl = 5;
#ifdef DEBUG
  lprintf(lvl-1, "|-----> Tile2D(%d)::getClosestReadLoc(%d)\n", id, dev_id_in);
#endif
  int dev_id_in_idx = idxize(dev_id_in);
  int pos_min = LOC_NUM;
  double link_cost_min = 10000000;
  for (int pos =0; pos < LOC_NUM; pos++){
    if (pos == dev_id_in_idx) continue;
    if (CacheLocId[pos] == -1){
      if (link_cost[dev_id_in_idx][pos] < link_cost_min){
        link_cost_min = link_cost[dev_id_in_idx][pos];
        pos_min = pos;
      }
    }
    else if (CacheLocId[pos] > -1){
      state temp = CacheGetBlockStateNoLock(pos, CacheLocId[pos]);
      if (temp == AVAILABLE || temp == R){
        if (link_cost[dev_id_in_idx][pos] < link_cost_min){
          link_cost_min = link_cost[dev_id_in_idx][pos];
          pos_min = pos;
        }
      }
    }
  }
#ifdef DDEBUG
  lprintf(lvl, "|-----> Tile2D(%d)::getClosestReadLoc(%d): Selecting cached tile in loc =%d \n", id, dev_id_in, pos_min);
#endif
  if (pos_min >= LOC_NUM) error("Tile2D(%d)::getClosestReadLoc(%d): No location found for tile - bug.", id, dev_id_in);
  else return deidxize(pos_min);
}

template<typename dtype> double Tile2D<dtype>::getMinLinkCost(short dev_id_in){
  short lvl = 5;
#ifdef DEBUG
  lprintf(lvl-1, "|-----> Tile2D(%d)::getMinLinkCost(%d)\n", id, dev_id_in);
#endif
  int dev_id_in_idx = idxize(dev_id_in);
  double link_cost_min = 10000000;
  for (int pos =0; pos < LOC_NUM; pos++){
    if (CacheLocId[pos] == -1){
      if (link_cost[dev_id_in_idx][pos] < link_cost_min)
        link_cost_min = link_cost[dev_id_in_idx][pos];
    }
    else if (CacheLocId[pos] > -1){
      state temp = CacheGetBlockStateNoLock(pos, CacheLocId[pos]);
      if (temp == AVAILABLE || temp == R || temp == EXCLUSIVE){
        if (link_cost[dev_id_in_idx][pos] < link_cost_min)
          link_cost_min = link_cost[dev_id_in_idx][pos];
      }
    }
  }
  return link_cost_min;
}

void CoCoUpdateLinkSpeed2D(CoControl_p autotuned_vals, CoCoModel_p* glob_model){
  for (int i = 0; i < autotuned_vals->dev_num; i++){
		short dev_id_idi = idxize(autotuned_vals->dev_ids[i]);
    for(int j = 0; j < LOC_NUM; j++){
      short dev_id_idj = idxize(j);
      if(dev_id_idi == dev_id_idj) link_cost[dev_id_idi][dev_id_idj] = 0;
      else link_cost[dev_id_idi][dev_id_idj] = t_com_predict(glob_model[dev_id_idi]->revlink[dev_id_idj], autotuned_vals->T*autotuned_vals->T*sizeof(VALUE_TYPE));
    }
  }
}
