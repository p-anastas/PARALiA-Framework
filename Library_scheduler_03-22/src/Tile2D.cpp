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

template<typename dtype>  Tile2D<dtype>::Tile2D(void * in_addr, int in_dim1, int in_dim2,
  int in_ldim, int inGrid1, int inGrid2, CBlock_p init_loc_block_p)
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
  short init_loc = CoCoGetPtrLoc(in_addr);
  short init_loc_idx = idxize(init_loc);
  WriteBackLoc = init_loc;
  for (int iloc = 0; iloc < LOC_NUM; iloc++){
    RunTileMap[iloc] = 0;
    if (iloc == init_loc_idx){
      WriteBackBlock = init_loc_block_p;
      StoreBlock[iloc] = init_loc_block_p;
      StoreBlock[iloc]->Adrs = in_addr;
      StoreBlock[iloc]->set_owner((void**)&StoreBlock[iloc]);
      StoreBlock[iloc]->set_state(EXCLUSIVE,false);
      ldim[iloc] = in_ldim;
      StoreBlock[iloc]->Available->record_to_queue(NULL);
    }
    else{
      StoreBlock[iloc] = NULL;
      ldim[iloc] = in_dim1;
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
  Tile2D_num--;
}

template<typename dtype> short Tile2D<dtype>::getWriteBackLoc(){
  return WriteBackLoc;
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
    if (pos == dev_id_in_idx || StoreBlock[pos] == NULL) {
      //if (StoreBlock[pos]!= NULL)
      //  error("Tile2D(%d)::getClosestReadLoc(%d): Should not be called, Tile already available in %d.\n",  id, dev_id_in, dev_id_in);
      continue;
    }
    StoreBlock[pos]->update_state(false);
    state temp = StoreBlock[pos]->State;
    if (temp == AVAILABLE || temp == SHARABLE || temp == EXCLUSIVE){
      if (link_cost[dev_id_in_idx][pos] < link_cost_min){
        link_cost_min = link_cost[dev_id_in_idx][pos];
        pos_min = pos;
      }
    }
  }
#ifdef DDEBUG
  lprintf(lvl, "|-----> Tile2D(%d)::getClosestReadLoc(%d): Selecting cached tile in loc =%d \n", id, dev_id_in, pos_min);
#endif
  if (pos_min >= LOC_NUM) error("Tile2D(%d)::getClosestReadLoc(%d): No location found for tile - bug.", id, dev_id_in);
  return deidxize(pos_min);
}

template<typename dtype> double Tile2D<dtype>::getMinLinkCost(short dev_id_in){
  short lvl = 5;
  int dev_id_in_idx = idxize(dev_id_in);
  double link_cost_min = 10000000;
  for (int pos =0; pos < LOC_NUM; pos++){
    StoreBlock[pos]->update_state(false);
    state temp = StoreBlock[pos]->State;
    if (temp == AVAILABLE || temp == SHARABLE || temp == EXCLUSIVE){
      if (link_cost[dev_id_in_idx][pos] < link_cost_min)
        link_cost_min = link_cost[dev_id_in_idx][pos];
    }
  }
  return link_cost_min;
}

void CoCoUpdateLinkSpeed2D(CoControl_p autotuned_vals, CoCoModel_p* glob_model){
  short lvl = 2;
  #ifdef DDEBUG
    lprintf(lvl, "|-----> CoCoUpdateLinkSpeed2D(dev_num = %d, LOC_NUM = %d)\n", autotuned_vals->dev_num, LOC_NUM);
  #endif
  for (int i = 0; i < autotuned_vals->dev_num; i++){
		short dev_id_idi = idxize(autotuned_vals->dev_ids[i]);
    for(int j = 0; j < LOC_NUM; j++){
      short dev_id_idj = idxize(j);
      if(dev_id_idi == dev_id_idj) link_cost[dev_id_idi][dev_id_idj] = 0;
      else link_cost[dev_id_idi][dev_id_idj] = t_com_predict(glob_model[dev_id_idi]->revlink[dev_id_idj], autotuned_vals->T*autotuned_vals->T*sizeof(VALUE_TYPE));
    }
    for(int j = 0; j < LOC_NUM; j++){
      short dev_id_idj = idxize(j);
      if(dev_id_idi == dev_id_idj) continue;
      int flag_normalize[LOC_NUM] = {0}, normalize_num = 1;
      double normalize_sum = link_cost[dev_id_idi][dev_id_idj];
      flag_normalize[j] = 1;
      for (int k = j + 1; k < LOC_NUM; k++)
        if(abs(link_cost[dev_id_idi][dev_id_idj] - link_cost[dev_id_idi][idxize(k)])
          /link_cost[dev_id_idi][dev_id_idj] < NORMALIZE_NEAR_SPLIT_LIMIT){
          flag_normalize[k] = 1;
          normalize_sum+=link_cost[dev_id_idi][idxize(k)];
          normalize_num++;
        }
      for (int k = j ; k < LOC_NUM; k++) if(flag_normalize[k]) link_cost[dev_id_idi][idxize(k)] = normalize_sum/normalize_num;
    }
  }
  #ifdef DEBUG
    lprintf(lvl-1, "<-----| CoCoUpdateLinkSpeed2D()\n");
  #endif
}
