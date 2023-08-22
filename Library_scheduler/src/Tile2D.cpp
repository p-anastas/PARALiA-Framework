///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The "Asset" related function implementations.
///

#include "Decomposer.hpp"
#include "linkmap.hpp"

int Tile2D_num = 0;

Tile2D::Tile2D(void * in_addr, int in_dim1, int in_dim2,
  int in_ldim, int inGrid1, int inGrid2, dtype_enum dtype_in, CBlock_p init_loc_block_p)
{
  short lvl = 3;

  #ifdef DDEBUG
    lprintf(lvl-1, "|-----> Tile2D(%d)::Tile2D(in_addr(%d),%d,%d,%d, %d, %d)\n",
      Tile2D_num, CoCoGetPtrLoc(in_addr), in_dim1, in_dim2, in_ldim, inGrid1, inGrid2);
  #endif
  dtype = dtype_in;
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
      StoreBlock[iloc]->set_owner((void**)&StoreBlock[iloc],false);
      ldim[iloc] = in_ldim;
      StoreBlock[iloc]->Available->record_to_queue(NULL);
    }
    else{
      StoreBlock[iloc] = NULL;
      ldim[iloc] = in_dim1;
    }
  }
  W_flag = R_flag = W_total = 0;

  RW_lock = -42;
  RW_lock_holders = 0;

  RW_master = init_loc;
  #ifdef DDEBUG
  	lprintf(lvl-1, "<-----|\n");
  #endif
}

Tile2D::~Tile2D()
{
  short lvl = 3;
  Tile2D_num--;
}

short Tile2D::getWriteBackLoc(){
  return WriteBackLoc;
}

short Tile2D::isLocked(){
#ifdef ENABLE_MUTEX_LOCKING
		error("Not sure how to do this correctly\n");
    return 0;
#else
		if(RW_lock) return 1;
    else return 0;
#endif
}

short Tile2D::getClosestReadLoc(short dev_id_in){
  short lvl = 5;
#ifdef DDEBUG
  lprintf(lvl-1, "|-----> Tile2D(%d)::getClosestReadLoc(%d)\n", id, dev_id_in);
#endif
  int dev_id_in_idx = idxize(dev_id_in);
  int pos_max = LOC_NUM;
  double link_bw_max = 0;
  for (int pos =0; pos < LOC_NUM; pos++){
    if (pos == dev_id_in_idx || StoreBlock[pos] == NULL) {
      //if (StoreBlock[pos]!= NULL)
      //  error("Tile2D(%d)::getClosestReadLoc(%d): Should not be called, Tile already available in %d.\n",  id, dev_id_in, dev_id_in);
      continue;
    }
    //StoreBlock[pos]->update_state(false);
    state temp = StoreBlock[pos]->State;
    if (temp == AVAILABLE || temp == SHARABLE || temp == NATIVE){
      event_status block_status = StoreBlock[pos]->Available->query_status();
#ifdef ALLOW_FETCH_RECORDED
      if(block_status == COMPLETE || block_status == CHECKED || block_status == RECORDED){
#else
      if(block_status == COMPLETE || block_status == CHECKED){
#endif
        double current_link_bw = final_estimated_link_bw[dev_id_in_idx][pos];
        if (block_status == RECORDED) current_link_bw-=current_link_bw*FETCH_UNAVAILABLE_PENALTY;
        if (current_link_bw > link_bw_max){
          link_bw_max = current_link_bw;
          pos_max = pos;
        }
        else if (current_link_bw == link_bw_max &&
        final_estimated_linkmap->link_uses[dev_id_in_idx][pos] < final_estimated_linkmap->link_uses[dev_id_in_idx][pos_max]){
          link_bw_max = current_link_bw;
          pos_max = pos;
        }
      }
    }
  }
#ifdef DEBUG
  lprintf(lvl, "|-----> Tile2D(%d)::getClosestReadLoc(%d): Selecting cached tile in loc =%d \n", id, dev_id_in, pos_max);
#endif
  if (pos_max >= LOC_NUM) error("Tile2D(%d)::getClosestReadLoc(%d): No location found for tile - bug.", id, dev_id_in);
  //Global_Cache[pos_max]->lock();
  CBlock_p temp_outblock = StoreBlock[pos_max];
  if(temp_outblock != NULL){
    temp_outblock->lock();
    //temp_outblock->update_state(true);
    state temp = temp_outblock->State;
    event_status block_status = temp_outblock->Available->query_status();
    if ((temp == AVAILABLE || temp == SHARABLE || temp == NATIVE) &&
#ifdef ALLOW_FETCH_RECORDED
    (block_status == COMPLETE || block_status == CHECKED|| block_status == RECORDED)){
#else
    (block_status == COMPLETE || block_status == CHECKED)){
#endif
      temp_outblock->add_reader(true);
      //Global_Cache[pos_max]->unlock();
      temp_outblock->unlock();
      #ifdef DDEBUG
        lprintf(lvl-1, "<-----|\n");
      #endif
      final_estimated_linkmap->link_uses[dev_id_in_idx][pos_max]++;
      return deidxize(pos_max);
    }
    else error("Tile2D(%d)::getClosestReadLoc(%d): pos_max = %d selected,\
      but something changed after locking its cache...fixme\n", id, dev_id_in, pos_max);
  }
  else error("Tile2D(%d)::getClosestReadLoc(%d): pos_max = %d selected,\
    but StoreBlock[pos_max] was NULL after locking its cache...fixme\n", id, dev_id_in, pos_max);
  return -666;
}

double Tile2D::getMinLinkCost(short dev_id_in){
  short lvl = 5;
  int dev_id_in_idx = idxize(dev_id_in);
  double link_bw_max = 0;
  for (int pos =0; pos < LOC_NUM; pos++){
    CBlock_p temp_outblock = StoreBlock[pos];
    if(temp_outblock == NULL) continue;
    //StoreBlock[pos]->update_state(false);
    state temp = temp_outblock->State;
    if (temp == AVAILABLE || temp == SHARABLE || temp == NATIVE){
      event_status block_status = temp_outblock->Available->query_status();
#ifdef ALLOW_FETCH_RECORDED
    if(block_status == COMPLETE || block_status == CHECKED || block_status == RECORDED){
#else
    if(block_status == COMPLETE || block_status == CHECKED){
#endif
        double current_link_bw = final_estimated_link_bw[dev_id_in_idx][pos];

        if (block_status == RECORDED) current_link_bw-=current_link_bw*FETCH_UNAVAILABLE_PENALTY;
        if (current_link_bw > link_bw_max) link_bw_max = current_link_bw;
      }
    }
  }
  return link_bw_max;
}
