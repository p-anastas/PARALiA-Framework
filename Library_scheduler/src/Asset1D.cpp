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

template<typename dtype> void Asset1D<dtype>::InitTileMap(int T, Buffer_p* init_loc_cache_p){
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

  Tile_map = (Tile1D<dtype>**) malloc(sizeof(Tile1D<dtype>*)*GridSz);

  int current_ctr, Ttmp;
  void* tile_addr = NULL;
  for (int itt = 0; itt < GridSz; itt++){
    if ( itt == GridSz - 1) Ttmp = TLast;
    else  Ttmp = T;
    current_ctr = itt;
    tile_addr = adrs + itt*T*inc;
    Tile_map[current_ctr] = new Tile1D<dtype>(tile_addr, Ttmp, inc, itt,
      init_loc_cache_p[idxize(CoCoGetPtrLoc(adrs))]->assign_Cblock(NATIVE, true));
  }
  #ifdef DEBUG
  	lprintf(lvl-1, "<-----|\n");
  #endif
}

template void Asset1D<double>::InitTileMap(int T, Buffer_p* init_loc_cache_p);
template void Asset1D<float>::InitTileMap(int T, Buffer_p* init_loc_cache_p);

template<typename dtype> void Asset1D<dtype>::DestroyTileMap(){
  int current_ctr;
  for (int itt = 0 ; itt < GridSz; itt++){
    current_ctr = itt;
    delete Tile_map[current_ctr];
  }
  free(Tile_map);
}

template void Asset1D<double>::DestroyTileMap();
template void Asset1D<float>::DestroyTileMap();

template<typename dtype> void Asset1D<dtype>::DrawTileMap(){
  fprintf(stderr, " Tile1D representation: \
                 \n ______________________ \
                 \n|      id[GridId]      |\
                 \n| - - - - - - - - - - -|\
                 \n|        (dim1)        |\
                 \n| - - - - - - - - - - -|\
                 \n| Read | Write | RW_M  |\n");
  for(int loctr = 0; loctr < LOC_NUM; loctr++)
    fprintf(stderr, "| - - - - - - - - - - -|\
                   \n| loc: %2d | CLI | RTM  |\n",
                   (loctr == LOC_NUM - 1) ? -1 : loctr );

  fprintf(stderr, "|______________________|\n\n");

  for (int itt = 0 ; itt < GridSz; itt++)
    fprintf(stderr, " ______________________ ");
  fprintf(stderr, "\n");
  for (int itt = 0 ; itt < GridSz; itt++)
    fprintf(stderr, "|     %4d[%6d]     |",
    Tile_map[itt]->id,
    Tile_map[itt]->GridId);
  fprintf(stderr, "\n");
  for (int itt = 0 ; itt < GridSz; itt++)
    fprintf(stderr, "| - - - - - - - - - - -|");
  fprintf(stderr, "\n");
  for (int itt = 0 ; itt < GridSz; itt++)
    fprintf(stderr, "|       (%6d)       |",
    Tile_map[itt]->dim);
  fprintf(stderr, "\n");
  for (int itt = 0 ; itt < GridSz; itt++)
    fprintf(stderr, "| - - - - - - - - - - -|");
  fprintf(stderr, "\n");
  for (int itt = 0 ; itt < GridSz; itt++)
    fprintf(stderr, "| R:%1d | W:%1d | WR_M:%3d |",
    Tile_map[itt]->R_flag,
    Tile_map[itt]->W_flag,
    Tile_map[itt]->RW_master);
  fprintf(stderr, "\n");
  for(int loctr = 0; loctr < LOC_NUM; loctr++){
    for (int itt = 0 ; itt < GridSz; itt++)
      fprintf(stderr, "| - - - - - - - - - - -|");
    fprintf(stderr, "\n");
    for (int itt = 0 ; itt < GridSz; itt++){
      if(Tile_map[itt]->StoreBlock[loctr])
        fprintf(stderr, "| loc: %2d | %3d | %3d  |",
                   (loctr == LOC_NUM - 1) ? -1 : loctr,
                   Tile_map[itt]->StoreBlock[loctr]->id,
                   Tile_map[itt]->RunTileMap[loctr]);
      else fprintf(stderr, "| loc: %2d | NaN | %3d  |",
                   (loctr == LOC_NUM - 1) ? -1 : loctr,
                   Tile_map[itt]->RunTileMap[loctr]);
    }
    fprintf(stderr, "\n");
  }
  for (int itt = 0 ; itt < GridSz; itt++)
   fprintf(stderr, "|______________________|");
  fprintf(stderr, "\n\n");
}

template void Asset1D<double>::DrawTileMap();
template void Asset1D<float>::DrawTileMap();
