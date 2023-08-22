///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The "Asset" related function implementations.
///

#include "Decomposer.hpp"
#include "linkmap.hpp"

Tile1D* Decom1D::getTile(int iloc){
  if(iloc >= GridSz) error("Decom1D::getTile : iloc >= GridSz (%d vs %d)\n", iloc, GridSz);
  return Tile_map[iloc];
}

Decom1D::Decom1D(void* in_adr, int in_dim, int in_inc, dtype_enum dtype_in){
  dim = in_dim;
  adrs = in_adr;
  loc = CoCoGetPtrLoc(in_adr);
  inc = in_inc;
  dtype = dtype_in;
}

void Decom1D::InitTileMap(int T, Buffer_p* init_loc_cache_p){
  short lvl = 2;

  #ifdef DEBUG
  	lprintf(lvl-1, "|-----> Decom1D::InitTileMap(%d)\n", T);
  #endif

  GridSz = dim/T;
  int TLast = dim%T;
  // TODO: Padding instead of resize so all data fit in buffer without complex mechanism.
  // Can degrade performance for small div sizes.
  if (TLast > 0) GridSz++;
  else TLast=T;

  Tile_map = (Tile1D**) malloc(sizeof(Tile1D*)*GridSz);

  int current_ctr, Ttmp;
  void* tile_addr = NULL;
  for (int itt = 0; itt < GridSz; itt++){
    if ( itt == GridSz - 1) Ttmp = TLast;
    else  Ttmp = T;
    current_ctr = itt;
    if (dtype == DOUBLE) tile_addr = ((double*)adrs) + itt*T*inc;
    else if(dtype == FLOAT) tile_addr = ((float*)adrs) + itt*T*inc;
    else error("Decom1D::InitTileMap: dtype not implemented");
    Tile_map[current_ctr] = new Tile1D(tile_addr, Ttmp, inc, itt, dtype, init_loc_cache_p[idxize(CoCoGetPtrLoc(adrs))]->assign_Cblock(NATIVE, true));
  }
  #ifdef DEBUG
  	lprintf(lvl-1, "<-----|\n");
  #endif
}


void Decom1D::DestroyTileMap(){
  int current_ctr;
  for (int itt = 0 ; itt < GridSz; itt++){
    current_ctr = itt;
    delete Tile_map[current_ctr];
  }
  free(Tile_map);
}

void Decom1D::DrawTileMap(){
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
