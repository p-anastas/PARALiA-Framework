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
                 \n| - - WR_properties - -|\
                 \n| - - - - - - - - - - -|\
                 \n| - - - loc_list  - - -|\
                 \n|______________________|\n\n");

  for (int itt = 0 ; itt < GridSz; itt++)
    fprintf(stderr, " ______________________ ");
  fprintf(stderr, "\n");
  for (int itt = 0 ; itt < GridSz; itt++)
    fprintf(stderr, "|     %4d[%6d]     |",
    Tile_map[itt]->id,
    Tile_map[itt]->GridId1);
  fprintf(stderr, "\n");
  for (int itt = 0 ; itt < GridSz; itt++)
    fprintf(stderr, "| - - - - - - - - - - -|");
  fprintf(stderr, "\n");
  for (int itt = 0 ; itt < GridSz; itt++)
    fprintf(stderr, "|       (%6d)       |",
    Tile_map[itt]->dim1);
  fprintf(stderr, "\n");
  for (int itt = 0 ; itt < GridSz; itt++)
    fprintf(stderr, "| - - - - - - - - - - -|");
  fprintf(stderr, "\n");
  for (int itt = 0 ; itt < GridSz; itt++)
    fprintf(stderr, "%s", Tile_map[itt]->get_WRP_string());
  fprintf(stderr, "\n");
  for(int loctr = 0; loctr < LOC_NUM; loctr++){
    for (int itt = 0 ; itt < GridSz; itt++)
      fprintf(stderr, "| - - - - - - - - - - -|");
    fprintf(stderr, "\n");
    for (int itt = 0 ; itt < GridSz; itt++)
      fprintf(stderr, "%s", printlist<int>(Tile_map[itt]->loc_map, LOC_NUM));
    fprintf(stderr, "\n");
  }
  for (int itt = 0 ; itt < GridSz; itt++)
   fprintf(stderr, "|______________________|");
  fprintf(stderr, "\n\n");
}
