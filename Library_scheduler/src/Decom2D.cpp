///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The "Asset" related function implementations.
///

#include "Decomposer.hpp"
#include "linkmap.hpp"

Tile2D* Decom2D::getTile(int iloc1, int iloc2){
  if(iloc1 >= GridSz1) error("Decom2D::getTile : iloc1 >= GridSz1 (%d vs %d)\n", iloc1, GridSz1);
  else if(iloc2 >= GridSz2) error("Decom2D::getTile : iloc2 >= GridSz2 (%d vs %d)\n", iloc2, GridSz2);
  return Tile_map[iloc1*GridSz2 + iloc2];
}

Decom2D::Decom2D( void* in_adr, int in_dim1, int in_dim2, int in_ldim, char in_transpose, dtype_enum dtype_in){
  ldim = in_ldim;
  dim1 = in_dim1;
  dim2 = in_dim2;
  adrs = in_adr;
  loc = CoCoGetPtrLoc(in_adr);
  transpose = in_transpose;
  dtype = dtype_in;
}

void Decom2D::InitTileMap(int T1, int T2, Buffer_p* init_loc_cache_p){
  short lvl = 2;

  #ifdef DEBUG
  	lprintf(lvl-1, "|-----> Decom2D::InitTileMap(%d,%d)\n", T1, T2);
  #endif

  GridSz1 = dim1/T1;
	GridSz2 = dim2/T2;
  int T1Last = dim1%T1, T2Last = dim2%T2;
  // TODO: Padding instead of resize so all data fit in buffer without complex mechanism.
  // Can degrade performance for small div sizes.
  if (T1Last > 0) GridSz1++;
  else T1Last=T1;
  if (T2Last > 0) GridSz2++;
  else T2Last=T2;
	//if (T1Last > T1/4) GridSz1++;
	//else T1Last+=T1;
  //if (T2Last > T2/4) GridSz2++;
  //else T2Last+=T2;


  Tile_map = (Tile2D**) malloc(sizeof(Tile2D*)*GridSz1*GridSz2);

  int current_ctr, T1tmp, T2tmp;
  void* tile_addr = NULL;
  int loc_idx = idxize(CoCoGetPtrLoc(adrs));
  for (int itt1 = 0; itt1 < GridSz1; itt1++){
    if ( itt1 == GridSz1 - 1) T1tmp = T1Last;
    else  T1tmp = T1;
		for (int itt2 = 0 ; itt2 < GridSz2; itt2++){
      if ( itt2 == GridSz2 - 1) T2tmp = T2Last;
      else  T2tmp = T2;
      current_ctr = itt1*GridSz2 + itt2;
      /// For column major format assumed with T1tmp = rows and T2tmp = cols
      if (transpose == 'N'){
         if (dtype == DOUBLE) tile_addr = ((double*)adrs) + (itt1*T1 + itt2*T2*ldim);
         else if(dtype == FLOAT) tile_addr = ((float*)adrs) + (itt1*T1 + itt2*T2*ldim);
         else error("Decom2D::InitTileMap: dtype not implemented");
         Tile_map[current_ctr] = new Tile2D(tile_addr, T1tmp, T2tmp, ldim, itt1, itt2, dtype, init_loc_cache_p[loc_idx]->assign_Cblock(NATIVE, true));
       }
      else if (transpose == 'T'){
        if (dtype == DOUBLE) tile_addr = ((double*)adrs) + (itt1*T1*ldim + itt2*T2);
         else if(dtype == FLOAT)  tile_addr = ((float*)adrs) + (itt1*T1*ldim + itt2*T2);
        else error("Decom2D::InitTileMap: dtype not implemented");
        Tile_map[current_ctr] = new Tile2D(tile_addr, T2tmp, T1tmp, ldim, itt2, itt1, dtype, init_loc_cache_p[loc_idx]->assign_Cblock(NATIVE, true));
      }
      else error("Decom2D::InitTileMap: Unknown transpose type\n");

     }
   }
   #ifdef DEBUG
   	lprintf(lvl-1, "<-----|\n");
   #endif
}

void Decom2D::DestroyTileMap(){
  int current_ctr;
  for (int itt1 = 0; itt1 < GridSz1; itt1++)
    for (int itt2 = 0 ; itt2 < GridSz2; itt2++){
      current_ctr = itt1*GridSz2 + itt2;
      delete Tile_map[current_ctr];
    }
  free(Tile_map);
}

void Decom2D::SyncTileMap(){
  for (int itt1 = 0; itt1 < GridSz1; itt1++)
    for (int itt2 = 0 ; itt2 < GridSz2; itt2++)
      Tile_map[itt1*GridSz2 + itt2]->writeback();

  for (int itt1 = 0; itt1 < GridSz1; itt1++)
    for (int itt2 = 0 ; itt2 < GridSz2; itt2++)
      Tile_map[itt1*GridSz2 + itt2]->StoreBlock[idxize(Tile_map[itt1*GridSz2 + itt2]->get_initial_location())]->Available->sync_barrier();
}

void Decom2D::DrawTileMap(){
  fprintf(stderr, " Tile2D representation: \
                 \n ______________________ \
                 \n| id[GridId1, GridId2] |\
                 \n| - - - - - - - - - - -|\
                 \n|    (dim1 X dim2)     |\
                 \n| - - - - - - - - - - -|\
                 \n| - - WR_properties - -|\
                 \n| - - - - - - - - - - -|\
                 \n| - - - loc_list  - - -|\
                 \n|______________________|\n\n");

  for (int itt1 = 0; itt1 < GridSz1; itt1++){
    for (int itt2 = 0 ; itt2 < GridSz2; itt2++)
      fprintf(stderr, " ______________________ ");
    fprintf(stderr, "\n");
    for (int itt2 = 0 ; itt2 < GridSz2; itt2++)
      fprintf(stderr, "|%4d[%6d,%6d]   |",
      Tile_map[itt1*GridSz2 + itt2]->id,
      Tile_map[itt1*GridSz2 + itt2]->GridId1,
      Tile_map[itt1*GridSz2 + itt2]->GridId2);
    fprintf(stderr, "\n");
    for (int itt2 = 0 ; itt2 < GridSz2; itt2++)
      fprintf(stderr, "| - - - - - - - - - - -|");
    fprintf(stderr, "\n");
    for (int itt2 = 0 ; itt2 < GridSz2; itt2++)
      fprintf(stderr, "|  (%6d X %6d)   |",
      Tile_map[itt1*GridSz2 + itt2]->dim1,
      Tile_map[itt1*GridSz2 + itt2]->dim2);
    fprintf(stderr, "\n");
    for (int itt2 = 0 ; itt2 < GridSz2; itt2++)
      fprintf(stderr, "| - - - - - - - - - - -|");
    fprintf(stderr, "\n");
    for (int itt2 = 0 ; itt2 < GridSz2; itt2++)
      fprintf(stderr, "%s", Tile_map[itt1*GridSz2 + itt2]->get_WRP_string());
    fprintf(stderr, "\n");
    for(int loctr = 0; loctr < LOC_NUM; loctr++){
      for (int itt2 = 0 ; itt2 < GridSz2; itt2++)
        fprintf(stderr, "| - - - - - - - - - - -|");
      fprintf(stderr, "\n");
      for (int itt2 = 0 ; itt2 < GridSz2; itt2++)
        fprintf(stderr, "%s", printlist<int>(Tile_map[itt1*GridSz2 + itt2]->loc_map, LOC_NUM));
      fprintf(stderr, "\n");
    }
    for (int itt2 = 0 ; itt2 < GridSz2; itt2++)
     fprintf(stderr, "|______________________|");
    fprintf(stderr, "\n\n");
  }
}
