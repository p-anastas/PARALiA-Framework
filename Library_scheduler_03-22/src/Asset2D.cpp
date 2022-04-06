///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The "Asset" related function implementations.
///

#include "Asset.hpp"
#include "unihelpers.hpp"

template class Asset2D<double>;

template<typename dtype> Tile2D<dtype>* Asset2D<dtype>::getTile(int iloc1, int iloc2){
  if(iloc1 >= GridSz1) error("Asset2D::getTile : iloc1 >= GridSz1 (%d vs %d)\n", iloc1, GridSz1);
  else if(iloc2 >= GridSz2) error("Asset2D::getTile : iloc2 >= GridSz2 (%d vs %d)\n", iloc2, GridSz2);
  return Tile_map[iloc1*GridSz2 + iloc2];
}

template<typename dtype> Asset2D<dtype>::Asset2D( void* in_adr, int in_dim1, int in_dim2, int in_ldim, char in_transpose){
  ldim = in_ldim;
  dim1 = in_dim1;
  dim2 = in_dim2;
  adrs = (dtype*) in_adr;
  loc = CoCoGetPtrLoc(in_adr);
  transpose = in_transpose;
}

template<typename dtype> void Asset2D<dtype>::InitTileMap(int T1, int T2, Cache_p* init_loc_cache_p){
  short lvl = 2;

  #ifdef DEBUG
  	lprintf(lvl-1, "|-----> Asset2D<dtype>::InitTileMap(%d,%d)\n", T1, T2);
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

  Tile_map = (Tile2D<dtype>**) malloc(sizeof(Tile2D<dtype>*)*GridSz1*GridSz2);

  int current_ctr, T1tmp, T2tmp;
  void* tile_addr = NULL;
  for (int itt1 = 0; itt1 < GridSz1; itt1++){
    if ( itt1 == GridSz1 - 1) T1tmp = T1Last;
    else  T1tmp = T1;
		for (int itt2 = 0 ; itt2 < GridSz2; itt2++){
      if ( itt2 == GridSz2 - 1) T2tmp = T2Last;
      else  T2tmp = T2;
      current_ctr = itt1*GridSz2 + itt2;
      /// For column major format assumed with T1tmp = rows and T2tmp = cols
      if (transpose == 'N'){
         tile_addr = adrs + itt1*T1 + itt2*T2*ldim;
         Tile_map[current_ctr] = new Tile2D<dtype>(tile_addr, T1tmp, T2tmp, ldim, itt1, itt2,
           init_loc_cache_p[idxize(CoCoGetPtrLoc(adrs))]->assign_Cblock());
       }
      else if (transpose == 'T'){
        tile_addr = adrs + itt1*T1*ldim + itt2*T2;
        Tile_map[current_ctr] = new Tile2D<dtype>(tile_addr, T2tmp, T1tmp, ldim, itt2, itt1,
           init_loc_cache_p[idxize(CoCoGetPtrLoc(adrs))]->assign_Cblock());
      }
      else error("Asset2D<dtype>::InitTileMap: Unknown transpose type\n");

     }
   }
   #ifdef DEBUG
   	lprintf(lvl-1, "<-----|\n");
   #endif
}

template void Asset2D<double>::InitTileMap(int T1, int T2, Cache_p* init_loc_cache_p);

template<typename dtype> void Asset2D<dtype>::DestroyTileMap(){
  int current_ctr;
  for (int itt1 = 0; itt1 < GridSz1; itt1++)
    for (int itt2 = 0 ; itt2 < GridSz2; itt2++){
      current_ctr = itt1*GridSz2 + itt2;
      delete Tile_map[current_ctr];
    }
  free(Tile_map);
}

template void Asset2D<double>::DestroyTileMap();

template<typename dtype> void Asset2D<dtype>::DrawTileMap(){
  fprintf(stderr, " Tile2D representation: \
                 \n ______________________ \
                 \n| id[GridId1, GridId2] |\
                 \n| - - - - - - - - - - -|\
                 \n|    (dim1 X dim2)     |\
                 \n| - - - - - - - - - - -|\
                 \n| Read | Write | RW_M  |\n");
  for(int loctr = 0; loctr < LOC_NUM; loctr++)
    fprintf(stderr, "| - - - - - - - - - - -|\
                   \n| loc: %2d | CLI | RTM  |\n",
                   (loctr == LOC_NUM - 1) ? -1 : loctr );

  fprintf(stderr, "|______________________|\n\n");

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
      fprintf(stderr, "| R:%1d | W:%1d | WR_M:%3d |",
      Tile_map[itt1*GridSz2 + itt2]->R_flag,
      Tile_map[itt1*GridSz2 + itt2]->W_flag,
      Tile_map[itt1*GridSz2 + itt2]->RW_master);
    fprintf(stderr, "\n");
    for(int loctr = 0; loctr < LOC_NUM; loctr++){
      for (int itt2 = 0 ; itt2 < GridSz2; itt2++)
        fprintf(stderr, "| - - - - - - - - - - -|");
      fprintf(stderr, "\n");
      for (int itt2 = 0 ; itt2 < GridSz2; itt2++){
        if(Tile_map[itt1*GridSz2 + itt2]->StoreBlock[loctr])
          fprintf(stderr, "| loc: %2d | %3d | %3d  |",
                     (loctr == LOC_NUM - 1) ? -1 : loctr,
                     Tile_map[itt1*GridSz2 + itt2]->StoreBlock[loctr]->id,
                     Tile_map[itt1*GridSz2 + itt2]->RunTileMap[loctr]);
        else fprintf(stderr, "| loc: %2d | NaN | %3d  |",
                     (loctr == LOC_NUM - 1) ? -1 : loctr,
                     Tile_map[itt1*GridSz2 + itt2]->RunTileMap[loctr]);
      }
      fprintf(stderr, "\n");
    }
    for (int itt2 = 0 ; itt2 < GridSz2; itt2++)
     fprintf(stderr, "|______________________|");
    fprintf(stderr, "\n\n");
  }
}

template void Asset2D<double>::DrawTileMap();
