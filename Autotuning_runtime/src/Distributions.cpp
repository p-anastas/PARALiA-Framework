///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The possible subkernel distributions to different execution units.
///

#include "unihelpers.hpp"
#include "CoCoPeLiaModel.hpp"

void CoCoDistributeSubkernelsRoundRobin(ATC_p autotune_controller, int Subkernel_num){
  int lvl = 6;
  if (Subkernel_num <= autotune_controller->active_unit_num){
    autotune_controller->active_unit_num = Subkernel_num;
    for (int d = 0 ; d < autotune_controller->active_unit_num; d++){
      autotune_controller->Subkernels_per_unit_num[d] = 1;
      autotune_controller->Subkernels_per_unit_list[d][0] = d;
    }
  }
  else{
  int rem_dev = Subkernel_num;
  for (int d = 0 ; d < autotune_controller->active_unit_num; d++){
     autotune_controller->Subkernels_per_unit_num[d] =
      (int) (1.0* autotune_controller->active_unit_score[d]* Subkernel_num);
     rem_dev-= autotune_controller->Subkernels_per_unit_num[d];
  }
  while(rem_dev!= 0){
    for (int d = 0 ; d < autotune_controller->active_unit_num; d++){
       if(rem_dev!= 0){
         autotune_controller->Subkernels_per_unit_num[d] += 1;
         rem_dev--;
       }
       else break;
    }
  }
  int total_sk_ctr = 0;
  short dev_sk_ctr_list[autotune_controller->active_unit_num];
  for(int devidx = 0; devidx < autotune_controller->active_unit_num; devidx++) dev_sk_ctr_list[devidx] = 0;
  while(total_sk_ctr<Subkernel_num){
    for(int devidx = 0; devidx < autotune_controller->active_unit_num; devidx++){
      if(total_sk_ctr == Subkernel_num) break;
      else if(dev_sk_ctr_list[devidx] == autotune_controller->Subkernels_per_unit_num[devidx]) continue;
      else{
        autotune_controller->Subkernels_per_unit_list[devidx][dev_sk_ctr_list[devidx]] = total_sk_ctr;
        dev_sk_ctr_list[devidx]++;
        total_sk_ctr++;
      }
    }
  }
#ifdef PDEBUG
    lprintf(lvl, "CoCoDistributeSubkernelsRoundRobin:\nDistributing %d Subkernels to %d devices\n",
      Subkernel_num, autotune_controller->active_unit_num);
    lprintf(lvl, "Device Ids : [ ");
    for (int i =0; i < autotune_controller->active_unit_num; i++) fprintf(stderr, "%d ", autotune_controller->active_unit_id_list[i]);
    lprintf(0, "]\n");
    lprintf(lvl, "Subker Num : [ ");
    for (int i =0; i < autotune_controller->active_unit_num; i++) fprintf(stderr, "%d ",
      autotune_controller->Subkernels_per_unit_num[i]);
    lprintf(0, "]\n");
    for (int i =0; i < autotune_controller->active_unit_num; i++){
      lprintf(lvl, "Subker Id list for dev_id = %d: [ ", autotune_controller->active_unit_id_list[i]);
      for (int j =0; j < autotune_controller->Subkernels_per_unit_num[i]; j++) fprintf(stderr, "%d ",
        autotune_controller->Subkernels_per_unit_list[i][j]);
      lprintf(0, "]\n");
    }
#endif
  }
}

void CoCoDistributeSubkernelsNaive(ATC_p autotune_controller, int Subkernel_num){
  int lvl = 6;
  if (Subkernel_num <= autotune_controller->active_unit_num){
    autotune_controller->active_unit_num = Subkernel_num;
    for (int d = 0 ; d < autotune_controller->active_unit_num; d++){
      autotune_controller->Subkernels_per_unit_num[d] = 1;
      autotune_controller->Subkernels_per_unit_list[d][0] = d;
    }
  }
  else{
    int total_sk_ctr = 0;
    int dev_offset;
    int rem_dev = Subkernel_num;
    for (int d = 0 ; d < autotune_controller->active_unit_num; d++){
       autotune_controller->Subkernels_per_unit_num[d] =
        (int) (1.0* autotune_controller->active_unit_score[d]* Subkernel_num);
       rem_dev-= autotune_controller->Subkernels_per_unit_num[d];
    }
    while(rem_dev!= 0){
      for (int d = 0 ; d < autotune_controller->active_unit_num; d++){
         if(rem_dev!= 0){
           autotune_controller->Subkernels_per_unit_num[d] += 1;
           rem_dev--;
         }
         else break;
      }
    }
#ifdef DEBUG
    lprintf(lvl, "Subkernel Split offset = %d\n", dev_offset);
#endif
    short dev_sk_ctr = 0, cur_dev_id_ctr = 0;
    while(total_sk_ctr<Subkernel_num && cur_dev_id_ctr < autotune_controller->active_unit_num){
      while(dev_sk_ctr == autotune_controller->Subkernels_per_unit_num[cur_dev_id_ctr]){
        dev_sk_ctr = 0;
        cur_dev_id_ctr++;
      }
      autotune_controller->Subkernels_per_unit_list[cur_dev_id_ctr][dev_sk_ctr] = total_sk_ctr;
      dev_sk_ctr++;
      total_sk_ctr++;
    }
  }
#ifdef PDEBUG
    lprintf(lvl, "CoCoDistributeSubkernelsNaive:\nDistributing %d Subkernels to %d devices\n",
      Subkernel_num, autotune_controller->active_unit_num);
    lprintf(lvl, "Device Ids : [ ");
    for (int i =0; i < autotune_controller->active_unit_num; i++) fprintf(stderr, "%d ", autotune_controller->active_unit_id_list[i]);
    lprintf(0, "]\n");
    lprintf(lvl, "Subker Num : [ ");
    for (int i =0; i < autotune_controller->active_unit_num; i++) fprintf(stderr, "%d ",
      autotune_controller->Subkernels_per_unit_num[i]);
    lprintf(0, "]\n");
    for (int i =0; i < autotune_controller->active_unit_num; i++){
      lprintf(lvl, "Subker Id list for dev_id = %d: [ ", autotune_controller->active_unit_id_list[i]);
      for (int j =0; j < autotune_controller->Subkernels_per_unit_num[i]; j++) fprintf(stderr, "%d ",
        autotune_controller->Subkernels_per_unit_list[i][j]);
      lprintf(0, "]\n");
    }
#endif
}

void CoCoDistributeSubkernelsRoundRobinChunk(ATC_p autotune_controller, int Subkernel_num, int Chunk_size){
  int lvl = 6;
  if (Subkernel_num <= autotune_controller->active_unit_num){
    autotune_controller->active_unit_num = Subkernel_num;
    for (int d = 0 ; d < autotune_controller->active_unit_num; d++){
      autotune_controller->Subkernels_per_unit_num[d] = 1;
      autotune_controller->Subkernels_per_unit_list[d][0] = d;
    }
  }
  else{
  int rem_dev = Subkernel_num;
  for (int d = 0 ; d < autotune_controller->active_unit_num; d++){
     autotune_controller->Subkernels_per_unit_num[d] =
      (int) (1.0* autotune_controller->active_unit_score[d]* Subkernel_num);
     rem_dev-= autotune_controller->Subkernels_per_unit_num[d];
  }
  while(rem_dev!= 0){
    for (int d = 0 ; d < autotune_controller->active_unit_num; d++){
       if(rem_dev!= 0){
         autotune_controller->Subkernels_per_unit_num[d] += 1;
         rem_dev--;
       }
       else break;
    }
  }
  int total_sk_ctr = 0, local_dim_ctr = 0;
  short dev_sk_ctr_list[autotune_controller->active_unit_num];
  for(int devidx = 0; devidx < autotune_controller->active_unit_num; devidx++) dev_sk_ctr_list[devidx] = 0;
  while(total_sk_ctr<Subkernel_num){
    for(int devidx = 0; devidx < autotune_controller->active_unit_num; devidx++){
      if(total_sk_ctr == Subkernel_num) break;
      else if(dev_sk_ctr_list[devidx] == autotune_controller->Subkernels_per_unit_num[devidx]) continue;
      else{
        autotune_controller->Subkernels_per_unit_list[devidx][dev_sk_ctr_list[devidx]] = total_sk_ctr;
        dev_sk_ctr_list[devidx]++;
        total_sk_ctr++;
      }
      while(total_sk_ctr%Chunk_size!=0){
        if(total_sk_ctr == Subkernel_num) break;
        else if(dev_sk_ctr_list[devidx] == autotune_controller->Subkernels_per_unit_num[devidx]) break;
        else{
          autotune_controller->Subkernels_per_unit_list[devidx][dev_sk_ctr_list[devidx]] = total_sk_ctr;
          dev_sk_ctr_list[devidx]++;
          total_sk_ctr++;
        }
      }
      if(total_sk_ctr == Subkernel_num) break;
    }
  }
#ifdef PDEBUG
    lprintf(lvl, "CoCoDistributeSubkernelsRoundRobinChunk:\nDistributing %d Subkernels to %d devices\n",
      Subkernel_num, autotune_controller->active_unit_num);
    lprintf(lvl, "Device Ids : [ ");
    for (int i =0; i < autotune_controller->active_unit_num; i++) fprintf(stderr, "%d ", autotune_controller->active_unit_id_list[i]);
    lprintf(0, "]\n");
    lprintf(lvl, "Subker Num : [ ");
    for (int i =0; i < autotune_controller->active_unit_num; i++) fprintf(stderr, "%d ",
      autotune_controller->Subkernels_per_unit_num[i]);
    lprintf(0, "]\n");
    for (int i =0; i < autotune_controller->active_unit_num; i++){
      lprintf(lvl, "Subker Id list for dev_id = %d: [ ", autotune_controller->active_unit_id_list[i]);
      for (int j =0; j < autotune_controller->Subkernels_per_unit_num[i]; j++) fprintf(stderr, "%d ",
        autotune_controller->Subkernels_per_unit_list[i][j]);
      lprintf(0, "]\n");
    }
#endif
  }
}

void CoCoDistributeSubkernelsRoundRobinChunkReverse(ATC_p autotune_controller, int Subkernel_num, int Chunk_size){
  int lvl = 6;
  if (Subkernel_num <= autotune_controller->active_unit_num){
    autotune_controller->active_unit_num = Subkernel_num;
    for (int d = 0 ; d < autotune_controller->active_unit_num; d++){
      autotune_controller->Subkernels_per_unit_num[d] = 1;
      autotune_controller->Subkernels_per_unit_list[d][0] = d;
    }
  }
  else{
  int rem_dev = Subkernel_num;
  for (int d = 0 ; d < autotune_controller->active_unit_num; d++){
     autotune_controller->Subkernels_per_unit_num[d] =
      (int) (1.0* autotune_controller->active_unit_score[d]* Subkernel_num);
     rem_dev-= autotune_controller->Subkernels_per_unit_num[d];
  }
  while(rem_dev!= 0){
    for (int d = 0 ; d < autotune_controller->active_unit_num; d++){
       if(rem_dev!= 0){
         autotune_controller->Subkernels_per_unit_num[d] += 1;
         rem_dev--;
       }
       else break;
    }
  }
  int total_sk_ctr = 0, total_sk_prev = 0;
  short dev_sk_ctr_list[autotune_controller->active_unit_num];
  for(int devidx = 0; devidx < autotune_controller->active_unit_num; devidx++) dev_sk_ctr_list[devidx] = 0;
  while(total_sk_ctr<Subkernel_num){

    for(int devidx = 0; devidx < autotune_controller->active_unit_num; devidx++){
      total_sk_prev = total_sk_ctr;
      if(total_sk_ctr == Subkernel_num) break;
      else if(dev_sk_ctr_list[devidx] == autotune_controller->Subkernels_per_unit_num[devidx]) continue;
      else{
        autotune_controller->Subkernels_per_unit_list[devidx][dev_sk_ctr_list[devidx]] = total_sk_ctr;
        dev_sk_ctr_list[devidx]++;
        total_sk_ctr++;
      }
      while(total_sk_ctr%Chunk_size!=0){
        if(total_sk_ctr == Subkernel_num) break;
        else if(dev_sk_ctr_list[devidx] == autotune_controller->Subkernels_per_unit_num[devidx]) break;
        else{
          autotune_controller->Subkernels_per_unit_list[devidx][dev_sk_ctr_list[devidx]] = total_sk_ctr;
          dev_sk_ctr_list[devidx]++;
          total_sk_ctr++;
        }
      }
      if (devidx%2 == 0){
        for(int local_ctr = dev_sk_ctr_list[devidx] - total_sk_ctr + total_sk_prev; local_ctr < dev_sk_ctr_list[devidx]; local_ctr++){
          if (local_ctr < dev_sk_ctr_list[devidx] - local_ctr - 1){
            int temp_sk_id = autotune_controller->Subkernels_per_unit_list[devidx][local_ctr];
            autotune_controller->Subkernels_per_unit_list[devidx][local_ctr] =
              autotune_controller->Subkernels_per_unit_list[devidx]
                [dev_sk_ctr_list[devidx] - local_ctr - 1];
            autotune_controller->Subkernels_per_unit_list[devidx]
              [dev_sk_ctr_list[devidx] - local_ctr - 1] = temp_sk_id;
          }
          else break;
        }
      }
      if(total_sk_ctr == Subkernel_num) break;
    }
  }
/*  for(int devidx = 0; devidx < autotune_controller->active_unit_num; devidx++){
    for(int local_ctr = 0; local_ctr < autotune_controller->Subkernels_per_unit_num[devidx]; local_ctr++){
      if (local_ctr < autotune_controller->Subkernels_per_unit_num[devidx] - local_ctr - 1){
        int temp_sk_id = autotune_controller->Subkernels_per_unit_list[devidx][local_ctr];
        autotune_controller->Subkernels_per_unit_list[devidx][local_ctr] =
          autotune_controller->Subkernels_per_unit_list[devidx]
            [autotune_controller->Subkernels_per_unit_num[devidx] - local_ctr - 1];
        autotune_controller->Subkernels_per_unit_list[devidx]
          [autotune_controller->Subkernels_per_unit_num[devidx] - local_ctr - 1] = temp_sk_id;
      }
      else break;
    }
  }
  */
#ifdef PDEBUG
    lprintf(lvl, "CoCoDistributeSubkernelsRoundRobinChunkReverse:\nDistributing %d Subkernels to %d devices\n",
      Subkernel_num, autotune_controller->active_unit_num);
    lprintf(lvl, "Device Ids : [ ");
    for (int i =0; i < autotune_controller->active_unit_num; i++) fprintf(stderr, "%d ", autotune_controller->active_unit_id_list[i]);
    lprintf(0, "]\n");
    lprintf(lvl, "Subker Num : [ ");
    for (int i =0; i < autotune_controller->active_unit_num; i++) fprintf(stderr, "%d ",
      autotune_controller->Subkernels_per_unit_num[i]);
    lprintf(0, "]\n");
    for (int i =0; i < autotune_controller->active_unit_num; i++){
      lprintf(lvl, "Subker Id list for dev_id = %d: [ ", autotune_controller->active_unit_id_list[i]);
      for (int j =0; j < autotune_controller->Subkernels_per_unit_num[i]; j++) fprintf(stderr, "%d ",
        autotune_controller->Subkernels_per_unit_list[i][j]);
      lprintf(0, "]\n");
    }
#endif
  }
}

void CoCoDistributeSubkernels2DBlockCyclic(ATC_p autotune_controller, int D1GridSz, int D2GridSz, int D3GridSz){
  int lvl = 6;
  long Subkernel_num = D1GridSz* D2GridSz* D3GridSz;
  if (Subkernel_num <= autotune_controller->active_unit_num){
    autotune_controller->active_unit_num = Subkernel_num;
    for (int d = 0 ; d < autotune_controller->active_unit_num; d++){
      autotune_controller->Subkernels_per_unit_num[d] = 1;
      autotune_controller->Subkernels_per_unit_list[d][0] = d;
    }
  }
  else{
  int rem_dev = Subkernel_num;
  for (int d = 0 ; d < autotune_controller->active_unit_num; d++){
     autotune_controller->Subkernels_per_unit_num[d] =
      (int) (1.0* autotune_controller->active_unit_score[d]* Subkernel_num);
     rem_dev-= autotune_controller->Subkernels_per_unit_num[d];
  }
  while(rem_dev!= 0){
    for (int d = 0 ; d < autotune_controller->active_unit_num; d++){
       if(rem_dev!= 0){
         autotune_controller->Subkernels_per_unit_num[d] += 1;
         rem_dev--;
       }
       else break;
    }
  }

/* 2D Bloc cyclic */
  int D1_parts = sqrt(autotune_controller->active_unit_num);
  int D2_parts = D1_parts;
  if (D1_parts ==0) { D2_parts = autotune_controller->active_unit_num; D1_parts = 1; }
  else {
    /* find the most square decomposition of autotune_controller->active_unit_num in D1_parts x D2_parts */
    int g;
    for (g = D1_parts+1; g>0; --g)
       if (autotune_controller->active_unit_num % g == 0) break;
    if (g==0) { D1_parts = autotune_controller->active_unit_num; D2_parts = 1; }
    //if (g==0) { D1_parts = 1; D2_parts = autotune_controller->active_unit_num; }
    else { D1_parts = g; D2_parts = autotune_controller->active_unit_num/g; }
  }
  if(D1GridSz < D1_parts || D2GridSz < D2_parts){
    warning("CoCoDistributeSubkernels2DBlockCyclic:\nGrid(%d,%d) smaller than {D1,D2}_parts = (%d,%d)\
    using CoCoDistributeSubkernelsRoundRobinChunk instead\n", D1GridSz, D2GridSz, D1_parts, D2_parts);
    CoCoDistributeSubkernelsRoundRobinChunk(autotune_controller, Subkernel_num, D3GridSz);
    return;
  }
  int D1GridSz_div = D1GridSz/D1_parts*D1_parts, D2GridSz_div = D2GridSz/D2_parts*D2_parts,
      D1GridSz_mod = D1GridSz%D1_parts, D2GridSz_mod = D2GridSz%D2_parts;
#ifdef PDEBUG
lprintf(lvl, "CoCoDistributeSubkernels2DBlockCyclic: Devices = %d, D1_parts = %d, D2_parts = %d\n",
  autotune_controller->active_unit_num, D1_parts, D2_parts);
#endif
  int sk_ctr, devidx, dev_sk_ctr_list[autotune_controller->active_unit_num] = {0};
  for (int D1 = 0; D1 < D1GridSz_div; D1++)
    for (int D2 = 0; D2 < D2GridSz_div; D2++)
        for (int D3 = 0; D3 < D3GridSz; D3++){
          sk_ctr = D1*D2GridSz*D3GridSz + D2*D3GridSz+D3;
          devidx = D1/(D1GridSz/D1_parts)*D2_parts + D2/(D2GridSz/D2_parts);
#ifdef DPDEBUG
          lprintf(lvl, "CoCoDistributeSubkernels2DBlockCyclic: sk_ctr[%d,%d,%d] = %d, devidx = %d\n",
            D1,D2,D3, sk_ctr, devidx);
#endif
          while(dev_sk_ctr_list[devidx] == autotune_controller->Subkernels_per_unit_num[devidx]){
            if(devidx == autotune_controller->active_unit_num - 1) devidx = 0;
            else devidx++;
          }
          autotune_controller->Subkernels_per_unit_list[devidx][dev_sk_ctr_list[devidx]] = sk_ctr;
          dev_sk_ctr_list[devidx]++;
        }

  devidx = 0;

  for (int D1 = 0; D1 < D1GridSz_div; D1++)
    for (int D2 = D2GridSz_div; D2 < D2GridSz_div + D2GridSz_mod; D2++){
      for (int D3 = 0; D3 < D3GridSz; D3++){
        sk_ctr = D1*D2GridSz*D3GridSz + D2*D3GridSz+D3;
#ifdef PDEBUG
        lprintf(lvl, "CoCoDistributeSubkernels2DBlockCyclic:\nsk_ctr[%d,%d,%d] = %d, devidx = %d\n",
          D1,D2,D3, sk_ctr, devidx);
#endif
        while(dev_sk_ctr_list[devidx] == autotune_controller->Subkernels_per_unit_num[devidx]){
          if(devidx == autotune_controller->active_unit_num - 1) devidx = 0;
          else devidx++;
        }
        autotune_controller->Subkernels_per_unit_list[devidx][dev_sk_ctr_list[devidx]] = sk_ctr;
        dev_sk_ctr_list[devidx]++;
      }
      if(devidx == autotune_controller->active_unit_num - 1) devidx = 0;
      else devidx++;
    }

  for (int D1 = D1GridSz_div; D1 < D1GridSz_div + D1GridSz_mod; D1++)
    for (int D2 = 0; D2 < D2GridSz_div; D2++){
      for (int D3 = 0; D3 < D3GridSz; D3++){
        sk_ctr = D1*D2GridSz*D3GridSz + D2*D3GridSz+D3;
#ifdef PDEBUG
        lprintf(lvl, "CoCoDistributeSubkernels2DBlockCyclic:\nsk_ctr[%d,%d,%d] = %d, devidx = %d\n",
          D1,D2,D3, sk_ctr, devidx);
#endif
        while(dev_sk_ctr_list[devidx] == autotune_controller->Subkernels_per_unit_num[devidx]){
          if(devidx == autotune_controller->active_unit_num - 1) devidx = 0;
          else devidx++;
        }
        autotune_controller->Subkernels_per_unit_list[devidx][dev_sk_ctr_list[devidx]] = sk_ctr;
        dev_sk_ctr_list[devidx]++;
      }
      if(devidx == autotune_controller->active_unit_num - 1) devidx = 0;
      else devidx++;
    }

  for (int D1 = D1GridSz_div; D1 < D1GridSz_div + D1GridSz_mod; D1++)
    for (int D2 = D2GridSz_div; D2 < D2GridSz_div + D2GridSz_mod; D2++){
        for (int D3 = 0; D3 < D3GridSz; D3++){
          sk_ctr = D1*D2GridSz*D3GridSz + D2*D3GridSz+D3;
#ifdef PDEBUG
          lprintf(lvl, "CoCoDistributeSubkernels2DBlockCyclic:\nsk_ctr[%d,%d,%d] = %d, devidx = %d\n",
            D1,D2,D3, sk_ctr, devidx);
#endif
          while(dev_sk_ctr_list[devidx] == autotune_controller->Subkernels_per_unit_num[devidx]){
            if(devidx == autotune_controller->active_unit_num - 1) devidx = 0;
            else devidx++;
          }
          autotune_controller->Subkernels_per_unit_list[devidx][dev_sk_ctr_list[devidx]] = sk_ctr;
          dev_sk_ctr_list[devidx]++;
        }
        if(devidx == autotune_controller->active_unit_num - 1) devidx = 0;
        else devidx++;
      }
#ifdef PDEBUG
    lprintf(lvl, "CoCoDistributeSubkernels2DBlockCyclic:\nDistributing %ld Subkernels to %d devices\n",
      Subkernel_num, autotune_controller->active_unit_num);
    lprintf(lvl, "Device Ids : [ ");
    for (int i =0; i < autotune_controller->active_unit_num; i++) fprintf(stderr, "%d ", autotune_controller->active_unit_id_list[i]);
    lprintf(0, "]\n");
    lprintf(lvl, "Subker Num : [ ");
    for (int i =0; i < autotune_controller->active_unit_num; i++) fprintf(stderr, "%d ",
      autotune_controller->Subkernels_per_unit_num[i]);
    lprintf(0, "]\n");
    for (int i =0; i < autotune_controller->active_unit_num; i++){
      lprintf(lvl, "Subker Id list for dev_id = %d: [ ", autotune_controller->active_unit_id_list[i]);
      for (int j =0; j < autotune_controller->Subkernels_per_unit_num[i]; j++) fprintf(stderr, "%d ",
        autotune_controller->Subkernels_per_unit_list[i][j]);
      lprintf(0, "]\n");
    }
#endif
  }
}
