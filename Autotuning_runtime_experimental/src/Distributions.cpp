///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The possible subkernel distributions to different execution units.
///

#include "unihelpers.hpp"
#include "CoCoPeLiaModel.hpp"

void CoCoDistributeSubkernelsRoundRobin(CoControl_p autotune_vals,
  tunableParams_p pred_p, int Subkernel_num){
  int lvl = 6;
  if (Subkernel_num <= autotune_vals->dev_num){
    autotune_vals->dev_num = Subkernel_num;
    for (int d = 0 ; d < autotune_vals->dev_num; d++){
      autotune_vals->Subkernels_per_dev[d] = 1;
      autotune_vals->Subkernel_dev_id_list[d][0] = d;
    }
  }
  else{
#ifdef MULTIDEVICE_REDUCTION_ENABLE
  int rem_dev = Subkernel_num;
  for (int d = 0 ; d < autotune_vals->dev_num; d++){
     autotune_vals->Subkernels_per_dev[d] =
      (int) (1.0* pred_p->rel_dev_score[d]* Subkernel_num);
     rem_dev-= autotune_vals->Subkernels_per_dev[d];
  }
  while(rem_dev!= 0){
    for (int d = 0 ; d < autotune_vals->dev_num; d++){
       if(rem_dev!= 0){
         autotune_vals->Subkernels_per_dev[d] += 1;
         rem_dev--;
       }
       else break;
    }
  }
#else
      error("CoCoDistributeSubkernelsRoundRobin: not implemented for undefined MULTIDEVICE_REDUCTION_ENABLE\n");
#endif
  int total_sk_ctr = 0;
  short dev_sk_ctr_list[autotune_vals->dev_num] = {0};
  while(total_sk_ctr<Subkernel_num){
    for(int devidx = 0; devidx < autotune_vals->dev_num; devidx++){
      if(total_sk_ctr == Subkernel_num) break;
      else if(dev_sk_ctr_list[devidx] == autotune_vals->Subkernels_per_dev[devidx]) continue;
      else{
        autotune_vals->Subkernel_dev_id_list[devidx][dev_sk_ctr_list[devidx]] = total_sk_ctr;
        dev_sk_ctr_list[devidx]++;
        total_sk_ctr++;
      }
    }
  }
#ifdef PDEBUG
    lprintf(lvl, "CoCoDistributeSubkernelsRoundRobin:\nDistributing %d Subkernels to %d devices\n",
      Subkernel_num, autotune_vals->dev_num);
    lprintf(lvl, "Device Ids : [ ");
    for (int i =0; i < autotune_vals->dev_num; i++) fprintf(stderr, "%d ", autotune_vals->dev_ids[i]);
    lprintf(0, "]\n");
    lprintf(lvl, "Subker Num : [ ");
    for (int i =0; i < autotune_vals->dev_num; i++) fprintf(stderr, "%d ",
      autotune_vals->Subkernels_per_dev[i]);
    lprintf(0, "]\n");
    for (int i =0; i < autotune_vals->dev_num; i++){
      lprintf(lvl, "Subker Id list for dev_id = %d: [ ", autotune_vals->dev_ids[i]);
      for (int j =0; j < autotune_vals->Subkernels_per_dev[i]; j++) fprintf(stderr, "%d ",
        autotune_vals->Subkernel_dev_id_list[i][j]);
      lprintf(0, "]\n");
    }
#endif
  }
}

void CoCoDistributeSubkernelsNaive(CoControl_p autotune_vals,
  tunableParams_p pred_p, int Subkernel_num){
  int lvl = 6;
  if (Subkernel_num <= autotune_vals->dev_num){
    autotune_vals->dev_num = Subkernel_num;
    for (int d = 0 ; d < autotune_vals->dev_num; d++){
      autotune_vals->Subkernels_per_dev[d] = 1;
      autotune_vals->Subkernel_dev_id_list[d][0] = d;
    }
  }
  else{
    int total_sk_ctr = 0;
    int dev_offset;
#ifdef MULTIDEVICE_REDUCTION_ENABLE
    int rem_dev = Subkernel_num;
    for (int d = 0 ; d < autotune_vals->dev_num; d++){
       autotune_vals->Subkernels_per_dev[d] =
        (int) (1.0* pred_p->rel_dev_score[d]* Subkernel_num);
       rem_dev-= autotune_vals->Subkernels_per_dev[d];
    }
    while(rem_dev!= 0){
      for (int d = 0 ; d < autotune_vals->dev_num; d++){
         if(rem_dev!= 0){
           autotune_vals->Subkernels_per_dev[d] += 1;
           rem_dev--;
         }
         else break;
      }
    }
#else
    error("CoCoDistributeSubkernelsNaive: dev_offset = 0 undefined without MULTIDEVICE_REDUCTION_ENABLE");
#endif
#ifdef DEBUG
    lprintf(lvl, "Subkernel Split offset = %d\n", dev_offset);
#endif
    short dev_sk_ctr = 0, cur_dev_id_ctr = 0;
    while(total_sk_ctr<Subkernel_num && cur_dev_id_ctr < autotune_vals->dev_num){
      while(dev_sk_ctr == autotune_vals->Subkernels_per_dev[cur_dev_id_ctr]){
        dev_sk_ctr = 0;
        cur_dev_id_ctr++;
      }
      autotune_vals->Subkernel_dev_id_list[cur_dev_id_ctr][dev_sk_ctr] = total_sk_ctr;
      dev_sk_ctr++;
      total_sk_ctr++;
    }
  }
#ifdef PDEBUG
    lprintf(lvl, "CoCoDistributeSubkernelsNaive:\nDistributing %d Subkernels to %d devices\n",
      Subkernel_num, autotune_vals->dev_num);
    lprintf(lvl, "Device Ids : [ ");
    for (int i =0; i < autotune_vals->dev_num; i++) fprintf(stderr, "%d ", autotune_vals->dev_ids[i]);
    lprintf(0, "]\n");
    lprintf(lvl, "Subker Num : [ ");
    for (int i =0; i < autotune_vals->dev_num; i++) fprintf(stderr, "%d ",
      autotune_vals->Subkernels_per_dev[i]);
    lprintf(0, "]\n");
    for (int i =0; i < autotune_vals->dev_num; i++){
      lprintf(lvl, "Subker Id list for dev_id = %d: [ ", autotune_vals->dev_ids[i]);
      for (int j =0; j < autotune_vals->Subkernels_per_dev[i]; j++) fprintf(stderr, "%d ",
        autotune_vals->Subkernel_dev_id_list[i][j]);
      lprintf(0, "]\n");
    }
#endif
}

void CoCoDistributeSubkernelsRoundRobinChunk(CoControl_p autotune_vals,
  tunableParams_p pred_p, int Subkernel_num, int Chunk_size){
  int lvl = 6;
  if (Subkernel_num <= autotune_vals->dev_num){
    autotune_vals->dev_num = Subkernel_num;
    for (int d = 0 ; d < autotune_vals->dev_num; d++){
      autotune_vals->Subkernels_per_dev[d] = 1;
      autotune_vals->Subkernel_dev_id_list[d][0] = d;
    }
  }
  else{
#ifdef MULTIDEVICE_REDUCTION_ENABLE
  int rem_dev = Subkernel_num;
  for (int d = 0 ; d < autotune_vals->dev_num; d++){
     autotune_vals->Subkernels_per_dev[d] =
      (int) (1.0* pred_p->rel_dev_score[d]* Subkernel_num);
     rem_dev-= autotune_vals->Subkernels_per_dev[d];
  }
  while(rem_dev!= 0){
    for (int d = 0 ; d < autotune_vals->dev_num; d++){
       if(rem_dev!= 0){
         autotune_vals->Subkernels_per_dev[d] += 1;
         rem_dev--;
       }
       else break;
    }
  }
#else
      error("CoCoDistributeSubkernelsRoundRobinChunk: not implemented for undefined MULTIDEVICE_REDUCTION_ENABLE\n");
#endif
  int total_sk_ctr = 0, local_dim_ctr = 0;
  short dev_sk_ctr_list[autotune_vals->dev_num] = {0};
  while(total_sk_ctr<Subkernel_num){
    for(int devidx = 0; devidx < autotune_vals->dev_num; devidx++){
      if(total_sk_ctr == Subkernel_num) break;
      else if(dev_sk_ctr_list[devidx] == autotune_vals->Subkernels_per_dev[devidx]) continue;
      else{
        autotune_vals->Subkernel_dev_id_list[devidx][dev_sk_ctr_list[devidx]] = total_sk_ctr;
        dev_sk_ctr_list[devidx]++;
        total_sk_ctr++;
      }
      while(total_sk_ctr%Chunk_size!=0){
        if(total_sk_ctr == Subkernel_num) break;
        else if(dev_sk_ctr_list[devidx] == autotune_vals->Subkernels_per_dev[devidx]) break;
        else{
          autotune_vals->Subkernel_dev_id_list[devidx][dev_sk_ctr_list[devidx]] = total_sk_ctr;
          dev_sk_ctr_list[devidx]++;
          total_sk_ctr++;
        }
      }
      if(total_sk_ctr == Subkernel_num) break;
    }
  }
#ifdef PDEBUG
    lprintf(lvl, "CoCoDistributeSubkernelsRoundRobinChunk:\nDistributing %d Subkernels to %d devices\n",
      Subkernel_num, autotune_vals->dev_num);
    lprintf(lvl, "Device Ids : [ ");
    for (int i =0; i < autotune_vals->dev_num; i++) fprintf(stderr, "%d ", autotune_vals->dev_ids[i]);
    lprintf(0, "]\n");
    lprintf(lvl, "Subker Num : [ ");
    for (int i =0; i < autotune_vals->dev_num; i++) fprintf(stderr, "%d ",
      autotune_vals->Subkernels_per_dev[i]);
    lprintf(0, "]\n");
    for (int i =0; i < autotune_vals->dev_num; i++){
      lprintf(lvl, "Subker Id list for dev_id = %d: [ ", autotune_vals->dev_ids[i]);
      for (int j =0; j < autotune_vals->Subkernels_per_dev[i]; j++) fprintf(stderr, "%d ",
        autotune_vals->Subkernel_dev_id_list[i][j]);
      lprintf(0, "]\n");
    }
#endif
  }
}

void CoCoDistributeSubkernelsRoundRobinChunkReverse(CoControl_p autotune_vals,
  tunableParams_p pred_p, int Subkernel_num, int Chunk_size){
  int lvl = 6;
  if (Subkernel_num <= autotune_vals->dev_num){
    autotune_vals->dev_num = Subkernel_num;
    for (int d = 0 ; d < autotune_vals->dev_num; d++){
      autotune_vals->Subkernels_per_dev[d] = 1;
      autotune_vals->Subkernel_dev_id_list[d][0] = d;
    }
  }
  else{
#ifdef MULTIDEVICE_REDUCTION_ENABLE
  int rem_dev = Subkernel_num;
  for (int d = 0 ; d < autotune_vals->dev_num; d++){
     autotune_vals->Subkernels_per_dev[d] =
      (int) (1.0* pred_p->rel_dev_score[d]* Subkernel_num);
     rem_dev-= autotune_vals->Subkernels_per_dev[d];
  }
  while(rem_dev!= 0){
    for (int d = 0 ; d < autotune_vals->dev_num; d++){
       if(rem_dev!= 0){
         autotune_vals->Subkernels_per_dev[d] += 1;
         rem_dev--;
       }
       else break;
    }
  }
#else
      error("CoCoDistributeSubkernelsRoundRobinChunkReverse: not implemented for undefined MULTIDEVICE_REDUCTION_ENABLE\n");
#endif
  int total_sk_ctr = 0, total_sk_prev = 0;
  int dev_sk_ctr_list[autotune_vals->dev_num] = {0};
  while(total_sk_ctr<Subkernel_num){

    for(int devidx = 0; devidx < autotune_vals->dev_num; devidx++){
      total_sk_prev = total_sk_ctr;
      if(total_sk_ctr == Subkernel_num) break;
      else if(dev_sk_ctr_list[devidx] == autotune_vals->Subkernels_per_dev[devidx]) continue;
      else{
        autotune_vals->Subkernel_dev_id_list[devidx][dev_sk_ctr_list[devidx]] = total_sk_ctr;
        dev_sk_ctr_list[devidx]++;
        total_sk_ctr++;
      }
      while(total_sk_ctr%Chunk_size!=0){
        if(total_sk_ctr == Subkernel_num) break;
        else if(dev_sk_ctr_list[devidx] == autotune_vals->Subkernels_per_dev[devidx]) break;
        else{
          autotune_vals->Subkernel_dev_id_list[devidx][dev_sk_ctr_list[devidx]] = total_sk_ctr;
          dev_sk_ctr_list[devidx]++;
          total_sk_ctr++;
        }
      }
      if (devidx%2 == 0){
        for(int local_ctr = dev_sk_ctr_list[devidx] - total_sk_ctr + total_sk_prev; local_ctr < dev_sk_ctr_list[devidx]; local_ctr++){
          if (local_ctr < dev_sk_ctr_list[devidx] - local_ctr - 1){
            int temp_sk_id = autotune_vals->Subkernel_dev_id_list[devidx][local_ctr];
            autotune_vals->Subkernel_dev_id_list[devidx][local_ctr] =
              autotune_vals->Subkernel_dev_id_list[devidx]
                [dev_sk_ctr_list[devidx] - local_ctr - 1];
            autotune_vals->Subkernel_dev_id_list[devidx]
              [dev_sk_ctr_list[devidx] - local_ctr - 1] = temp_sk_id;
          }
          else break;
        }
      }
      if(total_sk_ctr == Subkernel_num) break;
    }
  }
/*  for(int devidx = 0; devidx < autotune_vals->dev_num; devidx++){
    for(int local_ctr = 0; local_ctr < autotune_vals->Subkernels_per_dev[devidx]; local_ctr++){
      if (local_ctr < autotune_vals->Subkernels_per_dev[devidx] - local_ctr - 1){
        int temp_sk_id = autotune_vals->Subkernel_dev_id_list[devidx][local_ctr];
        autotune_vals->Subkernel_dev_id_list[devidx][local_ctr] =
          autotune_vals->Subkernel_dev_id_list[devidx]
            [autotune_vals->Subkernels_per_dev[devidx] - local_ctr - 1];
        autotune_vals->Subkernel_dev_id_list[devidx]
          [autotune_vals->Subkernels_per_dev[devidx] - local_ctr - 1] = temp_sk_id;
      }
      else break;
    }
  }
  */
#ifdef PDEBUG
    lprintf(lvl, "CoCoDistributeSubkernelsRoundRobinChunkReverse:\nDistributing %d Subkernels to %d devices\n",
      Subkernel_num, autotune_vals->dev_num);
    lprintf(lvl, "Device Ids : [ ");
    for (int i =0; i < autotune_vals->dev_num; i++) fprintf(stderr, "%d ", autotune_vals->dev_ids[i]);
    lprintf(0, "]\n");
    lprintf(lvl, "Subker Num : [ ");
    for (int i =0; i < autotune_vals->dev_num; i++) fprintf(stderr, "%d ",
      autotune_vals->Subkernels_per_dev[i]);
    lprintf(0, "]\n");
    for (int i =0; i < autotune_vals->dev_num; i++){
      lprintf(lvl, "Subker Id list for dev_id = %d: [ ", autotune_vals->dev_ids[i]);
      for (int j =0; j < autotune_vals->Subkernels_per_dev[i]; j++) fprintf(stderr, "%d ",
        autotune_vals->Subkernel_dev_id_list[i][j]);
      lprintf(0, "]\n");
    }
#endif
  }
}
