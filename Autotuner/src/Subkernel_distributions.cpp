///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The possible subkernel distributions to different execution units.
///

#include "linkmap.hpp"
#include "Autotuner.hpp"

void CoCoDistributeSubkernelsRoundRobin(ATC_p autotune_controller){
  #ifdef DEBUG
  	fprintf(stderr, "|-----> CoCoDistributeSubkernelsRoundRobin(%p)\n", autotune_controller);
  #endif
  if (autotune_controller->subkernel_num < autotune_controller->active_unit_num){ // < or <= here?
    int pred_active_unit_num = autotune_controller->active_unit_num;
    autotune_controller->active_unit_num = autotune_controller->subkernel_num;
    warning("CoCoDistributeSubkernelsRoundRobin: Problem with predicted active_unit_num(%d) < subkernel_num(%d) will be run with active_unit_num = %d\n",
    	pred_active_unit_num, autotune_controller->subkernel_num, autotune_controller->active_unit_num);
    for (int d = autotune_controller->active_unit_num; d < pred_active_unit_num; d++) autotune_controller->linkmap->reset_links(autotune_controller->active_unit_id_list[d]);
    for (int d = 0 ; d < autotune_controller->active_unit_num; d++){
      autotune_controller->Subkernels_per_unit_num[d] = 1;
      autotune_controller->Subkernels_per_unit_list[d][0] = d;
    }
  }
  else{
    int rem_dev = autotune_controller->subkernel_num;
    for (int d = 0 ; d < autotune_controller->active_unit_num; d++){
       autotune_controller->Subkernels_per_unit_num[d] =
        (int) (1.0* autotune_controller->active_unit_score[d]* autotune_controller->subkernel_num);
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
    while(total_sk_ctr<autotune_controller->subkernel_num){
      for(int devidx = 0; devidx < autotune_controller->active_unit_num; devidx++){
        if(total_sk_ctr == autotune_controller->subkernel_num) break;
        else if(dev_sk_ctr_list[devidx] == autotune_controller->Subkernels_per_unit_num[devidx]) continue;
        else{
          autotune_controller->Subkernels_per_unit_list[devidx][dev_sk_ctr_list[devidx]] = total_sk_ctr;
          dev_sk_ctr_list[devidx]++;
          total_sk_ctr++;
        }
      }
    }
  }
#ifdef PDEBUG
  fprintf(stderr, "CoCoDistributeSubkernelsRoundRobin:\nDistributing %ld Subkernels to %d devices\n",
    autotune_controller->subkernel_num, autotune_controller->active_unit_num);
  fprintf(stderr, "Device Ids : [ ");
  for (int i =0; i < autotune_controller->active_unit_num; i++) fprintf(stderr, "%d ", autotune_controller->active_unit_id_list[i]);
  lprintf(0, "]\n");
  fprintf(stderr, "Subker Num : [ ");
  for (int i =0; i < autotune_controller->active_unit_num; i++) fprintf(stderr, "%d ",
    autotune_controller->Subkernels_per_unit_num[i]);
  lprintf(0, "]\n");
  for (int i =0; i < autotune_controller->active_unit_num; i++){
    fprintf(stderr, "Subker Id list for dev_id = %d: [ ", autotune_controller->active_unit_id_list[i]);
    for (int j =0; j < autotune_controller->Subkernels_per_unit_num[i]; j++) fprintf(stderr, "%d ",
      autotune_controller->Subkernels_per_unit_list[i][j]);
    lprintf(0, "]\n");
  }
#endif
#ifdef DEBUG
	fprintf(stderr, "<-----|\n");
#endif
}

void CoCoDistributeSubkernelsNaive(ATC_p autotune_controller){
  #ifdef DEBUG
  	fprintf(stderr, "|-----> CoCoDistributeSubkernelsNaive(%p)\n", autotune_controller);
  #endif
  if (autotune_controller->subkernel_num <= autotune_controller->active_unit_num){
    int pred_active_unit_num = autotune_controller->active_unit_num;
    autotune_controller->active_unit_num = autotune_controller->subkernel_num;
    warning("CoCoDistributeSubkernelsNaive: Problem with predicted active_unit_num(%d) < subkernel_num(%d) will be run with active_unit_num = %d\n",
    	pred_active_unit_num, autotune_controller->subkernel_num, autotune_controller->active_unit_num);
    for (int d = autotune_controller->active_unit_num; d < pred_active_unit_num; d++) autotune_controller->linkmap->reset_links(autotune_controller->active_unit_id_list[d]);
    for (int d = 0 ; d < autotune_controller->active_unit_num; d++){
      autotune_controller->Subkernels_per_unit_num[d] = 1;
      autotune_controller->Subkernels_per_unit_list[d][0] = d;
    }
  }
  else{
    int total_sk_ctr = 0;
    int rem_dev = autotune_controller->subkernel_num;
    for (int d = 0 ; d < autotune_controller->active_unit_num; d++){
       autotune_controller->Subkernels_per_unit_num[d] =
        (int) (1.0* autotune_controller->active_unit_score[d]* autotune_controller->subkernel_num);
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
    short dev_sk_ctr = 0, cur_dev_id_ctr = 0;
    while(total_sk_ctr<autotune_controller->subkernel_num && cur_dev_id_ctr < autotune_controller->active_unit_num){
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
    fprintf(stderr, "CoCoDistributeSubkernelsNaive:\nDistributing %ld Subkernels to %d devices\n",
      autotune_controller->subkernel_num, autotune_controller->active_unit_num);
    fprintf(stderr, "Device Ids : [ ");
    for (int i =0; i < autotune_controller->active_unit_num; i++) fprintf(stderr, "%d ", autotune_controller->active_unit_id_list[i]);
    lprintf(0, "]\n");
    fprintf(stderr, "Subker Num : [ ");
    for (int i =0; i < autotune_controller->active_unit_num; i++) fprintf(stderr, "%d ",
      autotune_controller->Subkernels_per_unit_num[i]);
    lprintf(0, "]\n");
    for (int i =0; i < autotune_controller->active_unit_num; i++){
      fprintf(stderr, "Subker Id list for dev_id = %d: [ ", autotune_controller->active_unit_id_list[i]);
      for (int j =0; j < autotune_controller->Subkernels_per_unit_num[i]; j++) fprintf(stderr, "%d ",
        autotune_controller->Subkernels_per_unit_list[i][j]);
      lprintf(0, "]\n");
    }
#endif
#ifdef DEBUG
	fprintf(stderr, "<-----|\n");
#endif
}

void CoCoDistributeSubkernelsRoundRobinChunk(ATC_p autotune_controller,  int Chunk_size){
#ifdef DEBUG
  	fprintf(stderr, "|-----> CoCoDistributeSubkernelsRoundRobinChunk(%p, %d)\n", autotune_controller, Chunk_size);
#endif
#ifdef PDEBUG
fprintf(stderr, "CoCoDistributeSubkernelsRoundRobinChunk(%d): Devices = %d (scores = %s), sk_num = %d, sk_buckets = %d\n",
  Chunk_size, autotune_controller->active_unit_num, 
  printlist<double>(autotune_controller->active_unit_score, autotune_controller->active_unit_num),
  autotune_controller->subkernel_num, autotune_controller->subkernel_num/Chunk_size);
#endif
  if (autotune_controller->subkernel_num/Chunk_size <= autotune_controller->active_unit_num){
    int pred_active_unit_num = autotune_controller->active_unit_num;
    autotune_controller->active_unit_num = autotune_controller->subkernel_num/Chunk_size;
    warning("CoCoDistributeSubkernelsRoundRobinChunk: Problem with predicted active_unit_num(%d) < subkernel_num(%d) will be run with active_unit_num = %d\n",
    	pred_active_unit_num, autotune_controller->subkernel_num, autotune_controller->active_unit_num);
    for (int d = autotune_controller->active_unit_num; d < pred_active_unit_num; d++) autotune_controller->linkmap->reset_links(autotune_controller->active_unit_id_list[d]);
    for (int d = 0 ; d < autotune_controller->active_unit_num; d++){
      autotune_controller->Subkernels_per_unit_num[d] = Chunk_size;
      for (int idx3 = 0; idx3 < Chunk_size; idx3++)
        autotune_controller->Subkernels_per_unit_list[d][idx3] = d*Chunk_size + idx3;
    }
  }
  else{
  for (int d = 0 ; d < autotune_controller->active_unit_num; d++){
    autotune_controller->Subkernels_per_unit_num[d] = Chunk_size*
      (int) (1.0* autotune_controller->active_unit_score[d]* (autotune_controller->subkernel_num/Chunk_size));
  }
  int sks_accounted_for = 0;
  for (int d = 0 ; d < autotune_controller->active_unit_num; d++)
    sks_accounted_for += autotune_controller->Subkernels_per_unit_num[d];
#ifdef PDEBUG
fprintf(stderr, "Assigned kernel num to devices kernels (first pass): %s\n",
  printlist<int>(autotune_controller->Subkernels_per_unit_num, autotune_controller->active_unit_num));
#endif
  int sk_ctr = 0, dev_sk_ctr_list[autotune_controller->active_unit_num] = {0}, devidx = 0;
  for (int D1 = 0; D1 < autotune_controller->subkernel_num/Chunk_size; D1++){
    for (int D3 = 0; D3 < Chunk_size; D3++){
      int full_circle = autotune_controller->active_unit_num;
      while(dev_sk_ctr_list[devidx] == autotune_controller->Subkernels_per_unit_num[devidx] && sk_ctr < sks_accounted_for){ 
        if(!full_circle) error("CoCoDistributeSubkernels2DBlockCyclic: would enter infinite loop due to wrong Subkernels_per_unit_num, terminating\n");
        if(devidx == autotune_controller->active_unit_num - 1) devidx = 0;
        else devidx++;
        full_circle--;
      }
      autotune_controller->Subkernels_per_unit_list[devidx][dev_sk_ctr_list[devidx]] = sk_ctr;
      if(sk_ctr >= sks_accounted_for) autotune_controller->Subkernels_per_unit_num[devidx] ++;
      dev_sk_ctr_list[devidx]++;
#ifdef PDEBUG
      fprintf(stderr, "CoCoDistributeSubkernelsRoundRobinChunk: sk_ctr[%d,%d] = %d, devidx = %d\n",
        D1, D3, sk_ctr, devidx);
#endif
      sk_ctr++;
    }
    if(devidx == autotune_controller->active_unit_num - 1) devidx = 0;
    else devidx++;
  }
  }
#ifdef PDEBUG
    fprintf(stderr, "CoCoDistributeSubkernelsRoundRobinChunk:\nDistributing %ld Subkernels to %d devices\n",
      autotune_controller->subkernel_num, autotune_controller->active_unit_num);
    fprintf(stderr, "Device Ids : [ ");
    for (int i =0; i < autotune_controller->active_unit_num; i++) fprintf(stderr, "%d ", autotune_controller->active_unit_id_list[i]);
    lprintf(0, "]\n");
    fprintf(stderr, "Subker Num : [ ");
    for (int i =0; i < autotune_controller->active_unit_num; i++) fprintf(stderr, "%d ",
      autotune_controller->Subkernels_per_unit_num[i]);
    lprintf(0, "]\n");
    for (int i =0; i < autotune_controller->active_unit_num; i++){
      fprintf(stderr, "Subker Id list for dev_id = %d: [ ", autotune_controller->active_unit_id_list[i]);
      for (int j =0; j < autotune_controller->Subkernels_per_unit_num[i]; j++) fprintf(stderr, "%d ",
        autotune_controller->Subkernels_per_unit_list[i][j]);
      lprintf(0, "]\n");
    }
#endif
#ifdef DEBUG
	fprintf(stderr, "<-----|\n");
#endif
}

void CoCoDistributeSubkernelsRoundRobinChunkReverse(ATC_p autotune_controller,  int Chunk_size){
  #ifdef DEBUG
  	fprintf(stderr, "|-----> CoCoDistributeSubkernelsRoundRobinChunkReverse(%p, %d)\n", autotune_controller, Chunk_size);
  #endif
  if (autotune_controller->subkernel_num <= autotune_controller->active_unit_num){
    int pred_active_unit_num = autotune_controller->active_unit_num;
    autotune_controller->active_unit_num = autotune_controller->subkernel_num;
    warning("CoCoDistributeSubkernelsRoundRobinChunkReverse: Problem with predicted active_unit_num(%d) < subkernel_num(%d) will be run with active_unit_num = %d\n",
    	pred_active_unit_num, autotune_controller->subkernel_num, autotune_controller->active_unit_num);
    for (int d = autotune_controller->active_unit_num; d < pred_active_unit_num; d++) autotune_controller->linkmap->reset_links(autotune_controller->active_unit_id_list[d]);
    for (int d = 0 ; d < autotune_controller->active_unit_num; d++){
      autotune_controller->Subkernels_per_unit_num[d] = 1;
      autotune_controller->Subkernels_per_unit_list[d][0] = d;
    }
  }
  else{
    int rem_dev = autotune_controller->subkernel_num;
    for (int d = 0 ; d < autotune_controller->active_unit_num; d++){
       autotune_controller->Subkernels_per_unit_num[d] =
        (int) (1.0* autotune_controller->active_unit_score[d]* autotune_controller->subkernel_num);
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
    while(total_sk_ctr<autotune_controller->subkernel_num){

      for(int devidx = 0; devidx < autotune_controller->active_unit_num; devidx++){
        total_sk_prev = total_sk_ctr;
        if(total_sk_ctr == autotune_controller->subkernel_num) break;
        else if(dev_sk_ctr_list[devidx] == autotune_controller->Subkernels_per_unit_num[devidx]) continue;
        else{
          autotune_controller->Subkernels_per_unit_list[devidx][dev_sk_ctr_list[devidx]] = total_sk_ctr;
          dev_sk_ctr_list[devidx]++;
          total_sk_ctr++;
        }
        while(total_sk_ctr%Chunk_size!=0){
          if(total_sk_ctr == autotune_controller->subkernel_num) break;
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
        if(total_sk_ctr == autotune_controller->subkernel_num) break;
      }
    }
  }
#ifdef PDEBUG
  fprintf(stderr, "CoCoDistributeSubkernelsRoundRobinChunkReverse:\nDistributing %ld Subkernels to %d devices\n",
    autotune_controller->subkernel_num, autotune_controller->active_unit_num);
  fprintf(stderr, "Device Ids : [ ");
  for (int i =0; i < autotune_controller->active_unit_num; i++) fprintf(stderr, "%d ", autotune_controller->active_unit_id_list[i]);
  lprintf(0, "]\n");
  fprintf(stderr, "Subker Num : [ ");
  for (int i =0; i < autotune_controller->active_unit_num; i++) fprintf(stderr, "%d ",
    autotune_controller->Subkernels_per_unit_num[i]);
  lprintf(0, "]\n");
  for (int i =0; i < autotune_controller->active_unit_num; i++){
    fprintf(stderr, "Subker Id list for dev_id = %d: [ ", autotune_controller->active_unit_id_list[i]);
    for (int j =0; j < autotune_controller->Subkernels_per_unit_num[i]; j++) fprintf(stderr, "%d ",
      autotune_controller->Subkernels_per_unit_list[i][j]);
    lprintf(0, "]\n");
  }
#endif
#ifdef DEBUG
	fprintf(stderr, "<-----|\n");
#endif
}

void CoCoDistributeSubkernels2DBlockCyclic(ATC_p autotune_controller, int D1GridSz, int D2GridSz, int D3GridSz){
#ifdef DEBUG
  	fprintf(stderr, "|-----> CoCoDistributeSubkernels2DBlockCyclic(%p, %d, %d, %d)\n", autotune_controller, D1GridSz, D2GridSz, D3GridSz);
#endif
  if ((D2GridSz == D3GridSz) &&  (D2GridSz == 1)){
    warning("CoCoDistributeSubkernels2DBlockCyclic: D2GridSz==D3GridSz==1 -> using CoCoDistributeSubkernelsRoundRobin\n");
    return CoCoDistributeSubkernelsRoundRobin(autotune_controller);
  }

/* 2D Block cyclic */
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
  //TODO: reverse layout
  //int tmp = D1_parts;
  //D1_parts = D2_parts;
  //D2_parts = tmp;
  if(D1GridSz < D1_parts || D2GridSz < D2_parts){
    warning("CoCoDistributeSubkernels2DBlockCyclic:\nGrid(%d,%d) smaller than {D1,D2}_parts = (%d,%d)\
    using CoCoDistributeSubkernelsRoundRobinChunk instead\n", D1GridSz, D2GridSz, D1_parts, D2_parts);
    CoCoDistributeSubkernelsRoundRobinChunk(autotune_controller, D3GridSz);
    return;
  }
  int D1GridSz_div = D1GridSz/D1_parts*D1_parts, D2GridSz_div = D2GridSz/D2_parts*D2_parts,
      D1GridSz_mod = D1GridSz%D1_parts, D2GridSz_mod = D2GridSz%D2_parts;
#ifdef PDEBUG
fprintf(stderr, "CoCoDistributeSubkernels2DBlockCyclic(%d, %d, %d): Devices = %d (scores = %s), D1_parts = %d, D2_parts = %d\n",
  D1GridSz, D2GridSz, D3GridSz, autotune_controller->active_unit_num, 
  printlist<double>(autotune_controller->active_unit_score, autotune_controller->active_unit_num),D1_parts, D2_parts);
#endif

  if ((D1GridSz*D2GridSz) < autotune_controller->active_unit_num){
    warning("CoCoDistributeSubkernels2DBlockCyclic: D1GridSz*D2GridSz(%d) < autotune_controller->active_unit_num(%d)\n", 
      D1GridSz*D2GridSz, autotune_controller->active_unit_num);
    int pred_active_unit_num = D1GridSz*D2GridSz;
    autotune_controller->active_unit_num = autotune_controller->subkernel_num;
    warning("CoCoDistributeSubkernels2DBlockCyclic: Problem with predicted active_unit_num(%d) < subkernel_num(%d) will be run with active_unit_num = %d\n",
    	pred_active_unit_num, autotune_controller->subkernel_num, autotune_controller->active_unit_num);
    for (int d = autotune_controller->active_unit_num; d < pred_active_unit_num; d++) autotune_controller->linkmap->reset_links(autotune_controller->active_unit_id_list[d]);
    for (int d = 0 ; d < autotune_controller->active_unit_num; d++){
      autotune_controller->Subkernels_per_unit_num[d] = D3GridSz;
      for (int idx3 = 0; idx3 < D3GridSz; idx3++)
        autotune_controller->Subkernels_per_unit_list[d][idx3] = d*D3GridSz + idx3;
    }
  }
  else{
  int sks_accounted_for = 0;
  for (int d = 0 ; d < autotune_controller->active_unit_num; d++){
     autotune_controller->Subkernels_per_unit_num[d] = D3GridSz * (
      (int) (autotune_controller->active_unit_score[d]* D1GridSz_div*D2GridSz_div));
      /// TODO: this is a fix because int (some_double) does not work for all doubles as intended 
      /// Will disrupt non-homogeneous splits!
      sks_accounted_for+= autotune_controller->Subkernels_per_unit_num[d]; 
  }
  if(!D1GridSz_mod && !D2GridSz_mod && sks_accounted_for < autotune_controller->subkernel_num){
    warning("CoCoDistributeSubkernels2DBlockCyclic: Questionable remainder from first pass %d / %d sub-kernels\n",
      autotune_controller->subkernel_num - sks_accounted_for, autotune_controller->subkernel_num);
    int buckets =  D1GridSz_div*D2GridSz_div, 
        buckets_rem = (autotune_controller->subkernel_num - sks_accounted_for)/D3GridSz,
        buckets_intended = buckets/autotune_controller->active_unit_num; 
    for (int d = 0 ; d < autotune_controller->active_unit_num; d++) 
      if(autotune_controller->Subkernels_per_unit_num[d]/D3GridSz < buckets_intended && buckets_rem){
        autotune_controller->Subkernels_per_unit_num[d]+= D3GridSz; 
        buckets_rem--;
      }
  }
#ifdef PDEBUG
fprintf(stderr, "Assigned kernel num to devices kernels (first pass): %s\n",
  printlist<int>(autotune_controller->Subkernels_per_unit_num, autotune_controller->active_unit_num));
#endif
  int sk_ctr, dev_sk_ctr_list[autotune_controller->active_unit_num] = {0}, devidx = 0;
  for (int D1 = 0; D1 < D1GridSz_div; D1++)
    for (int D2 = 0; D2 < D2GridSz_div; D2++)
        for (int D3 = 0; D3 < D3GridSz; D3++){
          sk_ctr = D1*D2GridSz*D3GridSz + D2*D3GridSz+D3;
          devidx = D1/(D1GridSz/D1_parts)*D2_parts + D2/(D2GridSz/D2_parts);
#ifdef PDEBUG
          fprintf(stderr, "CoCoDistributeSubkernels2DBlockCyclic: sk_ctr[%d,%d,%d] = %d, devidx = %d\n",
            D1,D2,D3, sk_ctr, devidx);
#endif
          int full_circle = autotune_controller->active_unit_num; 
          while(dev_sk_ctr_list[devidx] == autotune_controller->Subkernels_per_unit_num[devidx]){
            if(!full_circle) error("CoCoDistributeSubkernels2DBlockCyclic: would enter infinite loop due to wrong Subkernels_per_unit_num, terminating\n");
            if(devidx == autotune_controller->active_unit_num - 1) devidx = 0;
            else devidx++;
            full_circle--; 
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
        fprintf(stderr, "CoCoDistributeSubkernels2DBlockCyclic: D1-mod part\n sk_ctr[%d,%d,%d] = %d, devidx = %d\n",
          D1,D2,D3, sk_ctr, devidx);
#endif
        autotune_controller->Subkernels_per_unit_list[devidx][dev_sk_ctr_list[devidx]] = sk_ctr;
        dev_sk_ctr_list[devidx]++;
      }
      autotune_controller->Subkernels_per_unit_num[devidx] += D3GridSz;
      if(devidx == autotune_controller->active_unit_num - 1) devidx = 0;
      else devidx++;
    }

  for (int D1 = D1GridSz_div; D1 < D1GridSz_div + D1GridSz_mod; D1++)
    for (int D2 = 0; D2 < D2GridSz_div; D2++){
      for (int D3 = 0; D3 < D3GridSz; D3++){
        sk_ctr = D1*D2GridSz*D3GridSz + D2*D3GridSz+D3;
#ifdef PDEBUG
        fprintf(stderr, "CoCoDistributeSubkernels2DBlockCyclic: D2-mod part\nsk_ctr[%d,%d,%d] = %d, devidx = %d\n",
          D1,D2,D3, sk_ctr, devidx);
#endif
        autotune_controller->Subkernels_per_unit_list[devidx][dev_sk_ctr_list[devidx]] = sk_ctr;
        dev_sk_ctr_list[devidx]++;
      }
      autotune_controller->Subkernels_per_unit_num[devidx] += D3GridSz;
      if(devidx == autotune_controller->active_unit_num - 1) devidx = 0;
      else devidx++;
    }

  for (int D1 = D1GridSz_div; D1 < D1GridSz_div + D1GridSz_mod; D1++)
    for (int D2 = D2GridSz_div; D2 < D2GridSz_div + D2GridSz_mod; D2++){
        for (int D3 = 0; D3 < D3GridSz; D3++){
          sk_ctr = D1*D2GridSz*D3GridSz + D2*D3GridSz+D3;
#ifdef PDEBUG
          fprintf(stderr, "CoCoDistributeSubkernels2DBlockCyclic: D1 & D2 mod part\nsk_ctr[%d,%d,%d] = %d, devidx = %d\n",
            D1,D2,D3, sk_ctr, devidx);
#endif
        autotune_controller->Subkernels_per_unit_list[devidx][dev_sk_ctr_list[devidx]] = sk_ctr;
        dev_sk_ctr_list[devidx]++;
      }
      autotune_controller->Subkernels_per_unit_num[devidx] += D3GridSz;
      if(devidx == autotune_controller->active_unit_num - 1) devidx = 0;
      else devidx++;
      }
    }
#ifdef PDEBUG
  fprintf(stderr, "CoCoDistributeSubkernels2DBlockCyclic:\nDistributing %ld Subkernels to %d devices\n",
    autotune_controller->subkernel_num, autotune_controller->active_unit_num);
  fprintf(stderr, "Device Ids : [ ");
  for (int i =0; i < autotune_controller->active_unit_num; i++) fprintf(stderr, "%d ", autotune_controller->active_unit_id_list[i]);
  lprintf(0, "]\n");
  fprintf(stderr, "Subker Num : [ ");
  for (int i =0; i < autotune_controller->active_unit_num; i++) fprintf(stderr, "%d ",
    autotune_controller->Subkernels_per_unit_num[i]);
  lprintf(0, "]\n");
  for (int i =0; i < autotune_controller->active_unit_num; i++){
    fprintf(stderr, "Subker Id list for dev_id = %d: [ ", autotune_controller->active_unit_id_list[i]);
    for (int j =0; j < autotune_controller->Subkernels_per_unit_num[i]; j++) fprintf(stderr, "%d ",
      autotune_controller->Subkernels_per_unit_list[i][j]);
    lprintf(0, "]\n");
  }
#endif
#ifdef DEBUG
	fprintf(stderr, "<-----|\n");
#endif

}
