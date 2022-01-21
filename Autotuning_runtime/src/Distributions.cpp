///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The possible subkernel distributions to different execution units.
///

#include "unihelpers.hpp"

void CoCoDistributeSubkernelsRoundRobin(int* Subkernel_dev_id_list,
  int* Subkernels_per_dev, short num_devices, int MGridSz, int NGridSz, int KGridSz){
  int lvl = 6;
  int Subkernel_num = MGridSz*NGridSz*KGridSz;
  if (Subkernel_num <= num_devices){
    num_devices = Subkernel_num;
    for (int d = 0 ; d < num_devices; d++){
      Subkernels_per_dev[d] = 1;
      Subkernel_dev_id_list[d*Subkernel_num] = d;
    }
  }
  else{
    int total_sk_ctr = 0;
#ifdef MULTIDEVICE_REDUCTION_ENABLE
  while(total_sk_ctr<Subkernel_num){
    for(int devidx = 0; devidx < num_devices; devidx++){
      if(total_sk_ctr == Subkernel_num) break;
      else{
        Subkernel_dev_id_list[devidx*Subkernel_num + Subkernels_per_dev[devidx]] = total_sk_ctr;
        Subkernels_per_dev[devidx]++;
        total_sk_ctr++;
      }
    }
  }
#else
      error("CoCoDistributeSubkernelsRoundRobin: not implemented for undefined MULTIDEVICE_REDUCTION_ENABLE\n");
#endif
  }
}

void CoCoDistributeSubkernelsNaive(int* Subkernel_dev_id_list,
  int* Subkernels_per_dev, short num_devices, int MGridSz, int NGridSz, int KGridSz){
  int lvl = 6;
  int Subkernel_num = MGridSz*NGridSz*KGridSz;
  if (Subkernel_num <= num_devices){
    num_devices = Subkernel_num;
    for (int d = 0 ; d < num_devices; d++){
      Subkernels_per_dev[d] = 1;
      Subkernel_dev_id_list[d*Subkernel_num] = d;
    }
  }
  else{
    int total_sk_ctr = 0;
    int dev_offset;
#ifdef MULTIDEVICE_REDUCTION_ENABLE
    dev_offset = Subkernel_num/num_devices;
    if (dev_offset);
    else dev_offset = Subkernel_num;
#else
    dev_offset = (MGridSz*NGridSz)/num_devices*KGridSz;
    if (dev_offset);
    else error("CoCoDistributeSubkernelsNaive: dev_offset = 0 undefined without MULTIDEVICE_REDUCTION_ENABLE");
  #endif
  #ifdef DEBUG
    lprintf(lvl, "Subkernel Split offset = %d\n", dev_offset);
  #endif
   /* int skitt;
    for(skitt = 0; skitt < dev_offset*num_devices; skitt++){
      Subkernel_dev_id_list[skitt/dev_offset*Subkernel_num + Subkernels_per_dev[skitt/dev_offset]]
        = skitt/dev_offset*dev_offset + Subkernels_per_dev[skitt/dev_offset];
      Subkernels_per_dev[skitt/dev_offset]++;
    }
    int cur_dev = 0;
    while(skitt<Subkernel_num){
      Subkernel_dev_id_list[cur_dev*Subkernel_num + Subkernels_per_dev[cur_dev]]
        = cur_dev*dev_offset + Subkernels_per_dev[cur_dev];
      Subkernels_per_dev[cur_dev]++;
      cur_dev++;
      skitt++;
      if(cur_dev == num_devices) warning("CoCoDistributeSubkernelsNaive: Weird modulo, probably bug\n");
    }*/

    while(total_sk_ctr<Subkernel_num){
      if(total_sk_ctr<dev_offset){
        Subkernel_dev_id_list[0*Subkernel_num + Subkernels_per_dev[0]] = 0*dev_offset + Subkernels_per_dev[0];
        Subkernels_per_dev[0]++;
      }
      else{
        Subkernel_dev_id_list[1*Subkernel_num + Subkernels_per_dev[1]] = 1*dev_offset + Subkernels_per_dev[1];
        Subkernels_per_dev[1]++;
      }
      total_sk_ctr++;
    }
  }
}

void CoCoDistributeSubkernelsRevLast(int* Subkernel_dev_id_list,
  int* Subkernels_per_dev, short num_devices, int MGridSz, int NGridSz, int KGridSz){
  int lvl = 6;
  int Subkernel_num = MGridSz*NGridSz*KGridSz;
  if (Subkernel_num <= num_devices){
    num_devices = Subkernel_num;
    for (int d = 0 ; d < num_devices; d++){
      Subkernels_per_dev[d] = 1;
      Subkernel_dev_id_list[d*Subkernel_num] = d;
    }
  }
  else{
    int total_sk_ctr = 0;
    int dev_offset;
#ifdef MULTIDEVICE_REDUCTION_ENABLE
    dev_offset = Subkernel_num/num_devices;
    if (dev_offset);
    else dev_offset = Subkernel_num;
#else
    dev_offset = (MGridSz*NGridSz)/num_devices*KGridSz;
    if (dev_offset);
    else error("CoCoDistributeSubkernelsNaive: dev_offset = 0 undefined without MULTIDEVICE_REDUCTION_ENABLE");
  #endif
  #ifdef DEBUG
    lprintf(lvl, "Subkernel Split offset = %d\n", dev_offset);
  #endif
    while(total_sk_ctr<Subkernel_num){
      if(total_sk_ctr<dev_offset){
        Subkernel_dev_id_list[0*Subkernel_num + Subkernels_per_dev[0]] = 0*dev_offset + Subkernels_per_dev[0];
        Subkernels_per_dev[0]++;
      }
      else{
        Subkernel_dev_id_list[1*Subkernel_num + Subkernels_per_dev[1]] = 1*dev_offset + Subkernels_per_dev[1];
        Subkernels_per_dev[1]++;
      }
      total_sk_ctr++;
    }
  }
}
