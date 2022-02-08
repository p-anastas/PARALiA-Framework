///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The possible subkernel distributions to different execution units.
///

#include "unihelpers.hpp"
#include "CoCoPeLiaModel.hpp"

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

#ifndef COCONTROL_H
typedef struct CoControl{
	int T = 0;
	int dev_num = -1;
	int dev_ids[LOC_NUM];
	int Subkernels_per_dev[LOC_NUM];
	int *Subkernel_dev_id_list;
	long long cache_limit = 0;
}* CoControl_p;
#endif

CoControl_p CoCoAutotuneParameters(const char* routine_name, void* initial_problem_wrap,
  CoControl_p predef_vals){
/*
  	/// Read predefined values for T or use Tile selection.
  	/// return: T size for datum
  	size_t T = 256;
  	double slowest_problem_t = 0;
  	CoCoModel_p model = NULL;
  	{
  		if(predef_vals->T <= 0){
  			/// For each asset: find datum dimension, taking into account shared dimensions for the problem (e.g. here M, N, K are shared between two matrices each)
  			/// 1) Ideally we would want a method to find the optimal Tm, Tn, Tk
  			/// 2) Currently only square for 2D (sufficient for CoCoPeLia and BLAS in the general case)
  			/// 3) Interesting point for exploration (how to find datum, performance impact etc.)
  			/// 4)  Basically its the tile selection part of CoCoPeLia, but for multiple devices.

  			/// Naive for multiple equivalent devices.
  			int slowest_problem_T = std::min((size_t) 1024, std::min((size_t) M, (size_t)std::min(N, K)));
  			tunableParams_p pred_p[num_devices];
  			for (int d = 0 ; d < num_devices; d++) if (dev_ids[d]!= -1){
  				model = CoCoPeLiaModelInit(dev_ids[d], "Dgemm", 'X', TransA, TransB,
  					M/num_devices, N, K,
  					(CoCoGetPtrLoc(A) == dev_ids[d])? 0 : 1, (CoCoGetPtrLoc(B) == dev_ids[d])? 0 : 1,
  					(CoCoGetPtrLoc(C) == dev_ids[d])? 0 : 1, (CoCoGetPtrLoc(A) == dev_ids[d])? 0 : 1,
  					(CoCoGetPtrLoc(B) == dev_ids[d])? 0 : 1, (CoCoGetPtrLoc(C) == dev_ids[d])? 0 : 1,
  					ldA, ldB, ldC);
  #ifdef TEST
  				cpu_timer = csecond() - cpu_timer;
  				lprintf(lvl, "Model Initialization(dev = %d): t_mod_init = %lf ms\n", dev_ids[d], cpu_timer*1000);
  				cpu_timer = csecond();
  #endif

  				pred_p[d] = CoCoPeLiaModelOptimizeTile(model, COCOPELIA_PIPELINE_EMULATE);
  				if (pred_p[d]->pred_t > slowest_problem_t){
  					slowest_problem_t = pred_p[d]->pred_t;
  					slowest_problem_T = pred_p[d]->T;
  				}
  #ifdef TEST
  				cpu_timer = csecond() - cpu_timer;
  				lprintf(lvl, "Model Selected T=%zu for dev = %d with t_predicted = %lf ms : t_mod_opt = %lf ms\n", pred_p[d]->T, dev_id[d], pred_p[d]->pred_t*1000, cpu_timer*1000);
  				cpu_timer = csecond();
  #endif

  			}
  			/// Extra: check if running in multiple GPUs seems to have a point performance-wise.
  			/// Currently only comparing single vs multi GPU
  			/// Can be extended to complex (e.g. 1 vs 2 vs 3 etc)
  			if (predef_vals->dev_num < 0 && num_devices > 1 && dev_ids[0] == 0) {
  				short best_dev_id = 0;
  			 	model = CoCoPeLiaModelInit(0, "Dgemm", 'X', TransA, TransB, M, N, K,
  				 (CoCoGetPtrLoc(A) == 0)? 0 : 1, (CoCoGetPtrLoc(B) == 0)? 0 : 1,
  				 (CoCoGetPtrLoc(C) == 0)? 0 : 1, (CoCoGetPtrLoc(A) == 0)? 0 : 1,
  				 (CoCoGetPtrLoc(B) == 0)? 0 : 1, (CoCoGetPtrLoc(C) == 0)? 0 : 1,
  				 ldA, ldB, ldC);

  				tunableParams_p pred_p_single_dev = CoCoPeLiaModelOptimizeTile(model, COCOPELIA_PIPELINE_EMULATE);

  #ifdef TEST
  			 cpu_timer = csecond() - cpu_timer;
  			 lprintf(lvl, "Model Selected T=%zu for single-device execution(%d) with t_predicted = %lf ms : t_mod_opt = %lf ms\n", pred_p_single_dev->T, best_dev_id, pred_p_single_dev->pred_t*1000, cpu_timer*1000);
  			 cpu_timer = csecond();
  #endif

  				/// How much performance improvent justifies adding one more GPU?
  				/// Aren't there better metrics for this?
  				if (slowest_problem_t > pred_p_single_dev->pred_t){
  				 	slowest_problem_T = pred_p_single_dev->T;
  				 	warning("Chose to run on only 1 device: Model implies %lf\% better performance\n",
  						(slowest_problem_t - pred_p_single_dev->pred_t)/slowest_problem_t*100);
  					slowest_problem_t = pred_p_single_dev->pred_t;
  					num_devices = 1;
  					dev_ids[0] = best_dev_id;
  			 	}
  			}

  			T = slowest_problem_T;
  #ifdef TEST
  			cpu_timer = csecond() - cpu_timer;
  			lprintf(lvl, "Model Selected T=%zu with t_predicted = %lf ms : t_mod_opt = %lf ms\n", T, slowest_problem_t*1000, cpu_timer*1000);
  			cpu_timer = csecond();
  #endif

  #ifdef DEBUG
  			lprintf(lvl, "Model Selected T=%zu : t_predicted = %lf ms\n", T, slowest_problem_t*1000);
  			lprintf(lvl, "====================================\n");
  #endif
  		}

  		if(used_vals == NULL) {
  			used_vals = (CoControl_p) malloc(sizeof(struct CoControl));
  			used_vals->dev_ids = NULL;
  		}
  		used_vals->T = T;
  		used_vals->cache_limit = predef_vals->cache_limit;
  	}

    int Subkernel_dev_id_list[num_devices*Subkernel_num] = {-1}, Subkernels_per_dev[num_devices] = {0};
    if (!strcmp(DISTRIBUTION, "ROUND-ROBIN"))
      CoCoDistributeSubkernelsRoundRobin(Subkernel_dev_id_list, Subkernels_per_dev, num_devices, MGridSz, NGridSz, KGridSz);
    else if (!strcmp(DISTRIBUTION, "SPLITD1-NAIVE"))
      CoCoDistributeSubkernelsNaive(Subkernel_dev_id_list, Subkernels_per_dev, num_devices, MGridSz, NGridSz, KGridSz);
    else error("CoCopeLiaDgemm: Unknown Subkernel Distribution %s\n", DISTRIBUTION);
  }
  */
}
