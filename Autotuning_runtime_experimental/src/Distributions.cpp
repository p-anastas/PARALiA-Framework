///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The possible subkernel distributions to different execution units.
///

#include "unihelpers.hpp"
#include "CoCoPeLiaModel.hpp"

void CoCoDistributeSubkernelsRoundRobin(CoControl_p autotune_vals,
  tunableParams_p best_pred_p, int MGridSz, int NGridSz, int KGridSz){
  int lvl = 6;
  warning("CoCoDistributeSubkernelsRoundRobin: not updated for best_pred_p addition\n");
  int Subkernel_num = MGridSz*NGridSz*KGridSz;
  if (Subkernel_num <= autotune_vals->dev_num){
    autotune_vals->dev_num = Subkernel_num;
    for (int d = 0 ; d < autotune_vals->dev_num; d++){
      autotune_vals->Subkernels_per_dev[d] = 1;
      autotune_vals->Subkernel_dev_id_list[d*Subkernel_num] = d;
    }
  }
  else{
    int total_sk_ctr = 0;
#ifdef MULTIDEVICE_REDUCTION_ENABLE
  while(total_sk_ctr<Subkernel_num){
    for(int devidx = 0; devidx < autotune_vals->dev_num; devidx++){
      if(total_sk_ctr == Subkernel_num) break;
      else{
        autotune_vals->Subkernel_dev_id_list[devidx*Subkernel_num +
          autotune_vals->Subkernels_per_dev[devidx]] = total_sk_ctr;
        autotune_vals->Subkernels_per_dev[devidx]++;
        total_sk_ctr++;
      }
    }
  }
#else
      error("CoCoDistributeSubkernelsRoundRobin: not implemented for undefined MULTIDEVICE_REDUCTION_ENABLE\n");
#endif
  }
}

void CoCoDistributeSubkernelsNaive(CoControl_p autotune_vals,
  tunableParams_p pred_p, int MGridSz, int NGridSz, int KGridSz){
  int lvl = 6;
  int Subkernel_num = MGridSz*NGridSz*KGridSz;
  if (Subkernel_num <= autotune_vals->dev_num){
    autotune_vals->dev_num = Subkernel_num;
    for (int d = 0 ; d < autotune_vals->dev_num; d++){
      autotune_vals->Subkernels_per_dev[d] = 1;
      autotune_vals->Subkernel_dev_id_list[d*Subkernel_num] = d;
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
   /* int skitt;
    for(skitt = 0; skitt < dev_offset*autotune_vals->dev_num; skitt++){
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
      if(cur_dev == autotune_vals->dev_num) warning("CoCoDistributeSubkernelsNaive: Weird modulo, probably bug\n");
    }*/
    short dev_sk_ctr = 0, cur_dev_id_ctr = 0;
    while(total_sk_ctr<Subkernel_num && cur_dev_id_ctr < autotune_vals->dev_num){
      while(dev_sk_ctr == autotune_vals->Subkernels_per_dev[cur_dev_id_ctr]){
        dev_sk_ctr = 0;
        cur_dev_id_ctr++;
      }
      autotune_vals->Subkernel_dev_id_list[cur_dev_id_ctr*Subkernel_num + dev_sk_ctr] = total_sk_ctr;
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
        autotune_vals->Subkernel_dev_id_list[i*Subkernel_num + j]);
      lprintf(0, "]\n");
    }
  #endif
}

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
  			tunableParams_p pred_p[autotune_vals->dev_num];
  			for (int d = 0 ; d < autotune_vals->dev_num; d++) if (dev_ids[d]!= -1){
  				model = CoCoPeLiaModelInit(dev_ids[d], "Dgemm", 'X', TransA, TransB,
  					M/autotune_vals->dev_num, N, K,
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
  			if (predef_vals->dev_num < 0 && autotune_vals->dev_num > 1 && dev_ids[0] == 0) {
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
  					autotune_vals->dev_num = 1;
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

    int Subkernel_dev_id_list[autotune_vals->dev_num*Subkernel_num] = {-1}, Subkernels_per_dev[autotune_vals->dev_num] = {0};
    if (!strcmp(DISTRIBUTION, "ROUND-ROBIN"))
      CoCoDistributeSubkernelsRoundRobin(Subkernel_dev_id_list, Subkernels_per_dev, autotune_vals->dev_num, MGridSz, NGridSz, KGridSz);
    else if (!strcmp(DISTRIBUTION, "SPLITD1-NAIVE"))
      CoCoDistributeSubkernelsNaive(Subkernel_dev_id_list, Subkernels_per_dev, autotune_vals->dev_num, MGridSz, NGridSz, KGridSz);
    else error("CoCopeLiaDgemm: Unknown Subkernel Distribution %s\n", DISTRIBUTION);
  }
  */
}
