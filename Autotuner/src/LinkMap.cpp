
#include <iostream>

#include "Autotuning_runtime.hpp"
#include "Model_functions.hpp"
#include "unihelpers.hpp"

LinkMap::LinkMap(){
	for (int i = 0; i < LOC_NUM*LOC_NUM; i++) ESPA_ETA_sorted_dec_ids[i] = i;
}

void LinkMap::copy(class LinkMap* other_linkmap)
{
  for (int i = 0; i < LOC_NUM; i++)
    for (int j = 0; j < LOC_NUM; j++){
      link_lat[i][j] = other_linkmap->link_lat[i][j];
      link_bw[i][j] = other_linkmap->link_bw[i][j];
      link_bw_shared[i][j] = other_linkmap->link_bw_shared[i][j];
      link_bw_shared_hops[i][j] = other_linkmap->link_bw_shared_hops[i][j];
      link_uses[i][j] = other_linkmap->link_uses[i][j];
      link_hop_num[i][j] = other_linkmap->link_hop_num[i][j];
      link_hop_route_num[i][j] = other_linkmap->link_hop_route_num[i][j];
      for (int k = 0; k < MAX_ALLOWED_HOPS; k++)
        for (int l = 0; l < MAX_HOP_ROUTES; l++)
            link_hop_route[i][j][k][l] = other_linkmap->link_hop_route[i][j][k][l];
			ESPA_bytes[i][j] = other_linkmap->ESPA_bytes[i][j];
      ESPA_ETA[i][j] = other_linkmap->ESPA_ETA[i][j];
      ESPA_ETA_max = other_linkmap->ESPA_ETA_max;
			ESPA_ETA_mean = other_linkmap->ESPA_ETA_mean;
			ESPA_ETA_var = other_linkmap->ESPA_ETA_var;
  }
	for (int i = 0; i < LOC_NUM*LOC_NUM; i++) ESPA_ETA_sorted_dec_ids[i] = other_linkmap->ESPA_ETA_sorted_dec_ids[i];
}

void LinkMap::reset()
{
  for (int i = 0; i < LOC_NUM; i++)
    for (int j = 0; j < LOC_NUM; j++){
      link_bw_shared[i][j] = 0;
      link_bw_shared_hops[i][j] = 0;
      link_uses[i][j] = 0;
      link_hop_num[i][j] = 0;
      link_hop_route_num[i][j] = 0;
      ESPA_bytes[i][j] = 0;
      ESPA_ETA[i][j] = 0;
      ESPA_ETA_max = ESPA_ETA_mean = ESPA_ETA_var = 0;
      for (int k = 0; k < MAX_ALLOWED_HOPS; k++)
        for (int l = 0; l < MAX_HOP_ROUTES; l++)
            link_hop_route[i][j][k][l] = 0;
    }
	for (int i = 0; i < LOC_NUM*LOC_NUM; i++) ESPA_ETA_sorted_dec_ids[i] = i;
}


void LinkMap::reset_links(int unit_id){
  	for (int i = 0; i < LOC_NUM; i++)
    	for (int j = 0; j < LOC_NUM; j++) 
    		if(link_hop_num[i][j])
    			for (int l = 0; l < MAX_HOP_ROUTES; l++)
    				for (int k = 0; k < MAX_ALLOWED_HOPS; k++)
    			 		///terminate all routes for links that (might) use unit_id as an intermediate hop. 
    	     			if(link_hop_route[i][j][k][l] == unit_id){
    	     				link_hop_num[i][j] = 0;
#ifdef PDEBUG
							lprintf(0, "\n|-----> LinkMap::reset_links(Terminating route [%d][%d] due to link_hop_route[%d][%d][%d][%d] = %d)\n\n",
								i, j, i, j, k, l, unit_id);
#endif
						}    	     				
    	     			
}


void normalize_2D_LOC_NUM(double link_bw [][LOC_NUM], int dim1, double split_limit){
  int already_normalized[dim1][LOC_NUM] = {{0}};
  for (int i = 0; i < dim1; i++){
    for(int j = 0; j < LOC_NUM; j++){
      if(i == j) continue;
      int flag_normalize[dim1][LOC_NUM] = {{0}};
      int normalize_num = 0;
      double normalize_sum = 0;
    for (int k = 0; k < dim1; k++)
      for (int l = 0; l < LOC_NUM; l++)
        if(already_normalized[k][l]) continue;
        else if(abs(link_bw[i][j] - link_bw[k][l])
          /link_bw[i][j] < split_limit){
            flag_normalize[k][l] = 1;
            normalize_sum+=link_bw[k][l];
            normalize_num++;
          }
    for (int k = 0; k < dim1; k++)
      for (int l = 0; l < LOC_NUM; l++)
        if(flag_normalize[k][l] && !already_normalized[k][l]) {
          link_bw[k][l] = normalize_sum/normalize_num;
          already_normalized[k][l] = 1;
        }
    }
  }
}

void LinkMap::update_link_weights(MD_p* list_of_models, int T){
short lvl = 1;
#ifdef DEBUG
    lprintf(0, "\n|-----> LinkMap::update_link_weights(list_of_models = %p, T = %d)\n\n",
    list_of_models, T);
#endif
#ifdef PDEBUG
    lprintf(0, "\n|-----> LinkMap::update_link_weights(list_of_models = %p, T = %d)\n\n",
    list_of_models, T);
#endif
  int pred_T_dim = 0;
  if(T < 1) pred_T_dim = 2048;
  else pred_T_dim = T;
  for (int i = 0; i < LOC_NUM; i++){
    for(int j = 0; j < LOC_NUM; j++){
      if(i == j) link_bw[i][j] = 0;
      else link_bw[i][j] = Gval_per_s(pred_T_dim*pred_T_dim*sizeof(VALUE_TYPE),
      t_com_predict(list_of_models[i]->revlink[j], pred_T_dim*pred_T_dim*sizeof(VALUE_TYPE)));
    }
  }
  normalize_2D_LOC_NUM(link_bw, LOC_NUM, NORMALIZE_NEAR_SPLIT_LIMIT);
  for(int i = 0; i< LOC_NUM; i++)	for(int j = 0; j< LOC_NUM; j++)
    final_estimated_link_bw[i][j] = link_bw[i][j];
#ifdef PDEBUG
  print_link_bw();
#endif
#ifdef DEBUG
  lprintf(lvl, "<-----| LinkMap::update_link_weights()\n");
#endif
}

void LinkMap::update_link_shared_weights(MD_p* unit_modeler_list,
  int* active_unit_id_list, int active_unit_num)
{
  short lvl = 3;
#ifdef DEBUG
    lprintf(lvl, "|-----> LinkMap::update_link_shared_weights(unit_modeler_list = %p, active_unit_id_list = %s)\n",
    unit_modeler_list, printlist<int>(active_unit_id_list, active_unit_num));
#endif
  int* datalocs = (int*) malloc(LOC_NUM*sizeof(int)), dataloc_num = 0;
  unit_modeler_list[0]->getDatalocs(&datalocs, &dataloc_num);
  //double send_ratio = unit_modeler_list[0]->getSendRatio(), recv_ratio = unit_modeler_list[0]->getRecvRatio();
  if (!dataloc_num)
    error("Called ATC::update_link_map_shared() without properly initalized model in unit_modeler_list[0]\n");
#ifdef PDEBUG
  lprintf(0, "\n|-----> LinkMap::update_link_shared_weights(unit_list = %s, datalocs = %s)\n\n",
    printlist<int>(active_unit_id_list, active_unit_num), printlist<int>(datalocs, dataloc_num));

#endif
  //int pred_T_dim = 0;
  //if(T < 1) pred_T_dim = 2048;
  //else pred_T_dim = T;
  for (int i = 0; i < LOC_NUM; i++){
    for(int j = 0; j < LOC_NUM; j++){
      if(i == j) link_bw_shared[i][j] = 0;
      else{
        //link_bw_shared[i][j] = Gval_per_s(pred_T_dim*pred_T_dim*sizeof(VALUE_TYPE),
        //t_com_predict(unit_modeler_list[i]->revlink[j], pred_T_dim*pred_T_dim*sizeof(VALUE_TYPE)));
        double link_slowdown_multiplier = 1.0;
        for (int k = 0; k < LOC_NUM; k++){
          for(int l = 0; l < LOC_NUM; l++){
            if ((k == l) || (i == k && j == l)) continue;
            if(!link_hop_num[k][l] && (is_in_list(deidxize(l),datalocs, dataloc_num) && is_in_list(deidxize(k),active_unit_id_list, active_unit_num))){
             link_slowdown_multiplier = fmax(link_slowdown_multiplier, unit_modeler_list[i]->link[j]->sl[k][l]);
#ifdef DPDEBUG
              if (unit_modeler_list[i]->link[j]->sl[k][l] != 1.0) lprintf(lvl, "ATC::update_link_map_shared():\
                \nFound link (%d -> %d) imposing potential recv-based slowdown to (%d -> %d) with sl = %Lf\n",
                deidxize(l), deidxize(k), deidxize(j), deidxize(i), unit_modeler_list[i]->link[j]->sl[k][l]);
#endif
            }
            else if(link_hop_num[k][l]){
              int hoplocs[link_hop_num[k][l] + 2];
              hoplocs[0] = k;
              hoplocs[link_hop_num[k][l]+1] = l;
              for (int idx = 0; idx < link_hop_num[k][l]; idx++) hoplocs[idx+1] = link_hop_route[k][l][0][idx];
              for (int idx = 0; idx < link_hop_num[k][l] + 1; idx++){
                if(is_in_list(deidxize(hoplocs[idx+1]),datalocs, dataloc_num) && is_in_list(deidxize(hoplocs[idx]),active_unit_id_list, active_unit_num))
                link_slowdown_multiplier = fmax(link_slowdown_multiplier,unit_modeler_list[i]->link[j]->sl[hoplocs[idx]][hoplocs[idx+1]]);
#ifdef DPDEBUG
                 if (unit_modeler_list[i]->link[j]->sl[hoplocs[idx]][hoplocs[idx+1]] != 1.0) lprintf(lvl, "ATC::update_link_map_shared():\
                   \nFound link (%d -> %d) imposing potential recv-based slowdown to (%d -> %d) with sl = %Lf\n",
                   deidxize(hoplocs[idx+1]), deidxize(hoplocs[idx]), deidxize(j), deidxize(i), unit_modeler_list[i]->link[j]->sl[hoplocs[idx]][hoplocs[idx+1]]);
#endif

              }
            }
/*            if(is_in_list(deidxize(k),datalocs, dataloc_num) && is_in_list(deidxize(l),active_unit_id_list, active_unit_num)){
              link_slowdown_multiplier = fmax(link_slowdown_multiplier, unit_modeler_list[i]->link[j]->sl[k][l]);
//#ifdef DPDEBUG
              if (unit_modeler_list[i]->link[j]->sl[k][l] != 1.0) lprintf(lvl, "ATC::update_link_map_shared():\
                \nFound link (%d -> %d) imposing potential send-based slowdown to (%d -> %d) with sl = %Lf\n",
                deidxize(l), deidxize(k), deidxize(j), deidxize(i), unit_modeler_list[i]->link[j]->sl[k][l]);
//#endif
            }*/
          }
        }
#ifdef PDEBUG
        if(link_slowdown_multiplier!= 1.00) lprintf(lvl, "ATC::update_link_map_shared():\
        \nAdjusting link_bw_shared[%d][%d] with link_slowdown_multiplier = %lf\n", i, j, link_slowdown_multiplier);
#endif
        if (link_slowdown_multiplier>2) link_slowdown_multiplier = 2;
        link_bw_shared[i][j] = link_bw[i][j] * (1/link_slowdown_multiplier);
      }
    }
    /// Normalize costs.
    for(int j = 0; j < LOC_NUM; j++){
      if(i == j) continue;
      int flag_normalize[LOC_NUM] = {0}, normalize_num = 1;
      double normalize_sum = link_bw_shared[i][j];
      flag_normalize[j] = 1;
      for (int k = j + 1; k < LOC_NUM; k++)
        if(link_bw_shared[i][j] && (abs(link_bw_shared[i][j] - link_bw_shared[i][k])
          /link_bw_shared[i][j] < NORMALIZE_NEAR_SPLIT_LIMIT )){
          flag_normalize[k] = 1;
          normalize_sum+=link_bw_shared[i][k];
          normalize_num++;
        }
      for (int k = j ; k < LOC_NUM; k++) if(flag_normalize[k]) link_bw_shared[i][k] = normalize_sum/normalize_num;
    }
  }
#ifdef PDEBUG
  print_link_bw_shared();
#endif
#ifdef DEBUG
  lprintf(lvl, "<-----| update_link_map_shared()\n");
#endif
}

#ifdef ENABLE_TRANSFER_HOPS
void LinkMap::init_hop_routes(MD_p* unit_modeler_list, int* active_unit_id_list, int active_unit_num){
  double safe_hop_penalty = HOP_PENALTY;
  for (int unit_idx = 0 ; unit_idx < LOC_NUM; unit_idx++)
  for (int unit_idy = 0 ; unit_idy < LOC_NUM; unit_idy++){
    for (int hops = 0 ; hops < MAX_ALLOWED_HOPS; hops++) for (int rt = 0 ; rt < MAX_HOP_ROUTES; rt++)
      link_hop_route[unit_idx][unit_idy][rt][hops] = -42;
    link_hop_num[unit_idx][unit_idy] = 0;
    double max_hop_bw = link_bw_shared[unit_idx][unit_idy];
    link_hop_route_num[unit_idx][unit_idy] = 0;
    if(MAX_ALLOWED_HOPS >= 1){
      for (int hop_idx = 0 ; hop_idx < LOC_NUM; hop_idx++)
      if(hop_idx!= unit_idx && hop_idx!= unit_idy && unit_idx!= unit_idy){
        if(!is_in_list(deidxize(hop_idx),active_unit_id_list, active_unit_num)) continue;
        double hop_bw =  fmin(link_bw_shared[unit_idx][hop_idx], link_bw_shared[hop_idx][unit_idy]);
        //  /unit_modeler_list[unit_idx]->link[hop_idx]->sl[hop_idx][unit_idy]; // FIXME: Might be reverse
        hop_bw-= safe_hop_penalty*hop_bw;
        if (hop_bw > max_hop_bw){
         max_hop_bw = hop_bw;
         link_hop_route_num[unit_idx][unit_idy] = 1;
         link_hop_num[unit_idx][unit_idy] = 1;
         link_hop_route[unit_idx][unit_idy][link_hop_route_num[unit_idx][unit_idy]-1][0] = hop_idx;
        }
        else if(hop_bw == max_hop_bw && link_hop_route_num[unit_idx][unit_idy] < MAX_HOP_ROUTES){
          link_hop_route[unit_idx][unit_idy][link_hop_route_num[unit_idx][unit_idy]++][0] = hop_idx;
        }
      }
    }
    if(MAX_ALLOWED_HOPS >= 2){
      for (int hop_idx = 0 ; hop_idx < LOC_NUM; hop_idx++)
      for (int hop_idy = 0 ; hop_idy < LOC_NUM; hop_idy++)
      if(hop_idx!= unit_idx && hop_idx!= unit_idy && unit_idx!= unit_idy &&
      hop_idy!= unit_idx && hop_idy!= unit_idy && hop_idy!= hop_idx){
        double hop_bw =  fmin(link_bw_shared[unit_idx][hop_idx], fmin(link_bw_shared[hop_idx][hop_idy], link_bw_shared[hop_idy][unit_idy]));
        hop_bw-= 2*safe_hop_penalty*hop_bw;
        if (hop_bw > max_hop_bw){
          max_hop_bw = hop_bw;
          link_hop_route_num[unit_idx][unit_idy] = 1;
          link_hop_route[unit_idx][unit_idy][0][0] = hop_idy;
          link_hop_route[unit_idx][unit_idy][0][1] = hop_idx;
          link_hop_num[unit_idx][unit_idy] = 2;
        }
      }
    }
    if (link_hop_num[unit_idx][unit_idy]){
    link_bw_shared_hops[unit_idx][unit_idy] = max_hop_bw;
#ifdef PDEBUG
		  	lprintf(0, "LinkMap::init_hop_routes: %d -> %d transfer sequence -> [ %d ] => ", unit_idy, unit_idx,
		  		link_hop_route[unit_idx][unit_idy][link_hop_route_num[unit_idx][unit_idy]-1][0]);
		  	lprintf(0, "Cost No-hop = %lf, Hop-adjusted = %lf (%3lf times faster)\n", link_bw_shared[unit_idx][unit_idy],
				link_bw_shared_hops[unit_idx][unit_idy], link_bw_shared_hops[unit_idx][unit_idy]/link_bw_shared[unit_idx][unit_idy]);
#endif
    }
    else link_bw_shared_hops[unit_idx][unit_idy] = link_bw_shared[unit_idx][unit_idy];
  }
}

void LinkMap::print_link_bw_shared_hops(){
  lprintf(0,"\n Link Shared-BW Hop Map:\n   |");
  for (int d2 = 0; d2 < LOC_NUM; d2++)
    lprintf(0, "  %2d  |", deidxize(d2));
  lprintf(0, "\n   |");
  for (int d2 = 0; d2 < LOC_NUM; d2++)
    lprintf(0, "-------");
  lprintf(0, "\n");
  for (int d1 = 0; d1 < LOC_NUM; d1++){
    lprintf(0, "%2d | ", deidxize(d1));
    for (int d2 = 0; d2 < LOC_NUM; d2++){
      lprintf(0, "%4.2lf | ", link_bw_shared_hops[d1][d2]);
    }
    lprintf(0, "\n");
  }
}

#ifdef ENABLE_ESPA

double LinkMap::update_ESPA_ETA_idx(MD_p* unit_modeler_list, int idxi, int idxj){
	ESPA_ETA[idxi][idxj] = t_com_predict_shared(unit_modeler_list[idxi]->link[idxj], ESPA_bytes[idxi][idxj]);
	return ESPA_ETA[idxi][idxj]; // Already stored in LinkMap, return for ease of use as getter.
}

double LinkMap::update_ESPA_ETA_max(){
	ESPA_ETA_max = 0;
  	for (int idxi = 0; idxi < LOC_NUM; idxi++) for (int idxj = 0; idxj < LOC_NUM; idxj++) if(ESPA_ETA[idxi][idxj] > ESPA_ETA_max) ESPA_ETA_max = ESPA_ETA[idxi][idxj];
  	return ESPA_ETA_max; // Already stored in LinkMap, return for ease of use as getter.
}

double LinkMap::update_ESPA_ETA_mean(){
	ESPA_ETA_mean = 0;
  	for (int idxi = 0; idxi < LOC_NUM; idxi++) for (int idxj = 0; idxj < LOC_NUM; idxj++) ESPA_ETA_mean += ESPA_ETA[idxi][idxj];
  	ESPA_ETA_mean /= LOC_NUM*LOC_NUM;
	return ESPA_ETA_mean; // Already stored in LinkMap, return for ease of use as getter.
}

double LinkMap::update_ESPA_ETA_mean_and_var(){
	ESPA_ETA_mean = 0;
  	for (int idxi = 0; idxi < LOC_NUM; idxi++) for (int idxj = 0; idxj < LOC_NUM; idxj++) ESPA_ETA_mean += ESPA_ETA[idxi][idxj];
  	ESPA_ETA_mean /= LOC_NUM*LOC_NUM;
	ESPA_ETA_var = 0;
	for (int idxi = 0; idxi < LOC_NUM; idxi++) for (int idxj = 0; idxj < LOC_NUM; idxj++)
		ESPA_ETA_var+= pow(ESPA_ETA[idxi][idxj] - ESPA_ETA_mean, 2);
	ESPA_ETA_var /= LOC_NUM*LOC_NUM;
	return ESPA_ETA_var; // Already stored in LinkMap, return for ease of use as getter.
}

void LinkMap::update_ESPA_ETA_sorted_dec_ids(){
	for (int i = 0; i < LOC_NUM*LOC_NUM; i++) ESPA_ETA_sorted_dec_ids[i] = i;
	for (int i = 0; i < LOC_NUM*LOC_NUM - 1; i++)
		for (int j = 0; j < LOC_NUM*LOC_NUM - i - 1; j++){
			int unit_idx = ESPA_ETA_sorted_dec_ids[j]/LOC_NUM, unit_idy = ESPA_ETA_sorted_dec_ids[j]%LOC_NUM;
			int unit_jdx = ESPA_ETA_sorted_dec_ids[j+1]/LOC_NUM, unit_jdy = ESPA_ETA_sorted_dec_ids[j+1]%LOC_NUM;
			if (ESPA_ETA[unit_idx][unit_idy] < ESPA_ETA[unit_jdx][unit_jdy]){
				int temp = ESPA_ETA_sorted_dec_ids[j];
				ESPA_ETA_sorted_dec_ids[j] = ESPA_ETA_sorted_dec_ids[j+1];
				ESPA_ETA_sorted_dec_ids[j+1] = temp;
			}
	}
}

void LinkMap::ESPA_init(MD_p* unit_modeler_list, int* active_unit_id_list, double* active_unit_score,
  int active_unit_num, int init_type){
  short lvl = 4;
#ifdef DEBUG
    lprintf(0, "\n|-----> link_map::ESPA_init(list_of_models = %p, list_of_units = %s, list_of_unit_percentages = %s, init_type = %d)\n\n",
      active_unit_id_list, printlist(active_unit_id_list, active_unit_num),
      (init_type)? printlist(active_unit_score, active_unit_num): "NULL->equal", init_type);
#endif
#ifdef PDEBUG
    lprintf(0, "\n|-----> link_map::ESPA_init() Initializing for list_of_units = %s, list_of_unit_percentages = %s, init_type = %d)\n\n",
      printlist(active_unit_id_list, active_unit_num),
      (init_type)? printlist(active_unit_score, active_unit_num): "NULL->equal", init_type);
#endif
  double* ESPA_unit_score = NULL;
  if (init_type && active_unit_score) ESPA_unit_score = active_unit_score;
  else if((init_type && !active_unit_score) || !init_type && active_unit_score) error("link_map::ESPA_init() called with init_type = %d and active_unit_score\n",
    init_type, active_unit_score);
  else{
    //TODO: Assume an equal problem data distribution if none given (yet)
    ESPA_unit_score = (double*) malloc(active_unit_num*sizeof(double));
    for (int i = 0; i < active_unit_num; i++) ESPA_unit_score[i] = 1.0/active_unit_num;
  }

  for (int idxi = 0; idxi < LOC_NUM; idxi++) for (int idxj = 0; idxj < LOC_NUM; idxj++) ESPA_bytes[idxi][idxj] = 0;

  for(int unit_idx = 0; unit_idx < active_unit_num; unit_idx++){
    MD_p model = NULL;
    for (int model_idx = 0; model_idx < LOC_NUM; model_idx++)
      if(unit_modeler_list[model_idx]->unit_id == active_unit_id_list[unit_idx]){
          model = unit_modeler_list[model_idx];
          break;
      }
    if (!model) error("LinkMap::ESPA_init: Model not found for active_unit_id_list[%d] = %d\n",
      unit_idx, active_unit_id_list[unit_idx]);

    long long recv_sz_RONLY = 0;
    int recv_num_RONLY = 0;
    for (int i = 0; i < model->V->numT; i++){
      long long recv_bytes = (long long) model->V->in[i]*(*model->V->Dim1[i])*(*model->V->Dim2[i])*
        model->V->dtype_sz*ESPA_unit_score[unit_idx];
      long long send_bytes = (long long) model->V->out[i]*(*model->V->Dim1[i])*(*model->V->Dim2[i])*
        model->V->dtype_sz*ESPA_unit_score[unit_idx];
			if(model->V->loc[i] != model->unit_id){
      	ESPA_bytes[idxize(model->unit_id)][idxize(model->V->loc[i])]+= recv_bytes;
      	ESPA_bytes[idxize(model->V->loc[i])][idxize(model->unit_id)]+= send_bytes;
	      if(!model->V->out[i]) {
	        recv_sz_RONLY+= recv_bytes;
	        recv_num_RONLY++;
	      }
			}
    }

    /// TODO: Extra transfers created from internal dims due to multi-unit spliting.
    /// Algorithm may vary for other BLAS3, but not at that bridge yet.
    /// The assumtion for extra transfers is made based on the 2D cyclic distribution,
    /// but the estimation is also useful for other distributions as a best case scenario (worse distributions -> more extra transfers).
    int D1_parts = sqrt(active_unit_num);
    int D2_parts = D1_parts;
    if (D1_parts ==0) { D2_parts = active_unit_num; D1_parts = 1; }
    else { /* find the most square decomposition of autotune_controller->active_unit_num in D1_parts x D2_parts */
      int g;
      for (g = D1_parts+1; g>0; --g) if (active_unit_num % g == 0) break;
      if (g==0) { D1_parts = active_unit_num; D2_parts = 1; }
      else { D1_parts = g; D2_parts = active_unit_num/g; }
    }
    double extra_transfer_ratio = (recv_num_RONLY)? (1.0*((D1_parts-1) + (D2_parts -1)))/recv_num_RONLY: 0;

#ifdef DPDEBUG
    lprintf(lvl, "LinkMap::ESPA_init(unit_num = %d) : D1_parts = %d, D2_parts = %d, extra_transfer_ratio = %lf\n",
    active_unit_num, D1_parts, D2_parts, extra_transfer_ratio);
#endif
    long long recv_sz_extra = extra_transfer_ratio * recv_sz_RONLY;
    /// Assume extra transfers are split equally between all other units
    for (int other_loc_idx = 0; other_loc_idx < active_unit_num; other_loc_idx++) if(active_unit_id_list[other_loc_idx]!=model->unit_id)
      ESPA_bytes[idxize(model->unit_id)][idxize(active_unit_id_list[other_loc_idx])]+= recv_sz_extra/(active_unit_num-1);
  }

  for (int idxi = 0; idxi < LOC_NUM; idxi++) for (int idxj = 0; idxj < LOC_NUM; idxj++) update_ESPA_ETA_idx(unit_modeler_list, idxi, idxj);
  update_ESPA_ETA_max();
  update_ESPA_ETA_mean_and_var();
  update_ESPA_ETA_sorted_dec_ids();

#ifdef PDEBUG
  print_ESPA();
#endif
#ifdef DEBUG
  lprintf(lvl, "<-----| LinkMap::ESPA_init()\n");
#endif
}

void LinkMap::ESPA_init_hop_routes(MD_p* unit_modeler_list, int* active_unit_id_list, double* active_unit_score,
  int active_unit_num, int init_type){
	int* datalocs = (int*) malloc(LOC_NUM*sizeof(int)), dataloc_num = 0;
	unit_modeler_list[0]->getDatalocs(&datalocs, &dataloc_num);
	double safe_hop_penalty = HOP_PENALTY;
	for (int unit_ctr = 0 ; unit_ctr < LOC_NUM*LOC_NUM; unit_ctr++){
		int unit_idx = ESPA_ETA_sorted_dec_ids[unit_ctr]/LOC_NUM, unit_idy = ESPA_ETA_sorted_dec_ids[unit_ctr]%LOC_NUM; // Sorted by ETA
		//int unit_idx = unit_ctr/LOC_NUM, unit_idy = unit_ctr%LOC_NUM; // Serial
		//printf("Checking unit combination ESPA_ETA_sorted_dec_ids[%d] = %d : [%d][%d]\n", unit_ctr, ESPA_ETA_sorted_dec_ids[unit_ctr], unit_idx, unit_idy);
		if ((is_in_list(deidxize(unit_idx),active_unit_id_list, active_unit_num) || is_in_list(deidxize(unit_idx),datalocs, dataloc_num)) &&
			  (is_in_list(deidxize(unit_idy),active_unit_id_list, active_unit_num) || is_in_list(deidxize(unit_idy),datalocs, dataloc_num))){
			for (int hops = 0 ; hops < MAX_ALLOWED_HOPS; hops++) for (int rt = 0 ; rt < MAX_HOP_ROUTES; rt++)
				link_hop_route[unit_idx][unit_idy][rt][hops] = -42;
			link_hop_num[unit_idx][unit_idy] = 0;
			link_hop_route_num[unit_idx][unit_idy] = 0;
			int best_hop_idx = -1;
			double max_hop_bw = link_bw_shared[unit_idx][unit_idy];
			double best_ETA = ESPA_ETA[unit_idx][unit_idy];
			long long prev_bytes = ESPA_bytes[unit_idx][unit_idy];
				if(MAX_ALLOWED_HOPS >= 1){
					for (int hop_idx = 0 ; hop_idx < LOC_NUM; hop_idx++)
					if(hop_idx!= unit_idx && hop_idx!= unit_idy && unit_idx!= unit_idy){
						if(!is_in_list(deidxize(hop_idx),active_unit_id_list, active_unit_num)) continue;

						double hop_bw =  fmin(link_bw_shared[unit_idx][hop_idx], link_bw_shared[hop_idx][unit_idy]);
		        		hop_bw-= safe_hop_penalty*hop_bw;

						double ESPA_ETA_tmp_1 = t_com_predict_shared(unit_modeler_list[hop_idx]->link[unit_idy], ESPA_bytes[hop_idx][unit_idy] + prev_bytes);

						double ESPA_ETA_tmp_2 = t_com_predict_shared(unit_modeler_list[unit_idx]->link[hop_idx], ESPA_bytes[unit_idx][hop_idx] + prev_bytes);

						double link_hop_ETA = fmax(ESPA_ETA_tmp_1,ESPA_ETA_tmp_2);

						if ((hop_bw > max_hop_bw || (hop_bw == max_hop_bw && link_hop_ETA < best_ETA)) && ( hop_bw >= 2*max_hop_bw || link_hop_ETA < ESPA_ETA_max)){ /// BW-focused
						//if (link_hop_ETA < best_ETA || (hop_bw > max_hop_bw && link_hop_ETA == best_ETA) ){ /// ETA-focused - not sufficient
								best_hop_idx = hop_idx;
								best_ETA = link_hop_ETA;
								max_hop_bw = hop_bw;
						}

					}

				if(best_hop_idx!= -1){
					link_hop_route_num[unit_idx][unit_idy] = 1;
					link_hop_num[unit_idx][unit_idy] = 1;
					link_hop_route[unit_idx][unit_idy][link_hop_route_num[unit_idx][unit_idy]-1][0] = best_hop_idx;

					ESPA_bytes[unit_idx][unit_idy] = 0;
					ESPA_ETA[unit_idx][unit_idy] = 0;

					ESPA_bytes[best_hop_idx][unit_idy] += prev_bytes;
					ESPA_bytes[unit_idx][best_hop_idx] += prev_bytes;
					update_ESPA_ETA_idx(unit_modeler_list, best_hop_idx, unit_idy);
					update_ESPA_ETA_idx(unit_modeler_list, unit_idx, best_hop_idx);

					update_ESPA_ETA_max();
					update_ESPA_ETA_mean_and_var();

					//printf("Hoping via %d -> link_bw_shared[%d][%d] = %lf, link_bw_shared[%d][%d] = %lf\n",
					//	best_hop_idx, unit_idx, best_hop_idx, link_bw_shared[unit_idx][best_hop_idx], best_hop_idx, unit_idy, link_bw_shared[best_hop_idx][unit_idy]);

					link_bw_shared_hops[unit_idx][unit_idy] = fmin(link_bw_shared[unit_idx][best_hop_idx], link_bw_shared[best_hop_idx][unit_idy]);
					link_bw_shared_hops[unit_idx][unit_idy]-= safe_hop_penalty*link_bw_shared_hops[unit_idx][unit_idy];
	#ifdef PDEBUG
			  		lprintf(0, "LinkMap::ESPA_init_hop_routes: %d -> %d transfer sequence -> [ %d ] => ", unit_idy, unit_idx,
			  			best_hop_idx);
			  		lprintf(0, "Cost No-hop = %lf, Hop-adjusted = %lf (%3lf times faster)\n", link_bw_shared[unit_idx][unit_idy],
						link_bw_shared_hops[unit_idx][unit_idy], link_bw_shared_hops[unit_idx][unit_idy]/link_bw_shared[unit_idx][unit_idy]);
	#endif
				}
				else link_bw_shared_hops[unit_idx][unit_idy] = link_bw_shared[unit_idx][unit_idy];
			}
			else if(MAX_ALLOWED_HOPS >= 2){
				error("LinkMap::ESPA_init_hop_routes: Allowed hops > 1 not implemented!\n");
			}
	  }
		else link_bw_shared_hops[unit_idx][unit_idy] = link_bw_shared[unit_idx][unit_idy];
	}
	update_ESPA_ETA_sorted_dec_ids();
#ifdef PDEBUG
  print_ESPA();
  print_link_bw_shared_hops();
#endif
}

/// Init type 0 -> Equal split, use additive ETA time prediction
/// Init type 1 -> Input split, use additive ETA time prediction
/// Init type 2 -> Input split, use max ETA time prediction
double LinkMap::ESPA_predict(MD_p unit_modeler, int T, int* active_unit_id_list,
			double* active_unit_score, int active_unit_num, int init_type){
	short lvl = 4;
	int used_unit_idx = -1;
	for(int unit_idx = 0; unit_idx < active_unit_num; unit_idx++) if(unit_modeler->unit_id == active_unit_id_list[unit_idx]) used_unit_idx = unit_idx;
	if (used_unit_idx == - 1) error("LinkMap::ESPA_predict: Model %p with unit_id = %d not present in given active_unit_id_list = %s\n",
		unit_modeler, unit_modeler->unit_id, printlist<int>(active_unit_id_list, active_unit_num));
	double penalty = PARALiaPerfPenaltyModifier(unit_modeler, T, active_unit_num);

	double* ESPA_unit_score = NULL;
	if (init_type && active_unit_score) ESPA_unit_score = active_unit_score;
	else if((init_type && !active_unit_score) || !init_type && active_unit_score) error("link_map::ESPA_predict() called with init_type = %d and active_unit_score\n",
	init_type, active_unit_score);
	else{
	//TODO: Assume an equal problem data distribution if none given (yet)
	ESPA_unit_score = (double*) malloc(active_unit_num*sizeof(double));
	for (int i = 0; i < active_unit_num; i++) ESPA_unit_score[i] = 1.0/active_unit_num;
	}
	double t_recv_full = 0, t_send_full = 0, t_exec_full = 0, t_total = 0;
	t_exec_full = ESPA_unit_score[used_unit_idx]*unit_modeler->getGPUexecFull();
	
	//warning("ESPA_predict is modified for heterogeneous scenario ratios, beware!!1!\n"); 
	//if(unit_modeler->unit_id == 0 || unit_modeler->unit_id == 1 || unit_modeler->unit_id == 4 || unit_modeler->unit_id == 6) t_exec_full*=7.0/3.0;
	//else if ( unit_modeler->unit_id == 2 || unit_modeler->unit_id == 3)  t_exec_full*=7.0/5.0;
	
	if ( t_exec_full < 0){
		warning("LinkMap::ESPA_predict: GPUexec3Model_predict submodel returned negative value, abort prediction");
		return -1.0;
	}

	int global_unit_idx = idxize(unit_modeler->unit_id);
			for (int other_unit_idx = 0; other_unit_idx < LOC_NUM; other_unit_idx++){
				/// t_recv_full estimation
				double tmp_link_cost = 0;
				int src_hop = other_unit_idx, dst_hop = global_unit_idx;
				for(int intermediate_hops = 0; intermediate_hops < link_hop_num[global_unit_idx][other_unit_idx]; intermediate_hops++){
					tmp_link_cost = fmax(tmp_link_cost, ESPA_ETA[link_hop_route[global_unit_idx][other_unit_idx][intermediate_hops][0]][src_hop]);
					src_hop = link_hop_route[global_unit_idx][other_unit_idx][intermediate_hops][0];
				}
				tmp_link_cost = fmax(tmp_link_cost, ESPA_ETA[dst_hop][src_hop]);
				if(1 == ESPA_COMMUNICATION_AGGREGATOR) t_recv_full+=tmp_link_cost;
				else if (0 == ESPA_COMMUNICATION_AGGREGATOR) t_recv_full = fmax(t_recv_full, tmp_link_cost);

				/// t_send_full estimation
				tmp_link_cost = 0;
				src_hop = global_unit_idx, dst_hop = other_unit_idx;
				for(int intermediate_hops = 0; intermediate_hops < link_hop_num[other_unit_idx][global_unit_idx]; intermediate_hops++){
					tmp_link_cost = fmax(tmp_link_cost, ESPA_ETA[link_hop_route[other_unit_idx][global_unit_idx][intermediate_hops][0]][src_hop]);
					src_hop = link_hop_route[other_unit_idx][global_unit_idx][intermediate_hops][0];
				}
				tmp_link_cost = fmax(tmp_link_cost, ESPA_ETA[dst_hop][src_hop]);
				if(1 == ESPA_COMMUNICATION_AGGREGATOR) t_send_full+=tmp_link_cost;
				else if (0 == ESPA_COMMUNICATION_AGGREGATOR) t_send_full = fmax(t_send_full, tmp_link_cost);
		}

		t_total = fmax(t_exec_full, fmax(t_recv_full, t_send_full));
#ifdef PDEBUG
		lprintf(0, "PARALia LinkMap::ESPA_predict (Unit = %d, Unit_ratio = %.2lf%%):\n"
		"\tt_recv_full: %lf ms\n"
		"\tt_exec_full: %lf ms (%lf GFlops/s)\n"
		"\tt_send_full: %lf ms\n"
		"\tt_total: %lf ms (%lf GFlops/s)\n\n"
		"\tt_total X penalty: %lf ms (%lf GFlops/s)\n\n",
		active_unit_id_list[used_unit_idx], 100*ESPA_unit_score[used_unit_idx], t_recv_full*1000,
		t_exec_full*1000, Gval_per_s(unit_modeler->getFlops()*ESPA_unit_score[used_unit_idx], t_exec_full),
		t_send_full*1000,
		t_total*1000, Gval_per_s(unit_modeler->getFlops()*ESPA_unit_score[used_unit_idx], t_total),
		t_total*penalty*1000, Gval_per_s(unit_modeler->getFlops()*ESPA_unit_score[used_unit_idx], t_total*penalty));
#endif

		return t_total*penalty;
}

void LinkMap::print_ESPA(){
  lprintf(0,"\n ESPA bytes Map:\n   |");
  for (int d2 = 0; d2 < LOC_NUM; d2++)
    lprintf(0, "  %2d  |", deidxize(d2));
  lprintf(0, "\n   |");
  for (int d2 = 0; d2 < LOC_NUM; d2++)
    lprintf(0, "-------");
  lprintf(0, "\n");
  for (int d1 = 0; d1 < LOC_NUM; d1++){
    lprintf(0, "%2d | ", deidxize(d1));
    for (int d2 = 0; d2 < LOC_NUM; d2++){
      lprintf(0, "%Le | ", ESPA_bytes[d1][d2]);
    }
    lprintf(0, "\n");
  }

  lprintf(0,"\n ESPA ETA Map:\n   |");
  for (int d2 = 0; d2 < LOC_NUM; d2++)
    lprintf(0, "  %2d  |", deidxize(d2));
  lprintf(0, "\n   |");
  for (int d2 = 0; d2 < LOC_NUM; d2++)
    lprintf(0, "-------");
  lprintf(0, "\n");
  for (int d1 = 0; d1 < LOC_NUM; d1++){
    lprintf(0, "%2d | ", deidxize(d1));
    for (int d2 = 0; d2 < LOC_NUM; d2++){
      lprintf(0, "%le | ", ESPA_ETA[d1][d2]);
    }
    lprintf(0, "\n");
  }

  lprintf(0,"\n ESPA Longest ETA Map (Top 10 from ESPA_ETA_sorted_dec_ids):\n");
  int top_print = ((LOC_NUM*LOC_NUM < 10) ? LOC_NUM*LOC_NUM : 10);
  for (int unit_ctr = 0 ; unit_ctr < top_print; unit_ctr++){
	int unit_idx = ESPA_ETA_sorted_dec_ids[unit_ctr]/LOC_NUM, unit_idy = ESPA_ETA_sorted_dec_ids[unit_ctr]%LOC_NUM;
	lprintf(0, "ESPA_ETA_sorted_dec_ids[%d] = %d : ESPA_ETA[%d][%d] = %lf\n", unit_ctr, ESPA_ETA_sorted_dec_ids[unit_ctr], unit_idx, unit_idy, ESPA_ETA[unit_idx][unit_idy]);
  }
}

#endif


#endif


void LinkMap::print_link_bw(){
  lprintf(0,"\n Link BW Map:\n   |");
  for (int d2 = 0; d2 < LOC_NUM; d2++)
    lprintf(0, "  %2d  |", deidxize(d2));
  lprintf(0, "\n   |");
  for (int d2 = 0; d2 < LOC_NUM; d2++)
    lprintf(0, "-------");
  lprintf(0, "\n");
  for (int d1 = 0; d1 < LOC_NUM; d1++){
    lprintf(0, "%2d | ", deidxize(d1));
    for (int d2 = 0; d2 < LOC_NUM; d2++){
      lprintf(0, "%4.2lf | ", link_bw[d1][d2]);
    }
    lprintf(0, "\n");
  }
}

void LinkMap::print_link_bw_shared(){
  lprintf(0,"\n Link Shared-BW Map:\n   |");
  for (int d2 = 0; d2 < LOC_NUM; d2++)
    lprintf(0, "  %2d  |", deidxize(d2));
  lprintf(0, "\n   |");
  for (int d2 = 0; d2 < LOC_NUM; d2++)
    lprintf(0, "-------");
  lprintf(0, "\n");
  for (int d1 = 0; d1 < LOC_NUM; d1++){
    lprintf(0, "%2d | ", deidxize(d1));
    for (int d2 = 0; d2 < LOC_NUM; d2++){
      lprintf(0, "%4.2lf | ", link_bw_shared[d1][d2]);
    }
    lprintf(0, "\n");
  }
}
