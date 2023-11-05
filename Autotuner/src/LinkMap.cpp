
#include <iostream>

#include "Autotuner.hpp"
#include "Model_functions.hpp"

LinkMap::LinkMap(){
    ;
}

void LinkMap::copy(class LinkMap* other_linkmap)
{
  for (int i = 0; i < LOC_NUM; i++)
    for (int j = 0; j < LOC_NUM; j++){
      link_lat[i][j] = other_linkmap->link_lat[i][j];
      link_bw[i][j] = other_linkmap->link_bw[i][j];
      link_bw_shared[i][j] = other_linkmap->link_bw_shared[i][j];
      link_bw_shared_hops[i][j] = other_linkmap->link_bw_shared_hops[i][j];
      link_active[i][j] = other_linkmap->link_active[i][j];
      link_uses[i][j] = other_linkmap->link_uses[i][j];
      link_hop_num[i][j] = other_linkmap->link_hop_num[i][j];
      link_hop_route_num[i][j] = other_linkmap->link_hop_route_num[i][j];
      for (int k = 0; k < MAX_ALLOWED_HOPS; k++)
        for (int l = 0; l < LOC_NUM; l++)
            link_hop_route[i][j][k][l] = other_linkmap->link_hop_route[i][j][k][l];

  }
}

void LinkMap::reset()
{
  for (int i = 0; i < LOC_NUM; i++)
    for (int j = 0; j < LOC_NUM; j++){
      link_bw_shared[i][j] = 0;
      link_bw_shared_hops[i][j] = 0;
      link_active[i][j] = 0; 
      link_uses[i][j] = 0;
      link_hop_num[i][j] = 0;
      link_hop_route_num[i][j] = 0;
      for (int k = 0; k < MAX_ALLOWED_HOPS; k++)
        for (int l = 0; l < MAX_HOP_ROUTES; l++)
            link_hop_route[i][j][k][l] = -42;
    }
}

void LinkMap::reset_links(int unit_id){
  	for (int i = 0; i < LOC_NUM; i++)
    	for (int j = 0; j < LOC_NUM; j++)
    		if(link_hop_num[i][j])
    			for (int l = 0; l < LOC_NUM; l++)
    				for (int k = 0; k < MAX_ALLOWED_HOPS; k++)
    			 		///terminate all routes for links that (might) use unit_id as an intermediate hop.
    	     			if(link_hop_route[i][j][k][l] == unit_id){
    	     				link_hop_num[i][j] = 0;
#ifdef PDEBUG
							fprintf(stderr, "\n|-----> LinkMap::reset_links(Terminating route [%d][%d] due to link_hop_route[%d][%d][%d][%d] = %d)\n\n",
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
    fprintf(stderr, "\n|-----> LinkMap::update_link_weights(list_of_models = %p, T = %d)\n\n",
    list_of_models, T);
#endif
#ifdef PDEBUG
    fprintf(stderr, "\n|-----> LinkMap::update_link_weights(list_of_models = %p, T = %d)\n\n",
    list_of_models, T);
#endif
  int pred_T_dim = 0;
  if(T < 1) pred_T_dim = 2048;
  else pred_T_dim = T;
  for (int i = 0; i < LOC_NUM; i++){
    for(int j = 0; j < LOC_NUM; j++){
      if(i == j) link_bw[i][j] = 0;
      else link_bw[i][j] = Gval_per_s(pred_T_dim*pred_T_dim*8, // TODO: assuming a tile2D double transfer (inconsequential since its a BW estimation)
      t_com_predict(list_of_models[i]->link[j], pred_T_dim*pred_T_dim*8));
    }
  }
  normalize_2D_LOC_NUM(link_bw, LOC_NUM, NORMALIZE_NEAR_SPLIT_LIMIT);
  for(int i = 0; i< LOC_NUM; i++)	for(int j = 0; j< LOC_NUM; j++)
    final_estimated_link_bw[i][j] = link_bw[i][j];
#ifdef PDEBUG
  print_link_bw();
#endif
#ifdef DEBUG
  fprintf(stderr, "<-----| LinkMap::update_link_weights()\n");
#endif
}

void LinkMap::update_link_shared_weights(MD_p* unit_modeler_list,
  int* active_unit_id_list, int active_unit_num)
{
  short lvl = 3;
#ifdef DEBUG
    fprintf(stderr, "|-----> LinkMap::update_link_shared_weights(unit_modeler_list = %p, active_unit_id_list = %s)\n",
    unit_modeler_list, printlist<int>(active_unit_id_list, active_unit_num));
#endif
  int* datalocs = (int*) malloc(LOC_NUM*sizeof(int)), dataloc_num = 0;
  unit_modeler_list[0]->getDatalocs(&datalocs, &dataloc_num);
  if (!dataloc_num)
    error("Called ATC::update_link_map_shared() without properly initalized model in unit_modeler_list[0]\n");
#ifdef PDEBUG
  fprintf(stderr, "\n|-----> LinkMap::update_link_shared_weights(unit_list = %s, datalocs = %s)\n\n",
    printlist<int>(active_unit_id_list, active_unit_num), printlist<int>(datalocs, dataloc_num));
#endif
#ifdef PDEBUG
  print_link_bw();
#endif
	if(!links_share_bandwidth_init){
		for(int i = 0; i < LOC_NUM; i++)
		for(int j = 0; j < LOC_NUM; j++)
		for(int k = 0; k < 2; k++) links_share_bandwidth[i][j][k] = -42;
		links_share_bandwidth_init = 1; 
	}

#ifndef ENABLE_LINK_BW_SHARING
	///TODO: ENABLE_LINK_BW_SHARING flag is disabled, but sharing-disabler mechanism is handmade

	// FIXME: Handmade distribution, for testing purposes
	links_share_bandwidth[0][LOC_NUM - 1][0] = 1;
	links_share_bandwidth[0][LOC_NUM - 1][1] = LOC_NUM - 1;
	links_share_bandwidth[1][LOC_NUM - 1][0] = 0;
	links_share_bandwidth[1][LOC_NUM - 1][1] = LOC_NUM - 1;

	links_share_bandwidth[2][LOC_NUM - 1][0] = 3;
	links_share_bandwidth[2][LOC_NUM - 1][1] = LOC_NUM - 1;
	links_share_bandwidth[3][LOC_NUM - 1][0] = 2;
	links_share_bandwidth[3][LOC_NUM - 1][1] = LOC_NUM - 1;

	links_share_bandwidth[4][LOC_NUM - 1][0] = 5;
	links_share_bandwidth[4][LOC_NUM - 1][1] = LOC_NUM - 1;
	links_share_bandwidth[5][LOC_NUM - 1][0] = 4;
	links_share_bandwidth[5][LOC_NUM - 1][1] = LOC_NUM - 1;

	links_share_bandwidth[6][LOC_NUM - 1][0] = 7;
	links_share_bandwidth[6][LOC_NUM - 1][1] = LOC_NUM - 1;
	links_share_bandwidth[7][LOC_NUM - 1][0] = 6;
	links_share_bandwidth[7][LOC_NUM - 1][1] = LOC_NUM - 1;
/*
	links_share_bandwidth[LOC_NUM - 1][0][0] = LOC_NUM - 1;
	links_share_bandwidth[LOC_NUM - 1][0][1] = 1;
	links_share_bandwidth[LOC_NUM - 1][1][0] = LOC_NUM - 1;
	links_share_bandwidth[LOC_NUM - 1][1][1] = 0;

	links_share_bandwidth[LOC_NUM - 1][2][0] = LOC_NUM - 1;
	links_share_bandwidth[LOC_NUM - 1][2][1] = 3;
	links_share_bandwidth[LOC_NUM - 1][3][0] = LOC_NUM - 1;
	links_share_bandwidth[LOC_NUM - 1][3][1] = 2;

	links_share_bandwidth[LOC_NUM - 1][4][0] = LOC_NUM - 1;
	links_share_bandwidth[LOC_NUM - 1][4][1] = 5;
	links_share_bandwidth[LOC_NUM - 1][5][0] = LOC_NUM - 1;
	links_share_bandwidth[LOC_NUM - 1][5][1] = 4;

	links_share_bandwidth[LOC_NUM - 1][6][0] = LOC_NUM - 1;
	links_share_bandwidth[LOC_NUM - 1][6][1] = 7;
	links_share_bandwidth[LOC_NUM - 1][7][0] = LOC_NUM - 1;
	links_share_bandwidth[LOC_NUM - 1][7][1] = 6;
*/
#endif
  for (int i = 0; i < LOC_NUM; i++){
    for(int j = 0; j < LOC_NUM; j++){
      if(i == j || 
      !( (is_in_list(deidxize(i),active_unit_id_list, active_unit_num) && is_in_list(deidxize(j),active_unit_id_list, active_unit_num))
      || (is_in_list(deidxize(i),datalocs, dataloc_num) && is_in_list(deidxize(j),active_unit_id_list, active_unit_num))
      || (is_in_list(deidxize(j),datalocs, dataloc_num) && is_in_list(deidxize(i),active_unit_id_list, active_unit_num))))
        link_bw_shared[i][j] = -1;
      else{
        if(is_in_list(deidxize(i),active_unit_id_list, active_unit_num)) link_active[i][j] = 2;
        else link_active[i][j] = 1;
        double link_slowdown_multiplier = 1.0;
        for (int k = 0; k < LOC_NUM; k++){
          for(int l = 0; l < LOC_NUM; l++){
            if ((k == l) || (i == k && j == l)) continue;
            if((is_in_list(deidxize(l),datalocs, dataloc_num) 
              && is_in_list(deidxize(k),active_unit_id_list, active_unit_num)) 
              && !(links_share_bandwidth[i][j][0] == k && links_share_bandwidth[i][j][1] == l)){
             link_slowdown_multiplier = link_slowdown_multiplier;// + (unit_modeler_list[i]->link[j]->sl[k][l] -1); //fmax(link_slowdown_multiplier, unit_modeler_list[i]->link[j]->sl[k][l]); //
#ifdef DPDEBUG
              if (unit_modeler_list[i]->link[j]->sl[k][l] != 1.0) fprintf(stderr, "ATC::update_link_map_shared():\
                \nFound link (%d -> %d) imposing potential recv-based slowdown to (%d -> %d) with sl = %Lf\n",
                deidxize(l), deidxize(k), deidxize(j), deidxize(i), unit_modeler_list[i]->link[j]->sl[k][l]);
#endif
            }
            /* Do not include output transfer to link slowdown calculation...
            else if((is_in_list(deidxize(k),datalocs, dataloc_num)
                && is_in_list(deidxize(l),active_unit_id_list, active_unit_num))
                && !(links_share_bandwidth[i][j][0] == k && links_share_bandwidth[i][j][1] == l)){
                link_slowdown_multiplier = fmax(link_slowdown_multiplier,unit_modeler_list[i]->link[j]->sl[k][l]);
#ifdef DPDEBUG
                if (unit_modeler_list[i]->link[j]->sl[k][l] != 1.0) fprintf(stderr, "ATC::update_link_map_shared():\
                   \nFound link (%d -> %d) imposing potential recv-based slowdown to (%d -> %d) with sl = %Lf\n",
                   deidxize(l), deidxize(k), deidxize(j), deidxize(i), unit_modeler_list[i]->link[j]->sl[k][l]);
#endif
            }*/
          }
        }
#ifdef PDEBUG
        if(link_slowdown_multiplier!= 1.00) fprintf(stderr, "ATC::update_link_map_shared():\
        \nAdjusting link_bw_shared[%d][%d] with link_slowdown_multiplier = %lf\n", i, j, link_slowdown_multiplier);
#endif
        //if (link_slowdown_multiplier>2) link_slowdown_multiplier = 2;
        if (links_share_bandwidth[i][j][0] != -42 
        && (i*LOC_NUM + j > links_share_bandwidth[i][j][0]*LOC_NUM + links_share_bandwidth[i][j][1]) 
        && (link_bw_shared[links_share_bandwidth[i][j][0]][links_share_bandwidth[i][j][1]]!=-1)) link_bw_shared[i][j] = -1;
        else link_bw_shared[i][j] = link_bw[i][j] * (1/link_slowdown_multiplier);
      }
    }
    /// Normalize costs.
    for(int j = 0; j < LOC_NUM; j++){
      if(i == j || 
      !( (is_in_list(deidxize(i),active_unit_id_list, active_unit_num) && is_in_list(deidxize(j),active_unit_id_list, active_unit_num))
      || (is_in_list(deidxize(i),datalocs, dataloc_num) && is_in_list(deidxize(j),active_unit_id_list, active_unit_num))
      || (is_in_list(deidxize(j),datalocs, dataloc_num) && is_in_list(deidxize(i),active_unit_id_list, active_unit_num)) )) continue;
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
#ifndef ENABLE_TRANSFER_HOPS
#ifdef PDEBUG
  print_link_active();
  print_link_bw_shared();
#endif
#endif
#ifdef DEBUG
  fprintf(stderr, "<-----| update_link_map_shared()\n");
#endif
}

/// Return the bandwidth of a link taking sharing into account
double LinkMap::linkmap_shared_bw_unroll(int dest, int src)
{
	long double bw_actual = 0;
	if (dest == src) error("linkmap_shared_bw_unroll src = dest = %d\n", src);
	else if (link_bw_shared[idxize(dest)][idxize(src)] != -1.0)
		bw_actual = link_bw_shared[idxize(dest)][idxize(src)];
	else if(links_share_bandwidth[idxize(dest)][idxize(src)][0] != -42 )
		bw_actual = link_bw_shared[links_share_bandwidth[idxize(dest)][idxize(src)][0]]
													[links_share_bandwidth[idxize(dest)][idxize(src)][1]];
	else error("linkmap_shared_bw_unroll: link_bw_shared[%d][%d] = %.1lf and is not shared.\n", 
		dest, src, link_bw_shared[idxize(dest)][idxize(src)]);
	return bw_actual;
}

int LinkMap::link_causes_slowdowns(MD_p suspicious_link_modeler, int sus_dest, int sus_src){
  for (int unit_idx = 0 ; unit_idx < LOC_NUM; unit_idx++)
  for (int unit_idy = 0 ; unit_idy < LOC_NUM; unit_idy++) if(link_active[unit_idx][unit_idy]){
    if(suspicious_link_modeler->link[idxize(sus_src)]->sl[unit_idx][unit_idy] > 1.0 + NORMALIZE_NEAR_SPLIT_LIMIT) return 1;
  }
  return 0;
}

#ifdef ENABLE_TRANSFER_HOPS
void LinkMap::update_link_hop_shared_weights(MD_p* unit_modeler_list, int* active_unit_id_list, int active_unit_num){
  if (MAX_ALLOWED_HOPS > 1) error("LinkMap::update_link_hop_shared_weights:"
                                  "Not implemented for MAX_ALLOWED_HOPS = %d\n", MAX_ALLOWED_HOPS);
  for (int unit_idx = 0 ; unit_idx < LOC_NUM; unit_idx++)
  for (int unit_idy = 0 ; unit_idy < LOC_NUM; unit_idy++){
    link_bw_shared_hops[unit_idx][unit_idy] = link_bw_shared[unit_idx][unit_idy];
    if (link_active[unit_idx][unit_idy]){
      int best_list[LOC_NUM], tie_list_num = 0; 
      int dest_loc = deidxize(unit_idx), src_loc = deidxize(unit_idy); 
      double hop_bw_best = linkmap_shared_bw_unroll(dest_loc, src_loc);
      for(int uidx = 0; uidx < LOC_NUM; uidx++)
        if (link_active[uidx][idxize(src_loc)] && link_active[idxize(dest_loc)][uidx]
          && !link_causes_slowdowns(unit_modeler_list[uidx], deidxize(uidx), src_loc) 
          && !link_causes_slowdowns(unit_modeler_list[idxize(dest_loc)], dest_loc, deidxize(uidx))
          && is_in_list(deidxize(uidx), active_unit_id_list, active_unit_num)){
          double hop_est_bw = (1 - HOP_PENALTY) * std::min(linkmap_shared_bw_unroll(deidxize(uidx),src_loc), 
            linkmap_shared_bw_unroll(dest_loc, deidxize(uidx)));
          if (hop_est_bw  > hop_bw_best){
            hop_bw_best = hop_est_bw;
            best_list[0] = deidxize(uidx);
            tie_list_num = 1; 
          }
          else if (hop_est_bw  == hop_bw_best){
            best_list[tie_list_num++] = deidxize(uidx);
          }
        }
      if (tie_list_num){
          link_bw_shared_hops[unit_idx][unit_idy] = hop_bw_best;
          link_hop_num[unit_idx][unit_idy] = 1; 
          link_hop_route_num[unit_idx][unit_idy] = tie_list_num;
          for (int i = 0; i < tie_list_num; i++) link_hop_route[unit_idx][unit_idy][0][i] = best_list[i];
  #ifdef PDEBUG
          fprintf(stderr, "LinkMap::init_hop_routes: %d -> %d transfer sequence -> [ %d ] => ", unit_idy, unit_idx,
            link_hop_route[unit_idx][unit_idy][0][link_hop_route_num[unit_idx][unit_idy]-1]);
          fprintf(stderr, "Cost No-hop = %lf, Hop-adjusted = %lf (%3lf times faster)\n", link_bw_shared[unit_idx][unit_idy],
          link_bw_shared_hops[unit_idx][unit_idy], link_bw_shared_hops[unit_idx][unit_idy]/link_bw_shared[unit_idx][unit_idy]);
  #endif    
      }
    }
  }
  int* datalocs = (int*) malloc(LOC_NUM*sizeof(int)), dataloc_num = 0;
  unit_modeler_list[0]->getDatalocs(&datalocs, &dataloc_num);
  for (int i = 0 ; i < LOC_NUM; i++)
  for (int j = 0 ; j < LOC_NUM; j++){
    if (link_active[i][j] && !link_hop_num[i][j]){
        double link_slowdown_multiplier = 1.0;
        for (int k = 0; k < LOC_NUM; k++){
          for(int l = 0; l < LOC_NUM; l++){
            if ((k == l) || (i == k && j == l)) continue;
            if((is_in_list(deidxize(l),datalocs, dataloc_num) 
              && is_in_list(deidxize(k),active_unit_id_list, active_unit_num)) 
              && !(links_share_bandwidth[i][j][0] == k && links_share_bandwidth[i][j][1] == l)
              && !link_hop_num[k][l]){
             link_slowdown_multiplier = fmax(link_slowdown_multiplier, unit_modeler_list[i]->link[j]->sl[k][l]);
#ifdef DPDEBUG
              if (unit_modeler_list[i]->link[j]->sl[k][l] != 1.0) fprintf(stderr, "ATC::update_link_hop_shared_weights():\
                \nFound link (%d -> %d) imposing potential recv-based slowdown to (%d -> %d) with sl = %Lf\n",
                deidxize(l), deidxize(k), deidxize(j), deidxize(i), unit_modeler_list[i]->link[j]->sl[k][l]);
#endif
            }
            /* Do not include output transfer to link slowdown calculation...
            else if((is_in_list(deidxize(k),datalocs, dataloc_num)
                && is_in_list(deidxize(l),active_unit_id_list, active_unit_num))
                && !(links_share_bandwidth[i][j][0] == k && links_share_bandwidth[i][j][1] == l)){
                link_slowdown_multiplier = fmax(link_slowdown_multiplier,unit_modeler_list[i]->link[j]->sl[k][l]);
#ifdef DPDEBUG
                if (unit_modeler_list[i]->link[j]->sl[k][l] != 1.0) fprintf(stderr, "ATC::update_link_map_shared():\
                   \nFound link (%d -> %d) imposing potential recv-based slowdown to (%d -> %d) with sl = %Lf\n",
                   deidxize(l), deidxize(k), deidxize(j), deidxize(i), unit_modeler_list[i]->link[j]->sl[k][l]);
#endif
            }*/
          }
        }
#ifdef PDEBUG
        if(link_slowdown_multiplier!= 1.00) fprintf(stderr, "ATC::update_link_hop_shared_weights():\
        \nAdjusting link_bw_shared[%d][%d] with link_slowdown_multiplier = %lf\n", i, j, link_slowdown_multiplier);
#endif
        if (link_slowdown_multiplier>2) link_slowdown_multiplier = 2;
        if (link_bw_shared[i][j] != -1) link_bw_shared_hops[i][j] = link_bw_shared[i][j] = link_bw[i][j] * (1/link_slowdown_multiplier);
      }
    }
#ifdef PDEBUG
  print_link_active();
  print_link_bw_shared();
  print_link_bw_shared_hops();
#endif
}
#endif

void LinkMap::print_link_bw_shared_hops(){
  fprintf(stderr,"\n Link Shared-BW Hop Map:\n   |");
  for (int d2 = 0; d2 < LOC_NUM; d2++)
    fprintf(stderr, "  %2d  |", deidxize(d2));
  fprintf(stderr, "\n   |");
  for (int d2 = 0; d2 < LOC_NUM; d2++)
    fprintf(stderr, "-------");
  fprintf(stderr, "\n");
  for (int d1 = 0; d1 < LOC_NUM; d1++){
    fprintf(stderr, "%2d | ", deidxize(d1));
    for (int d2 = 0; d2 < LOC_NUM; d2++){
      fprintf(stderr, "%4.2lf | ", link_bw_shared_hops[d1][d2]);
    }
    fprintf(stderr, "\n");
  }
}

/*
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
    fprintf(stderr, "\n|-----> link_map::ESPA_init(list_of_models = %p, list_of_units = %s, list_of_unit_percentages = %s, init_type = %d)\n\n",
      active_unit_id_list, printlist(active_unit_id_list, active_unit_num),
      (init_type)? printlist(active_unit_score, active_unit_num): "NULL->equal", init_type);
#endif
#ifdef PDEBUG
    fprintf(stderr, "\n|-----> link_map::ESPA_init() Initializing for list_of_units = %s, list_of_unit_percentages = %s, init_type = %d)\n\n",
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
    long long recv_sz_RONLY[5] = 0;
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
	        recv_sz_RONLY[recv_num_RONLY] = recv_bytes;
	        recv_num_RONLY++;
	      }
			}
    }

    if(!strcmp(func, "Dgemm") || !strcmp(func, "Sgemm") ){
      /// TODO: Extra transfers created from internal dims due to multi-unit spliting.
      /// Algorithm may vary for other BLAS3, but not at that bridge yet.
      /// The assumtion for extra transfers is made based on the 2D cyclic distribution,
      /// but the estimation is also useful for other distributions as a best case scenario (worse distributions -> more extra transfers).
      int D1_parts = sqrt(active_unit_num);
      int D2_parts = D1_parts;
      if (D1_parts ==0) { D2_parts = active_unit_num; D1_parts = 1; }
      else { // find the most square decomposition of autotune_controller->active_unit_num in D1_parts x D2_parts 
        int g;
        for (g = D1_parts+1; g>0; --g) if (active_unit_num % g == 0) break;
        if (g==0) { D1_parts = active_unit_num; D2_parts = 1; }
        else { D1_parts = g; D2_parts = active_unit_num/g; }
      }
  #ifdef DPDEBUG
      fprintf(stderr, "LinkMap::ESPA_init(unit_num = %d) : D1_parts = %d, D2_parts = %d\n",
      active_unit_num, D1_parts, D2_parts);
  #endif
      /// Assume extra transfers are split equally between all other units on the decomposition dims
      int row_dec = unit_idx/D1_parts, col_dec = unit_idx%D1_parts; 
      for (int other_loc_idx = 0; other_loc_idx < active_unit_num; other_loc_idx++)
        if(active_unit_id_list[other_loc_idx]!=model->unit_id){
          int other_row_dec = other_loc_idx/D1_parts, other_col_dec = other_loc_idx%D1_parts;
          if(other_row_dec == row_dec) 
            ESPA_bytes[idxize(model->unit_id)][idxize(active_unit_id_list[other_loc_idx])]+= recv_sz_RONLY[0];
          if(other_col_dec == col_dec) 
            ESPA_bytes[idxize(model->unit_id)][idxize(active_unit_id_list[other_loc_idx])]+= recv_sz_RONLY[1];
        }
    }
  }

  for (int idxi = 0; idxi < LOC_NUM; idxi++) for (int idxj = 0; idxj < LOC_NUM; idxj++) 
    update_ESPA_ETA_idx(unit_modeler_list, idxi, idxj);
  update_ESPA_ETA_max();
  update_ESPA_ETA_mean_and_var();
  update_ESPA_ETA_sorted_dec_ids();

#ifdef PDEBUG
  print_ESPA();
#endif
#ifdef DEBUG
  fprintf(stderr, "<-----| LinkMap::ESPA_init()\n");
#endif
}

void LinkMap::ESPA_estimate_hop_routes(MD_p* unit_modeler_list, int* active_unit_id_list, double* active_unit_score,
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
			int best_hop_idx = -42;
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

				if(best_hop_idx!= -42){
					link_hop_route_num[unit_idx][unit_idy] = 1;
					link_hop_num[unit_idx][unit_idy] = 1;
					link_hop_route[unit_idx][unit_idy][link_hop_route_num[unit_idx][unit_idy]-1][0] = deidxize(best_hop_idx);

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
			  		fprintf(stderr, "LinkMap::ESPA_init_hop_routes: %d -> %d transfer sequence -> [ %d ] => ", unit_idy, unit_idx,
			  			deidxize(best_hop_idx));
			  		fprintf(stderr, "Cost No-hop = %lf, Hop-adjusted = %lf (%3lf times faster)\n", link_bw_shared[unit_idx][unit_idy],
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
	else if((init_type && !active_unit_score) || !init_type && active_unit_score) error("link_map::ESPA_predict() called with init_type = %d and active_unit_score = %s\n",
	init_type, printlist<double>(active_unit_score,active_unit_num));
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
		fprintf(stderr, "PARALia LinkMap::ESPA_predict (Unit = %d, Unit_ratio = %.2lf%%):\n"
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
  fprintf(stderr,"\n ESPA bytes Map:\n   |");
  for (int d2 = 0; d2 < LOC_NUM; d2++)
    fprintf(stderr, "  %2d  |", deidxize(d2));
  fprintf(stderr, "\n   |");
  for (int d2 = 0; d2 < LOC_NUM; d2++)
    fprintf(stderr, "-------");
  fprintf(stderr, "\n");
  for (int d1 = 0; d1 < LOC_NUM; d1++){
    fprintf(stderr, "%2d | ", deidxize(d1));
    for (int d2 = 0; d2 < LOC_NUM; d2++){
      fprintf(stderr, "%Le | ", ESPA_bytes[d1][d2]);
    }
    fprintf(stderr, "\n");
  }

  fprintf(stderr,"\n ESPA ETA Map:\n   |");
  for (int d2 = 0; d2 < LOC_NUM; d2++)
    fprintf(stderr, "  %2d  |", deidxize(d2));
  fprintf(stderr, "\n   |");
  for (int d2 = 0; d2 < LOC_NUM; d2++)
    fprintf(stderr, "-------");
  fprintf(stderr, "\n");
  for (int d1 = 0; d1 < LOC_NUM; d1++){
    fprintf(stderr, "%2d | ", deidxize(d1));
    for (int d2 = 0; d2 < LOC_NUM; d2++){
      fprintf(stderr, "%le | ", ESPA_ETA[d1][d2]);
    }
    fprintf(stderr, "\n");
  }

  fprintf(stderr,"\n ESPA Longest ETA Map (Top 10 from ESPA_ETA_sorted_dec_ids):\n");
  int top_print = ((LOC_NUM*LOC_NUM < 10) ? LOC_NUM*LOC_NUM : 10);
  for (int unit_ctr = 0 ; unit_ctr < top_print; unit_ctr++){
	int unit_idx = ESPA_ETA_sorted_dec_ids[unit_ctr]/LOC_NUM, unit_idy = ESPA_ETA_sorted_dec_ids[unit_ctr]%LOC_NUM;
	fprintf(stderr, "ESPA_ETA_sorted_dec_ids[%d] = %d : ESPA_ETA[%d][%d] = %lf\n", unit_ctr, ESPA_ETA_sorted_dec_ids[unit_ctr], unit_idx, unit_idy, ESPA_ETA[unit_idx][unit_idy]);
  }
}
*/

void LinkMap::print_link_active(){
  fprintf(stderr,"\n Link Active Map:\n   |");
  for (int d2 = 0; d2 < LOC_NUM; d2++)
    fprintf(stderr, " %2d |", deidxize(d2));
  fprintf(stderr, "\n   |");
  for (int d2 = 0; d2 < LOC_NUM; d2++)
    fprintf(stderr, "-----");
  fprintf(stderr, "\n");
  for (int d1 = 0; d1 < LOC_NUM; d1++){
    fprintf(stderr, "%2d | ", deidxize(d1));
    for (int d2 = 0; d2 < LOC_NUM; d2++){
      fprintf(stderr, "%2d | ", link_active[d1][d2]);
    }
    fprintf(stderr, "\n");
  }
}

void LinkMap::print_link_bw(){
  fprintf(stderr,"\n Link BW Map:\n   |");
  for (int d2 = 0; d2 < LOC_NUM; d2++)
    fprintf(stderr, "  %2d  |", deidxize(d2));
  fprintf(stderr, "\n   |");
  for (int d2 = 0; d2 < LOC_NUM; d2++)
    fprintf(stderr, "-------");
  fprintf(stderr, "\n");
  for (int d1 = 0; d1 < LOC_NUM; d1++){
    fprintf(stderr, "%2d | ", deidxize(d1));
    for (int d2 = 0; d2 < LOC_NUM; d2++){
      fprintf(stderr, "%4.2lf | ", link_bw[d1][d2]);
    }
    fprintf(stderr, "\n");
  }
}

void LinkMap::print_link_bw_shared(){
  fprintf(stderr,"\n Link Shared-BW Map:\n   |");
  for (int d2 = 0; d2 < LOC_NUM; d2++)
    fprintf(stderr, "  %2d  |", deidxize(d2));
  fprintf(stderr, "\n   |");
  for (int d2 = 0; d2 < LOC_NUM; d2++)
    fprintf(stderr, "-------");
  fprintf(stderr, "\n");
  for (int d1 = 0; d1 < LOC_NUM; d1++){
    fprintf(stderr, "%2d | ", deidxize(d1));
    for (int d2 = 0; d2 < LOC_NUM; d2++){
      fprintf(stderr, "%4.2lf | ", link_bw_shared[d1][d2]);
    }
    fprintf(stderr, "\n");
  }
}
