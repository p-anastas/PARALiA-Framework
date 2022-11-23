
#include <iostream>

#include "Autotuning_runtime.hpp"
#include "unihelpers.hpp"

LinkMap::LinkMap(){}

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
    }
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
      for (int k = 0; k < MAX_ALLOWED_HOPS; k++)
        for (int l = 0; l < MAX_HOP_ROUTES; l++)
            link_hop_route[i][j][k][l] = 0;
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
    lprintf(lvl, "|-----> LinkMap::update_link_weights(list_of_models = %p, T = %d)\n",
    list_of_models, T);
#endif
#ifdef PDEBUG
    lprintf(lvl, "|-----> LinkMap::update_link_weights(list_of_models = %p, T = %d)\n",
    list_of_models, T);
    print_link_bw();
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
  lprintf(lvl, "|-----> LinkMap::update_link_shared_weights(unit_list = %s, datalocs = %s)\n",
    printlist<int>(active_unit_id_list, active_unit_num), printlist<int>(datalocs, dataloc_num));
  print_link_bw_shared();
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
             link_slowdown_multiplier = link_slowdown_multiplier + (unit_modeler_list[i]->link[j]->sl[k][l] - 1);
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
                link_slowdown_multiplier = link_slowdown_multiplier +  (unit_modeler_list[i]->link[j]->sl[hoplocs[idx]][hoplocs[idx+1]] - 1);
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
#ifdef DPDEBUG
        if(link_slowdown_multiplier!= 1.00) lprintf(lvl, "ATC::update_link_map_shared():\
        \nAdjusting link_bw_shared[%d][%d] with link_slowdown_multiplier = %lf\n", i, j, link_slowdown_multiplier);
#endif
        if(link_slowdown_multiplier>2) link_slowdown_multiplier = 2; 
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
        if(abs(link_bw_shared[i][j] - link_bw_shared[i][k])
          /link_bw_shared[i][j] < NORMALIZE_NEAR_SPLIT_LIMIT){
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
#ifdef DSTEST
      lprintf(1, "InitHopMap: %2d->%2d transfer sequence -> \n", unit_idy, unit_idx);
      for (int routes = 0; routes < link_hop_route_num[unit_idx][unit_idy]; routes++){
        lprintf(1,"Route %d => [ ", routes);
        for (int hops = 0; hops < link_hop_num[unit_idx][unit_idy]; hops++)
          lprintf(0, "%d ", link_hop_route[unit_idx][unit_idy][routes][hops]);
        lprintf(0,"]\n");
      }
      lprintf(0, "Cost No-hop = %lf, Hop-adjusted = %lf (%3lf times faster)\n", link_bw_shared[unit_idx][unit_idy],
        link_bw_shared_hops[unit_idx][unit_idy], link_bw_shared_hops[unit_idx][unit_idy]/link_bw_shared[unit_idx][unit_idy]);
#endif
    }
    else link_bw_shared_hops[unit_idx][unit_idy] = link_bw_shared[unit_idx][unit_idy];
  }
}

#ifdef ENABLE_ESPA
void LinkMap::ESPA_init(MD_p* unit_modeler_list, int* active_unit_id_list, double* active_unit_score,
  int active_unit_num, int init_type){
  short lvl = 4;
#ifdef DEBUG
    lprintf(lvl, "|-----> link_map::ESPA_init(list_of_models = %p, list_of_units = %s, list_of_unit_percentages = %s, init_type = %d)\n",
      active_unit_id_list, printlist(active_unit_id_list, active_unit_num),
      (init_type)? printlist(active_unit_score, active_unit_num): "NULL->equal", init_type);
#endif
#ifdef PDEBUG
    lprintf(lvl, "|-----> link_map::ESPA_init() Initializing for list_of_units = %s, list_of_unit_percentages = %s, init_type = %d)\n",
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
      ESPA_bytes[idxize(model->unit_id)][idxize(model->V->loc[i])]+= recv_bytes;
      ESPA_bytes[idxize(model->V->loc[i])][idxize(model->unit_id)]+= send_bytes;

      if(!model->V->out[i] && model->V->loc[i] != model->unit_id) {
        recv_sz_RONLY+= recv_bytes;
        recv_num_RONLY++;
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
  for (int idxi = 0; idxi < LOC_NUM; idxi++) for (int idxj = 0; idxj < LOC_NUM; idxj++)
    ESPA_ETA[idxi][idxj] = t_com_predict_shared(unit_modeler_list[idxi]->link[idxj], ESPA_bytes[idxi][idxj]);
#ifdef PDEBUG
  print_ESPA();
#endif
#ifdef DEBUG
  lprintf(lvl, "<-----| LinkMap::ESPA_init()\n");
#endif
}
#endif

#endif

void LinkMap::ESPA_init_hop_routes(MD_p* unit_modeler_list, int* active_unit_id_list, double* active_unit_score,
  int active_unit_num, int init_type){
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
#ifdef DSTEST
      lprintf(1, "InitHopMap: %2d->%2d transfer sequence -> \n", unit_idy, unit_idx);
      for (int routes = 0; routes < link_hop_route_num[unit_idx][unit_idy]; routes++){
        lprintf(1,"Route %d => [ ", routes);
        for (int hops = 0; hops < link_hop_num[unit_idx][unit_idy]; hops++)
          lprintf(0, "%d ", link_hop_route[unit_idx][unit_idy][routes][hops]);
        lprintf(0,"]\n");
      }
      lprintf(0, "Cost No-hop = %lf, Hop-adjusted = %lf (%3lf times faster)\n", link_bw_shared[unit_idx][unit_idy],
        link_bw_shared_hops[unit_idx][unit_idy], link_bw_shared_hops[unit_idx][unit_idy]/link_bw_shared[unit_idx][unit_idy]);
#endif
    }
    else link_bw_shared_hops[unit_idx][unit_idy] = link_bw_shared[unit_idx][unit_idy];
  }
}

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

#ifdef ENABLE_TRANSFER_HOPS
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
}

#endif
