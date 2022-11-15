
#include <iostream>

#include "Autotuning_runtime.hpp"
#include "unihelpers.hpp"

double link_bw[LOC_NUM][LOC_NUM];
double link_shared_bw[LOC_NUM][LOC_NUM];
double link_bw_hop[LOC_NUM][LOC_NUM];
double link_shared_bw_hop[LOC_NUM][LOC_NUM];


void InitHopMap(MD_p* unit_modeler_list, double link_bw_in [][LOC_NUM], double link_bw_out [][LOC_NUM], int* active_unit_id_list, int active_unit_num){
  double safe_hop_penalty = HOP_PENALTY;
  if ( HOP_PENALTY <= FETCH_UNAVAILABLE_PENALTY){
#ifdef PDEBUG
    warning("HOP_PENALTY(=%lf) should be larger than FETCH_UNAVAILABLE_PENALTY(=%lf) unless you like potantially infinite transfer circles.\
    \nIf you do, feel free to implement them cause currently they are not supported (e.g. X->Y->Z must always be assumed more expensive than Y->Z)\n",
      HOP_PENALTY, FETCH_UNAVAILABLE_PENALTY);
#endif
    safe_hop_penalty = FETCH_UNAVAILABLE_PENALTY*1.01;
  }
  for (int unit_idx = 0 ; unit_idx < LOC_NUM; unit_idx++)
  for (int unit_idy = 0 ; unit_idy < LOC_NUM; unit_idy++){
    for (int hops = 0 ; hops < MAX_ALLOWED_HOPS; hops++) for (int rt = 0 ; rt < MAX_HOP_ROUTES; rt++)
      link_hop_route[unit_idx][unit_idy][rt][hops] = -42;
    link_hop_num[unit_idx][unit_idy] = 0;
    double max_hop_bw = link_bw_in[unit_idx][unit_idy];
    link_hop_route_num[unit_idx][unit_idy] = 0;
    if(MAX_ALLOWED_HOPS >= 1){
      for (int hop_idx = 0 ; hop_idx < LOC_NUM; hop_idx++)
      if(hop_idx!= unit_idx && hop_idx!= unit_idy && unit_idx!= unit_idy){
        if(!is_in_list(deidxize(hop_idx),active_unit_id_list, active_unit_num)) continue;
        double hop_bw =  fmin(link_bw_in[unit_idx][hop_idx], link_bw_in[hop_idx][unit_idy]);
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
        double hop_bw =  fmin(link_bw_in[unit_idx][hop_idx], fmin(link_bw_in[hop_idx][hop_idy], link_bw_in[hop_idy][unit_idy]));
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
    link_bw_out[unit_idx][unit_idy] = max_hop_bw;
#ifdef DSTEST
      lprintf(1, "InitHopMap: %2d->%2d transfer sequence -> \n", unit_idy, unit_idx);
      for (int routes = 0; routes < link_hop_route_num[unit_idx][unit_idy]; routes++){
        lprintf(1,"Route %d => [ ", routes);
        for (int hops = 0; hops < link_hop_num[unit_idx][unit_idy]; hops++)
          lprintf(0, "%d ", link_hop_route[unit_idx][unit_idy][routes][hops]);
        lprintf(0,"]\n");
      }
      lprintf(0, "Cost No-hop = %lf, Hop-adjusted = %lf (%3lf times faster)\n", link_bw_in[unit_idx][unit_idy],
        link_bw_out[unit_idx][unit_idy], link_bw_out[unit_idx][unit_idy]/link_bw_in[unit_idx][unit_idy]);
#endif
    }
    else link_bw_out[unit_idx][unit_idy] = link_bw_in[unit_idx][unit_idy];
  }
}

void ATC::update_link_map_shared(){
  short lvl = 3;
#ifdef DEBUG
    lprintf(lvl, "|-----> update_link_map_shared(LOC_NUM = %d)\n", LOC_NUM);
#endif
  int* datalocs = (int*) malloc(LOC_NUM*sizeof(int)), dataloc_num = 0;
  unit_modeler_list[0]->getDatalocs(&datalocs, &dataloc_num);
  //double send_ratio = unit_modeler_list[0]->getSendRatio(), recv_ratio = unit_modeler_list[0]->getRecvRatio();
  if (!dataloc_num)
    error("Called ATC::update_link_map_shared() without properly initalized model in unit_modeler_list[0]\n");
#ifdef PDEBUG
    lprintf(lvl, "ATC::update_link_map_shared( %s, %d )\n", printlist<int>(datalocs, dataloc_num), dataloc_num);
    print();
#endif
  //int pred_T_dim = 0;
  //if(T < 1) pred_T_dim = 2048;
  //else pred_T_dim = T;
  for (int i = 0; i < LOC_NUM; i++){
    for(int j = 0; j < LOC_NUM; j++){
      if(i == j) link_shared_bw[i][j] = 0;
      else{
        //link_shared_bw[i][j] = Gval_per_s(pred_T_dim*pred_T_dim*sizeof(VALUE_TYPE),
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
                link_slowdown_multiplier = fmax(link_slowdown_multiplier, unit_modeler_list[i]->link[j]->sl[hoplocs[idx]][hoplocs[idx+1]]);
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
        \nAdjusting link_shared_bw[%d][%d] with link_slowdown_multiplier = %lf\n", i, j, link_slowdown_multiplier);
#endif
        link_shared_bw[i][j] = link_bw[i][j] * (1/link_slowdown_multiplier);
      }
    }
    /// Normalize costs.
    for(int j = 0; j < LOC_NUM; j++){
      if(i == j) continue;
      int flag_normalize[LOC_NUM] = {0}, normalize_num = 1;
      double normalize_sum = link_shared_bw[i][j];
      flag_normalize[j] = 1;
      for (int k = j + 1; k < LOC_NUM; k++)
        if(abs(link_shared_bw[i][j] - link_shared_bw[i][k])
          /link_shared_bw[i][j] < NORMALIZE_NEAR_SPLIT_LIMIT){
          flag_normalize[k] = 1;
          normalize_sum+=link_shared_bw[i][k];
          normalize_num++;
        }
      for (int k = j ; k < LOC_NUM; k++) if(flag_normalize[k]) link_shared_bw[i][k] = normalize_sum/normalize_num;
    }
  }
#ifdef PDEBUG
  link_shared_bw_map_print();
#endif
#ifdef ENABLE_TRANSFER_HOPS
  InitHopMap(unit_modeler_list, link_shared_bw, link_shared_bw_hop, active_unit_id_list, active_unit_num);
#ifdef PDEBUG
  link_shared_bw_hop_map_print();
#endif
#endif
#ifdef DEBUG
  lprintf(lvl, "<-----| update_link_map_shared()\n");
#endif
}


link_map::link_map(MD_p* list_of_models, int* list_of_units, int* list_of_unit_percentages,
  int unit_num, int init_type){
  short lvl = 4;
#ifdef DEBUG
    lprintf(lvl, "|-----> link_map::link_map(list_of_models = %p, list_of_units = %s, list_of_unit_percentages = %s, init_type = %d)\n",
      list_of_models, printlist(list_of_units, unit_num), printlist(list_of_unit_percentages, unit_num), init_type);
#endif
  for (int itt_dest = 0; itt_dest < LOC_NUM; itt_dest++)
    for (int itt_src = 0; itt_src < LOC_NUM; itt_src++){
      link_model[itt_dest][itt_src] = list_of_models[itt_dest]->link[itt_src];
      hop_num[itt_dest][itt_src] = 0;
      for (int i = 0; i < model->V->numT; i++){
        long long tmp_recv_sz = (long long) model->V->in[i]*(*model->V->Dim1[i])*(*model->V->Dim2[i])*
          model->V->dtype_sz*active_unit_score[used_unit_idx];
        long long tmp_send_sz =  (long long) model->V->out[i]*(*model->V->Dim1[i])*(*model->V->Dim2[i])*
          model->V->dtype_sz*active_unit_score[used_unit_idx];
      }
    }
  long long recv_sz = 0, recv_sz_RONLY = 0, send_sz = 0;
  int recv_num_RONLY = 0;

    long long tmp_recv_sz = (long long) model->V->in[i]*(*model->V->Dim1[i])*(*model->V->Dim2[i])*
      model->V->dtype_sz*active_unit_score[used_unit_idx];
    long long tmp_send_sz =  (long long) model->V->out[i]*(*model->V->Dim1[i])*(*model->V->Dim2[i])*
      model->V->dtype_sz*active_unit_score[used_unit_idx];
    double t_recv_tmp = t_com_predict_shared(model->link[idxize(model->V->loc[i])], tmp_recv_sz);
    double t_send_tmp = t_com_predict_shared(model->link[idxize(model->V->out_loc[i])], tmp_send_sz);
    if(t_recv_tmp < 0 || t_send_tmp < 0 ){
        warning("PARALiaPredictLinkHeteroBLAS3: t_com_predict submodel idx = %d\
          returned negative value, abort prediction", idxize(model->V->loc[i]));
        return -1.0;
    }
    recv_sz += tmp_recv_sz;
    send_sz += tmp_send_sz;
    if(!model->V->out[i] && model->V->loc[i] != model->unit_id) { recv_sz_RONLY+= recv_sz; recv_num_RONLY++; }
    t_recv_full+= t_recv_tmp;
    t_send_full+= t_send_tmp;
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
  lprintf(lvl, "PARALiaPredictLinkHeteroBLAS3(unit_num = %d) : D1_parts = %d, D2_parts = %d, extra_transfer_ratio = %lf\n",
  active_unit_num, D1_parts, D2_parts, extra_transfer_ratio);
  #endif
  long long recv_sz_extra = extra_transfer_ratio * recv_sz_RONLY;
  //double t_recv_extra_optimistic = model->predictBestFriends_t(extra_transfer_ratio, recv_sz_extra, active_unit_num, active_unit_id_list);
  double t_recv_extra_pesimistic = model->predictAvgBw_t(recv_sz_extra, active_unit_num, active_unit_id_list);

  double t_recv_extra = t_recv_extra_pesimistic; // (t_recv_extra_optimistic + t_recv_extra_pesimistic)/2;

}

void ATC::update_link_map(){
short lvl =1;
#ifdef DEBUG
    lprintf(lvl, "|-----> update_link_map(LOC_NUM = %d)\n", LOC_NUM);
#endif
#ifdef PDEBUG
    print();
#endif
  int pred_T_dim = 0;
  if(T < 1) pred_T_dim = 2048;
  else pred_T_dim = T;
  for (int i = 0; i < LOC_NUM; i++){
    for(int j = 0; j < LOC_NUM; j++){
      if(i == j) link_bw[i][j] = 0;
      else link_bw[i][j] = Gval_per_s(pred_T_dim*pred_T_dim*sizeof(VALUE_TYPE),
      t_com_predict(unit_modeler_list[i]->revlink[j], pred_T_dim*pred_T_dim*sizeof(VALUE_TYPE)));
    }
  }
  normalize_2D_LOC_NUM(link_bw, LOC_NUM, NORMALIZE_NEAR_SPLIT_LIMIT);
#ifdef PDEBUG
  link_bw_map_print();
#endif
#ifdef ENABLE_TRANSFER_HOPS
  InitHopMap(unit_modeler_list, link_bw, link_bw_hop, active_unit_id_list, active_unit_num);
#ifdef PDEBUG
  link_bw_hop_map_print();
#endif
#endif
#ifdef DEBUG
  lprintf(lvl, "<-----| update_link_map()\n");
#endif
}

void link_bw_map_print(){
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

void link_shared_bw_map_print(){
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
      lprintf(0, "%4.2lf | ", link_shared_bw[d1][d2]);
    }
    lprintf(0, "\n");
  }
}

#ifdef ENABLE_TRANSFER_HOPS
void link_bw_hop_map_print(){
  lprintf(0,"\n Link BW Hop Map:\n   |");
  for (int d2 = 0; d2 < LOC_NUM; d2++)
    lprintf(0, "  %2d  |", deidxize(d2));
  lprintf(0, "\n   |");
  for (int d2 = 0; d2 < LOC_NUM; d2++)
    lprintf(0, "-------");
  lprintf(0, "\n");
  for (int d1 = 0; d1 < LOC_NUM; d1++){
    lprintf(0, "%2d | ", deidxize(d1));
    for (int d2 = 0; d2 < LOC_NUM; d2++){
      lprintf(0, "%4.2lf | ", link_bw_hop[d1][d2]);
    }
    lprintf(0, "\n");
  }
}

void link_shared_bw_hop_map_print(){
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
      lprintf(0, "%4.2lf | ", link_shared_bw_hop[d1][d2]);
    }
    lprintf(0, "\n");
  }
}
#endif
