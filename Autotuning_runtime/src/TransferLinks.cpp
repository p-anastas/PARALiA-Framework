
#include <iostream>

#include "Autotuning_runtime.hpp"
#include "unihelpers.hpp"

double link_bw[LOC_NUM][LOC_NUM];
double link_shared_bw[LOC_NUM][LOC_NUM];

#ifdef ENABLE_TRANSFER_HOPS
short link_hop_num[LOC_NUM][LOC_NUM];
short link_hop_route[LOC_NUM][LOC_NUM][MAX_ALLOWED_HOPS];
double link_bw_hop[LOC_NUM][LOC_NUM];

void InitHopMap(double link_bw [][LOC_NUM], double link_bw_out [][LOC_NUM]){
  double safe_hop_penalty = HOP_PENALTY;
  if ( HOP_PENALTY <= FETCH_UNAVAILABLE_PENALTY){
    warning("HOP_PENALTY(=%lf) should be larger than FETCH_UNAVAILABLE_PENALTY(=%lf) unless you like potantially infinite transfer circles.\
    \nIf you do, feel free to implement them cause currently they are not supported (e.g. X->Y->Z must always be assumed more expensive than Y->Z)\n",
      HOP_PENALTY, FETCH_UNAVAILABLE_PENALTY);
    safe_hop_penalty = FETCH_UNAVAILABLE_PENALTY*1.01;
  }
  for (int unit_idx = 0 ; unit_idx < LOC_NUM; unit_idx++)
  for (int unit_idy = 0 ; unit_idy < LOC_NUM; unit_idy++){
    for (int hops = 0 ; hops < MAX_ALLOWED_HOPS; hops++) link_hop_route[unit_idx][unit_idy][hops] = -42;
    link_hop_num[unit_idx][unit_idy] = 0;
    double max_hop_bw = link_bw[unit_idx][unit_idy];
    if(MAX_ALLOWED_HOPS >= 1){
      for (int hop_idx = 0 ; hop_idx < LOC_NUM; hop_idx++)
      if(hop_idx!= unit_idx && hop_idx!= unit_idy && unit_idx!= unit_idy){
        double hop_bw =  fmin(link_bw[unit_idx][hop_idx], link_bw[hop_idx][unit_idy]);
        hop_bw-= safe_hop_penalty*hop_bw;
        if (hop_bw > max_hop_bw){
         max_hop_bw = hop_bw;
         link_hop_route[unit_idx][unit_idy][0] = hop_idx;
         link_hop_num[unit_idx][unit_idy] = 1;
        }
      }
    }
    if(MAX_ALLOWED_HOPS >= 2){
      for (int hop_idx = 0 ; hop_idx < LOC_NUM; hop_idx++)
      for (int hop_idy = 0 ; hop_idy < LOC_NUM; hop_idy++)
      if(hop_idx!= unit_idx && hop_idx!= unit_idy && unit_idx!= unit_idy &&
      hop_idy!= unit_idx && hop_idy!= unit_idy && hop_idy!= hop_idx){
        double hop_bw =  fmin(link_bw[unit_idx][hop_idx], fmin(link_bw[hop_idx][hop_idy], link_bw[hop_idy][unit_idy]));
        hop_bw-= 2*safe_hop_penalty*hop_bw;
        if (hop_bw > max_hop_bw){
          max_hop_bw = hop_bw;
          link_hop_route[unit_idx][unit_idy][0] = hop_idy;
          link_hop_route[unit_idx][unit_idy][1] = hop_idx;
          link_hop_num[unit_idx][unit_idy] = 2;
        }
      }
    }
    if (link_hop_num[unit_idx][unit_idy]){
    link_bw_out[unit_idx][unit_idy] = max_hop_bw;
#ifdef DSTEST
      lprintf(1, "InitHopMap: %2d->%2d transfer sequence -> %s -> ", unit_idy, unit_idx,
        printlist(link_hop_route[unit_idx][unit_idy], link_hop_num[unit_idx][unit_idy]));
      lprintf(0, "Cost No-hop = %lf, Hop-adjusted = %lf (%3lf times faster)\n", link_bw[unit_idx][unit_idy],
        link_bw_out[unit_idx][unit_idy], link_bw_out[unit_idx][unit_idy]/link_bw[unit_idx][unit_idy]);
#endif
    }
    else link_bw_out[unit_idx][unit_idy] = link_bw[unit_idx][unit_idy];
  }
}
#endif

void ATC::update_link_shared_weights(){
  short lvl = 3;
#ifdef DEBUG
    lprintf(lvl, "|-----> update_link_shared_weights(LOC_NUM = %d)\n", LOC_NUM);
#endif
  int* datalocs = (int*) malloc(LOC_NUM*sizeof(int)), dataloc_num = 0;
  unit_modeler_list[0]->getDatalocs(&datalocs, &dataloc_num);
  //double send_ratio = unit_modeler_list[0]->getSendRatio(), recv_ratio = unit_modeler_list[0]->getRecvRatio();
  if (!dataloc_num)
    error("Called ATC::update_link_shared_weights() without properly initalized model in unit_modeler_list[0]\n");
#ifdef PDEBUG
    lprintf(lvl, "ATC::update_link_shared_weights( %s, %d )\n", printlist<int>(datalocs, dataloc_num), dataloc_num);
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
            if((is_in_list(deidxize(l),datalocs, dataloc_num) && is_in_list(deidxize(k),active_unit_id_list, active_unit_num)) &&
              (deidxize(l) == -1 || deidxize(k) == -1) && (deidxize(i) == -1 || deidxize(j) == -1)){ /// FIXME:This should have been a check regarding both link passing from PCIe/ non-NVLINK connections and its not modelling.
             link_slowdown_multiplier = fmax(link_slowdown_multiplier, unit_modeler_list[i]->link[j]->sl[k][l]);
#ifdef DPDEBUG
              if (unit_modeler_list[i]->link[j]->sl[k][l] != 1.0) lprintf(lvl, "ATC::update_link_shared_weights():\
                \nFound link (%d -> %d) imposing potential recv-based slowdown to (%d -> %d) with sl = %Lf\n",
                deidxize(l), deidxize(k), deidxize(j), deidxize(i), unit_modeler_list[i]->link[j]->sl[k][l]);
#endif
            }
/*            if(is_in_list(deidxize(k),datalocs, dataloc_num) && is_in_list(deidxize(l),active_unit_id_list, active_unit_num)){
              link_slowdown_multiplier = fmax(link_slowdown_multiplier, unit_modeler_list[i]->link[j]->sl[k][l]);
//#ifdef DPDEBUG
              if (unit_modeler_list[i]->link[j]->sl[k][l] != 1.0) lprintf(lvl, "ATC::update_link_shared_weights():\
                \nFound link (%d -> %d) imposing potential send-based slowdown to (%d -> %d) with sl = %Lf\n",
                deidxize(l), deidxize(k), deidxize(j), deidxize(i), unit_modeler_list[i]->link[j]->sl[k][l]);
//#endif
            }*/
          }
        }
#ifdef DPDEBUG
        if(link_slowdown_multiplier!= 1.00) lprintf(lvl, "ATC::update_link_shared_weights():\
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
//#ifdef PDEBUG
  link_shared_bw_map_print();
//#endif
#ifdef DEBUG
  lprintf(lvl, "<-----| update_link_shared_weights()\n");
#endif
}

void ATC::update_link_weights(){
  short lvl = 3;
#ifdef DEBUG
    lprintf(lvl, "|-----> update_link_weights(LOC_NUM = %d)\n", LOC_NUM);
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
    for(int j = 0; j < LOC_NUM; j++){
      if(i == j) continue;
      int flag_normalize[LOC_NUM] = {0}, normalize_num = 1;
      double normalize_sum = link_bw[i][j];
      flag_normalize[j] = 1;
      for (int k = j + 1; k < LOC_NUM; k++)
        if(abs(link_bw[i][j] - link_bw[i][k])
          /link_bw[i][j] < NORMALIZE_NEAR_SPLIT_LIMIT){
          flag_normalize[k] = 1;
          normalize_sum+=link_bw[i][k];
          normalize_num++;
        }
      for (int k = j ; k < LOC_NUM; k++) if(flag_normalize[k]) link_bw[i][k] = normalize_sum/normalize_num;
    }
  }
#ifdef ENABLE_TRANSFER_HOPS
  InitHopMap(link_bw, link_bw_hop);
#endif
#ifdef PDEBUG
  link_bw_map_print();
#endif
#ifdef DEBUG
  lprintf(lvl, "<-----| update_link_weights()\n");
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
