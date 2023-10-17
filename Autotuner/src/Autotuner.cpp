///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The Autotuner controller functions.
///

#include "Autotuner.hpp"
#include "Subkernel_distributions.hpp"

#include <float.h> /// For DBL_MAX

/********************** Initialization/Modification ***************************/
ATC::ATC(){
	short lvl = 2;
#ifdef DEBUG
		fprintf(stderr,  "|-----> ATC::ATC\n");
#endif
	active_unit_id_list = (int*) malloc(LOC_NUM*sizeof(int));
	active_unit_score = (double*) malloc(LOC_NUM*sizeof(double));
	Subkernels_per_unit_num = (int*) malloc(LOC_NUM*sizeof(int));
	Subkernels_per_unit_list = (int**) malloc(LOC_NUM*sizeof(int*));
	for (int d = 0; d < LOC_NUM; d++){
		active_unit_score[d] = -42.0;
		Subkernels_per_unit_list[d] = NULL;
		Subkernels_per_unit_num[d] = 0;
	}
	T = active_unit_num = subkernel_num = -1;
	pred_t = pred_J = power_delay = energy_delay = -1.0;
	cache_limit = 0;

	unit_modeler_list = (MD_p*) malloc(LOC_NUM*sizeof(MD_p));
	linkmap = new LinkMap();
#ifdef DEBUG
	fprintf(stderr,  "<-----|\n");
#endif
}

ATC::~ATC(){
	short lvl = 2;
#ifdef DEBUG
		fprintf(stderr,  "|-----> ATC::~ATC\n");
#endif
	free(active_unit_id_list);
	free(active_unit_score);
	free(Subkernels_per_unit_num);
	for (int d = 0; d < LOC_NUM; d++) free(Subkernels_per_unit_list[d]);
	free(Subkernels_per_unit_list);
#ifdef DEBUG
	fprintf(stderr,  "<-----|\n");
#endif
}

void ATC::init_modelers(const char* routine_name, void* initial_problem_wrap){
	for(int i = 0; i < LOC_NUM; i++) unit_modeler_list[i] = new Modeler(deidxize(i), routine_name, initial_problem_wrap);
}
void ATC::reset(){
	short lvl = 4;
#ifdef DEBUG
		fprintf(stderr,  "|-----> ATC::reset\n");
#endif
	for (int d = 0; d < LOC_NUM; d++){
		free(Subkernels_per_unit_list[d]);
		Subkernels_per_unit_list[d] = NULL;
		Subkernels_per_unit_num[d] = 0;
	}
	T = active_unit_num = subkernel_num = -1;
	pred_t = -1.0;
	cache_limit = 0;
#ifdef DEBUG
	fprintf(stderr,  "<-----|\n");
#endif
}

int ATC::diff_intialized_params_ATC(ATC_p other_ATC){
	short lvl = 3;
	#ifdef DEBUG
		fprintf(stderr,  "|-----> ATC::diff_intialized_params_ATC(other_ATC = %p)\n", other_ATC);
	#endif
	int result = 0;
	if(other_ATC->T != -1 && other_ATC->T != T){
		result++;
#ifdef PDEBUG
		fprintf(stderr,  "ATC::diff_intialized_params_ATC(): other_ATC->T = %ld, T = %ld\n", other_ATC->T, T);
#endif
		}
	if(other_ATC->active_unit_num != -1 && other_ATC->active_unit_num != active_unit_num){
		result++;
#ifdef PDEBUG
		fprintf(stderr,  "ATC::diff_intialized_params_ATC(): other_ATC->active_unit_num = %d, active_unit_num = %d\n",
			other_ATC->active_unit_num, active_unit_num);
#endif
	}
	else if(other_ATC->active_unit_num != -1 && other_ATC->active_unit_num == active_unit_num){
		for (int ctr = 0; ctr < active_unit_num; ctr++) if(other_ATC->active_unit_id_list[ctr] != active_unit_id_list[ctr]){
			result++;
#ifdef PDEBUG
		fprintf(stderr,  "ATC::diff_intialized_params_ATC(): other_ATC->active_unit_id_list[%d] = %d, active_unit_id_list[%d] = %d\n",
			ctr, other_ATC->active_unit_id_list[ctr], ctr, active_unit_id_list[ctr]);
#endif
			break;
		}
	}
	if(other_ATC->cache_limit != 0 && other_ATC->cache_limit != cache_limit){
		result++;
#ifdef PDEBUG
		fprintf(stderr,  "ATC::diff_intialized_params_ATC(): other_ATC->cache_limit = %lld, cache_limit = %lld\n",
			other_ATC->cache_limit, cache_limit);
#endif
	}
#ifdef DEBUG
	fprintf(stderr,  "<-----|\n");
#endif
	return result;
}

void ATC::mimic_ATC(ATC_p other_ATC){
	short lvl = 3;
	#ifdef DEBUG
		fprintf(stderr,  "|-----> ATC::mimic_ATC(other_ATC = %p)\n", other_ATC);
	#endif
	T = other_ATC->T;
	active_unit_num = other_ATC->active_unit_num;
	for (int d = 0; d < other_ATC->active_unit_num; d++) active_unit_id_list[d] = other_ATC->active_unit_id_list[d];
	for (int d = 0; d < other_ATC->active_unit_num; d++) active_unit_score[d] = other_ATC->active_unit_score[d];
	pred_t = other_ATC->pred_t;
	pred_J = other_ATC->pred_J;
	power_delay = other_ATC->power_delay;
	energy_delay = other_ATC->energy_delay;
	cache_limit = other_ATC->cache_limit;
	linkmap->copy(other_ATC->linkmap);

	if(subkernel_num != -1){
		for (int d = 0; d < LOC_NUM; d++){
			//fprintf(stderr,"Subkernels_per_unit_list[%d] = %p\n", d, Subkernels_per_unit_list[d]);
			//free(Subkernels_per_unit_list[d]);
			//Subkernels_per_unit_num[d] = 0;
			;//Subkernels_per_unit_list[d] = NULL;
		} /// TODO: Got some "double free 2cache" error when used in many different mimicked ATCs ->
			/// potential problem here  ATC::update_sk_num resizing Subkernels_per_unit_list[d] might be solution and/or relevant.
		subkernel_num = -1;
	}

	if (other_ATC->subkernel_num != -1){
		for (int d = 0; d < other_ATC->active_unit_num; d++){
			Subkernels_per_unit_num[d] = other_ATC->Subkernels_per_unit_num[d];
			free(Subkernels_per_unit_list[d]);
			Subkernels_per_unit_list[d] = (int*) malloc(other_ATC->subkernel_num*sizeof(int));
			for (int sk = 0; sk < other_ATC->subkernel_num; sk++)
				Subkernels_per_unit_list[d][sk] = other_ATC->Subkernels_per_unit_list[d][sk];
		}
		subkernel_num = other_ATC->subkernel_num;
	}

	unit_modeler_list = other_ATC->unit_modeler_list;
#ifdef DEBUG
	fprintf(stderr,  "<-----|\n");
#endif
}

void ATC::update_sk_num(long long int subkernel_num_in){
	short lvl = 3;
	#ifdef DEBUG
		fprintf(stderr,  "|-----> ATC::update_sk_num\n");
	#endif
	int prev_sk_num = subkernel_num;
	subkernel_num = subkernel_num_in;
	if (prev_sk_num == -1)  for (int d = 0; d < LOC_NUM; d++){
		Subkernels_per_unit_list[d] = (int*) malloc(subkernel_num*sizeof(int));
		for (int sk = 0; sk < subkernel_num; sk++) Subkernels_per_unit_list[d][sk] = -1;
	}
	else if (prev_sk_num < subkernel_num) for (int d = 0; d < LOC_NUM; d++){
		free(Subkernels_per_unit_list[d]);
		Subkernels_per_unit_list[d] = (int*) malloc(subkernel_num*sizeof(int));
		for (int sk = 0; sk < subkernel_num; sk++) Subkernels_per_unit_list[d][sk] = -1;
	}
#ifdef DEBUG
	fprintf(stderr,  "<-----|\n");
#endif
}

void ATC::distribute_subkernels(int D1GridSz, int D2GridSz, int D3GridSz){
	if (!strcmp(DISTRIBUTION, "ROUND-ROBIN"))
		CoCoDistributeSubkernelsRoundRobin(this);
	else if (!strcmp(DISTRIBUTION, "SPLIT-NAIVE"))
		CoCoDistributeSubkernelsNaive(this);
	else if (!strcmp(DISTRIBUTION, "SPLIT-CHUNKS-ROBIN"))
		CoCoDistributeSubkernelsRoundRobinChunk(this, D3GridSz);
	else if (!strcmp(DISTRIBUTION, "SPLIT-CHUNKS-ROBIN-REVERSE"))
		CoCoDistributeSubkernelsRoundRobinChunkReverse(this, D3GridSz);
	else if (!strcmp(DISTRIBUTION, "2D-BLOCK-CYCLIC"))
		CoCoDistributeSubkernels2DBlockCyclic(this, D1GridSz, D2GridSz, D3GridSz);
	else error("ATC::distribute_subkernels: Unknown Subkernel Distribution %s\n", DISTRIBUTION);
	for (int i = 0; i < active_unit_num; i++)
	if(!Subkernels_per_unit_num[i]) {
		free(Subkernels_per_unit_list[i]);
		for (int i_move = i; i_move < active_unit_num - 1; i_move++){
			active_unit_score[i_move] = active_unit_score[i_move+1];
			active_unit_id_list[i_move] = active_unit_id_list[i_move+1];
			Subkernels_per_unit_num[i_move] = Subkernels_per_unit_num[i_move+1];
			Subkernels_per_unit_list[i_move] = Subkernels_per_unit_list[i_move+1];
		}
		i--;
		active_unit_num--;
	}
#ifdef PDEBUG
    print();
#endif
}
/******************************************************************************/
/****************************** Autotuning ************************************/

void ATC::normalize_split(){
	short lvl = 4;
	double cpu_timer = csecond();
#ifdef DEBUG
	fprintf(stderr,  "|-----> ATC::normalize_split\n");
#endif
	for (int i = 0; i < active_unit_num; i++)
	if(active_unit_score[i] < MINIMUM_UNIT_CONTRIBUTION){
		for (int j = 0; j < active_unit_num; j++)
			if (i != j) active_unit_score[j] = active_unit_score[j]/(1 - active_unit_score[i]);
		active_unit_score[i] = 0;
	}
	else {
	      int flag_normalize[active_unit_num] = {0}, normalize_num = 1;
	      double normalize_sum = active_unit_score[i];
	      flag_normalize[i] = 1;
	      for (int j = i + 1; j < active_unit_num; j++)
				if(abs(active_unit_score[i] - active_unit_score[j])/active_unit_score[i]/active_unit_num < NORMALIZE_NEAR_SPLIT_LIMIT){
		//printf("Normalizing active_unit_score[%d] and active_unit_score[%d] to %lf\n", i, j, (active_unit_score[i] + active_unit_score[j])/2);
					//active_unit_score[j] = active_unit_score[i] = (active_unit_score[i] + active_unit_score[j])/2;
		flag_normalize[j] = 1;
		normalize_sum+=active_unit_score[j];
		normalize_num++;
	      }
	      for (int j = i ; j < active_unit_num; j++) if(flag_normalize[j]) active_unit_score[j] = normalize_sum/normalize_num;
	}
	for (int i = 0; i < active_unit_num; i++)
	if(active_unit_score[i] == 0.0 ) {
		for (int i_move = i; i_move < active_unit_num - 1; i_move++){
			active_unit_score[i_move] = active_unit_score[i_move+1];
			active_unit_id_list[i_move] = active_unit_id_list[i_move+1];
		}
		i--;
		active_unit_num--;
	}
#ifdef DEBUG
	fprintf(stderr,  "<-----|\n");
#endif
}

double ATC::predict_reuse_map(){
		switch(unit_modeler_list[0]->problem){
		case BLAS1:
			return 0;
		case BLAS2:
			return 0;
		case BLAS3:
			return PredictHeteroBestReuseMapBLAS3_v2(unit_modeler_list, T, 
				active_unit_num, active_unit_id_list, active_unit_score);
		default:
			error("PredictHeteroFullOverlap_v2: Invalid Problem %s", printProblem(unit_modeler_list[0]->problem));
	}
	return 0;
}

double ATC::autotune_problem(const char* routine_name, void* initial_problem_wrap){
	short lvl = 3;
	double cpu_timer = csecond();
#ifdef DEBUG
	fprintf(stderr,  "|-----> ATC::autotune_problem(%s, %p)\n", routine_name,
		initial_problem_wrap);
	print();
#endif

	init_modelers(routine_name, initial_problem_wrap);
	linkmap->update_link_weights(unit_modeler_list, T);

	int autotune_eval_devices = 0;
	if (active_unit_num > 0){
		if (active_unit_id_list){
//#ifdef PDEBUG
		fprintf(stderr,  "Running on %d devices with dev_ids=[ ", active_unit_num);
		for (int i =0; i < active_unit_num; i++) fprintf(stderr, "%d ", active_unit_id_list[i]);
		fprintf(stderr, "]\n");
//#endif
		}
		else{
			autotune_eval_devices = 1;
//#ifdef PDEBUG
		fprintf(stderr,  "Running on %d devices with tunable dev_ids\n", active_unit_num);
//#endif
		}
	}
	else{
#ifdef ENABLE_CPU_WORKLOAD
		active_unit_num = LOC_NUM;
#else
		active_unit_num = LOC_NUM - 1;
#endif
		autotune_eval_devices = 1;
		for (int i =0; i < active_unit_num; i++)
			active_unit_id_list[i] = deidxize(i);
	}

	if (autotune_eval_devices){
		ATC_p temp_controller = new ATC();
		temp_controller->mimic_ATC(this);
		int explored_cases = pow(2,LOC_NUM);
		pred_t = pred_J = DBL_MAX;
		int max_unit_num = active_unit_num, initial_T = T;
		double tile_selection_t = 0, split_selection_t = 0;
		for (int case_id = 1; case_id < explored_cases; case_id++){
			translate_binary_to_unit_list(case_id, &temp_controller->active_unit_num, temp_controller->active_unit_id_list);
			if(temp_controller->active_unit_num > max_unit_num) continue;
#ifdef SDEBUG
				fprintf(stderr, "==============================================\n");
				fprintf(stderr, "Autotune devices (iter %d): Tuning for active_unit_id_list = %s\n", case_id,
					printlist<int>(temp_controller->active_unit_id_list, temp_controller->active_unit_num));
#endif
			temp_controller->linkmap->reset();
			temp_controller->linkmap->update_link_shared_weights(temp_controller->unit_modeler_list,
					temp_controller->active_unit_id_list, temp_controller->active_unit_num);
			for(int i = 0; i< LOC_NUM; i++)	for(int j = 0; j< LOC_NUM; j++){
				final_estimated_link_bw[i][j] = temp_controller->linkmap->link_bw_shared[i][j];
				final_link_active[i][j] = temp_controller->linkmap->link_active[i][j];
			}

			if(initial_T <= 0) tile_selection_t += temp_controller->optimize_tile();
			/// Remove case that could not find a proper tile. 
			if(temp_controller->T >= 0) split_selection_t += temp_controller->optimize_split();
			if(initial_T <= 0) tile_selection_t += temp_controller->optimize_tile();
			/// Remove case that could not find a proper tile. 
			if(temp_controller->T < 0){
				temp_controller->pred_t = pred_t*100;
				temp_controller->pred_J = pred_J*100;
				temp_controller->power_delay = power_delay/100; 
				temp_controller->energy_delay = energy_delay/100; 
			}
#ifndef ENABLE_POWA
			if (temp_controller->pred_t +
				temp_controller->pred_t*((temp_controller->active_unit_num-active_unit_num)*MINIMUM_UNIT_CONTRIBUTION) < pred_t) mimic_ATC(temp_controller);
#ifdef SDEBUG
			fprintf(stderr, "] -> pred_t = %lf, best_pred_t = %lf\n", temp_controller->pred_t,  pred_t);
#endif
#else
			if (!strcmp(PREDICT_OPTIMIZE_TARGET,"PERF")){
				if ((temp_controller->pred_t + temp_controller->pred_t*
					((temp_controller->active_unit_num-active_unit_num)*MINIMUM_UNIT_CONTRIBUTION) < pred_t)
					|| ((temp_controller->pred_t + temp_controller->pred_t*
					((temp_controller->active_unit_num-active_unit_num)*MINIMUM_UNIT_CONTRIBUTION) == pred_t) && (
					temp_controller->pred_t_pesimistic + temp_controller->pred_t_pesimistic*
						((temp_controller->active_unit_num-active_unit_num)*MINIMUM_UNIT_CONTRIBUTION) < pred_t_pesimistic
					)))
						mimic_ATC(temp_controller);
#ifdef SDEBUG
				fprintf(stderr, "] -> T = %d, pred_t = %lf, pred_t_pesimistic = %lf, best_pred_t = %lf\n", 
					temp_controller->T, temp_controller->pred_t, temp_controller->pred_t_pesimistic, pred_t);
#endif
			}
			else if(!strcmp(PREDICT_OPTIMIZE_TARGET,"ENERGY")){
				if ((temp_controller->pred_J - temp_controller->pred_J*
					((temp_controller->active_unit_num-active_unit_num)*MINIMUM_UNIT_CONTRIBUTION) < pred_J)
					|| ((temp_controller->pred_J - temp_controller->pred_J*
					((temp_controller->active_unit_num-active_unit_num)*MINIMUM_UNIT_CONTRIBUTION) == pred_J) && (
					temp_controller->pred_J_pesimistic - temp_controller->pred_J_pesimistic*
						((temp_controller->active_unit_num-active_unit_num)*MINIMUM_UNIT_CONTRIBUTION) < pred_J_pesimistic
					)))
						mimic_ATC(temp_controller);
#ifdef SDEBUG
				fprintf(stderr, "] -> T = %d, pred_J = %lf, pred_J_pesimistic = %lf, best_pred_J = %lf\n", 
					temp_controller->T, temp_controller->pred_J, temp_controller->pred_J_pesimistic, pred_J);
#endif
			}
			else if(!strcmp(PREDICT_OPTIMIZE_TARGET,"POWER-DELAY")){
				if ((temp_controller->power_delay - temp_controller->power_delay*
					((temp_controller->active_unit_num-active_unit_num)*MINIMUM_UNIT_CONTRIBUTION) > power_delay)
					|| ((temp_controller->power_delay - temp_controller->power_delay*
					((temp_controller->active_unit_num-active_unit_num)*MINIMUM_UNIT_CONTRIBUTION) == power_delay) && (
					temp_controller->power_delay_pesimistic - temp_controller->power_delay_pesimistic*
						((temp_controller->active_unit_num-active_unit_num)*MINIMUM_UNIT_CONTRIBUTION) > power_delay_pesimistic
					)))
						mimic_ATC(temp_controller);
#ifdef SDEBUG
				fprintf(stderr, "] -> T = %d, power_delay = %e, power_delay_pesimistic = %e, best_power_delay = %e\n", 
					temp_controller->T, temp_controller->power_delay, temp_controller->power_delay_pesimistic, power_delay);
#endif
			}
			else if(!strcmp(PREDICT_OPTIMIZE_TARGET,"ENERGY-DELAY")){
				if ((temp_controller->energy_delay - temp_controller->energy_delay*
					((temp_controller->active_unit_num-active_unit_num)*MINIMUM_UNIT_CONTRIBUTION) > energy_delay)
					|| ((temp_controller->energy_delay - temp_controller->energy_delay*
					((temp_controller->active_unit_num-active_unit_num)*MINIMUM_UNIT_CONTRIBUTION) == energy_delay) && (
					temp_controller->energy_delay_pesimistic - temp_controller->energy_delay_pesimistic*
						((temp_controller->active_unit_num-active_unit_num)*MINIMUM_UNIT_CONTRIBUTION) > energy_delay_pesimistic
					)))
						mimic_ATC(temp_controller);
#ifdef SDEBUG
				fprintf(stderr, "-> T = %d, energy_delay = %e, energy_delay_pesimistic = %e, best_energy_delay = %e\n", 
					temp_controller->T, temp_controller->energy_delay, temp_controller->energy_delay_pesimistic, energy_delay);
#endif
			} // Example for choosing U1(tpred = X, En = J1) vs U2(tpred = Y, En = J2) units with PERPER_LIMIT: if ( X/Y >= PERPER_LIMIT*J2/J1) U2 else U1
			else if(!strcmp(PREDICT_OPTIMIZE_TARGET,"PERF-PER-J")){
				double PER_score = -42;
				if (pred_t == DBL_MAX) mimic_ATC(temp_controller);
				else if (temp_controller->pred_J == pred_J){
					if (temp_controller->pred_t < pred_t) mimic_ATC(temp_controller);
				}
				else if (temp_controller->pred_t == pred_t){
					if (temp_controller->pred_J < pred_J) mimic_ATC(temp_controller);
				}
				else if (temp_controller->pred_t > pred_t && temp_controller->pred_J > pred_J);
				else if (temp_controller->pred_t < pred_t && temp_controller->pred_J < pred_J) mimic_ATC(temp_controller);
				else if (temp_controller->pred_t < pred_t && temp_controller->pred_J > pred_J){
					PER_score = (pred_t/temp_controller->pred_t - 1 )/(temp_controller->pred_J/pred_J - 1);
					if (PER_score >= PERPER_LIMIT) mimic_ATC(temp_controller);
				}
				else if (temp_controller->pred_t > pred_t && temp_controller->pred_J < pred_J){
					PER_score = (temp_controller->pred_t/pred_t - 1 )/(pred_J/temp_controller->pred_J - 1);
					if (PER_score < PERPER_LIMIT) mimic_ATC(temp_controller);
				}
#ifdef SDEBUG
			fprintf(stderr, "] -> t = %lf per J = %lf : PER_Score = %lf, best_t = %lf per J = %lf\n", 
				temp_controller->pred_t, temp_controller->pred_J,  PER_score, pred_t, pred_J);

#endif
			}
			else error("PREDICT_OPTIMIZE_TARGET = %s not implemented\n", PREDICT_OPTIMIZE_TARGET);
#endif
#ifdef SDEBUG
			fprintf(stderr, "-> active_unit_id_list = %s with active_unit_score = %s\n",
				printlist<int>(temp_controller->active_unit_id_list, temp_controller->active_unit_num),
				printlist<double>(temp_controller->active_unit_score, temp_controller->active_unit_num));
			fprintf(stderr, "==============================================\n");
#endif
		}
		if (T < 0) error("ATC::autotune_problem() - returned negative T -> failed to decompose dims\n");
	}
	else{
		int initial_T = T;
		double tile_selection_t = 0, split_selection_t = 0;
		linkmap->update_link_shared_weights(unit_modeler_list,
				active_unit_id_list, active_unit_num);
		for(int i = 0; i< LOC_NUM; i++)	for(int j = 0; j< LOC_NUM; j++){
			final_estimated_link_bw[i][j] = linkmap->link_bw_shared[i][j];
			final_link_active[i][j] = linkmap->link_active[i][j];
		}
		if(initial_T <= 0) tile_selection_t += optimize_tile();
		// TODO: Must decide if workload ratio should be tuned when there is a predefined number of devices... Currently == off for paper
		split_homogeneously = 1;
		split_selection_t += optimize_split();
	}
	for(int i = 0; i< LOC_NUM; i++)	for(int j = 0; j< LOC_NUM; j++){
		final_estimated_link_bw[i][j] = linkmap->link_bw_shared[i][j];
		final_link_active[i][j] = linkmap->link_active[i][j];
	}
	final_estimated_linkmap = linkmap;
#ifdef SDEBUG
	final_estimated_linkmap->print_link_active();
  	final_estimated_linkmap->print_link_bw_shared();
#endif
	MD_p model = unit_modeler_list[idxize(active_unit_id_list[0])];
	update_sk_num(model->getSKNum(T));
	distribute_subkernels(model->D1/T + (model->D1%T ? 1 : 0), model->D2/T  + (model->D2%T ? 1 : 0), model->D3/T  + (model->D3%T ? 1 : 0));

	cpu_timer = csecond() - cpu_timer;
	fprintf(stderr, "====================================\n");
	fprintf(stderr, "ATC::autotune_problem: Autotuning complete-> t_autotune = %lf ms\n", cpu_timer*1000);
	fprintf(stderr, "autotune_controller: T=%ld,  active_unit_num=%d, Problem split = %s -> %s : pred_t = %lf ms, pred_J = %lf kJ\n",
		T, active_unit_num, printlist<int>(active_unit_id_list, active_unit_num),
		printlist<int>(Subkernels_per_unit_num, active_unit_num), pred_t*1000, pred_J/1000);
	fprintf(stderr, "====================================\n");
#ifdef DEBUG
	fprintf(stderr,  "<-----|\n");
#endif
	return cpu_timer;
}

double ATC::optimize_tile(){
	double timer = csecond();
#ifdef DEBUG
fprintf(stderr,  "|-----> ATC::optimize_tile( autotune_controller{ T=%ld, active_unit_num=%d, Problem split = %s -> %s : t_pred = %lf ms}, unit_modeler_list =%p)\n",
	T, active_unit_num, printlist<int>(active_unit_id_list, active_unit_num),
	printlist<double>(active_unit_score, active_unit_num), pred_t*1000, unit_modeler_list);
#endif
	int best_idx = -1;
	double temp_score = 0;

	if (active_unit_num <= 0)
	error("ATC::optimize_tile: Called with active_unit_num = %d\n", active_unit_num);
	// All models are created for the same initial problem, therefore min_T, max_T and dimension related values are the same.
	short first_model_idx = idxize(active_unit_id_list[0]);
	MD_p model = unit_modeler_list[first_model_idx];

/// TODO: Maybe define the lowest suggested Tile sizes?

#define MIN_DESIRED_SK_DEV 64
	long int min_sk = MIN_DESIRED_SK_DEV*active_unit_num, max_sk = 10000;
#ifdef PDEBUG
	fprintf(stderr,  "min_sk = %ld\n", min_sk);
#endif

/// Conditions: 
/// 1-HARD: NO-Wshare: Spliting creates subkernels that can be distributed equally to devices without WR-sharing
/// 2-HARD: SK-num: subkernels per device must be >= MIN_DESIRED_SK_DEV
/// 3-LIGHT: NO-remainder: split should not create remainder Tiles if possible
/// TODO: For future could also take into account specific "good" tiles per device.

	int D1_dummy = model->D1, D2_dummy = (model->D2 == -1)? D1_dummy: model->D2, D3_dummy = (model->D3 == -1)? D1_dummy: model->D3;
	int candidate_T = gcd(D1_dummy, D2_dummy, D3_dummy);
	if(candidate_T == 1) candidate_T = std::min(D1_dummy, std::min(D2_dummy, D3_dummy));
	int ctr = 1;
	while(ctr < candidate_T && ((D1_dummy/(candidate_T/ctr) + ((D1_dummy%(candidate_T/ctr))? 1:0))
	 *(D2_dummy/(candidate_T/ctr) + ((D2_dummy%(candidate_T/ctr))? 1:0)) % active_unit_num
	|| (((D1_dummy%(candidate_T/ctr))? 1:0) + 
		((D2_dummy%(candidate_T/ctr))? 1:0) + 
		((D3_dummy%(candidate_T/ctr))? 1:0))
	|| (candidate_T%ctr)))
		ctr++;
	candidate_T/=ctr; 
#ifdef PDEBUG
	fprintf(stderr,  "Updated candidate_T = %d with ctr = %d for NO-Wshare + NO-remainder conditions\n", 
		candidate_T, ctr);
#endif
	ctr = 1; 
	while(ctr < candidate_T && (model->getSKNum(candidate_T/ctr) < min_sk ||
	((D1_dummy/(candidate_T/ctr) + ((D1_dummy%(candidate_T/ctr))? 1:0))
	*(D2_dummy/(candidate_T/ctr) + ((D2_dummy%(candidate_T/ctr))? 1:0))) % active_unit_num
	|| (((D1_dummy%(candidate_T/ctr))? 1:0) + 
		((D2_dummy%(candidate_T/ctr))? 1:0) + 
		((D3_dummy%(candidate_T/ctr))? 1:0))
	|| (candidate_T%ctr)))
		ctr++;
	candidate_T/=ctr; 
#ifdef PDEBUG
	fprintf(stderr,  "Updated candidate_T = %d with ctr = %d for NO-Wshare + NO-remainder + SK-num conditions\n", 
		candidate_T, ctr);
#endif
	if (model->getSKNum(candidate_T) > max_sk){
#ifdef PDEBUG
		warning("Finding a no-remainder split failed - performance might degrade\n");
#endif
		candidate_T = gcd(D1_dummy, D2_dummy, D3_dummy);
		if(candidate_T == 1) candidate_T = std::min(D1_dummy, std::min(D2_dummy, D3_dummy));
		ctr = 1;
		while(ctr < candidate_T  && (((D1_dummy/(candidate_T/ctr) + ((D1_dummy%(candidate_T/ctr))? 1:0))
		*(D2_dummy/(candidate_T/ctr) + ((D2_dummy%(candidate_T/ctr))? 1:0)))%active_unit_num
		|| (candidate_T%ctr)))
			ctr++;
		//if(!candidate_T%ctr){
			candidate_T/=ctr; 
#ifdef PDEBUG
			fprintf(stderr,  "Updated candidate_T = %d with ctr = %d for NO-Wshare condition\n", candidate_T, ctr);
#endif
			ctr = 1; 
			while(ctr < candidate_T && (model->getSKNum(candidate_T/ctr) < min_sk ||
			((D1_dummy/(candidate_T/ctr) + ((D1_dummy%(candidate_T/ctr))? 1:0))
			*(D2_dummy/(candidate_T/ctr) + ((D2_dummy%(candidate_T/ctr))? 1:0)))%active_unit_num
			|| (candidate_T%ctr)))
				ctr++;
			if(candidate_T/ctr > 1){
				candidate_T/=ctr; 
#ifdef PDEBUG
				fprintf(stderr,  "Updated candidate_T = %d with ctr = %d for NO-Wshare + SK-num conditions\n", candidate_T, ctr);
#endif
			}
			else{
				ctr = 1; 
				while(ctr < candidate_T && 
				(((D1_dummy/(candidate_T/ctr) + ((D1_dummy%(candidate_T/ctr))? 1:0))
				*(D2_dummy/(candidate_T/ctr) + ((D2_dummy%(candidate_T/ctr))? 1:0)))%active_unit_num
				|| (candidate_T%ctr)))
					ctr++;
				candidate_T/=ctr;
#ifdef PDEBUG
				fprintf(stderr,  "Updated candidate_T = %d with ctr = %d for NO-Wshare condition\n", candidate_T, ctr);
#endif 			
			}
		//}
		//else candidate_T = -1;
	}
	T = candidate_T;
	if (candidate_T <= 1 || model->getSKNum(candidate_T) > max_sk){
#ifdef PDEBUG
		warning("Default method for obtaining T failed for active_unit_num = %d\n", active_unit_num);
#endif
		T = -1; 
	}
#ifdef PDEBUG
	fprintf(stderr,  "====================================\n");
	fprintf(stderr,  "Predict T=%ld : No t_pred provided\n", T);
#endif
	timer = csecond() - timer;
#ifdef TEST
	fprintf(stderr,  "Tile selection time:%lf ms\n", timer*1000);
#endif
#ifdef DEBUG
	fprintf(stderr,  "<-----|\n");
#endif
	return timer;
}

//TODO: optimize split is considerably naive, further consideration needed for PARALiA 2.0 in heterogeneous systems.
double ATC::optimize_split(){
	short lvl = 3;
	double timer = csecond();
#ifdef DEBUG
	fprintf(stderr,  "|-----> ATC::optimize_split( autotune_controller{ T=%ld, active_unit_num=%d, Problem split = %s -> %s : t_pred = %lf ms}, unit_modeler_list =%p)\n",
		T, active_unit_num, printlist<int>(active_unit_id_list, active_unit_num),
		printlist<double>(active_unit_score, active_unit_num), pred_t*1000, unit_modeler_list);
#endif

	if (active_unit_num <= 0)
	error("ATC::optimize_split: Called with active_unit_num = %d\n", active_unit_num);
	// All models are created for the same initial problem, therefore min_T, max_T and dimension related values are the same.
	short first_model_idx = idxize(active_unit_id_list[0]);
	MD_p model = unit_modeler_list[first_model_idx];

	long int max_allowed_T = 0, ctr = 0;
	max_allowed_T = model->getMaxT();
#ifdef PDEBUG
		fprintf(stderr,  "max_allowed_T = %ld\n", max_allowed_T);
#endif
	if (T > max_allowed_T)
		error("ATC::optimize_split: Give T = %ld > max_allowed_T = %ld\n", T, max_allowed_T);

	long int sk_num = model->getSKNum(T);

	double min_overlap_t = 10000000, temp_score = 0;

	for(int idx = 0; idx < active_unit_num; idx++)
		active_unit_score[idx] = 1.0/active_unit_num;

	for(int split_itter = 0; split_itter < RATIO_TUNE_ITTER; split_itter++){
		double temp_overlap_t = 0, total_J = 0, temp_overlap_t_pesimistic = 0, total_J_pesimistic = 0;
		double active_unit_score_new[LOC_NUM];
		temp_score = 0;
		for(int idx = 0; idx < active_unit_num; idx++){
			int cur_dev_id = active_unit_id_list[idx], cur_dev_idx = idxize(cur_dev_id);
			model = unit_modeler_list[cur_dev_idx];
			ModelType used_model = NO_OVERLAP;
			switch(model->problem){
				case BLAS1:
					used_model = HETERO_BIDIRECTIONAL;
					break;
				case BLAS2:
					used_model = HETERO_BIDIRECTIONAL;
					break;
				case BLAS3:
					used_model = HETERO_FULL_OVERLAP_v2;
					break;
				default:
					error("ATC::optimize_tileAndSplit:model->problem switch default reached\n");
			}
			double* scores = model->predict_v2(used_model, T, active_unit_num, active_unit_id_list, active_unit_score);
			double extra_reuse_dim_t = predict_reuse_map();
			//double 
			double tmp_score = fmax(scores[0], fmax(scores[1], fmax(scores[2], extra_reuse_dim_t))), 
				tmp_score_pesimistic = scores[0] + scores[1] + scores[2] + extra_reuse_dim_t;
	#ifndef ENABLE_POWA
			double temp_t = tmp_score;
			active_unit_score_new[idx] = temp_t;
			if (active_unit_score_new[idx] != 0) //active_unit_score_new[idx] = active_unit_score[idx]/active_unit_score_new[idx]; this was wrong?
				active_unit_score_new[idx] = 1/active_unit_score_new[idx];
			else warning("ATC::optimize_split: active_unit_score_new[%d] == 0\n", idx);
			temp_score+= active_unit_score_new[idx];
			if(temp_t > 0) temp_overlap_t = fmax(temp_overlap_t, temp_t);
			else error("model->predict(%p(dev_id = %d, (idx = %d )), T = %ld): negative prediction temp_t = %lf\n",
				model, cur_dev_id, cur_dev_idx, T, temp_t);
	#ifdef PDEBUG
			fprintf(stderr,  "model->predict(%p) for dev_id = %d (idx = %d ) with T = %ld: temp_overlap_t = %lf, temp_t = %lf\n",
				model, cur_dev_id, cur_dev_idx, T, temp_overlap_t, temp_t);
	#endif
	#else
			double temp_t = tmp_score, temp_t_pesimistic = tmp_score_pesimistic;
			double temp_J = temp_t*unit_modeler_list[cur_dev_idx]->getGPUexecWatts(),
				temp_J_pesimistic = temp_t_pesimistic*unit_modeler_list[cur_dev_idx]->getGPUexecWatts();
			long int temp_flops = active_unit_score[idx]*model->getFlops();
			double temp_PDP = (temp_flops/temp_t)/unit_modeler_list[cur_dev_idx]->getGPUexecWatts();
			double temp_EDP = (temp_flops/temp_t)*(temp_flops/temp_t)/unit_modeler_list[cur_dev_idx]->getGPUexecWatts();
			
			if(temp_t > 0) temp_overlap_t = fmax(temp_overlap_t, temp_t);
			else error("model->predict(%p(dev_id = %d, (idx = %d )), T = %ld): negative prediction temp_t = %lf\n",
				model, cur_dev_id, cur_dev_idx, T, temp_t);
			if(temp_t_pesimistic > 0) temp_overlap_t_pesimistic = fmax(temp_overlap_t_pesimistic, temp_t_pesimistic);
			else error("model->predict(%p(dev_id = %d, (idx = %d )), T = %ld): negative prediction temp_t_pesimistic = %lf\n",
				model, cur_dev_id, cur_dev_idx, T, temp_t_pesimistic);
			
			total_J += temp_J;
			total_J_pesimistic += temp_J_pesimistic;
	#ifdef PDEBUG
			fprintf(stderr,  "model->predict(%p) for dev_id = %d (idx = %d ) with T = %ld: temp_overlap_t = %lf, temp_t = %lf\
			total_J = %lf, temp_J = %lf\n",
				model, cur_dev_id, cur_dev_idx, T, temp_overlap_t, temp_t, total_J, temp_J);
	#endif
			if (!strcmp(PREDICT_OPTIMIZE_TARGET,"PERF")) active_unit_score_new[idx] = temp_t;
			else if (!strcmp(PREDICT_OPTIMIZE_TARGET,"ENERGY")) active_unit_score_new[idx] = temp_J;
			else if (!strcmp(PREDICT_OPTIMIZE_TARGET,"POWER-DELAY")) active_unit_score_new[idx] = 1/temp_PDP;
			else if (!strcmp(PREDICT_OPTIMIZE_TARGET,"ENERGY-DELAY")) active_unit_score_new[idx] = 1/temp_EDP;
			else if (!strcmp(PREDICT_OPTIMIZE_TARGET,"PERF-PER-J")) active_unit_score_new[idx] = temp_t;
			else error("PREDICT_OPTIMIZE_TARGET = %s not implemented\n", PREDICT_OPTIMIZE_TARGET);

			temp_score+= 1/((active_unit_score[idx]) ? active_unit_score_new[idx]/active_unit_score[idx] : 0);
	#endif
		}
	#ifndef ENABLE_POWA
		pred_t = temp_overlap_t;
	#else
		pred_t = temp_overlap_t;
		pred_J = total_J;
		long int total_flops = model->getFlops();
		power_delay = (total_flops/temp_overlap_t)/(total_J/temp_overlap_t);
		energy_delay = (total_flops/temp_overlap_t)*(total_flops/temp_overlap_t)/(total_J/temp_overlap_t);

		pred_t_pesimistic = temp_overlap_t_pesimistic;
		pred_J_pesimistic = total_J_pesimistic;
		power_delay_pesimistic = (total_flops/temp_overlap_t_pesimistic)/(total_J_pesimistic/temp_overlap_t_pesimistic);
		energy_delay_pesimistic = (total_flops/temp_overlap_t_pesimistic)*(total_flops/temp_overlap_t_pesimistic)
								/(total_J_pesimistic/temp_overlap_t_pesimistic);

	#endif
		for(int idx = 0; idx < active_unit_num; idx++){
			if (split_homogeneously) active_unit_score[idx] = 1.0/active_unit_num;
			else active_unit_score[idx] = 1/((active_unit_score[idx]) ? active_unit_score_new[idx]/active_unit_score[idx] : 0)/temp_score;
	#ifdef PDEBUG
			fprintf(stderr,  "Recalibrating Relative score for unit_id = %d (idx = %d ): active_unit_score = %e\n",
					active_unit_id_list[idx], idx, active_unit_score[idx]);
	#endif
		}

		normalize_split();
		for(int idx = 0; idx < active_unit_num; idx++){;
	#ifdef PDEBUG
			fprintf(stderr,  "Normalized Relative score for unit_id = %d (idx = %d ): active_unit_score = %e\n",
					active_unit_id_list[idx], idx, active_unit_score[idx]);
	#endif
		}
	}
#ifdef PDEBUG
	fprintf(stderr,  "====================================\n");
	fprintf(stderr,  "Best %d percentages : [ ", active_unit_num);
	for (int i =0; i < active_unit_num; i++) fprintf(stderr, "%.5lf ", active_unit_score[i]);
	fprintf(stderr, "]\n");
	fprintf(stderr,  "====================================\n");
#endif
	timer = csecond() - timer;
#ifdef TEST
	fprintf(stderr,  "Optimization time:%lf ms\n", timer*1000);
#endif
#ifdef DEBUG
	fprintf(stderr,  "<-----|\n");
#endif
	return timer;
}
/******************************************************************************/
/**************************** Helper Fuctions *********************************/
void ATC::print(){
	//int dev_ids_token = 0;
	//int ctr = 0, itter = 0;
	//if (active_unit_num > 0) for (int i = 0; i < active_unit_num; i++) dev_ids_token+=pow(10,idxize(active_unit_id_list[i]));
	fprintf(stderr, "Autotune controller:\n->T = %ld\n->active_unit_num = %d\n->active_unit_id_list = %s\n->active_unit_score = %s\
	\n->pred_t = %lf\n->subkernel_num = %ld\n->Subkernels_per_unit_num = %s\n", T, active_unit_num,
	printlist<int>(active_unit_id_list, active_unit_num),
	printlist<double>(active_unit_score, active_unit_num),
	pred_t, subkernel_num,
	printlist<int>(Subkernels_per_unit_num, active_unit_num));
 	if(subkernel_num != -1){
	for (int d = 0; d < active_unit_num; d++) fprintf(stderr, "Subkernels_per_unit_list[%d] = %s\n", d,
		printlist<int>(Subkernels_per_unit_list[d], subkernel_num));
	}
	fprintf(stderr, "\n");
	return;
}

const char* ATC::print_csv(){
	char* outstring = (char*) malloc(256*sizeof(char));
	int dev_ids_token = 0;
	int ctr = 0, itter = 0;
	if (active_unit_num > 0) for (int i = 0; i < active_unit_num; i++) dev_ids_token+=pow(10,idxize(active_unit_id_list[i]));
	sprintf(outstring, "%ld,%d,%d,%lld",  T, active_unit_num, dev_ids_token, cache_limit);
	return outstring;
}

/******************************************************************************/

/* FIXME: DEPRECATED - kept for future model-based T selection update if needed
double ATC::optimize_tile_modelBased(ATC_p autotune_controller, short used_devs, short* used_dev_ids,
	int* dev_idx_ignore, MD_p* unit_modeler_list){
	short lvl = 3;
#ifdef DEBUG
	fprintf(stderr,  "||-----> ATC::optimize_tile_modelBased(used_devs=%d, used_dev_ids= [ ", used_devs);
	for (int i =0; i < used_devs; i++) fprintf(stderr, "%d ", used_dev_ids[i]);
	fprintf(stderr, "]\n");
#endif
#ifdef TEST
	double timer = csecond();
#endif
	short first_model_idx = (used_dev_ids[0] == -1) ? LOC_NUM - 1 : used_dev_ids[0];
	MD_p model = unit_modeler_list[first_model_idx];
	active_unit_score = NULL;
	int best_idx = -1;
	double temp_score = 0;

	long int min_T = 0, max_allowed_T = 0, ctr = 0;
	max_allowed_T = CoCopeLiaMaxT(model);
	min_T = CoCopeLiaMinT(model);
#ifdef PDEBUG
		fprintf(stderr,  "min_T = %ld, max_allowed_T = %ld\n",
			min_T, max_allowed_T);
#endif
	if (min_T > max_allowed_T){
		outparams->T = max_allowed_T;
		// FIXME: Undefined expected performance for tiles < than the smaller microbenchmark
		outparams->pred_t = 0;
#ifdef PDEBUG
		fprintf(stderr,  "min_T = %ld > max_allowed_T = %ld: returning T = %ld",
			min_T, max_allowed_T, max_allowed_T);
#endif
		return outparams;
	}
	double temp_t, min_overlap_t = 10000000;
	long int prev_trial_T = 0;

	int lines = CoCoPeLiaGPUexecGetLines(model);
	for (ctr = 0 ; ctr < lines ; ctr++){
		long int trial_T = CoCoPeLiaGPUexecGetElem(model, ctr);
		if (trial_T > max_allowed_T) break;
		if (trial_T ==  prev_trial_T) continue;

		double temp_overlap_t = 0;
		for(int idx = 0; idx < used_devs; idx++){
			if(dev_idx_ignore[idx]) continue;
			short cur_dev_id = used_dev_ids[idx], cur_dev_idx = (cur_dev_id == -1)? LOC_NUM - 1 : cur_dev_id;
			model = unit_modeler_list[cur_dev_idx];
			ModelType used_model;
			switch(model->problem){
				case BLAS1:
					used_model = COCOPELIA_BIDIRECTIONAL;
					break;
				case BLAS2:
					used_model = COCOPELIA_BIDIRECTIONAL;
					break;
				case BLAS3:
					used_model = COCOPELIA_REUSE;
					break;
				default:
					error("ATC::optimize_tile_modelBased:\
					model->problem switch default reached\n");
			}
				temp_t = CoCoPeLiaModelPredict(model, trial_T, used_model);
				double imb_time_multiplier = 1.0, reduce_time_multiplier = 1.0;
#ifdef TILE_IMBALANCE_PENALTY
					if (model->D1 != -1 && (model->D1%trial_T)) imb_time_multiplier+=TILE_IMBALANCE_PENALTY;
					if (model->D2 != -1 && (model->D2%trial_T)) imb_time_multiplier+=TILE_IMBALANCE_PENALTY;
					if (model->D3 != -1 && (model->D3%trial_T)) imb_time_multiplier+=TILE_IMBALANCE_PENALTY;
#endif
#ifdef REDUCE_PENALTY
					if ((model->D1/trial_T + ((model->D1%trial_T)? 1 : 0))*(model->D2/trial_T + ((model->D2%trial_T)? 1 : 0))
						*(model->D3/trial_T + ((model->D3%trial_T)? 1 : 0)) % used_devs) reduce_time_multiplier+=REDUCE_PENALTY;
#endif
			temp_t *= imb_time_multiplier *reduce_time_multiplier;
			if(temp_t > 0) temp_overlap_t = fmax(temp_overlap_t, temp_t);
			else error("CoCoPeLiaModelPredict(%p(dev_id = %d, (idx = %d )), trial_T = %ld): negative prediction temp_t = %lf\n",
				model, cur_dev_id, cur_dev_idx, trial_T, temp_t);
#ifdef PDEBUG
			fprintf(stderr,  "CoCoPeLiaModelPredict(%p) for dev_id = %d (idx = %d ) with trial_T = %ld: temp_overlap_t = %lf, temp_t = %lf\n",
				model, cur_dev_id, cur_dev_idx, trial_T, temp_overlap_t, temp_t);
#endif
		}
		if (temp_overlap_t < min_overlap_t){
			min_overlap_t = temp_overlap_t;
			min_T = trial_T;
		}
		prev_trial_T = trial_T;
	}
	outparams->T = min_T;
	outparams->pred_t = min_overlap_t;
#ifdef PDEBUG
	fprintf(stderr,  "====================================\n");
	fprintf(stderr,  "Predict T=%ld : t_pred = %lf\n", outparams->T, outparams->pred_t);
#endif
#ifdef TEST
	timer = csecond() - timer;
	fprintf(stderr,  "Tile selection time:%lf ms\n", timer*1000);
	fprintf(stderr,  "<-----|\n");
#endif
#ifdef DEBUG
	fprintf(stderr,  "outparams->T = %ld\n : outparams->pred_t = %lf ms\n", outparams->T, outparams->pred_t);
	fprintf(stderr,  "<-----|\n");
#endif
	return outparams;
}
*/
